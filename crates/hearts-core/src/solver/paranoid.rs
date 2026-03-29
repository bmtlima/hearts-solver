use std::sync::OnceLock;

use crate::game_state::GameState;
use crate::solver::transposition::ZobristKeys;
use crate::trick::PlayerIndex;
use crate::types::{Card, Rank, Suit};

// ── Global ZobristKeys (computed once per process) ───────────────────────

static ZOBRIST_KEYS: OnceLock<ZobristKeys> = OnceLock::new();

fn global_keys() -> &'static ZobristKeys {
    ZOBRIST_KEYS.get_or_init(|| ZobristKeys::new(0xDEAD))
}

// ── Public API ───────────────────────────────────────────────────────────

/// Paranoid solver for Hearts.
///
/// Treats the game as 2-player zero-sum: the AI (specified player) minimizes
/// its own score, while all 3 opponents form a coalition that maximizes
/// the AI's score with shared perfect information.
///
/// Uses a transposition table internally for memoization.
pub fn paranoid_solve(state: &mut GameState, ai_player: PlayerIndex) -> i32 {
    let keys = global_keys();
    let mut tt = ParanoidTT::new(20);
    let hands_hash = keys.hash_hands(&state.hands);
    paranoid_recursive(state, ai_player, i32::MIN + 1, i32::MAX, hands_hash, &mut tt, keys)
}

/// Paranoid solve with an externally-provided transposition table.
pub fn paranoid_solve_with_tt(
    state: &mut GameState,
    ai_player: PlayerIndex,
    tt: &mut ParanoidTT,
    keys: &ZobristKeys,
) -> i32 {
    let hands_hash = keys.hash_hands(&state.hands);
    paranoid_recursive(state, ai_player, i32::MIN + 1, i32::MAX, hands_hash, tt, keys)
}

/// Find the best move for the AI player using paranoid search.
/// Shares a single TT across all move evaluations for better memoization.
pub fn paranoid_best_move(
    state: &mut GameState,
    ai_player: PlayerIndex,
) -> (Card, i32) {
    let keys = global_keys();
    let mut tt = ParanoidTT::new(20);
    let hands_hash = keys.hash_hands(&state.hands);
    let legal = state.legal_moves();
    let mut best_card = None;
    let mut best_score = i32::MAX;

    for card in legal.cards() {
        let player_idx = state.current_player.index();
        let card_key = keys.card_keys[player_idx][ZobristKeys::card_index(card)];
        let new_hands_hash = hands_hash ^ card_key;

        let undo = state.play_card_with_undo(card);
        let score = paranoid_recursive(
            state, ai_player, i32::MIN + 1, i32::MAX, new_hands_hash, &mut tt, keys,
        );
        state.undo_card(&undo);

        if score < best_score {
            best_score = score;
            best_card = Some(card);
        }
    }

    (best_card.unwrap(), best_score)
}

// ── Paranoid TT with bound types and best-move storage ───────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone)]
struct ParanoidTTEntry {
    hash: u64,
    score: i32,
    bound: Bound,
    best_move: Option<Card>,
}

pub struct ParanoidTT {
    entries: Vec<Option<ParanoidTTEntry>>,
    mask: usize,
}

impl ParanoidTT {
    pub fn new(size_bits: u32) -> Self {
        let size = 1usize << size_bits;
        ParanoidTT {
            entries: vec![None; size],
            mask: size - 1,
        }
    }

    fn probe(&self, hash: u64) -> Option<&ParanoidTTEntry> {
        let idx = hash as usize & self.mask;
        self.entries[idx].as_ref().filter(|e| e.hash == hash)
    }

    fn store(&mut self, hash: u64, score: i32, bound: Bound, best_move: Option<Card>) {
        let idx = hash as usize & self.mask;
        self.entries[idx] = Some(ParanoidTTEntry {
            hash,
            score,
            bound,
            best_move,
        });
    }

    pub fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = None;
        }
    }
}

// ── Core recursive search ────────────────────────────────────────────────

fn paranoid_recursive(
    state: &mut GameState,
    ai_player: PlayerIndex,
    mut alpha: i32,
    mut beta: i32,
    hands_hash: u64,
    tt: &mut ParanoidTT,
    keys: &ZobristKeys,
) -> i32 {
    if state.is_game_over() {
        return state.final_scores()[ai_player.index()];
    }

    // Compute full hash: incremental hands_hash + context from scratch
    let hash = hands_hash
        ^ keys.hash_context(
            state.current_player,
            state.current_trick.played_cards(),
            &state.points_taken,
        );

    // TT probe
    let mut tt_best_move: Option<Card> = None;

    if let Some(entry) = tt.probe(hash) {
        tt_best_move = entry.best_move;
        match entry.bound {
            Bound::Exact => return entry.score,
            Bound::Lower => {
                if entry.score >= beta {
                    return entry.score;
                }
                if entry.score > alpha {
                    alpha = entry.score;
                }
            }
            Bound::Upper => {
                if entry.score <= alpha {
                    return entry.score;
                }
                if entry.score < beta {
                    beta = entry.score;
                }
            }
        }
    }

    let legal = state.legal_moves();
    let is_ai_turn = state.current_player == ai_player;

    // Move ordering: stack-allocated array, heuristic sort, TT best move first
    let mut moves = [Card::new(Suit::Clubs, Rank::Two); 13];
    let mut n_moves = 0;
    for card in legal.cards() {
        moves[n_moves] = card;
        n_moves += 1;
    }
    let moves = &mut moves[..n_moves];
    order_moves(moves, state);
    if let Some(tm) = tt_best_move {
        if let Some(pos) = moves.iter().position(|&m| m == tm) {
            moves.swap(0, pos);
        }
    }

    let orig_alpha = alpha;
    let orig_beta = beta;

    if is_ai_turn {
        // AI minimizes its own score (MIN node).
        let mut best = i32::MAX;
        let mut best_card: Option<Card> = None;
        for card in moves.iter() {
            // Compute new hands_hash BEFORE play (need current player index)
            let player_idx = state.current_player.index();
            let card_key = keys.card_keys[player_idx][ZobristKeys::card_index(*card)];
            let new_hands_hash = hands_hash ^ card_key;

            let undo = state.play_card_with_undo(*card);
            let score = paranoid_recursive(state, ai_player, alpha, beta, new_hands_hash, tt, keys);
            state.undo_card(&undo);

            if score < best {
                best = score;
                best_card = Some(*card);
            }
            if best < beta {
                beta = best;
            }
            if alpha >= beta {
                break;
            }
        }

        let bound = if best <= orig_alpha {
            Bound::Upper
        } else if best >= orig_beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        tt.store(hash, best, bound, best_card);

        best
    } else {
        // Opponent coalition maximizes AI's score (MAX node).
        let mut best = i32::MIN;
        let mut best_card: Option<Card> = None;
        for card in moves.iter() {
            let player_idx = state.current_player.index();
            let card_key = keys.card_keys[player_idx][ZobristKeys::card_index(*card)];
            let new_hands_hash = hands_hash ^ card_key;

            let undo = state.play_card_with_undo(*card);
            let score = paranoid_recursive(state, ai_player, alpha, beta, new_hands_hash, tt, keys);
            state.undo_card(&undo);

            if score > best {
                best = score;
                best_card = Some(*card);
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }

        let bound = if best <= orig_alpha {
            Bound::Upper
        } else if best >= orig_beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        tt.store(hash, best, bound, best_card);

        best
    }
}

/// Heuristic move ordering for better alpha-beta pruning.
fn order_moves(moves: &mut [Card], state: &GameState) {
    let is_leading = state.current_trick.is_empty();
    let led_suit = state.current_trick.led_suit();

    moves.sort_by_key(|card| {
        if is_leading {
            let suit_penalty = if card.suit() == Suit::Hearts { 100 } else { 0 };
            suit_penalty + card.rank() as i32
        } else if let Some(led) = led_suit {
            if card.suit() == led {
                card.rank() as i32
            } else {
                if card.suit() == Suit::Spades && card.rank() == Rank::Queen {
                    -100
                } else if card.suit() == Suit::Hearts {
                    -(card.rank() as i32)
                } else {
                    50 + card.rank() as i32
                }
            }
        } else {
            0
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::DeckConfig;
    use crate::solver::{brute_force, maxn};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn valid_scores_on_tiny() {
        let dp = DeckConfig::Tiny.total_points();
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);
            let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let ai = state.current_player;
            let score = paranoid_solve(&mut state, ai);
            assert!(
                score >= 0 && score <= dp,
                "seed {}: paranoid score {} out of range [0, {}]",
                seed, score, dp
            );
        }
    }

    #[test]
    fn paranoid_is_pessimistic_vs_maxn() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);

            let mut maxn_state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let ai = maxn_state.current_player;
            let maxn_scores = maxn::maxn_solve(&mut maxn_state);
            let maxn_ai_score = maxn_scores[ai.index()];

            let mut par_state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let par_score = paranoid_solve(&mut par_state, ai);

            assert!(
                par_score >= maxn_ai_score,
                "seed {}: paranoid {} < maxn {} for P{} (should be >=)",
                seed, par_score, maxn_ai_score, ai.index()
            );
        }
    }

    #[test]
    fn paranoid_matches_brute_force_pessimism_on_tiny() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);

            let bf_state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let bf_scores = brute_force::brute_force_solve(&bf_state);

            let mut par_state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let ai = par_state.current_player;
            let par_score = paranoid_solve(&mut par_state, ai);

            assert!(
                par_score >= bf_scores[ai.index()],
                "seed {}: paranoid {} < brute_force {} for P{} (should be >=)",
                seed, par_score, bf_scores[ai.index()], ai.index()
            );
        }
    }

    #[test]
    fn paranoid_with_external_tt_matches() {
        let keys = ZobristKeys::new(42);
        let mut tt = ParanoidTT::new(16);

        for seed in 0..50 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);

            let mut state1 = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let ai = state1.current_player;
            let score_default = paranoid_solve(&mut state1, ai);

            tt.clear();
            let mut state2 = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let score_ext_tt = paranoid_solve_with_tt(&mut state2, ai, &mut tt, &keys);

            assert_eq!(
                score_default, score_ext_tt,
                "seed {}: default={} != external_tt={}",
                seed, score_default, score_ext_tt
            );
        }
    }

    #[test]
    fn paranoid_faster_than_maxn_on_small() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Small.deal(&mut rng);

        let start_maxn = std::time::Instant::now();
        let mut state1 = GameState::new_with_deal(hands.clone(), DeckConfig::Small);
        let _ = maxn::maxn_solve(&mut state1);
        let maxn_time = start_maxn.elapsed();

        let start_par = std::time::Instant::now();
        let mut state2 = GameState::new_with_deal(hands, DeckConfig::Small);
        let ai = state2.current_player;
        let _ = paranoid_solve(&mut state2, ai);
        let par_time = start_par.elapsed();

        println!("Small: Max^n {:?}, Paranoid {:?}", maxn_time, par_time);
    }

    #[test]
    fn paranoid_handles_moon() {
        let dp = DeckConfig::Tiny.total_points();
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);
            let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let ai = state.current_player;
            let score = paranoid_solve(&mut state, ai);
            assert!(
                score >= 0 && score <= dp,
                "seed {}: paranoid score {} out of range [0, {}]",
                seed, score, dp
            );
        }
    }

    #[test]
    fn paranoid_medium_timing() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Medium.deal(&mut rng);

        let start = std::time::Instant::now();
        let mut state = GameState::new_with_deal(hands, DeckConfig::Medium);
        let ai = state.current_player;
        let score = paranoid_solve(&mut state, ai);
        let elapsed = start.elapsed();

        println!("Medium paranoid: score={}, time={:?}", score, elapsed);
    }

    #[test]
    fn medium_solve_by_trick() {
        // Profile solve time at each trick start on Medium deck
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Medium.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Medium);

        let mut trick = 0;
        let mut card_in_trick = 0;

        while !state.is_game_over() {
            let legal = state.legal_moves();
            let cards_remaining: u32 = state.hands.iter().map(|h| h.count()).sum();

            if card_in_trick == 0 {
                let ai = state.current_player;
                let start = std::time::Instant::now();
                let mut s = state.clone();
                let score = paranoid_solve(&mut s, ai);
                let elapsed = start.elapsed();
                println!(
                    "trick={} cards_left={} legal={} score={} time={:?}",
                    trick, cards_remaining, legal.count(), score, elapsed
                );
            }

            let card = legal.cards().next().unwrap();
            state.play_card(card);
            card_in_trick += 1;
            if card_in_trick == 4 {
                card_in_trick = 0;
                trick += 1;
            }
        }
    }

    #[test]
    fn medium_pimc_by_trick() {
        // Measure actual PIMC decision time at each trick on Medium
        use crate::search::pimc;

        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Medium.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Medium);

        let mut trick = 0;
        let mut card_in_trick = 0;

        while !state.is_game_over() {
            let legal = state.legal_moves();
            let cards_remaining: u32 = state.hands.iter().map(|h| h.count()).sum();

            if card_in_trick == 0 && legal.count() > 1 {
                let player = state.current_player;
                let n_worlds = 30;

                let start = std::time::Instant::now();
                let mut pimc_rng = StdRng::seed_from_u64(trick as u64 * 1000);
                let card = pimc::pimc_choose(
                    &state, player, n_worlds,
                    pimc::SolverType::Paranoid, &mut pimc_rng,
                );
                let elapsed = start.elapsed();

                println!(
                    "trick={} cards_left={} legal={} chose={} pimc_n={} time={:?}",
                    trick, cards_remaining, legal.count(), card, n_worlds, elapsed
                );

                state.play_card(card);
            } else {
                let card = legal.cards().next().unwrap();
                state.play_card(card);
            }

            card_in_trick += 1;
            if card_in_trick == 4 {
                card_in_trick = 0;
                trick += 1;
            }
        }

        println!("Final scores: {:?}", state.final_scores());
    }

    #[test]
    fn full_solve_by_trick() {
        // Profile paranoid solve time at each trick on Full deck
        // to find the viable DD cutoff point
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Full.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Full);

        let mut trick = 0;
        let mut card_in_trick = 0;

        println!("trick | cards_left | legal | solve_time");
        println!("------|------------|-------|----------");

        while !state.is_game_over() {
            let legal = state.legal_moves();
            let cards_remaining: u32 = state.hands.iter().map(|h| h.count()).sum();

            if card_in_trick == 0 {
                // Only attempt solve if cards_remaining <= 28 (otherwise too slow)
                if cards_remaining <= 32 {
                    let ai = state.current_player;
                    let start = std::time::Instant::now();
                    let mut s = state.clone();
                    let score = paranoid_solve(&mut s, ai);
                    let elapsed = start.elapsed();
                    println!(
                        "  {:>2}  |     {:>2}     |   {:>2}  | {:?}",
                        trick, cards_remaining, legal.count(), elapsed
                    );
                } else {
                    println!(
                        "  {:>2}  |     {:>2}     |   {:>2}  | SKIPPED (too slow)",
                        trick, cards_remaining, legal.count()
                    );
                }
            }

            let card = legal.cards().next().unwrap();
            state.play_card(card);
            card_in_trick += 1;
            if card_in_trick == 4 {
                card_in_trick = 0;
                trick += 1;
            }
        }
    }
}
