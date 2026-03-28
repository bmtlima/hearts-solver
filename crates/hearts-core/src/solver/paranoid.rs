use std::cell::RefCell;

use crate::game_state::GameState;
use crate::solver::transposition::ZobristKeys;
use crate::trick::PlayerIndex;

/// Paranoid solver for Hearts.
///
/// Treats the game as 2-player zero-sum: the AI (specified player) minimizes
/// its own score, while all 3 opponents form a coalition that maximizes
/// the AI's score with shared perfect information.
///
/// This enables standard alpha-beta pruning with full α/β bounds, giving
/// massive speedup over Max^n. The tradeoff: paranoid search is pessimistic —
/// it assumes opponents coordinate perfectly against you.
///
/// Returns the AI player's score under optimal paranoid play.
pub fn paranoid_solve(state: &mut GameState, ai_player: PlayerIndex) -> i32 {
    paranoid_recursive(state, ai_player, i32::MIN + 1, i32::MAX, None)
}

/// Paranoid solve with transposition table.
pub fn paranoid_solve_with_tt(
    state: &mut GameState,
    ai_player: PlayerIndex,
    tt: &mut ParanoidTT,
    keys: &ZobristKeys,
) -> i32 {
    let ctx = TTContext {
        tt: RefCell::new(tt),
        keys,
    };
    paranoid_recursive(state, ai_player, i32::MIN + 1, i32::MAX, Some(&ctx))
}

// ── Paranoid-specific transposition table with bound types ──────────────

/// Bound type for alpha-beta TT entries.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    /// The stored value is a lower bound (true value >= stored).
    Lower,
    /// The stored value is an upper bound (true value <= stored).
    Upper,
}

#[derive(Clone)]
struct ParanoidTTEntry {
    hash: u64,
    score: i32,
    bound: Bound,
}

/// A fixed-size transposition table for paranoid (alpha-beta) search.
/// Stores bound type alongside the score so that entries from pruned
/// subtrees are handled correctly on re-probe.
pub struct ParanoidTT {
    entries: Vec<Option<ParanoidTTEntry>>,
    mask: usize,
}

impl ParanoidTT {
    /// Create a table with 2^size_bits entries.
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

    fn store(&mut self, hash: u64, score: i32, bound: Bound) {
        let idx = hash as usize & self.mask;
        self.entries[idx] = Some(ParanoidTTEntry { hash, score, bound });
    }

    pub fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = None;
        }
    }
}

struct TTContext<'a> {
    tt: RefCell<&'a mut ParanoidTT>,
    keys: &'a ZobristKeys,
}

fn paranoid_recursive(
    state: &mut GameState,
    ai_player: PlayerIndex,
    mut alpha: i32,
    mut beta: i32,
    ctx: Option<&TTContext>,
) -> i32 {
    if state.is_game_over() {
        return state.final_scores()[ai_player.index()];
    }

    // TT probe
    let hash = ctx.map(|c| {
        c.keys.hash_position(
            &state.hands,
            state.current_player,
            state.current_trick.played_cards(),
        )
    });

    if let (Some(h), Some(c)) = (hash, ctx) {
        if let Some(entry) = c.tt.borrow().probe(h) {
            if entry.bound == Bound::Exact {
                return entry.score;
            }
        }
    }

    let legal = state.legal_moves();
    let is_ai_turn = state.current_player == ai_player;

    let orig_alpha = alpha;
    let orig_beta = beta;

    if is_ai_turn {
        // AI minimizes its own score (MIN node).
        // At MIN nodes only beta is tightened; alpha stays unchanged.
        let mut best = i32::MAX;
        for card in legal.cards() {
            let undo = state.play_card_with_undo(card);
            let score = paranoid_recursive(state, ai_player, alpha, beta, ctx);
            state.undo_card(&undo);

            if score < best {
                best = score;
            }
            if best < beta {
                beta = best;
            }
            if alpha >= beta {
                break;
            }
        }

        // TT store: MIN node bound classification
        //   - best <= orig_alpha → cutoff by ancestor MAX; UPPER bound
        //   - best >= orig_beta  → no move lowered beta; LOWER bound
        //   - otherwise          → EXACT
        if let (Some(h), Some(c)) = (hash, ctx) {
            let bound = if best <= orig_alpha {
                Bound::Upper
            } else if best >= orig_beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            c.tt.borrow_mut().store(h, best, bound);
        }

        best
    } else {
        // Opponent coalition maximizes AI's score (MAX node).
        // At MAX nodes only alpha is tightened; beta stays unchanged.
        let mut best = i32::MIN;
        for card in legal.cards() {
            let undo = state.play_card_with_undo(card);
            let score = paranoid_recursive(state, ai_player, alpha, beta, ctx);
            state.undo_card(&undo);

            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }

        // TT store: MAX node bound classification
        //   - best <= orig_alpha → no move beat alpha; UPPER bound
        //   - best >= orig_beta  → cutoff by ancestor MIN; LOWER bound
        //   - otherwise          → EXACT
        if let (Some(h), Some(c)) = (hash, ctx) {
            let bound = if best <= orig_alpha {
                Bound::Upper
            } else if best >= orig_beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            c.tt.borrow_mut().store(h, best, bound);
        }

        best
    }
}

/// Find the best move for the AI player using paranoid search.
pub fn paranoid_best_move(
    state: &mut GameState,
    ai_player: PlayerIndex,
) -> (crate::types::Card, i32) {
    let legal = state.legal_moves();
    let mut best_card = None;
    let mut best_score = i32::MAX;

    for card in legal.cards() {
        let undo = state.play_card_with_undo(card);
        let score = paranoid_solve(state, ai_player);
        state.undo_card(&undo);

        if score < best_score {
            best_score = score;
            best_card = Some(card);
        }
    }

    (best_card.unwrap(), best_score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::DeckConfig;
    use crate::solver::maxn;
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
        // Paranoid score >= Max^n score for the AI player
        // (paranoid assumes worst-case opponent coordination)
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
    fn paranoid_with_tt_matches() {
        use crate::solver::transposition::ZobristKeys;
        let keys = ZobristKeys::new(42);
        let mut tt = ParanoidTT::new(16);

        for seed in 0..50 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);

            let mut state1 = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let ai = state1.current_player;
            let score_no_tt = paranoid_solve(&mut state1, ai);

            tt.clear();
            let mut state2 = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let score_tt = paranoid_solve_with_tt(&mut state2, ai, &mut tt, &keys);

            assert_eq!(
                score_no_tt, score_tt,
                "seed {}: paranoid no_tt={} != tt={}",
                seed, score_no_tt, score_tt
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
}
