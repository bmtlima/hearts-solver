use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;

use crate::belief::observation::Observation;
use crate::belief::sampler::sample_worlds;
use crate::card_set::CardSet;
use crate::game::Player;
use crate::game_state::GameState;
use crate::solver::paranoid::ParanoidTT;
use crate::solver::transposition::{TranspositionTable, ZobristKeys};
use crate::solver::{maxn, paranoid};
use crate::trick::PlayerIndex;
use crate::types::Card;

/// Which solver to use for double-dummy evaluation.
#[derive(Clone, Copy, Debug)]
pub enum SolverType {
    MaxN,
    Paranoid,
    /// Paranoid with regression eval cutoff at the given card count.
    ParanoidDepthLimited(u32),
}

/// Choose the best card via PIMC — sequential version.
pub fn pimc_choose(
    state: &GameState,
    player: PlayerIndex,
    n_worlds: usize,
    solver: SolverType,
    rng: &mut impl Rng,
) -> Card {
    let obs = Observation::from_game_state(state, player);
    let legal = state.legal_moves();
    let moves: Vec<Card> = legal.cards().collect();

    if moves.len() == 1 {
        return moves[0];
    }

    // Sample worlds sequentially (deterministic with seed)
    let worlds = sample_worlds(&obs, n_worlds, rng);
    if worlds.is_empty() {
        return moves[0];
    }

    let total_scores = score_moves_sequential(state, player, &moves, &worlds, solver);
    pick_best_move(&moves, &total_scores)
}

/// Choose the best card via PIMC — parallel version using rayon.
///
/// Worlds are sampled sequentially (deterministic), then evaluated in parallel.
pub fn pimc_choose_parallel(
    state: &GameState,
    player: PlayerIndex,
    n_worlds: usize,
    solver: SolverType,
    rng: &mut impl Rng,
) -> Card {
    let obs = Observation::from_game_state(state, player);
    let legal = state.legal_moves();
    let moves: Vec<Card> = legal.cards().collect();

    if moves.len() == 1 {
        return moves[0];
    }

    // Sample worlds sequentially (deterministic)
    let worlds = sample_worlds(&obs, n_worlds, rng);
    if worlds.is_empty() {
        return moves[0];
    }

    let total_scores = score_moves_parallel(state, player, &moves, &worlds, solver);
    pick_best_move(&moves, &total_scores)
}

/// Score all moves across all worlds — sequential with early termination.
fn score_moves_sequential(
    state: &GameState,
    player: PlayerIndex,
    moves: &[Card],
    worlds: &[[CardSet; 4]],
    solver: SolverType,
) -> Vec<f64> {
    let mut total_scores = vec![0.0f64; moves.len()];
    let total_points = state.deck_config.total_points() as f64;

    for (wi, world) in worlds.iter().enumerate() {
        for (mi, &mov) in moves.iter().enumerate() {
            let score = evaluate_move(state, player, mov, world, solver);
            total_scores[mi] += score as f64;
        }

        // Early termination: after half the worlds, check if one move dominates
        if wi == worlds.len() / 2 && worlds.len() >= 4 && moves.len() > 1 {
            let mut sorted: Vec<f64> = total_scores.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let gap = sorted[1] - sorted[0];
            let remaining = (worlds.len() - wi - 1) as f64;
            // Worst case: best move scores max points in all remaining worlds,
            // second-best scores 0. If the gap exceeds that swing, best is locked in.
            if gap > remaining * total_points {
                break;
            }
        }
    }
    total_scores
}

/// Score all moves across all worlds — parallel over worlds with per-thread TT.
fn score_moves_parallel(
    state: &GameState,
    player: PlayerIndex,
    moves: &[Card],
    worlds: &[[CardSet; 4]],
    solver: SolverType,
) -> Vec<f64> {
    let n_moves = moves.len();

    // Each world produces a Vec<i32> of scores per move.
    // Each thread gets its own TT to avoid contention.
    let per_world_scores: Vec<Vec<i32>> = worlds
        .par_iter()
        .map(|world| {
            moves
                .iter()
                .map(|&mov| evaluate_move_with_tt(state, player, mov, world, solver))
                .collect()
        })
        .collect();

    // Aggregate
    let mut total_scores = vec![0.0f64; n_moves];
    for scores in &per_world_scores {
        for (mi, &s) in scores.iter().enumerate() {
            total_scores[mi] += s as f64;
        }
    }
    total_scores
}

/// Pick the move with the lowest total score.
fn pick_best_move(moves: &[Card], total_scores: &[f64]) -> Card {
    let (best_idx, _) = total_scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    moves[best_idx]
}

/// Evaluate a single move in a single sampled world.
pub fn evaluate_move(
    state: &GameState,
    player: PlayerIndex,
    mov: Card,
    world: &[CardSet; 4],
    solver: SolverType,
) -> i32 {
    let mut eval_state = state.clone();
    eval_state.hands = *world;
    // Ensure AI always uses its actual hand, not the sampled version
    eval_state.hands[player.index()] = state.hands[player.index()];
    eval_state.play_card(mov);

    match solver {
        SolverType::Paranoid => paranoid::paranoid_solve(&mut eval_state, player),
        SolverType::MaxN => {
            let scores = maxn::maxn_solve(&mut eval_state);
            scores[player.index()]
        }
        SolverType::ParanoidDepthLimited(cutoff) => {
            paranoid::paranoid_solve_depth_limited(&mut eval_state, player, cutoff) as i32
        }
    }
}

/// Evaluate a single move with a thread-local transposition table.
fn evaluate_move_with_tt(
    state: &GameState,
    player: PlayerIndex,
    mov: Card,
    world: &[CardSet; 4],
    solver: SolverType,
) -> i32 {
    let mut eval_state = state.clone();
    eval_state.hands = *world;
    eval_state.hands[player.index()] = state.hands[player.index()];
    eval_state.play_card(mov);

    match solver {
        SolverType::Paranoid => {
            let keys = ZobristKeys::new(0xCAFE);
            let mut tt = ParanoidTT::new(16);
            paranoid::paranoid_solve_with_tt(&mut eval_state, player, &mut tt, &keys)
        }
        SolverType::MaxN => {
            let keys = ZobristKeys::new(0xCAFE);
            let mut tt = TranspositionTable::new(16);
            let scores = maxn::maxn_solve_with_tt(&mut eval_state, &mut tt, &keys);
            scores[player.index()]
        }
        SolverType::ParanoidDepthLimited(cutoff) => {
            paranoid::paranoid_solve_depth_limited(&mut eval_state, player, cutoff) as i32
        }
    }
}

/// PIMC bot implementing the Player trait.
/// Uses parallel evaluation when `parallel` is true.
pub struct PIMCBot {
    pub n_worlds: usize,
    pub solver: SolverType,
    pub parallel: bool,
    rng: StdRng,
}

impl PIMCBot {
    pub fn new(n_worlds: usize, solver: SolverType, rng: StdRng) -> Self {
        PIMCBot {
            n_worlds,
            solver,
            parallel: false,
            rng,
        }
    }

    pub fn new_parallel(n_worlds: usize, solver: SolverType, rng: StdRng) -> Self {
        PIMCBot {
            n_worlds,
            solver,
            parallel: true,
            rng,
        }
    }
}

impl Player for PIMCBot {
    fn choose_card(&mut self, state: &GameState, legal_moves: CardSet) -> Card {
        if legal_moves.count() == 1 {
            return legal_moves.cards().next().unwrap();
        }
        if self.parallel {
            pimc_choose_parallel(
                state,
                state.current_player,
                self.n_worlds,
                self.solver,
                &mut self.rng,
            )
        } else {
            pimc_choose(
                state,
                state.current_player,
                self.n_worlds,
                self.solver,
                &mut self.rng,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_set::CardSet;
    use crate::deck::DeckConfig;
    use crate::game::GameRunner;
    use crate::bots::random_bot::RandomBot;
    use crate::bots::rule_bot::RuleBot;
    use crate::trick::PlayerIndex;
    use crate::types::{Card, Rank, Suit};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn pimc_finds_obvious_dump() {
        let hands = [
            CardSet::from_cards([
                Card::new(Suit::Spades, Rank::Queen),
                Card::new(Suit::Diamonds, Rank::King),
                Card::new(Suit::Hearts, Rank::Jack),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::King),
                Card::new(Suit::Spades, Rank::King),
                Card::new(Suit::Hearts, Rank::Queen),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Ace),
                Card::new(Suit::Diamonds, Rank::Ace),
                Card::new(Suit::Hearts, Rank::King),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Queen),
                Card::new(Suit::Spades, Rank::Ace),
                Card::new(Suit::Hearts, Rank::Ace),
            ]),
        ];

        let mut state = GameState::new(hands, DeckConfig::Tiny, PlayerIndex::P3);
        state.is_first_trick = false;
        state.play_card(Card::new(Suit::Clubs, Rank::Queen));

        let mut rng = StdRng::seed_from_u64(42);
        let card = pimc_choose(&state, PlayerIndex::P0, 50, SolverType::Paranoid, &mut rng);
        assert_eq!(card, Card::new(Suit::Spades, Rank::Queen));
    }

    #[test]
    fn pimc_beats_random_bot() {
        let mut pimc_total = 0i64;
        let mut rand_total = 0i64;
        let games = 500;

        for seed in 0..games {
            let mut rng = StdRng::seed_from_u64(seed);
            let players: [Box<dyn Player>; 4] = [
                Box::new(PIMCBot::new(30, SolverType::Paranoid, StdRng::seed_from_u64(seed * 10))),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 1))),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 2))),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 3))),
            ];
            let mut runner = GameRunner::new_with_deal(DeckConfig::Tiny, players, &mut rng);
            let scores = runner.play_game();
            pimc_total += scores[0] as i64;
            rand_total += (scores[1] + scores[2] + scores[3]) as i64;
        }

        let pimc_avg = pimc_total as f64 / games as f64;
        let rand_avg = rand_total as f64 / (games as f64 * 3.0);

        assert!(
            pimc_avg < rand_avg,
            "PIMC avg {:.2} should be less than Random avg {:.2}",
            pimc_avg, rand_avg
        );
    }

    #[test]
    fn pimc_competitive_with_rule_bot() {
        let mut pimc_total = 0i64;
        let mut rule_total = 0i64;
        let games = 200;

        for seed in 0..games {
            let mut rng = StdRng::seed_from_u64(seed);
            let players: [Box<dyn Player>; 4] = [
                Box::new(PIMCBot::new(30, SolverType::Paranoid, StdRng::seed_from_u64(seed * 10))),
                Box::new(RuleBot::new()),
                Box::new(RuleBot::new()),
                Box::new(RuleBot::new()),
            ];
            let mut runner = GameRunner::new_with_deal(DeckConfig::Tiny, players, &mut rng);
            let scores = runner.play_game();
            pimc_total += scores[0] as i64;
            rule_total += (scores[1] + scores[2] + scores[3]) as i64;
        }

        let pimc_avg = pimc_total as f64 / games as f64;
        let rule_avg = rule_total as f64 / (games as f64 * 3.0);

        println!("PIMC avg: {:.2}, RuleBot avg: {:.2}", pimc_avg, rule_avg);
    }

    #[test]
    fn pimc_decision_time() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);

        let start = std::time::Instant::now();
        let mut pimc_rng = StdRng::seed_from_u64(99);
        let _card = pimc_choose(
            &state,
            state.current_player,
            100,
            SolverType::Paranoid,
            &mut pimc_rng,
        );
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 1000, "PIMC n=100 on Tiny took {:?}", elapsed);
    }

    #[test]
    fn parallel_same_result_as_sequential() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);

        // Same seed → same sampled worlds → same result
        let mut rng1 = StdRng::seed_from_u64(99);
        let seq_card = pimc_choose(&state, state.current_player, 50, SolverType::Paranoid, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(99);
        let par_card = pimc_choose_parallel(&state, state.current_player, 50, SolverType::Paranoid, &mut rng2);

        assert_eq!(seq_card, par_card, "parallel and sequential should pick same move");
    }

    #[test]
    fn parallel_pimc_speedup() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Small.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Small);

        let mut rng1 = StdRng::seed_from_u64(99);
        let start_seq = std::time::Instant::now();
        let _seq = pimc_choose(&state, state.current_player, 50, SolverType::Paranoid, &mut rng1);
        let seq_time = start_seq.elapsed();

        let mut rng2 = StdRng::seed_from_u64(99);
        let start_par = std::time::Instant::now();
        let _par = pimc_choose_parallel(&state, state.current_player, 50, SolverType::Paranoid, &mut rng2);
        let par_time = start_par.elapsed();

        println!("Small n=50: sequential {:?}, parallel {:?}", seq_time, par_time);
        // Parallel should be faster (or at least not dramatically slower on single-core CI)
    }
}
