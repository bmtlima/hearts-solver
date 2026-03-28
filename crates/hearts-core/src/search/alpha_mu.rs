use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;

use crate::belief::observation::Observation;
use crate::belief::sampler::sample_worlds;
use crate::card_set::CardSet;
use crate::game::Player;
use crate::game_state::GameState;
use crate::search::pimc::{evaluate_move, SolverType};
use crate::trick::PlayerIndex;
use crate::types::Card;

/// Compute score matrix: scores[world_idx][move_idx] = AI's score.
/// Parallel over worlds.
fn compute_score_matrix(
    state: &GameState,
    player: PlayerIndex,
    moves: &[Card],
    worlds: &[[CardSet; 4]],
    solver: SolverType,
) -> Vec<Vec<i32>> {
    worlds
        .par_iter()
        .map(|world| {
            moves
                .iter()
                .map(|&mov| evaluate_move(state, player, mov, world, solver))
                .collect()
        })
        .collect()
}

/// Pure Alpha-Mu: worst-case aggregation over world partitions.
///
/// 1. Build score matrix (world × move)
/// 2. Partition worlds by which move is optimal in each
/// 3. For each candidate move, compute worst-case score across all partitions
/// 4. Pick move with best (lowest) worst-case
pub fn alpha_mu_choose(
    state: &GameState,
    player: PlayerIndex,
    n_worlds: usize,
    solver: SolverType,
    rng: &mut impl Rng,
) -> Card {
    alpha_mu_choose_softened(state, player, n_worlds, solver, 1.0, rng)
}

/// Softened Alpha-Mu with parameter λ.
///
/// λ=0.0: pure average (equivalent to PIMC)
/// λ=1.0: pure worst-case (equivalent to Alpha-Mu)
///
/// For each candidate move c, value is computed per-partition then summed:
///   value(c) = Σ_P weight(P) * ((1-λ)*avg(scores[w][c] for w in P) + λ*max(scores[w][c] for w in P))
pub fn alpha_mu_choose_softened(
    state: &GameState,
    player: PlayerIndex,
    n_worlds: usize,
    solver: SolverType,
    lambda: f64,
    rng: &mut impl Rng,
) -> Card {
    let obs = Observation::from_game_state(state, player);
    let legal = state.legal_moves();
    let moves: Vec<Card> = legal.cards().collect();

    if moves.len() == 1 {
        return moves[0];
    }

    let worlds = sample_worlds(&obs, n_worlds, rng);
    if worlds.is_empty() {
        return moves[0];
    }

    let n_worlds = worlds.len();
    let n_moves = moves.len();

    // Score matrix: scores[w][m]
    let scores = compute_score_matrix(state, player, &moves, &worlds, solver);

    // Partition worlds by optimal move
    // optimal[w] = index of best move in world w
    let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); n_moves];
    for (w, world_scores) in scores.iter().enumerate() {
        let best_m = world_scores
            .iter()
            .enumerate()
            .min_by_key(|(_, &s)| s)
            .unwrap()
            .0;
        partitions[best_m].push(w);
    }

    // For each candidate move, compute aggregated value
    let mut best_move_idx = 0;
    let mut best_value = f64::MAX;

    for (mi, _) in moves.iter().enumerate() {
        let mut total_value = 0.0f64;

        for (_, partition_worlds) in partitions.iter().enumerate() {
            if partition_worlds.is_empty() {
                continue;
            }

            let weight = partition_worlds.len() as f64 / n_worlds as f64;

            // Compute avg and worst-case of move mi across this partition
            let mut sum = 0.0f64;
            let mut worst = i32::MIN;
            for &w in partition_worlds {
                let s = scores[w][mi];
                sum += s as f64;
                if s > worst {
                    worst = s;
                }
            }
            let avg = sum / partition_worlds.len() as f64;

            // Blend: (1-λ)*avg + λ*worst
            let blended = (1.0 - lambda) * avg + lambda * worst as f64;
            total_value += weight * blended;
        }

        if total_value < best_value {
            best_value = total_value;
            best_move_idx = mi;
        }
    }

    moves[best_move_idx]
}

/// Alpha-Mu bot implementing the Player trait.
pub struct AlphaMuBot {
    pub n_worlds: usize,
    pub solver: SolverType,
    pub lambda: f64, // 0.0 = PIMC averaging, 1.0 = pure Alpha-Mu
    rng: StdRng,
}

impl AlphaMuBot {
    pub fn new(n_worlds: usize, solver: SolverType, lambda: f64, rng: StdRng) -> Self {
        AlphaMuBot {
            n_worlds,
            solver,
            lambda,
            rng,
        }
    }
}

impl Player for AlphaMuBot {
    fn choose_card(&mut self, state: &GameState, legal_moves: CardSet) -> Card {
        if legal_moves.count() == 1 {
            return legal_moves.cards().next().unwrap();
        }
        alpha_mu_choose_softened(
            state,
            state.current_player,
            self.n_worlds,
            self.solver,
            self.lambda,
            &mut self.rng,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::random_bot::RandomBot;
    use crate::deck::DeckConfig;
    use crate::game::GameRunner;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn alpha_mu_finds_obvious_dump() {
        // Same test as PIMC: void in clubs, should dump Qs
        use crate::types::{Rank, Suit};
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
        let card = alpha_mu_choose(&state, PlayerIndex::P0, 50, SolverType::Paranoid, &mut rng);
        assert_eq!(card, Card::new(Suit::Spades, Rank::Queen));
    }

    #[test]
    fn lambda_0_matches_pimc_result() {
        // λ=0 should produce same aggregation as PIMC (pure average)
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);

        let mut rng1 = StdRng::seed_from_u64(99);
        let pimc_card = crate::search::pimc::pimc_choose(
            &state,
            state.current_player,
            50,
            SolverType::Paranoid,
            &mut rng1,
        );

        let mut rng2 = StdRng::seed_from_u64(99);
        let am_card = alpha_mu_choose_softened(
            &state,
            state.current_player,
            50,
            SolverType::Paranoid,
            0.0,
            &mut rng2,
        );

        assert_eq!(pimc_card, am_card, "λ=0 should match PIMC");
    }

    #[test]
    fn lambda_1_is_pure_alpha_mu() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);

        let mut rng1 = StdRng::seed_from_u64(99);
        let pure_card = alpha_mu_choose(
            &state,
            state.current_player,
            50,
            SolverType::Paranoid,
            &mut rng1,
        );

        let mut rng2 = StdRng::seed_from_u64(99);
        let lambda1_card = alpha_mu_choose_softened(
            &state,
            state.current_player,
            50,
            SolverType::Paranoid,
            1.0,
            &mut rng2,
        );

        assert_eq!(pure_card, lambda1_card, "λ=1 should match pure Alpha-Mu");
    }

    #[test]
    fn alpha_mu_beats_random_bot() {
        let mut am_total = 0i64;
        let mut rand_total = 0i64;
        let games = 500;

        for seed in 0..games {
            let mut rng = StdRng::seed_from_u64(seed);
            let players: [Box<dyn Player>; 4] = [
                Box::new(AlphaMuBot::new(
                    30,
                    SolverType::Paranoid,
                    0.5,
                    StdRng::seed_from_u64(seed * 10),
                )),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 1))),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 2))),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 3))),
            ];
            let mut runner = GameRunner::new_with_deal(DeckConfig::Tiny, players, &mut rng);
            let scores = runner.play_game();
            am_total += scores[0] as i64;
            rand_total += (scores[1] + scores[2] + scores[3]) as i64;
        }

        let am_avg = am_total as f64 / games as f64;
        let rand_avg = rand_total as f64 / (games as f64 * 3.0);

        assert!(
            am_avg < rand_avg,
            "AlphaMu avg {:.2} should be less than Random avg {:.2}",
            am_avg, rand_avg
        );
    }

    #[test]
    fn alpha_mu_not_dominated() {
        // The chosen move should not be the worst in ALL worlds
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let player = state.current_player;
        let legal = state.legal_moves();
        let moves: Vec<Card> = legal.cards().collect();

        if moves.len() <= 1 {
            return; // nothing to test
        }

        let obs = Observation::from_game_state(&state, player);
        let mut sample_rng = StdRng::seed_from_u64(99);
        let worlds = sample_worlds(&obs, 50, &mut sample_rng);
        let scores = compute_score_matrix(&state, player, &moves, &worlds, SolverType::Paranoid);

        let mut choose_rng = StdRng::seed_from_u64(99);
        let chosen = alpha_mu_choose(&state, player, 50, SolverType::Paranoid, &mut choose_rng);
        let chosen_idx = moves.iter().position(|&m| m == chosen).unwrap();

        // Check: chosen move is NOT worst in every single world
        let mut worst_in_all = true;
        for world_scores in &scores {
            let chosen_score = world_scores[chosen_idx];
            let is_worst = world_scores.iter().all(|&s| s <= chosen_score);
            if !is_worst {
                worst_in_all = false;
                break;
            }
        }
        assert!(!worst_in_all, "Alpha-Mu chose a dominated move");
    }

    #[test]
    fn parameter_sweep() {
        let lambdas = [0.0, 0.25, 0.5, 0.75, 1.0];
        let games = 200;

        for &lambda in &lambdas {
            let mut total = 0i64;
            for seed in 0..games {
                let mut rng = StdRng::seed_from_u64(seed);
                let players: [Box<dyn Player>; 4] = [
                    Box::new(AlphaMuBot::new(
                        20,
                        SolverType::Paranoid,
                        lambda,
                        StdRng::seed_from_u64(seed * 10),
                    )),
                    Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 1))),
                    Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 2))),
                    Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 10 + 3))),
                ];
                let mut runner = GameRunner::new_with_deal(DeckConfig::Tiny, players, &mut rng);
                let scores = runner.play_game();
                total += scores[0] as i64;
            }
            let avg = total as f64 / games as f64;
            println!("λ={:.2}: AlphaMu avg score = {:.2}", lambda, avg);
        }
    }
}
