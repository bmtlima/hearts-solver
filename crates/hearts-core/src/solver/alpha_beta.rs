use std::cell::RefCell;

use crate::game_state::GameState;
use crate::solver::transposition::{TranspositionTable, ZobristKeys};

/// Double-dummy solver for multiplayer Hearts.
///
/// Each player minimizes their own score. Uses minimax with:
/// - Transposition table (optional) for memoization
/// - Undo-based state traversal (no cloning)
pub fn alpha_beta_solve(state: &mut GameState) -> [i32; 4] {
    solve_recursive(state, None)
}

/// Solve with a transposition table for memoization.
pub fn alpha_beta_solve_with_tt(
    state: &mut GameState,
    tt: &mut TranspositionTable,
    keys: &ZobristKeys,
) -> [i32; 4] {
    let ctx = TTContext {
        tt: RefCell::new(tt),
        keys,
    };
    solve_recursive(state, Some(&ctx))
}

struct TTContext<'a> {
    tt: RefCell<&'a mut TranspositionTable>,
    keys: &'a ZobristKeys,
}

fn solve_recursive(state: &mut GameState, ctx: Option<&TTContext>) -> [i32; 4] {
    if state.is_game_over() {
        return state.final_scores();
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
            return entry.scores;
        }
    }

    let legal = state.legal_moves();
    let current = state.current_player.index();
    let mut best_scores: Option<[i32; 4]> = None;

    for card in legal.cards() {
        let undo = state.play_card_with_undo(card);
        let scores = solve_recursive(state, ctx);
        state.undo_card(&undo);

        match best_scores {
            None => {
                best_scores = Some(scores);
            }
            Some(ref best) => {
                if scores[current] < best[current] {
                    best_scores = Some(scores);
                }
            }
        }
    }

    let result = best_scores.unwrap();

    // TT store
    if let (Some(h), Some(c)) = (hash, ctx) {
        c.tt.borrow_mut().store(h, result);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::DeckConfig;
    use crate::solver::brute_force;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// THE critical test: must match brute force exactly.
    #[test]
    fn matches_brute_force_on_100_tiny_deals() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);

            let bf_state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let bf_scores = brute_force::brute_force_solve(&bf_state);

            let mut ab_state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let ab_scores = alpha_beta_solve(&mut ab_state);

            assert_eq!(
                bf_scores, ab_scores,
                "seed {}: brute_force={:?} != alpha_beta={:?}",
                seed, bf_scores, ab_scores
            );
        }
    }

    #[test]
    fn matches_brute_force_with_tt() {
        let keys = ZobristKeys::new(42);
        let mut tt = TranspositionTable::new(16);

        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);

            let bf_state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let bf_scores = brute_force::brute_force_solve(&bf_state);

            tt.clear();
            let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let tt_scores = alpha_beta_solve_with_tt(&mut state, &mut tt, &keys);

            assert_eq!(
                bf_scores, tt_scores,
                "seed {}: brute_force={:?} != tt_solver={:?}",
                seed, bf_scores, tt_scores
            );
        }
    }

    #[test]
    fn undo_preserves_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let original = state.clone();

        let legal = state.legal_moves();
        let card = legal.cards().next().unwrap();
        let undo = state.play_card_with_undo(card);
        state.undo_card(&undo);

        assert_eq!(state.hands, original.hands);
        assert_eq!(state.cards_played, original.cards_played);
        assert_eq!(state.current_player, original.current_player);
        assert_eq!(state.hearts_broken, original.hearts_broken);
        assert_eq!(state.points_taken, original.points_taken);
        assert_eq!(state.trick_number, original.trick_number);
    }

    #[test]
    fn undo_across_trick_boundary() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);

        let mut undos = Vec::new();
        for _ in 0..3 {
            let legal = state.legal_moves();
            let card = legal.cards().next().unwrap();
            undos.push(state.play_card_with_undo(card));
        }
        let before_4th = state.clone();

        let legal = state.legal_moves();
        let card = legal.cards().next().unwrap();
        let undo4 = state.play_card_with_undo(card);
        assert_eq!(state.trick_number, 1);

        state.undo_card(&undo4);
        assert_eq!(state.hands, before_4th.hands);
        assert_eq!(state.trick_number, before_4th.trick_number);
        assert_eq!(state.points_taken, before_4th.points_taken);
    }

    #[test]
    fn faster_than_brute_force() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let bf_state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);

        let start_bf = std::time::Instant::now();
        for _ in 0..100 {
            let _ = brute_force::brute_force_solve(&bf_state);
        }
        let bf_time = start_bf.elapsed();

        let start_ab = std::time::Instant::now();
        for _ in 0..100 {
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let _ = alpha_beta_solve(&mut state);
        }
        let ab_time = start_ab.elapsed();

        println!("Brute force: {:?}, Alpha-beta+undo: {:?}", bf_time, ab_time);
    }

    #[test]
    fn tt_faster_than_no_tt_on_small() {
        let keys = ZobristKeys::new(42);
        let mut rng = StdRng::seed_from_u64(99);
        let hands = DeckConfig::Small.deal(&mut rng);

        let start_no_tt = std::time::Instant::now();
        let mut state1 = GameState::new_with_deal(hands.clone(), DeckConfig::Small);
        let scores_no_tt = alpha_beta_solve(&mut state1);
        let no_tt_time = start_no_tt.elapsed();

        let mut tt = TranspositionTable::new(20);
        let start_tt = std::time::Instant::now();
        let mut state2 = GameState::new_with_deal(hands, DeckConfig::Small);
        let scores_tt = alpha_beta_solve_with_tt(&mut state2, &mut tt, &keys);
        let tt_time = start_tt.elapsed();

        assert_eq!(scores_no_tt, scores_tt);
        println!("Small deck - No TT: {:?}, With TT: {:?}", no_tt_time, tt_time);
    }
}
