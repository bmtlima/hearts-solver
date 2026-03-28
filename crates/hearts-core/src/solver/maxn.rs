use crate::game_state::GameState;
use crate::solver::transposition::{TranspositionTable, ZobristKeys};

/// Max^n solver for multiplayer Hearts.
///
/// Each player minimizes their own score independently. This is the correct
/// solver for multiplayer games but does NOT support standard alpha-beta
/// pruning bounds across players — only weak pruning is possible.
/// Matches brute-force results exactly.
pub fn maxn_solve(state: &mut GameState) -> [i32; 4] {
    solve_recursive(state, None)
}

/// Solve with a transposition table for memoization.
pub fn maxn_solve_with_tt(
    state: &mut GameState,
    tt: &mut TranspositionTable,
    keys: &ZobristKeys,
) -> [i32; 4] {
    solve_recursive(state, Some((tt, keys)))
}

fn solve_recursive(
    state: &mut GameState,
    mut tt_ctx: Option<(&mut TranspositionTable, &ZobristKeys)>,
) -> [i32; 4] {
    if state.is_game_over() {
        return state.final_scores();
    }

    // TT probe
    let hash = tt_ctx.as_ref().map(|(_, keys)| {
        keys.hash_position(
            &state.hands,
            state.current_player,
            state.current_trick.played_cards(),
            &state.points_taken,
        )
    });

    if let (Some(h), Some((ref tt, _))) = (hash, &tt_ctx) {
        if let Some(entry) = tt.probe(h) {
            return entry.scores;
        }
    }

    let legal = state.legal_moves();
    let current = state.current_player.index();
    let mut best_scores: Option<[i32; 4]> = None;

    // Split tt_ctx so we can pass it into recursion while borrowing
    // We need to reborrow on each iteration since we pass &mut into recursion
    for card in legal.cards() {
        let undo = state.play_card_with_undo(card);
        let scores = match tt_ctx {
            Some((ref mut tt, keys)) => solve_recursive(state, Some((tt, keys))),
            None => solve_recursive(state, None),
        };
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
    if let (Some(h), Some((ref mut tt, _))) = (hash, tt_ctx) {
        tt.store(h, result);
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

    #[test]
    fn matches_brute_force_on_100_tiny_deals() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);

            let bf_state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let bf_scores = brute_force::brute_force_solve(&bf_state);

            let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let scores = maxn_solve(&mut state);

            assert_eq!(
                bf_scores, scores,
                "seed {}: brute_force={:?} != maxn={:?}",
                seed, bf_scores, scores
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
            let scores = maxn_solve_with_tt(&mut state, &mut tt, &keys);

            assert_eq!(
                bf_scores, scores,
                "seed {}: brute_force={:?} != maxn_tt={:?}",
                seed, bf_scores, scores
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
    fn tt_on_small() {
        let keys = ZobristKeys::new(42);
        let mut rng = StdRng::seed_from_u64(99);
        let hands = DeckConfig::Small.deal(&mut rng);

        let mut state1 = GameState::new_with_deal(hands.clone(), DeckConfig::Small);
        let scores_no_tt = maxn_solve(&mut state1);

        let mut tt = TranspositionTable::new(20);
        let mut state2 = GameState::new_with_deal(hands, DeckConfig::Small);
        let scores_tt = maxn_solve_with_tt(&mut state2, &mut tt, &keys);

        assert_eq!(scores_no_tt, scores_tt);
    }
}
