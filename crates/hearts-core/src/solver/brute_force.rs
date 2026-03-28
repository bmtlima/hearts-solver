use crate::game_state::GameState;

/// Exhaustive minimax solver — no pruning, no transposition table.
/// Each player minimizes their own score.
/// Only practical for Tiny (3 cards/player) games.
///
/// Returns the final score vector under optimal play from all 4 players.
pub fn brute_force_solve(state: &GameState) -> [i32; 4] {
    if state.is_game_over() {
        return state.final_scores();
    }

    let legal = state.legal_moves();
    let current = state.current_player.index();
    let mut best_scores: Option<[i32; 4]> = None;

    for card in legal.cards() {
        let mut child = state.clone();
        child.play_card(card);
        let scores = brute_force_solve(&child);

        if best_scores.is_none() || scores[current] < best_scores.unwrap()[current] {
            best_scores = Some(scores);
        }
    }

    best_scores.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_set::CardSet;
    use crate::deck::DeckConfig;
    use crate::types::{Card, Rank, Suit};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn known_tiny_deal_1() {
        // P0: Qc, Qs, Jh
        // P1: Kc, Kd, Ks
        // P2: Ac, Ad, Qh
        // P3: As, Kh, Ah
        let hands = [
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Queen),
                Card::new(Suit::Spades, Rank::Queen),
                Card::new(Suit::Hearts, Rank::Jack),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::King),
                Card::new(Suit::Diamonds, Rank::King),
                Card::new(Suit::Spades, Rank::King),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Ace),
                Card::new(Suit::Diamonds, Rank::Ace),
                Card::new(Suit::Hearts, Rank::Queen),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Spades, Rank::Ace),
                Card::new(Suit::Hearts, Rank::King),
                Card::new(Suit::Hearts, Rank::Ace),
            ]),
        ];
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let scores = brute_force_solve(&state);
        let total: i32 = scores.iter().sum();
        let dp = DeckConfig::Tiny.total_points();
        assert!(total == dp || total == dp * 3);
    }

    #[test]
    fn scores_always_valid_sum() {
        let dp = DeckConfig::Tiny.total_points();
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);
            let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
            let scores = brute_force_solve(&state);
            let total: i32 = scores.iter().sum();
            assert!(
                total == dp || total == dp * 3,
                "seed {}: scores {:?} sum to {}",
                seed, scores, total
            );
        }
    }

    #[test]
    fn brute_force_completes_in_time() {
        // Just a smoke test that it finishes reasonably fast
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let start = std::time::Instant::now();
        let _scores = brute_force_solve(&state);
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 100, "took {:?}", elapsed);
    }
}
