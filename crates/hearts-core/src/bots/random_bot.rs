use rand::rngs::StdRng;
use rand::Rng;

use crate::card_set::CardSet;
use crate::game::Player;
use crate::game_state::GameState;
use crate::types::Card;

/// Bot that picks a uniformly random legal move.
pub struct RandomBot {
    rng: StdRng,
}

impl RandomBot {
    pub fn new(rng: StdRng) -> Self {
        RandomBot { rng }
    }
}

impl Player for RandomBot {
    fn choose_card(&mut self, _state: &GameState, legal_moves: CardSet) -> Card {
        let count = legal_moves.count() as usize;
        let idx = self.rng.gen_range(0..count);
        legal_moves.cards().nth(idx).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::DeckConfig;
    use crate::game::GameRunner;
    use rand::SeedableRng;

    fn make_random_players() -> [Box<dyn Player>; 4] {
        [
            Box::new(RandomBot::new(StdRng::seed_from_u64(10))),
            Box::new(RandomBot::new(StdRng::seed_from_u64(20))),
            Box::new(RandomBot::new(StdRng::seed_from_u64(30))),
            Box::new(RandomBot::new(StdRng::seed_from_u64(40))),
        ]
    }

    #[test]
    fn random_bot_covers_all_moves() {
        // With 3 legal moves over 1000 trials, each should appear at least once
        let legal = CardSet::from_cards([
            crate::types::Card::new(crate::types::Suit::Clubs, crate::types::Rank::Two),
            crate::types::Card::new(crate::types::Suit::Clubs, crate::types::Rank::Three),
            crate::types::Card::new(crate::types::Suit::Clubs, crate::types::Rank::Four),
        ]);
        let state = GameState::new(
            [CardSet::empty(); 4],
            DeckConfig::Tiny,
            crate::trick::PlayerIndex::P0,
        );
        let mut bot = RandomBot::new(StdRng::seed_from_u64(99));
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            seen.insert(bot.choose_card(&state, legal));
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn thousand_tiny_games_random_bots() {
        for seed in 0..1000 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut runner =
                GameRunner::new_with_deal(DeckConfig::Tiny, make_random_players(), &mut rng);
            let scores = runner.play_game();
            let total: i32 = scores.iter().sum();
            let dp = DeckConfig::Tiny.total_points();
            assert!(
                total == dp || total == dp * 3,
                "seed {}: scores {:?} sum to {}",
                seed,
                scores,
                total
            );
        }
    }
}
