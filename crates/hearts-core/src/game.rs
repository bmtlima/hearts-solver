use rand::Rng;

use crate::card_set::CardSet;
use crate::deck::DeckConfig;
use crate::game_state::GameState;
use crate::types::Card;

/// Trait for a Hearts player (bot or human).
pub trait Player {
    fn choose_card(&mut self, state: &GameState, legal_moves: CardSet) -> Card;
}

/// Runs a complete Hearts game with 4 players.
pub struct GameRunner {
    pub state: GameState,
    players: [Box<dyn Player>; 4],
}

impl GameRunner {
    /// Create a game from a random deal.
    pub fn new_with_deal(
        deck_config: DeckConfig,
        players: [Box<dyn Player>; 4],
        rng: &mut impl Rng,
    ) -> Self {
        let hands = deck_config.deal(rng);
        let state = GameState::new_with_deal(hands, deck_config);
        GameRunner { state, players }
    }

    /// Create a game from predetermined hands.
    pub fn new_with_hands(
        hands: [CardSet; 4],
        deck_config: DeckConfig,
        players: [Box<dyn Player>; 4],
    ) -> Self {
        let state = GameState::new_with_deal(hands, deck_config);
        GameRunner { state, players }
    }

    /// Play the full game, returning final scores.
    pub fn play_game(&mut self) -> [i32; 4] {
        while !self.state.is_game_over() {
            let legal = self.state.legal_moves();
            let idx = self.state.current_player.index();
            let card = self.players[idx].choose_card(&self.state, legal);
            debug_assert!(legal.contains(card), "player chose illegal card: {}", card);
            self.state.play_card(card);
        }
        self.state.final_scores()
    }
}

/// Simplest possible "bot": always plays the first legal move.
pub struct FirstLegalBot;

impl Player for FirstLegalBot {
    fn choose_card(&mut self, _state: &GameState, legal_moves: CardSet) -> Card {
        legal_moves.cards().next().unwrap()
    }
}

pub fn make_first_legal_players() -> [Box<dyn Player>; 4] {
    [
        Box::new(FirstLegalBot),
        Box::new(FirstLegalBot),
        Box::new(FirstLegalBot),
        Box::new(FirstLegalBot),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn first_legal_bot_completes_tiny_game() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut runner =
            GameRunner::new_with_deal(DeckConfig::Tiny, make_first_legal_players(), &mut rng);
        let scores = runner.play_game();
        let total: i32 = scores.iter().sum();
        let dp = DeckConfig::Tiny.total_points();
        assert!(total == dp || total == dp * 3);
    }

    #[test]
    fn hundred_tiny_games_all_complete() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut runner =
                GameRunner::new_with_deal(DeckConfig::Tiny, make_first_legal_players(), &mut rng);
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

    #[test]
    fn hundred_full_games_all_complete() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut runner =
                GameRunner::new_with_deal(DeckConfig::Full, make_first_legal_players(), &mut rng);
            let scores = runner.play_game();
            let total: i32 = scores.iter().sum();
            assert!(
                total == 26 || total == 78,
                "seed {}: scores {:?} sum to {}",
                seed,
                scores,
                total
            );
        }
    }
}
