use crate::card_set::CardSet;
use crate::deck::DeckConfig;
use crate::game_state::GameState;
use crate::trick::PlayerIndex;

/// What a single player can observe about the game state.
pub struct Observation {
    pub viewer: PlayerIndex,
    pub my_hand: CardSet,
    pub cards_played: CardSet,
    /// voids[player_idx][suit_idx] = true if player is known void in that suit.
    pub voids: [[bool; 4]; 4],
    pub deck_config: DeckConfig,
    pub trick_number: usize,
    pub cards_per_player: usize,
    /// Which players have already played in the current (incomplete) trick.
    pub played_in_current_trick: [bool; 4],
}

impl Observation {
    /// Extract an observation from a game state for the given viewer.
    pub fn from_game_state(state: &GameState, viewer: PlayerIndex) -> Self {
        let mut voids = [[false; 4]; 4];

        // Infer voids from completed tricks
        for tr in &state.trick_history {
            let led_suit = tr.cards[0].1.suit();
            for &(player, card) in &tr.cards[1..] {
                if card.suit() != led_suit {
                    voids[player.index()][led_suit.index()] = true;
                }
            }
        }

        // Also check current in-progress trick
        let played = state.current_trick.played_cards();
        let mut played_in_current_trick = [false; 4];
        if let Some(led_suit) = state.current_trick.led_suit() {
            for &(player, card) in played {
                played_in_current_trick[player.index()] = true;
                // Only non-leader cards can reveal void info
                if card.suit() != led_suit && player != played[0].0 {
                    voids[player.index()][led_suit.index()] = true;
                }
            }
        }

        Observation {
            viewer,
            my_hand: state.hands[viewer.index()],
            cards_played: state.cards_played,
            voids,
            deck_config: state.deck_config,
            trick_number: state.trick_number,
            cards_per_player: state.deck_config.cards_per_player(),
            played_in_current_trick,
        }
    }

    /// Cards not in the viewer's hand and not yet played — distributed among opponents.
    pub fn unknown_cards(&self) -> CardSet {
        self.deck_config.deck_cards() - self.my_hand - self.cards_played
    }

    /// How many cards each player currently holds.
    pub fn hand_sizes(&self) -> [usize; 4] {
        let base = self.cards_per_player - self.trick_number;
        let mut sizes = [0usize; 4];
        for p in 0..4 {
            sizes[p] = if self.played_in_current_trick[p] {
                base - 1
            } else {
                base
            };
        }
        // For the viewer, we know exactly from the hand
        sizes[self.viewer.index()] = self.my_hand.count() as usize;
        sizes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::DeckConfig;
    use crate::game_state::GameState;
    use crate::trick::PlayerIndex;
    #[allow(unused_imports)]
    use crate::types::{Card, Rank, Suit};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn fresh_deal_observation() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
        let viewer = PlayerIndex::P0;
        let obs = Observation::from_game_state(&state, viewer);

        assert_eq!(obs.my_hand, hands[0]);
        assert!(obs.cards_played.is_empty());
        assert_eq!(obs.trick_number, 0);
        // No voids known yet
        for p in 0..4 {
            for s in 0..4 {
                assert!(!obs.voids[p][s]);
            }
        }
    }

    #[test]
    fn unknown_cards_count() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let obs = Observation::from_game_state(&state, PlayerIndex::P0);

        // Unknown = total deck - my hand - played (0)
        let unknown = obs.unknown_cards();
        assert_eq!(unknown.count(), 12 - 3); // Tiny: 12 total, 3 in hand
    }

    #[test]
    fn void_inference_from_trick_history() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);

        // Play a full trick — first player leads, others follow or void
        let legal = state.legal_moves();
        let first_card = legal.cards().next().unwrap();
        let _led_suit = first_card.suit();
        state.play_card(first_card);

        // Play remaining 3 cards
        for _ in 0..3 {
            let legal = state.legal_moves();
            let card = legal.cards().next().unwrap();
            state.play_card(card);
        }

        // Now check observation for void inference
        let obs = Observation::from_game_state(&state, PlayerIndex::P0);
        assert_eq!(obs.trick_number, 1);
        assert_eq!(obs.cards_played.count(), 4);

        // Check trick_history was populated
        assert_eq!(state.trick_history.len(), 1);
        let tr = &state.trick_history[0];
        let trick_led_suit = tr.cards[0].1.suit();

        // Any player who played off-suit should be marked void
        for &(player, card) in &tr.cards[1..] {
            if card.suit() != trick_led_suit {
                assert!(
                    obs.voids[player.index()][trick_led_suit.index()],
                    "P{} played off-suit but not marked void in {:?}",
                    player.index(), trick_led_suit
                );
            }
        }
    }

    #[test]
    fn after_two_tricks_cards_played_count() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Small.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Small);

        // Play 2 full tricks (8 cards)
        for _ in 0..8 {
            let legal = state.legal_moves();
            let card = legal.cards().next().unwrap();
            state.play_card(card);
        }

        let obs = Observation::from_game_state(&state, PlayerIndex::P0);
        assert_eq!(obs.cards_played.count(), 8);
        assert_eq!(obs.trick_number, 2);
    }
}
