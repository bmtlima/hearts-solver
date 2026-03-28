use crate::card_set::CardSet;
use crate::game::Player;
use crate::game_state::GameState;
use crate::types::{Card, Rank, Suit};

/// Heuristic rule-based bot that plays reasonable Hearts.
pub struct RuleBot;

impl RuleBot {
    pub fn new() -> Self {
        RuleBot
    }

    fn queen_of_spades() -> Card {
        Card::new(Suit::Spades, Rank::Queen)
    }
}

impl Player for RuleBot {
    fn choose_card(&mut self, state: &GameState, legal_moves: CardSet) -> Card {
        if legal_moves.count() == 1 {
            return legal_moves.cards().next().unwrap();
        }

        if state.current_trick.is_empty() {
            self.choose_lead(state, legal_moves)
        } else {
            let led_suit = state.current_trick.led_suit().unwrap();
            let in_suit = legal_moves.cards_of_suit(led_suit);
            if !in_suit.is_empty() {
                self.choose_follow(state, legal_moves, led_suit)
            } else {
                self.choose_slough(state, legal_moves)
            }
        }
    }
}

impl RuleBot {
    /// When leading a trick.
    fn choose_lead(&self, state: &GameState, legal_moves: CardSet) -> Card {
        // Prefer leading low cards in short non-heart suits to probe for voids
        let mut best: Option<Card> = None;
        let mut best_key: (usize, u8) = (usize::MAX, u8::MAX);

        for card in legal_moves.cards() {
            if card.suit() == Suit::Hearts && !state.hearts_broken {
                continue; // shouldn't be in legal_moves anyway, but guard
            }
            let suit_count = state.hands[state.current_player.index()]
                .cards_of_suit(card.suit())
                .count() as usize;
            let key = (suit_count, card.rank() as u8);
            if key < best_key {
                best_key = key;
                best = Some(card);
            }
        }

        best.unwrap_or_else(|| legal_moves.cards().next().unwrap())
    }

    /// When following suit.
    fn choose_follow(&self, state: &GameState, legal_moves: CardSet, led_suit: Suit) -> Card {
        let in_suit = legal_moves.cards_of_suit(led_suit);
        let qs = Self::queen_of_spades();

        // If Qs is legal and we're following spades led by someone else, dump it
        if led_suit == Suit::Spades && in_suit.contains(qs) {
            // Dump Qs if someone else might win the trick
            let current_high = self.trick_high_card(state, led_suit);
            if current_high.map_or(false, |c| c.rank() > Rank::Queen) {
                return qs;
            }
        }

        // Try to duck: play highest card below the current winner
        let current_high = self.trick_high_card(state, led_suit);
        if let Some(high) = current_high {
            // Find highest card that's still below the current winner
            let mut duck: Option<Card> = None;
            for card in in_suit.cards() {
                if card.rank() < high.rank() {
                    if duck.map_or(true, |d| card.rank() > d.rank()) {
                        duck = Some(card);
                    }
                }
            }
            if let Some(d) = duck {
                return d;
            }
        }

        // Can't duck — play lowest card in suit
        in_suit
            .cards()
            .min_by_key(|c| c.rank() as u8)
            .unwrap()
    }

    /// When void in the led suit (sloughing).
    fn choose_slough(&self, _state: &GameState, legal_moves: CardSet) -> Card {
        let qs = Self::queen_of_spades();

        // Dump Qs first
        if legal_moves.contains(qs) {
            return qs;
        }

        // Dump highest heart
        let hearts = legal_moves.cards_of_suit(Suit::Hearts);
        if !hearts.is_empty() {
            return hearts
                .cards()
                .max_by_key(|c| c.rank() as u8)
                .unwrap();
        }

        // Dump highest card in longest suit
        let mut best: Option<Card> = None;
        let mut best_key: (usize, u8) = (0, 0); // (suit_len desc, rank desc)
        for card in legal_moves.cards() {
            let suit_len = legal_moves.cards_of_suit(card.suit()).count() as usize;
            let key = (suit_len, card.rank() as u8);
            if key > best_key {
                best_key = key;
                best = Some(card);
            }
        }
        best.unwrap()
    }

    /// Highest card of the led suit played so far in the current trick.
    fn trick_high_card(&self, state: &GameState, led_suit: Suit) -> Option<Card> {
        state
            .current_trick
            .played_cards()
            .iter()
            .map(|(_, c)| *c)
            .filter(|c| c.suit() == led_suit)
            .max_by_key(|c| c.rank() as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_set::CardSet;
    use crate::deck::DeckConfig;
    use crate::game::{GameRunner, Player};
    use crate::game_state::GameState;
    use crate::trick::PlayerIndex;
    use crate::types::{Card, Rank, Suit};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rule_players() -> [Box<dyn Player>; 4] {
        [
            Box::new(RuleBot::new()),
            Box::new(RuleBot::new()),
            Box::new(RuleBot::new()),
            Box::new(RuleBot::new()),
        ]
    }

    #[test]
    fn dumps_queen_of_spades_when_void() {
        let legal = CardSet::from_cards([
            Card::new(Suit::Spades, Rank::Queen),
            Card::new(Suit::Hearts, Rank::Two),
            Card::new(Suit::Diamonds, Rank::Ace),
        ]);
        // Simulating being void in the led suit
        let state = GameState::new([CardSet::empty(); 4], DeckConfig::Tiny, PlayerIndex::P0);
        // Trick in progress (not leading), void scenario — choose_slough should be called
        let bot = RuleBot::new();
        // Direct call to slough
        let card = bot.choose_slough(&state, legal);
        assert_eq!(card, Card::new(Suit::Spades, Rank::Queen));
    }

    #[test]
    fn does_not_lead_hearts_when_unbroken() {
        let hand = CardSet::from_cards([
            Card::new(Suit::Clubs, Rank::Three),
            Card::new(Suit::Hearts, Rank::Ace),
            Card::new(Suit::Hearts, Rank::King),
        ]);
        let hands = [hand, CardSet::empty(), CardSet::empty(), CardSet::empty()];
        let mut state = GameState::new(hands, DeckConfig::Tiny, PlayerIndex::P0);
        state.is_first_trick = false;
        state.hearts_broken = false;

        let legal = state.legal_moves(); // should exclude hearts
        let mut bot = RuleBot::new();
        let card = bot.choose_card(&state, legal);
        assert_eq!(card.suit(), Suit::Clubs);
    }

    #[test]
    fn thousand_full_games_rule_bots() {
        for seed in 0..1000 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut runner =
                GameRunner::new_with_deal(DeckConfig::Full, make_rule_players(), &mut rng);
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

    #[test]
    fn rule_bot_beats_random_bot() {
        use crate::bots::random_bot::RandomBot;

        let mut rule_total = 0i64;
        let mut rand_total = 0i64;
        let games = 1000;

        for seed in 0..games {
            let mut rng = StdRng::seed_from_u64(seed);
            // P0,P1 = RuleBot; P2,P3 = RandomBot
            let players: [Box<dyn Player>; 4] = [
                Box::new(RuleBot::new()),
                Box::new(RuleBot::new()),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 100 + 1))),
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed * 100 + 2))),
            ];
            let mut runner = GameRunner::new_with_deal(DeckConfig::Full, players, &mut rng);
            let scores = runner.play_game();
            rule_total += (scores[0] + scores[1]) as i64;
            rand_total += (scores[2] + scores[3]) as i64;
        }

        let rule_avg = rule_total as f64 / (games as f64 * 2.0);
        let rand_avg = rand_total as f64 / (games as f64 * 2.0);

        assert!(
            rule_avg < rand_avg,
            "RuleBot avg {:.1} should be less than RandomBot avg {:.1}",
            rule_avg,
            rand_avg
        );
    }
}
