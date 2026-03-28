use crate::card_set::CardSet;
use crate::types::{Card, Rank, Suit};

/// Deck size configuration for reduced-deck development.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeckConfig {
    /// 3 cards/player, 12 total
    Tiny,
    /// 5 cards/player, 20 total
    Small,
    /// 8 cards/player, 32 total
    Medium,
    /// 13 cards/player, 52 total
    Full,
}

impl DeckConfig {
    pub fn cards_per_player(self) -> usize {
        match self {
            DeckConfig::Tiny => 3,
            DeckConfig::Small => 5,
            DeckConfig::Medium => 8,
            DeckConfig::Full => 13,
        }
    }

    pub fn total_cards(self) -> usize {
        self.cards_per_player() * 4
    }

    /// The set of cards in play for this configuration.
    ///
    /// Card selection rules:
    /// - Always keep Qs, Ks, As (spade danger cards)
    /// - Keep hearts as much as possible
    /// - Strip lowest ranks first
    pub fn deck_cards(self) -> CardSet {
        match self {
            DeckConfig::Full => CardSet::full(),
            DeckConfig::Tiny => deck_tiny(),
            DeckConfig::Small => deck_small(),
            DeckConfig::Medium => deck_medium(),
        }
    }

    /// Total point value of all cards in this deck configuration.
    pub fn total_points(self) -> i32 {
        self.deck_cards().cards().map(|c| c.point_value()).sum()
    }

    /// The card that must be led on the first trick (lowest club in the deck).
    pub fn first_lead_card(self) -> Card {
        let deck = self.deck_cards();
        for rank in Rank::ALL {
            let card = Card::new(Suit::Clubs, rank);
            if deck.contains(card) {
                return card;
            }
        }
        unreachable!("deck has no clubs")
    }

    /// Shuffle the deck and deal cards_per_player() cards to each of 4 players.
    pub fn deal(self, rng: &mut impl Rng) -> [CardSet; 4] {
        let mut cards: Vec<Card> = self.deck_cards().cards().collect();
        cards.shuffle(rng);
        let cpp = self.cards_per_player();
        let mut hands = [CardSet::empty(); 4];
        for (i, card) in cards.into_iter().enumerate() {
            hands[i / cpp].insert(card);
        }
        hands
    }
}

/// Tiny (12 cards, 3/player):
/// Clubs: Q, K, A  (3)
/// Diamonds: K, A  (2)
/// Spades: Q, K, A (3) — must keep Qs, Ks, As
/// Hearts: J, Q, K, A (4)
/// = 12 total
fn deck_tiny() -> CardSet {
    CardSet::from_cards([
        Card::new(Suit::Clubs, Rank::Queen),
        Card::new(Suit::Clubs, Rank::King),
        Card::new(Suit::Clubs, Rank::Ace),
        Card::new(Suit::Diamonds, Rank::King),
        Card::new(Suit::Diamonds, Rank::Ace),
        Card::new(Suit::Spades, Rank::Queen),
        Card::new(Suit::Spades, Rank::King),
        Card::new(Suit::Spades, Rank::Ace),
        Card::new(Suit::Hearts, Rank::Jack),
        Card::new(Suit::Hearts, Rank::Queen),
        Card::new(Suit::Hearts, Rank::King),
        Card::new(Suit::Hearts, Rank::Ace),
    ])
}

/// Small (20 cards, 5/player):
/// Clubs: T, J, Q, K, A (5)
/// Diamonds: J, Q, K, A  (4)
/// Spades: T, J, Q, K, A (5)
/// Hearts: 9, T, J, Q, K, A (6)
/// = 20 total
fn deck_small() -> CardSet {
    CardSet::from_cards([
        Card::new(Suit::Clubs, Rank::Ten),
        Card::new(Suit::Clubs, Rank::Jack),
        Card::new(Suit::Clubs, Rank::Queen),
        Card::new(Suit::Clubs, Rank::King),
        Card::new(Suit::Clubs, Rank::Ace),
        Card::new(Suit::Diamonds, Rank::Jack),
        Card::new(Suit::Diamonds, Rank::Queen),
        Card::new(Suit::Diamonds, Rank::King),
        Card::new(Suit::Diamonds, Rank::Ace),
        Card::new(Suit::Spades, Rank::Ten),
        Card::new(Suit::Spades, Rank::Jack),
        Card::new(Suit::Spades, Rank::Queen),
        Card::new(Suit::Spades, Rank::King),
        Card::new(Suit::Spades, Rank::Ace),
        Card::new(Suit::Hearts, Rank::Nine),
        Card::new(Suit::Hearts, Rank::Ten),
        Card::new(Suit::Hearts, Rank::Jack),
        Card::new(Suit::Hearts, Rank::Queen),
        Card::new(Suit::Hearts, Rank::King),
        Card::new(Suit::Hearts, Rank::Ace),
    ])
}

/// Medium (32 cards, 8/player):
/// Clubs: 7, 8, 9, T, J, Q, K, A (8)
/// Diamonds: 8, 9, T, J, Q, K, A (7)
/// Spades: 7, 8, 9, T, J, Q, K, A (8)
/// Hearts: 6, 7, 8, 9, T, J, Q, K, A (9)
/// = 32 total
fn deck_medium() -> CardSet {
    CardSet::from_cards([
        Card::new(Suit::Clubs, Rank::Seven),
        Card::new(Suit::Clubs, Rank::Eight),
        Card::new(Suit::Clubs, Rank::Nine),
        Card::new(Suit::Clubs, Rank::Ten),
        Card::new(Suit::Clubs, Rank::Jack),
        Card::new(Suit::Clubs, Rank::Queen),
        Card::new(Suit::Clubs, Rank::King),
        Card::new(Suit::Clubs, Rank::Ace),
        Card::new(Suit::Diamonds, Rank::Eight),
        Card::new(Suit::Diamonds, Rank::Nine),
        Card::new(Suit::Diamonds, Rank::Ten),
        Card::new(Suit::Diamonds, Rank::Jack),
        Card::new(Suit::Diamonds, Rank::Queen),
        Card::new(Suit::Diamonds, Rank::King),
        Card::new(Suit::Diamonds, Rank::Ace),
        Card::new(Suit::Spades, Rank::Seven),
        Card::new(Suit::Spades, Rank::Eight),
        Card::new(Suit::Spades, Rank::Nine),
        Card::new(Suit::Spades, Rank::Ten),
        Card::new(Suit::Spades, Rank::Jack),
        Card::new(Suit::Spades, Rank::Queen),
        Card::new(Suit::Spades, Rank::King),
        Card::new(Suit::Spades, Rank::Ace),
        Card::new(Suit::Hearts, Rank::Six),
        Card::new(Suit::Hearts, Rank::Seven),
        Card::new(Suit::Hearts, Rank::Eight),
        Card::new(Suit::Hearts, Rank::Nine),
        Card::new(Suit::Hearts, Rank::Ten),
        Card::new(Suit::Hearts, Rank::Jack),
        Card::new(Suit::Hearts, Rank::Queen),
        Card::new(Suit::Hearts, Rank::King),
        Card::new(Suit::Hearts, Rank::Ace),
    ])
}

// Re-export rand traits needed for deal()
use rand::seq::SliceRandom;
use rand::Rng;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn full_deck_count() {
        assert_eq!(DeckConfig::Full.deck_cards().count(), 52);
    }

    #[test]
    fn tiny_deck_count() {
        assert_eq!(DeckConfig::Tiny.deck_cards().count(), 12);
    }

    #[test]
    fn small_deck_count() {
        assert_eq!(DeckConfig::Small.deck_cards().count(), 20);
    }

    #[test]
    fn medium_deck_count() {
        assert_eq!(DeckConfig::Medium.deck_cards().count(), 32);
    }

    #[test]
    fn all_configs_contain_key_spades() {
        for config in [DeckConfig::Tiny, DeckConfig::Small, DeckConfig::Medium, DeckConfig::Full] {
            let deck = config.deck_cards();
            assert!(deck.contains(Card::new(Suit::Spades, Rank::Queen)), "{:?} missing Qs", config);
            assert!(deck.contains(Card::new(Suit::Spades, Rank::King)), "{:?} missing Ks", config);
            assert!(deck.contains(Card::new(Suit::Spades, Rank::Ace)), "{:?} missing As", config);
        }
    }

    #[test]
    fn all_configs_have_hearts() {
        for config in [DeckConfig::Tiny, DeckConfig::Small, DeckConfig::Medium, DeckConfig::Full] {
            let deck = config.deck_cards();
            assert!(deck.has_suit(Suit::Hearts), "{:?} has no hearts", config);
        }
    }

    #[test]
    fn deal_distributes_correctly() {
        for config in [DeckConfig::Tiny, DeckConfig::Small, DeckConfig::Medium, DeckConfig::Full] {
            let mut rng = StdRng::seed_from_u64(42);
            let hands = config.deal(&mut rng);
            let cpp = config.cards_per_player() as u32;

            for (i, hand) in hands.iter().enumerate() {
                assert_eq!(hand.count(), cpp, "{:?} player {} got {} cards", config, i, hand.count());
            }

            let union = hands[0] | hands[1] | hands[2] | hands[3];
            assert_eq!(union, config.deck_cards());
        }
    }

    #[test]
    fn deal_deterministic_with_seed() {
        let mut rng1 = StdRng::seed_from_u64(123);
        let mut rng2 = StdRng::seed_from_u64(123);
        let hands1 = DeckConfig::Small.deal(&mut rng1);
        let hands2 = DeckConfig::Small.deal(&mut rng2);
        assert_eq!(hands1, hands2);
    }

    #[test]
    fn first_lead_card_in_deck() {
        for config in [DeckConfig::Tiny, DeckConfig::Small, DeckConfig::Medium, DeckConfig::Full] {
            let card = config.first_lead_card();
            assert_eq!(card.suit(), Suit::Clubs);
            assert!(config.deck_cards().contains(card));
        }
    }

    #[test]
    fn full_first_lead_is_2c() {
        assert_eq!(DeckConfig::Full.first_lead_card(), Card::new(Suit::Clubs, Rank::Two));
    }
}
