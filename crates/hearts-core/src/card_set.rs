use std::fmt;
use std::ops::{BitAnd, BitOr, Sub};

use crate::types::{Card, Rank, Suit};

/// A set of cards represented as a 64-bit bitboard.
///
/// Bits 0..12 = Clubs, 13..25 = Diamonds, 26..38 = Spades, 39..51 = Hearts.
/// Each bit within a suit group corresponds to a rank (bit 0 = Two, bit 12 = Ace).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CardSet(u64);

impl CardSet {
    const SUIT_BITS: u32 = 13;
    const SUIT_MASK: u64 = (1 << 13) - 1; // 0x1FFF

    pub const fn empty() -> Self {
        CardSet(0)
    }

    /// All 52 cards.
    pub const fn full() -> Self {
        CardSet(
            Self::SUIT_MASK
                | (Self::SUIT_MASK << Self::SUIT_BITS)
                | (Self::SUIT_MASK << (2 * Self::SUIT_BITS))
                | (Self::SUIT_MASK << (3 * Self::SUIT_BITS)),
        )
    }

    pub fn from_cards(iter: impl IntoIterator<Item = Card>) -> Self {
        let mut set = Self::empty();
        for card in iter {
            set.insert(card);
        }
        set
    }

    fn bit_pos(card: Card) -> u32 {
        card.suit().index() as u32 * Self::SUIT_BITS + card.bit_index() as u32
    }

    pub fn insert(&mut self, card: Card) {
        self.0 |= 1u64 << Self::bit_pos(card);
    }

    pub fn remove(&mut self, card: Card) {
        self.0 &= !(1u64 << Self::bit_pos(card));
    }

    pub fn contains(self, card: Card) -> bool {
        self.0 & (1u64 << Self::bit_pos(card)) != 0
    }

    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn raw(self) -> u64 {
        self.0
    }

    pub const fn from_raw(bits: u64) -> Self {
        CardSet(bits)
    }

    /// 13-bit mask for a single suit.
    pub fn suit_mask(self, suit: Suit) -> u16 {
        ((self.0 >> (suit.index() as u32 * Self::SUIT_BITS)) & Self::SUIT_MASK) as u16
    }

    /// Subset containing only cards of the given suit.
    pub fn cards_of_suit(self, suit: Suit) -> CardSet {
        let shift = suit.index() as u32 * Self::SUIT_BITS;
        CardSet((self.0 >> shift & Self::SUIT_MASK) << shift)
    }

    /// Whether this set contains any card of the given suit.
    pub fn has_suit(self, suit: Suit) -> bool {
        self.suit_mask(suit) != 0
    }

    pub fn union(self, other: CardSet) -> CardSet {
        CardSet(self.0 | other.0)
    }

    pub fn intersection(self, other: CardSet) -> CardSet {
        CardSet(self.0 & other.0)
    }

    pub fn difference(self, other: CardSet) -> CardSet {
        CardSet(self.0 & !other.0)
    }

    /// Iterator over all cards in the set, ordered by suit then rank.
    pub fn cards(self) -> CardIter {
        CardIter(self.0)
    }
}

impl BitOr for CardSet {
    type Output = CardSet;
    fn bitor(self, rhs: CardSet) -> CardSet {
        self.union(rhs)
    }
}

impl BitAnd for CardSet {
    type Output = CardSet;
    fn bitand(self, rhs: CardSet) -> CardSet {
        self.intersection(rhs)
    }
}

impl Sub for CardSet {
    type Output = CardSet;
    fn sub(self, rhs: CardSet) -> CardSet {
        self.difference(rhs)
    }
}

impl fmt::Display for CardSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first_suit = true;
        for suit in Suit::ALL {
            let suit_cards: Vec<Card> = self.cards().filter(|c| c.suit() == suit).collect();
            if suit_cards.is_empty() {
                continue;
            }
            if !first_suit {
                write!(f, " ")?;
            }
            first_suit = false;
            for (i, card) in suit_cards.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", card)?;
            }
        }
        Ok(())
    }
}

/// Iterator over cards in a `CardSet`, yielding them in bit order (Clubs Two first).
pub struct CardIter(u64);

impl Iterator for CardIter {
    type Item = Card;

    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros();
        self.0 &= self.0 - 1; // clear lowest set bit
        let suit_idx = bit / CardSet::SUIT_BITS;
        let rank_idx = bit % CardSet::SUIT_BITS;
        Some(Card::new(Suit::ALL[suit_idx as usize], Rank::ALL[rank_idx as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Rank, Suit, ALL_CARDS};

    #[test]
    fn insert_and_count() {
        let mut set = CardSet::empty();
        let cards = [
            Card::new(Suit::Clubs, Rank::Two),
            Card::new(Suit::Hearts, Rank::Ace),
            Card::new(Suit::Spades, Rank::Queen),
            Card::new(Suit::Diamonds, Rank::Ten),
            Card::new(Suit::Clubs, Rank::King),
        ];
        for c in cards {
            set.insert(c);
        }
        assert_eq!(set.count(), 5);
        for c in cards {
            assert!(set.contains(c));
        }
        assert!(!set.contains(Card::new(Suit::Hearts, Rank::Two)));
    }

    #[test]
    fn full_and_empty() {
        assert_eq!(CardSet::full().count(), 52);
        assert_eq!(CardSet::empty().count(), 0);
        assert!(CardSet::empty().is_empty());
        assert!(!CardSet::full().is_empty());
    }

    #[test]
    fn union_disjoint() {
        let a = CardSet::from_cards([
            Card::new(Suit::Clubs, Rank::Two),
            Card::new(Suit::Clubs, Rank::Three),
        ]);
        let b = CardSet::from_cards([
            Card::new(Suit::Hearts, Rank::Ace),
            Card::new(Suit::Hearts, Rank::King),
        ]);
        let u = a | b;
        assert_eq!(u.count(), 4);
    }

    #[test]
    fn intersection_disjoint_is_empty() {
        let a = CardSet::from_cards([Card::new(Suit::Clubs, Rank::Two)]);
        let b = CardSet::from_cards([Card::new(Suit::Hearts, Rank::Ace)]);
        assert!((a & b).is_empty());
    }

    #[test]
    fn cards_of_suit_filters() {
        let mut set = CardSet::empty();
        set.insert(Card::new(Suit::Hearts, Rank::Ace));
        set.insert(Card::new(Suit::Hearts, Rank::King));
        set.insert(Card::new(Suit::Clubs, Rank::Two));
        set.insert(Card::new(Suit::Spades, Rank::Queen));

        let hearts = set.cards_of_suit(Suit::Hearts);
        assert_eq!(hearts.count(), 2);
        assert!(hearts.contains(Card::new(Suit::Hearts, Rank::Ace)));
        assert!(hearts.contains(Card::new(Suit::Hearts, Rank::King)));
        assert!(!hearts.contains(Card::new(Suit::Clubs, Rank::Two)));
    }

    #[test]
    fn remove_card() {
        let mut set = CardSet::from_cards([
            Card::new(Suit::Clubs, Rank::Two),
            Card::new(Suit::Clubs, Rank::Three),
        ]);
        assert_eq!(set.count(), 2);
        set.remove(Card::new(Suit::Clubs, Rank::Two));
        assert_eq!(set.count(), 1);
        assert!(!set.contains(Card::new(Suit::Clubs, Rank::Two)));
        assert!(set.contains(Card::new(Suit::Clubs, Rank::Three)));
    }

    #[test]
    fn iterator_yields_inserted_cards() {
        let cards = [
            Card::new(Suit::Diamonds, Rank::Jack),
            Card::new(Suit::Hearts, Rank::Two),
            Card::new(Suit::Clubs, Rank::Ace),
        ];
        let set = CardSet::from_cards(cards);
        let mut collected: Vec<Card> = set.cards().collect();
        collected.sort();
        let mut expected = cards.to_vec();
        expected.sort();
        assert_eq!(collected, expected);
    }

    #[test]
    fn suit_mask_single_card() {
        let set = CardSet::from_cards([Card::new(Suit::Spades, Rank::Queen)]);
        let mask = set.suit_mask(Suit::Spades);
        assert_eq!(mask, 1 << Rank::Queen.index());
        assert_eq!(set.suit_mask(Suit::Clubs), 0);
    }

    #[test]
    fn has_suit() {
        let set = CardSet::from_cards([
            Card::new(Suit::Hearts, Rank::Ace),
            Card::new(Suit::Clubs, Rank::Two),
        ]);
        assert!(set.has_suit(Suit::Hearts));
        assert!(set.has_suit(Suit::Clubs));
        assert!(!set.has_suit(Suit::Diamonds));
        assert!(!set.has_suit(Suit::Spades));
    }

    #[test]
    fn difference() {
        let all = CardSet::full();
        let one = CardSet::from_cards([Card::new(Suit::Clubs, Rank::Two)]);
        let diff = all - one;
        assert_eq!(diff.count(), 51);
        assert!(!diff.contains(Card::new(Suit::Clubs, Rank::Two)));
    }

    #[test]
    fn from_all_cards_equals_full() {
        let set = CardSet::from_cards(ALL_CARDS);
        assert_eq!(set, CardSet::full());
    }
}
