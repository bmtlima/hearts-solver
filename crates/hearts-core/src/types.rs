use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum Suit {
    Clubs = 0,
    Diamonds = 1,
    Spades = 2,
    Hearts = 3,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Spades, Suit::Hearts];

    pub fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ch = match self {
            Suit::Clubs => 'c',
            Suit::Diamonds => 'd',
            Suit::Spades => 's',
            Suit::Hearts => 'h',
        };
        write!(f, "{}", ch)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum Rank {
    Two = 0,
    Three = 1,
    Four = 2,
    Five = 3,
    Six = 4,
    Seven = 5,
    Eight = 6,
    Nine = 7,
    Ten = 8,
    Jack = 9,
    Queen = 10,
    King = 11,
    Ace = 12,
}

impl Rank {
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    pub fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Rank::Two => "2",
            Rank::Three => "3",
            Rank::Four => "4",
            Rank::Five => "5",
            Rank::Six => "6",
            Rank::Seven => "7",
            Rank::Eight => "8",
            Rank::Nine => "9",
            Rank::Ten => "T",
            Rank::Jack => "J",
            Rank::Queen => "Q",
            Rank::King => "K",
            Rank::Ace => "A",
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Card {
    suit: Suit,
    rank: Rank,
}

impl Card {
    pub const fn new(suit: Suit, rank: Rank) -> Self {
        Card { suit, rank }
    }

    pub fn suit(self) -> Suit {
        self.suit
    }

    pub fn rank(self) -> Rank {
        self.rank
    }

    /// Bit index within the suit (0 = Two, 12 = Ace).
    pub fn bit_index(self) -> u8 {
        self.rank as u8
    }

    /// Point value: Hearts = 1, Queen of Spades = 13, everything else = 0.
    pub fn point_value(self) -> i32 {
        match (self.suit, self.rank) {
            (Suit::Hearts, _) => 1,
            (Suit::Spades, Rank::Queen) => 13,
            _ => 0,
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank, self.suit)
    }
}

/// All 52 cards, ordered by suit (Clubs, Diamonds, Spades, Hearts) then rank (Two..Ace).
pub const ALL_CARDS: [Card; 52] = {
    let mut cards = [Card::new(Suit::Clubs, Rank::Two); 52];
    let suits = [Suit::Clubs, Suit::Diamonds, Suit::Spades, Suit::Hearts];
    let ranks = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];
    let mut i = 0;
    while i < 4 {
        let mut j = 0;
        while j < 13 {
            cards[i * 13 + j] = Card::new(suits[i], ranks[j]);
            j += 1;
        }
        i += 1;
    }
    cards
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queen_of_spades_worth_13() {
        assert_eq!(Card::new(Suit::Spades, Rank::Queen).point_value(), 13);
    }

    #[test]
    fn all_hearts_worth_1() {
        for rank in Rank::ALL {
            assert_eq!(Card::new(Suit::Hearts, rank).point_value(), 1);
        }
    }

    #[test]
    fn non_point_cards_worth_0() {
        for rank in Rank::ALL {
            assert_eq!(Card::new(Suit::Clubs, rank).point_value(), 0);
            assert_eq!(Card::new(Suit::Diamonds, rank).point_value(), 0);
        }
        // Spades other than Queen
        for rank in Rank::ALL {
            if rank != Rank::Queen {
                assert_eq!(Card::new(Suit::Spades, rank).point_value(), 0);
            }
        }
    }

    #[test]
    fn all_cards_has_52_unique() {
        assert_eq!(ALL_CARDS.len(), 52);
        let mut seen = std::collections::HashSet::new();
        for card in ALL_CARDS {
            assert!(seen.insert(card), "duplicate card: {}", card);
        }
    }

    #[test]
    fn bit_index_range() {
        assert_eq!(Card::new(Suit::Clubs, Rank::Two).bit_index(), 0);
        assert_eq!(Card::new(Suit::Clubs, Rank::Ace).bit_index(), 12);
        for rank in Rank::ALL {
            let idx = Card::new(Suit::Hearts, rank).bit_index();
            assert!(idx <= 12);
        }
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", Card::new(Suit::Spades, Rank::Queen)), "Qs");
        assert_eq!(format!("{}", Card::new(Suit::Hearts, Rank::Ace)), "Ah");
        assert_eq!(format!("{}", Card::new(Suit::Clubs, Rank::Two)), "2c");
        assert_eq!(format!("{}", Card::new(Suit::Diamonds, Rank::Ten)), "Td");
    }

    #[test]
    fn total_point_value_is_26() {
        let total: i32 = ALL_CARDS.iter().map(|c| c.point_value()).sum();
        assert_eq!(total, 26);
    }
}
