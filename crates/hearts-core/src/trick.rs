use std::fmt;

use crate::types::{Card, Rank, Suit};

/// Index of a player (0..3).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct PlayerIndex(u8);

impl PlayerIndex {
    pub const P0: PlayerIndex = PlayerIndex(0);
    pub const P1: PlayerIndex = PlayerIndex(1);
    pub const P2: PlayerIndex = PlayerIndex(2);
    pub const P3: PlayerIndex = PlayerIndex(3);

    pub const fn new(idx: u8) -> Self {
        debug_assert!(idx < 4);
        PlayerIndex(idx)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Next player clockwise (0→1→2→3→0).
    pub fn next(self) -> PlayerIndex {
        PlayerIndex((self.0 + 1) % 4)
    }

    pub fn all() -> [PlayerIndex; 4] {
        [Self::P0, Self::P1, Self::P2, Self::P3]
    }
}

impl fmt::Display for PlayerIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.0)
    }
}

/// A trick: up to 4 cards played by 4 players, led by a specific player.
#[derive(Clone, Debug)]
pub struct Trick {
    leader: PlayerIndex,
    cards: [(PlayerIndex, Card); 4],
    len: u8,
}

impl Trick {
    pub fn new(leader: PlayerIndex) -> Self {
        // Placeholder cards — only entries 0..len are valid
        let placeholder = Card::new(Suit::Clubs, Rank::Two);
        Trick {
            leader,
            cards: [
                (leader, placeholder),
                (leader, placeholder),
                (leader, placeholder),
                (leader, placeholder),
            ],
            len: 0,
        }
    }

    pub fn leader(&self) -> PlayerIndex {
        self.leader
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_complete(&self) -> bool {
        self.len == 4
    }

    /// The suit of the first card played (the led suit).
    pub fn led_suit(&self) -> Option<Suit> {
        if self.len > 0 {
            Some(self.cards[0].1.suit())
        } else {
            None
        }
    }

    /// Play a card into this trick.
    pub fn play(&mut self, player: PlayerIndex, card: Card) {
        debug_assert!(!self.is_complete(), "trick already complete");
        self.cards[self.len as usize] = (player, card);
        self.len += 1;
    }

    /// Remove the last card played (for undo).
    pub fn unplay(&mut self) {
        debug_assert!(self.len > 0, "trick is empty");
        self.len -= 1;
    }

    /// The (player, card) pairs played so far.
    pub fn played_cards(&self) -> &[(PlayerIndex, Card)] {
        &self.cards[..self.len as usize]
    }

    /// Winner of a complete trick: the player who played the highest card of the led suit.
    pub fn winner(&self) -> (PlayerIndex, Card) {
        debug_assert!(self.is_complete(), "trick not complete");
        let led = self.led_suit().unwrap();
        self.cards[..4]
            .iter()
            .filter(|(_, c)| c.suit() == led)
            .max_by_key(|(_, c)| c.rank())
            .copied()
            .unwrap()
    }

    /// Total point value of all cards in this trick.
    pub fn points(&self) -> i32 {
        self.cards[..self.len as usize]
            .iter()
            .map(|(_, c)| c.point_value())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Card, Rank, Suit};

    #[test]
    fn player_index_next_wraps() {
        assert_eq!(PlayerIndex::P0.next(), PlayerIndex::P1);
        assert_eq!(PlayerIndex::P1.next(), PlayerIndex::P2);
        assert_eq!(PlayerIndex::P2.next(), PlayerIndex::P3);
        assert_eq!(PlayerIndex::P3.next(), PlayerIndex::P0);
    }

    #[test]
    fn trick_hearts_led_winner_is_highest_heart() {
        let mut trick = Trick::new(PlayerIndex::P0);
        trick.play(PlayerIndex::P0, Card::new(Suit::Hearts, Rank::Five));
        trick.play(PlayerIndex::P1, Card::new(Suit::Hearts, Rank::Three));
        trick.play(PlayerIndex::P2, Card::new(Suit::Hearts, Rank::King));
        trick.play(PlayerIndex::P3, Card::new(Suit::Hearts, Rank::Jack));

        let (winner, card) = trick.winner();
        assert_eq!(winner, PlayerIndex::P2);
        assert_eq!(card, Card::new(Suit::Hearts, Rank::King));
    }

    #[test]
    fn off_suit_cards_dont_win() {
        let mut trick = Trick::new(PlayerIndex::P0);
        trick.play(PlayerIndex::P0, Card::new(Suit::Clubs, Rank::Five));
        trick.play(PlayerIndex::P1, Card::new(Suit::Hearts, Rank::Ace)); // off-suit, doesn't count
        trick.play(PlayerIndex::P2, Card::new(Suit::Clubs, Rank::Ten));
        trick.play(PlayerIndex::P3, Card::new(Suit::Clubs, Rank::Three));

        let (winner, _) = trick.winner();
        assert_eq!(winner, PlayerIndex::P2); // Tc is highest club
    }

    #[test]
    fn points_with_queen_and_hearts() {
        let mut trick = Trick::new(PlayerIndex::P0);
        trick.play(PlayerIndex::P0, Card::new(Suit::Spades, Rank::Queen)); // 13
        trick.play(PlayerIndex::P1, Card::new(Suit::Hearts, Rank::Two));   // 1
        trick.play(PlayerIndex::P2, Card::new(Suit::Hearts, Rank::Three)); // 1
        trick.play(PlayerIndex::P3, Card::new(Suit::Clubs, Rank::Ace));    // 0

        assert_eq!(trick.points(), 15);
    }

    #[test]
    fn points_no_point_cards() {
        let mut trick = Trick::new(PlayerIndex::P0);
        trick.play(PlayerIndex::P0, Card::new(Suit::Clubs, Rank::Two));
        trick.play(PlayerIndex::P1, Card::new(Suit::Clubs, Rank::Three));
        trick.play(PlayerIndex::P2, Card::new(Suit::Clubs, Rank::Four));
        trick.play(PlayerIndex::P3, Card::new(Suit::Clubs, Rank::Five));

        assert_eq!(trick.points(), 0);
    }

    #[test]
    fn is_complete_transitions() {
        let mut trick = Trick::new(PlayerIndex::P0);
        assert!(!trick.is_complete());
        assert!(trick.is_empty());

        trick.play(PlayerIndex::P0, Card::new(Suit::Clubs, Rank::Two));
        assert!(!trick.is_complete());
        assert!(!trick.is_empty());

        trick.play(PlayerIndex::P1, Card::new(Suit::Clubs, Rank::Three));
        trick.play(PlayerIndex::P2, Card::new(Suit::Clubs, Rank::Four));
        assert!(!trick.is_complete());

        trick.play(PlayerIndex::P3, Card::new(Suit::Clubs, Rank::Five));
        assert!(trick.is_complete());
    }

    #[test]
    fn led_suit() {
        let mut trick = Trick::new(PlayerIndex::P0);
        assert_eq!(trick.led_suit(), None);

        trick.play(PlayerIndex::P0, Card::new(Suit::Diamonds, Rank::Jack));
        assert_eq!(trick.led_suit(), Some(Suit::Diamonds));
    }
}
