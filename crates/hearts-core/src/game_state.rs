use crate::card_set::CardSet;
use crate::deck::DeckConfig;
use crate::trick::{PlayerIndex, Trick};
use crate::types::{Card, Suit};

/// Result of completing a trick.
#[derive(Clone, Debug)]
pub struct TrickResult {
    pub winner: PlayerIndex,
    pub points: i32,
    /// The 4 cards played in this trick, in play order (needed for void inference).
    pub cards: [(PlayerIndex, Card); 4],
}

/// Information needed to undo a play_card call.
#[derive(Clone, Debug)]
pub struct UndoInfo {
    card: Card,
    player: PlayerIndex,
    hearts_broken_before: bool,
    is_first_trick_before: bool,
    /// If the play completed a trick, stores the previous trick and trick_number.
    completed_trick: Option<CompletedTrickUndo>,
}

#[derive(Clone, Debug)]
struct CompletedTrickUndo {
    trick: Trick,
    winner: PlayerIndex,
    points: i32,
    trick_number: usize,
}

/// Full game state for a Hearts hand.
#[derive(Clone, Debug)]
pub struct GameState {
    pub hands: [CardSet; 4],
    pub current_trick: Trick,
    pub points_taken: [i32; 4],
    pub hearts_broken: bool,
    pub current_player: PlayerIndex,
    pub deck_config: DeckConfig,
    pub trick_number: usize,
    pub cards_played: CardSet,
    pub is_first_trick: bool,
    pub trick_history: Vec<TrickResult>,
}

impl GameState {
    pub fn new(hands: [CardSet; 4], deck_config: DeckConfig, first_leader: PlayerIndex) -> Self {
        GameState {
            hands,
            current_trick: Trick::new(first_leader),
            points_taken: [0; 4],
            hearts_broken: false,
            current_player: first_leader,
            deck_config,
            trick_number: 0,
            cards_played: CardSet::empty(),
            is_first_trick: true,
            trick_history: Vec::new(),
        }
    }

    /// Create a game state where the first leader is the player holding the first lead card.
    pub fn new_with_deal(hands: [CardSet; 4], deck_config: DeckConfig) -> Self {
        let first_card = deck_config.first_lead_card();
        let leader = PlayerIndex::all()
            .into_iter()
            .find(|&p| hands[p.index()].contains(first_card))
            .expect("no player holds the first lead card");
        Self::new(hands, deck_config, leader)
    }

    /// The set of legal cards the current player can play.
    pub fn legal_moves(&self) -> CardSet {
        let hand = self.hands[self.current_player.index()];

        if hand.is_empty() {
            return CardSet::empty();
        }

        // If leading (trick is empty)
        if self.current_trick.is_empty() {
            return self.legal_leads(hand);
        }

        // Following: must follow suit if possible
        let led_suit = self.current_trick.led_suit().unwrap();
        let in_suit = hand.cards_of_suit(led_suit);

        if !in_suit.is_empty() {
            // Must follow suit
            if self.is_first_trick {
                // On first trick, can't play point cards if we have non-point alternatives in suit
                let non_point = CardSet::from_cards(in_suit.cards().filter(|c| c.point_value() == 0));
                if non_point.is_empty() {
                    in_suit
                } else {
                    non_point
                }
            } else {
                in_suit
            }
        } else {
            // Void in led suit: can play anything, but first trick restrictions apply
            if self.is_first_trick {
                // Can't play hearts or Qs on first trick (unless that's all we have)
                let non_point = CardSet::from_cards(hand.cards().filter(|c| c.point_value() == 0));
                if non_point.is_empty() {
                    hand // forced: only point cards
                } else {
                    non_point
                }
            } else {
                hand
            }
        }
    }

    fn legal_leads(&self, hand: CardSet) -> CardSet {
        if self.is_first_trick {
            // First trick: must lead the first lead card if we have it
            let first_card = self.deck_config.first_lead_card();
            if hand.contains(first_card) {
                return CardSet::from_cards([first_card]);
            }
        }

        if !self.hearts_broken {
            // Can't lead hearts unless that's all we have
            let non_hearts = hand - hand.cards_of_suit(Suit::Hearts);
            if non_hearts.is_empty() {
                hand // forced: only hearts
            } else {
                non_hearts
            }
        } else {
            hand
        }
    }

    /// Play a card, advancing the game state.
    /// Returns Some(TrickResult) if this completed a trick, None otherwise.
    pub fn play_card(&mut self, card: Card) -> Option<TrickResult> {
        let player = self.current_player;

        // Remove card from hand
        self.hands[player.index()].remove(card);
        self.cards_played.insert(card);

        // Track hearts broken
        if card.suit() == Suit::Hearts {
            self.hearts_broken = true;
        }

        // Add to current trick
        self.current_trick.play(player, card);

        if self.current_trick.is_complete() {
            // Resolve trick
            let (winner, _) = self.current_trick.winner();
            let points = self.current_trick.points();
            self.points_taken[winner.index()] += points;

            let played = self.current_trick.played_cards();
            let cards = [played[0], played[1], played[2], played[3]];
            let result = TrickResult { winner, points, cards };
            self.trick_history.push(result.clone());

            // Start new trick
            self.trick_number += 1;
            self.is_first_trick = false;
            self.current_trick = Trick::new(winner);
            self.current_player = winner;

            Some(result)
        } else {
            self.current_player = player.next();
            None
        }
    }

    /// Play a card and return undo information for backtracking.
    pub fn play_card_with_undo(&mut self, card: Card) -> UndoInfo {
        let player = self.current_player;
        let hearts_broken_before = self.hearts_broken;
        let is_first_trick_before = self.is_first_trick;

        self.hands[player.index()].remove(card);
        self.cards_played.insert(card);

        if card.suit() == Suit::Hearts {
            self.hearts_broken = true;
        }

        self.current_trick.play(player, card);

        let completed_trick = if self.current_trick.is_complete() {
            let (winner, _) = self.current_trick.winner();
            let points = self.current_trick.points();
            self.points_taken[winner.index()] += points;

            let saved_trick = self.current_trick.clone();
            let trick_number = self.trick_number;

            // Push to history (will be popped on undo)
            let played = self.current_trick.played_cards();
            let cards = [played[0], played[1], played[2], played[3]];
            self.trick_history.push(TrickResult { winner, points, cards });

            self.trick_number += 1;
            self.is_first_trick = false;
            self.current_trick = Trick::new(winner);
            self.current_player = winner;

            Some(CompletedTrickUndo {
                trick: saved_trick,
                winner,
                points,
                trick_number,
            })
        } else {
            self.current_player = player.next();
            None
        };

        UndoInfo {
            card,
            player,
            hearts_broken_before,
            is_first_trick_before,
            completed_trick,
        }
    }

    /// Undo a previous play_card_with_undo call.
    pub fn undo_card(&mut self, undo: &UndoInfo) {
        if let Some(ref ct) = undo.completed_trick {
            // Restore the completed trick, then remove the last card to revert
            // back to the state just before the 4th card was played.
            self.current_trick = ct.trick.clone();
            self.current_trick.unplay();
            self.trick_number = ct.trick_number;
            self.points_taken[ct.winner.index()] -= ct.points;
            self.trick_history.pop();
        } else {
            self.current_trick.unplay();
        }

        self.current_player = undo.player;
        self.hands[undo.player.index()].insert(undo.card);
        self.cards_played.remove(undo.card);
        self.hearts_broken = undo.hearts_broken_before;
        self.is_first_trick = undo.is_first_trick_before;
    }

    pub fn is_game_over(&self) -> bool {
        self.hands.iter().all(|h| h.is_empty())
    }

    /// Final scores, applying moon-shot rules.
    /// Only meaningful when the game is over.
    pub fn final_scores(&self) -> [i32; 4] {
        let total = self.deck_config.total_points();
        if let Some(shooter) = self.moon_shooter() {
            let mut scores = [total; 4];
            scores[shooter.index()] = 0;
            scores
        } else {
            self.points_taken
        }
    }

    /// Returns the player who shot the moon, if any.
    /// A player shot the moon if they took ALL point cards in the deck.
    pub fn moon_shooter(&self) -> Option<PlayerIndex> {
        let total = self.deck_config.total_points();
        for p in PlayerIndex::all() {
            if self.points_taken[p.index()] == total {
                return Some(p);
            }
        }
        None
    }

    /// Whether it's still theoretically possible for `player` to shoot the moon.
    /// True if `player` has taken all point cards that have been taken so far.
    pub fn could_shoot_moon(&self, player: PlayerIndex) -> bool {
        let their_points = self.points_taken[player.index()];
        let total_points: i32 = self.points_taken.iter().sum();
        // They have all the points taken so far, and some points have been taken
        their_points == total_points && total_points > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rank;

    // Helper: build a Tiny game with specific hands
    fn tiny_hands() -> [CardSet; 4] {
        // Tiny deck: Qc,Kc,Ac, Kd,Ad, Qs,Ks,As, Jh,Qh,Kh,Ah
        [
            // P0: Qc, Qs, Jh
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Queen),
                Card::new(Suit::Spades, Rank::Queen),
                Card::new(Suit::Hearts, Rank::Jack),
            ]),
            // P1: Kc, Kd, Ks
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::King),
                Card::new(Suit::Diamonds, Rank::King),
                Card::new(Suit::Spades, Rank::King),
            ]),
            // P2: Ac, Ad, Qh
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Ace),
                Card::new(Suit::Diamonds, Rank::Ace),
                Card::new(Suit::Hearts, Rank::Queen),
            ]),
            // P3: As, Kh, Ah
            CardSet::from_cards([
                Card::new(Suit::Spades, Rank::Ace),
                Card::new(Suit::Hearts, Rank::King),
                Card::new(Suit::Hearts, Rank::Ace),
            ]),
        ]
    }

    #[test]
    fn follow_suit_required() {
        let hands = tiny_hands();
        // P0 leads (has Qc, the lowest club in Tiny deck = Qc)
        let state = GameState::new(hands, DeckConfig::Tiny, PlayerIndex::P0);
        // First trick, P0 must lead Qc (first lead card)
        let moves = state.legal_moves();
        assert_eq!(moves.count(), 1);
        assert!(moves.contains(Card::new(Suit::Clubs, Rank::Queen)));
    }

    #[test]
    fn void_in_suit_play_anything_except_points_on_first_trick() {
        let hands = tiny_hands();
        let mut state = GameState::new(hands, DeckConfig::Tiny, PlayerIndex::P0);

        // P0 leads Qc (forced)
        state.play_card(Card::new(Suit::Clubs, Rank::Queen));
        // P1 has Kc, must follow suit
        let p1_moves = state.legal_moves();
        assert!(p1_moves.contains(Card::new(Suit::Clubs, Rank::King)));
        state.play_card(Card::new(Suit::Clubs, Rank::King));

        // P2 has Ac, must follow suit
        state.play_card(Card::new(Suit::Clubs, Rank::Ace));

        // P3 has no clubs (As, Kh, Ah) — void, but first trick: can't play hearts
        let p3_moves = state.legal_moves();
        // P3 should only be able to play As (no points)
        assert!(p3_moves.contains(Card::new(Suit::Spades, Rank::Ace)));
        assert!(!p3_moves.contains(Card::new(Suit::Hearts, Rank::King)));
        assert!(!p3_moves.contains(Card::new(Suit::Hearts, Rank::Ace)));
    }

    #[test]
    fn first_trick_forced_all_points() {
        // Player has only point cards and is void in led suit
        let hands = [
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Queen),
                Card::new(Suit::Clubs, Rank::King),
                Card::new(Suit::Clubs, Rank::Ace),
            ]),
            // P1 has only hearts and Qs — all points
            CardSet::from_cards([
                Card::new(Suit::Hearts, Rank::Jack),
                Card::new(Suit::Spades, Rank::Queen),
                Card::new(Suit::Hearts, Rank::Queen),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Diamonds, Rank::King),
                Card::new(Suit::Diamonds, Rank::Ace),
                Card::new(Suit::Spades, Rank::King),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Spades, Rank::Ace),
                Card::new(Suit::Hearts, Rank::King),
                Card::new(Suit::Hearts, Rank::Ace),
            ]),
        ];
        let mut state = GameState::new(hands, DeckConfig::Tiny, PlayerIndex::P0);
        state.play_card(Card::new(Suit::Clubs, Rank::Queen));

        // P1 void in clubs, only has point cards → all 3 are legal
        let p1_moves = state.legal_moves();
        assert_eq!(p1_moves.count(), 3);
    }

    #[test]
    fn hearts_cannot_lead_until_broken() {
        let hands = [
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Queen),
                Card::new(Suit::Hearts, Rank::Jack),
                Card::new(Suit::Hearts, Rank::Queen),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::King),
                Card::new(Suit::Diamonds, Rank::King),
                Card::new(Suit::Spades, Rank::King),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Ace),
                Card::new(Suit::Diamonds, Rank::Ace),
                Card::new(Suit::Spades, Rank::Queen),
            ]),
            CardSet::from_cards([
                Card::new(Suit::Spades, Rank::Ace),
                Card::new(Suit::Hearts, Rank::King),
                Card::new(Suit::Hearts, Rank::Ace),
            ]),
        ];
        let mut state = GameState::new(hands, DeckConfig::Tiny, PlayerIndex::P0);

        // First trick: P0 leads Qc (forced)
        state.play_card(Card::new(Suit::Clubs, Rank::Queen));
        state.play_card(Card::new(Suit::Clubs, Rank::King));
        state.play_card(Card::new(Suit::Clubs, Rank::Ace));
        // P3 void in clubs, first trick, can only play As (not hearts)
        state.play_card(Card::new(Suit::Spades, Rank::Ace));
        // P2 wins with Ac

        // Second trick: P2 leads. Has Ad, Qs. Hearts not broken.
        assert!(!state.hearts_broken);
        let leads = state.legal_moves();
        assert!(leads.contains(Card::new(Suit::Diamonds, Rank::Ace)));
        assert!(leads.contains(Card::new(Suit::Spades, Rank::Queen)));
    }

    #[test]
    fn hearts_broken_allows_lead() {
        let mut state = GameState::new(tiny_hands(), DeckConfig::Tiny, PlayerIndex::P0);
        state.hearts_broken = true;
        state.is_first_trick = false;

        // P0 has Qc, Qs, Jh — all should be legal leads now
        let leads = state.legal_moves();
        assert_eq!(leads.count(), 3);
        assert!(leads.contains(Card::new(Suit::Hearts, Rank::Jack)));
    }

    #[test]
    fn trick_winner_becomes_next_leader() {
        let mut state = GameState::new(tiny_hands(), DeckConfig::Tiny, PlayerIndex::P0);

        // Trick 1: P0 leads Qc
        state.play_card(Card::new(Suit::Clubs, Rank::Queen));
        state.play_card(Card::new(Suit::Clubs, Rank::King));
        state.play_card(Card::new(Suit::Clubs, Rank::Ace)); // P2 wins
        let result = state.play_card(Card::new(Suit::Spades, Rank::Ace)); // P3 plays off-suit
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.winner, PlayerIndex::P2);
        assert_eq!(state.current_player, PlayerIndex::P2);
    }

    #[test]
    fn points_accumulate() {
        let mut state = GameState::new(tiny_hands(), DeckConfig::Tiny, PlayerIndex::P0);

        // Trick 1: Qc, Kc, Ac, As — no points (P3 plays As off-suit)
        state.play_card(Card::new(Suit::Clubs, Rank::Queen));
        state.play_card(Card::new(Suit::Clubs, Rank::King));
        state.play_card(Card::new(Suit::Clubs, Rank::Ace));
        let r = state.play_card(Card::new(Suit::Spades, Rank::Ace));
        assert_eq!(r.unwrap().points, 0);
        assert_eq!(state.points_taken, [0, 0, 0, 0]);
    }

    #[test]
    fn complete_tiny_game() {
        let hands = [
            // P0: Qc, Kd, Jh
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Queen),
                Card::new(Suit::Diamonds, Rank::King),
                Card::new(Suit::Hearts, Rank::Jack),
            ]),
            // P1: Kc, Ad, Ks
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::King),
                Card::new(Suit::Diamonds, Rank::Ace),
                Card::new(Suit::Spades, Rank::King),
            ]),
            // P2: Ac, Qs, Qh
            CardSet::from_cards([
                Card::new(Suit::Clubs, Rank::Ace),
                Card::new(Suit::Spades, Rank::Queen),
                Card::new(Suit::Hearts, Rank::Queen),
            ]),
            // P3: As, Kh, Ah
            CardSet::from_cards([
                Card::new(Suit::Spades, Rank::Ace),
                Card::new(Suit::Hearts, Rank::King),
                Card::new(Suit::Hearts, Rank::Ace),
            ]),
        ];
        let mut state = GameState::new(hands, DeckConfig::Tiny, PlayerIndex::P0);

        // Trick 1: P0 leads Qc (first lead card), P1:Kc, P2:Ac, P3:As(void)
        state.play_card(Card::new(Suit::Clubs, Rank::Queen));
        state.play_card(Card::new(Suit::Clubs, Rank::King));
        state.play_card(Card::new(Suit::Clubs, Rank::Ace));
        state.play_card(Card::new(Suit::Spades, Rank::Ace));
        // P2 wins (Ac highest club), 0 points

        // Trick 2: P2 leads Qs, P3:Kh(void), P0:Jh(void), P1:Ks
        state.play_card(Card::new(Suit::Spades, Rank::Queen));
        state.play_card(Card::new(Suit::Hearts, Rank::King));
        state.play_card(Card::new(Suit::Hearts, Rank::Jack));
        state.play_card(Card::new(Suit::Spades, Rank::King));
        // P1 wins (Ks highest spade), takes Qs(13) + Kh(1) + Jh(1) = 15 points

        // Trick 3: P1 leads Ad, P2:Qh(void), P3:Ah(void), P0:Kd
        state.play_card(Card::new(Suit::Diamonds, Rank::Ace));
        state.play_card(Card::new(Suit::Hearts, Rank::Queen));
        state.play_card(Card::new(Suit::Hearts, Rank::Ace));
        state.play_card(Card::new(Suit::Diamonds, Rank::King));
        // P1 wins (Ad highest diamond), takes Qh(1) + Ah(1) = 2 points

        assert!(state.is_game_over());
        // P1 took all 17 points (Qs=13 + Kh+Jh+Qh+Ah=4) = moon shot!
        assert_eq!(state.points_taken, [0, 17, 0, 0]);
        let scores = state.final_scores();
        assert_eq!(scores, [17, 0, 17, 17]); // moon: shooter gets 0, others get 17
    }

    #[test]
    fn scores_sum_to_26_or_78() {
        // Play a game and verify score invariant
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Tiny.deal(&mut rng);
            let mut state = GameState::new_with_deal(hands, DeckConfig::Tiny);

            // Play game: always pick first legal move
            while !state.is_game_over() {
                let moves = state.legal_moves();
                let card = moves.cards().next().unwrap();
                state.play_card(card);
            }

            let scores = state.final_scores();
            let total: i32 = scores.iter().sum();
            let deck_points = DeckConfig::Tiny.total_points();
            let moon_total = deck_points * 3; // shooter gets 0, 3 others get deck_points
            assert!(
                total == deck_points || total == moon_total,
                "seed {}: scores {:?} sum to {} (expected {} or {})",
                seed, scores, total, deck_points, moon_total
            );
        }
    }
}
