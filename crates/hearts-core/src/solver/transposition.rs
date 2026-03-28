use crate::card_set::CardSet;
use crate::trick::PlayerIndex;
use crate::types::Card;

/// Zobrist hash key for game positions.
/// Pre-computed random values for each (player, card) pair.
pub struct ZobristKeys {
    /// keys[player][card_bit_position]
    card_keys: [[u64; 52]; 4],
    /// Key for current player
    player_keys: [u64; 4],
    /// Keys for cards in the current trick (position 0..3, card bit)
    trick_keys: [[u64; 52]; 4],
}

impl ZobristKeys {
    pub fn new(seed: u64) -> Self {
        // Simple LCG-based PRNG for generating keys
        let mut state = seed;
        let mut next = || -> u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };

        let mut card_keys = [[0u64; 52]; 4];
        for p in 0..4 {
            for c in 0..52 {
                card_keys[p][c] = next();
            }
        }

        let mut player_keys = [0u64; 4];
        for p in 0..4 {
            player_keys[p] = next();
        }

        let mut trick_keys = [[0u64; 52]; 4];
        for pos in 0..4 {
            for c in 0..52 {
                trick_keys[pos][c] = next();
            }
        }

        ZobristKeys {
            card_keys,
            player_keys,
            trick_keys,
        }
    }

    fn card_index(card: Card) -> usize {
        card.suit().index() * 13 + card.bit_index() as usize
    }

    pub fn hash_position(
        &self,
        hands: &[CardSet; 4],
        current_player: PlayerIndex,
        trick_cards: &[(PlayerIndex, Card)],
    ) -> u64 {
        let mut h = 0u64;

        // Hash each player's hand
        for p in 0..4 {
            for card in hands[p].cards() {
                h ^= self.card_keys[p][Self::card_index(card)];
            }
        }

        // Hash current player
        h ^= self.player_keys[current_player.index()];

        // Hash cards in current trick
        for (pos, (_, card)) in trick_cards.iter().enumerate() {
            h ^= self.trick_keys[pos][Self::card_index(*card)];
        }

        h
    }
}

/// A fixed-size transposition table with always-replace policy.
pub struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    mask: usize,
}

#[derive(Clone)]
pub struct TTEntry {
    pub hash: u64,
    pub scores: [i32; 4],
}

impl TranspositionTable {
    /// Create a table with 2^size_bits entries.
    pub fn new(size_bits: u32) -> Self {
        let size = 1usize << size_bits;
        TranspositionTable {
            entries: vec![None; size],
            mask: size - 1,
        }
    }

    pub fn probe(&self, hash: u64) -> Option<&TTEntry> {
        let idx = hash as usize & self.mask;
        self.entries[idx].as_ref().filter(|e| e.hash == hash)
    }

    pub fn store(&mut self, hash: u64, scores: [i32; 4]) {
        let idx = hash as usize & self.mask;
        self.entries[idx] = Some(TTEntry { hash, scores });
    }

    pub fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_set::CardSet;
    use crate::trick::PlayerIndex;
    use crate::types::{Card, Rank, Suit};

    #[test]
    fn tt_probe_empty_returns_none() {
        let tt = TranspositionTable::new(10);
        assert!(tt.probe(12345).is_none());
    }

    #[test]
    fn tt_store_and_probe() {
        let mut tt = TranspositionTable::new(10);
        tt.store(12345, [1, 2, 3, 4]);
        let entry = tt.probe(12345).unwrap();
        assert_eq!(entry.scores, [1, 2, 3, 4]);
    }

    #[test]
    fn tt_collision_replaces() {
        let mut tt = TranspositionTable::new(10);
        let size = 1 << 10;
        // Two hashes that map to the same bucket
        let h1 = 42u64;
        let h2 = h1 + size as u64;
        tt.store(h1, [1, 2, 3, 4]);
        tt.store(h2, [5, 6, 7, 8]);
        // h1 should be gone (replaced)
        assert!(tt.probe(h1).is_none());
        assert_eq!(tt.probe(h2).unwrap().scores, [5, 6, 7, 8]);
    }

    #[test]
    fn zobrist_different_positions_different_hashes() {
        let keys = ZobristKeys::new(42);
        let hands1 = [
            CardSet::from_cards([Card::new(Suit::Clubs, Rank::Ace)]),
            CardSet::empty(),
            CardSet::empty(),
            CardSet::empty(),
        ];
        let hands2 = [
            CardSet::empty(),
            CardSet::from_cards([Card::new(Suit::Clubs, Rank::Ace)]),
            CardSet::empty(),
            CardSet::empty(),
        ];
        let h1 = keys.hash_position(&hands1, PlayerIndex::P0, &[]);
        let h2 = keys.hash_position(&hands2, PlayerIndex::P0, &[]);
        assert_ne!(h1, h2);
    }
}
