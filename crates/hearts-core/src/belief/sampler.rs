use rand::seq::SliceRandom;
use rand::Rng;

use crate::belief::observation::Observation;
use crate::card_set::CardSet;
use crate::types::Card;

/// Sample a single consistent world from the observation.
/// Returns [CardSet; 4] where the viewer's hand is unchanged and
/// opponents' hands respect void constraints and hand sizes.
/// Returns None if constraints are unsatisfiable (shouldn't happen in valid games).
pub fn sample_world(obs: &Observation, rng: &mut impl Rng) -> Option<[CardSet; 4]> {
    let viewer = obs.viewer.index();
    let unknown: Vec<Card> = obs.unknown_cards().cards().collect();

    // Use hand_sizes() which correctly accounts for current-trick state
    let target_sizes = obs.hand_sizes();

    let opp_indices: Vec<usize> = (0..4).filter(|&p| p != viewer).collect();

    // Rejection sampling: shuffle unknowns, try to deal respecting voids
    for _attempt in 0..1000 {
        let mut shuffled = unknown.clone();
        shuffled.shuffle(rng);

        if let Some(hands) = try_deal(&shuffled, obs, &opp_indices, &target_sizes) {
            return Some(hands);
        }
    }

    None // couldn't satisfy constraints in 1000 attempts
}

fn try_deal(
    shuffled: &[Card],
    obs: &Observation,
    opp_indices: &[usize],
    target_sizes: &[usize; 4],
) -> Option<[CardSet; 4]> {
    let viewer = obs.viewer.index();
    let mut hands = [CardSet::empty(); 4];
    hands[viewer] = obs.my_hand;

    let mut pool: Vec<Card> = shuffled.to_vec();

    for &p in opp_indices {
        let need = target_sizes[p];
        let mut dealt = 0;
        let mut used_indices = Vec::new();

        for (i, &card) in pool.iter().enumerate() {
            if dealt >= need {
                break;
            }
            if !obs.voids[p][card.suit().index()] {
                hands[p].insert(card);
                used_indices.push(i);
                dealt += 1;
            }
        }

        if dealt < need {
            return None;
        }

        // Remove used cards from pool (in reverse to preserve indices)
        for &i in used_indices.iter().rev() {
            pool.swap_remove(i);
        }
    }

    Some(hands)
}

/// Sample multiple consistent worlds.
pub fn sample_worlds(
    obs: &Observation,
    n: usize,
    rng: &mut impl Rng,
) -> Vec<[CardSet; 4]> {
    let mut worlds = Vec::with_capacity(n);
    for _ in 0..n {
        if let Some(world) = sample_world(obs, rng) {
            worlds.push(world);
        }
    }
    worlds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::observation::Observation;
    use crate::deck::DeckConfig;
    use crate::game_state::GameState;
    use crate::trick::PlayerIndex;
    use crate::types::Suit;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn sampled_world_has_correct_hand_sizes() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let obs = Observation::from_game_state(&state, PlayerIndex::P0);

        let mut sample_rng = StdRng::seed_from_u64(99);
        let world = sample_world(&obs, &mut sample_rng).unwrap();

        for p in 0..4 {
            assert_eq!(world[p].count(), 3, "P{} has {} cards", p, world[p].count());
        }
    }

    #[test]
    fn viewer_hand_unchanged() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
        let obs = Observation::from_game_state(&state, PlayerIndex::P0);

        let mut sample_rng = StdRng::seed_from_u64(99);
        let world = sample_world(&obs, &mut sample_rng).unwrap();

        assert_eq!(world[0], hands[0]);
    }

    #[test]
    fn union_equals_deck() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let obs = Observation::from_game_state(&state, PlayerIndex::P0);

        let mut sample_rng = StdRng::seed_from_u64(99);
        let world = sample_world(&obs, &mut sample_rng).unwrap();

        let union = world[0] | world[1] | world[2] | world[3];
        assert_eq!(union, DeckConfig::Tiny.deck_cards());
    }

    #[test]
    fn respects_void_constraints() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Small.deal(&mut rng);
        let mut state = GameState::new_with_deal(hands, DeckConfig::Small);

        // Play a full trick to potentially create void info
        for _ in 0..4 {
            let legal = state.legal_moves();
            let card = legal.cards().next().unwrap();
            state.play_card(card);
        }

        let obs = Observation::from_game_state(&state, PlayerIndex::P0);

        let mut sample_rng = StdRng::seed_from_u64(99);
        for _ in 0..100 {
            if let Some(world) = sample_world(&obs, &mut sample_rng) {
                // Check void constraints
                for p in 0..4 {
                    for s in Suit::ALL {
                        if obs.voids[p][s.index()] {
                            assert!(
                                !world[p].has_suit(s),
                                "P{} is void in {:?} but got cards in that suit",
                                p, s
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn roughly_uniform_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Tiny.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Tiny);
        let obs = Observation::from_game_state(&state, PlayerIndex::P0);

        let mut counts = [[0u32; 52]; 4]; // counts[player][card_bit] = times dealt
        let mut sample_rng = StdRng::seed_from_u64(0);
        let n = 10000;
        for _ in 0..n {
            let world = sample_world(&obs, &mut sample_rng).unwrap();
            for p in 1..4 {
                // only check opponents
                for card in world[p].cards() {
                    let bit = card.suit().index() * 13 + card.bit_index() as usize;
                    counts[p][bit] += 1;
                }
            }
        }

        // Each unknown card should appear roughly equally among the 3 opponents
        // With 9 unknown cards and 3 cards per opponent, each card should appear
        // in each opponent's hand ~n/3 times (with variance)
        let unknown = obs.unknown_cards();
        for card in unknown.cards() {
            let bit = card.suit().index() * 13 + card.bit_index() as usize;
            let total: u32 = counts[1][bit] + counts[2][bit] + counts[3][bit];
            assert_eq!(total, n as u32, "card {} appeared {} times, expected {}", card, total, n);
        }
    }

    #[test]
    fn sampling_performance() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Small.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Small);
        let obs = Observation::from_game_state(&state, PlayerIndex::P0);

        let mut sample_rng = StdRng::seed_from_u64(0);
        let start = std::time::Instant::now();
        let worlds = sample_worlds(&obs, 500, &mut sample_rng);
        let elapsed = start.elapsed();

        assert_eq!(worlds.len(), 500);
        assert!(elapsed.as_millis() < 100, "500 samples took {:?}", elapsed);
    }
}
