use crate::game_state::GameState;
use crate::trick::PlayerIndex;
use crate::types::{Card, Rank, Suit};

pub const FEATURE_COUNT: usize = 29;

pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "ai_points_taken",
    "opp_max_points",
    "opp_total_points",
    "hearts_remaining",
    "ai_has_qs",
    "ai_has_qs_exposed",
    "ai_has_qs_protected",
    "ai_has_as",
    "ai_has_ks",
    "qs_already_played",
    "ai_spade_count",
    "ai_heart_count",
    "ai_top_hearts",
    "ai_void_hearts",
    "hearts_in_play",
    "ai_void_count",
    "ai_top_card_count",
    "ai_top2_card_count",
    "opp_void_count",
    "ai_longest_suit",
    "ai_shortest_nonvoid_suit",
    "ai_has_lead",
    "ai_has_qs_x_opp_voids",
    "ai_has_as_x_qs_in_play",
    "ai_has_ks_x_qs_in_play",
    "ai_top_cards_x_hearts_remaining",
    "ai_has_lead_x_top_card_count",
    "ai_took_all_penalties",
    "opp_took_all_penalties",
];

/// Trained weights from Ridge regression (alpha=10.0) on 1000 samples.
/// Layout: [bias, weight_0, weight_1, ..., weight_28]
pub const EVAL_WEIGHTS: [f64; FEATURE_COUNT + 1] = [
    // bias
    11.20764576,
    // ai_points_taken
    0.3828982067,
    // opp_max_points
    0.0389682987,
    // opp_total_points
    -0.6346433063,
    // hearts_remaining
    0.1210047547,
    // ai_has_qs
    -1.084591773,
    // ai_has_qs_exposed
    -1.254299086,
    // ai_has_qs_protected
    0.8494268645,
    // ai_has_as
    0.613124761,
    // ai_has_ks
    -0.002765325076,
    // qs_already_played
    -0.01005694961,
    // ai_spade_count
    -0.2506423959,
    // ai_heart_count
    0.02479826141,
    // ai_top_hearts
    0.3492681061,
    // ai_void_hearts
    -0.916423496,
    // hearts_in_play
    0.1210047547,
    // ai_void_count
    -0.584125682,
    // ai_top_card_count
    0.5482331229,
    // ai_top2_card_count
    0.8878017958,
    // opp_void_count
    0.4669857151,
    // ai_longest_suit
    -0.4242201288,
    // ai_shortest_nonvoid_suit
    -0.05115283403,
    // ai_has_lead
    0.0,
    // ai_has_qs_x_opp_voids
    0.531469218,
    // ai_has_as_x_qs_in_play
    0.1943008282,
    // ai_has_ks_x_qs_in_play
    -0.3073955427,
    // ai_top_cards_x_hearts_remaining
    -0.1513057531,
    // ai_has_lead_x_top_card_count
    0.5482331229,
    // ai_took_all_penalties
    -0.6096681491,
    // opp_took_all_penalties
    0.6491504831,
];

/// Extract all 29 features from a game state for the given AI player.
pub fn extract_features(state: &GameState, ai_player: PlayerIndex) -> [f64; FEATURE_COUNT] {
    let ai = ai_player.index();
    let ai_hand = state.hands[ai];

    let qs = Card::new(Suit::Spades, Rank::Queen);
    let as_card = Card::new(Suit::Spades, Rank::Ace);
    let ks = Card::new(Suit::Spades, Rank::King);

    // ── Basic points ────────────────────────────────────────────────────
    let ai_points_taken = state.points_taken[ai] as f64;

    let mut opp_max_points = 0i32;
    let mut opp_total_points = 0i32;
    for p in PlayerIndex::all() {
        if p == ai_player {
            continue;
        }
        let pts = state.points_taken[p.index()];
        opp_total_points += pts;
        if pts > opp_max_points {
            opp_max_points = pts;
        }
    }

    // ── Cards remaining across all hands ────────────────────────────────
    let all_remaining = state.hands[0] | state.hands[1] | state.hands[2] | state.hands[3];
    let hearts_remaining = all_remaining.cards_of_suit(Suit::Hearts).count() as f64;

    // ── Queen of Spades features ────────────────────────────────────────
    let has_qs = ai_hand.contains(qs);
    let ai_spade_count_raw =
        ai_hand.cards_of_suit(Suit::Spades).count() as i32 - has_qs as i32;
    let ai_has_qs = has_qs as i32 as f64;
    let ai_has_qs_exposed = (has_qs && ai_spade_count_raw <= 2) as i32 as f64;
    let ai_has_qs_protected = (has_qs && ai_spade_count_raw >= 4) as i32 as f64;
    let ai_has_as = ai_hand.contains(as_card) as i32 as f64;
    let ai_has_ks = ai_hand.contains(ks) as i32 as f64;
    let qs_already_played = state.cards_played.contains(qs) as i32 as f64;
    let ai_spade_count = ai_spade_count_raw as f64;

    // ── Heart exposure ──────────────────────────────────────────────────
    let ai_hearts = ai_hand.cards_of_suit(Suit::Hearts);
    let ai_heart_count = ai_hearts.count() as f64;
    let opp_hearts = all_remaining.cards_of_suit(Suit::Hearts) - ai_hearts;
    let ai_top_hearts = if opp_hearts.is_empty() {
        ai_hearts.count() as f64
    } else {
        let opp_max_heart_rank = opp_hearts.cards().map(|c| c.rank()).max().unwrap();
        ai_hearts
            .cards()
            .filter(|c| c.rank() > opp_max_heart_rank)
            .count() as f64
    };
    let ai_void_hearts = ai_hearts.is_empty() as i32 as f64;
    let hearts_in_play = hearts_remaining;

    // ── Suit control ────────────────────────────────────────────────────
    let mut ai_void_count = 0u32;
    let mut ai_longest_suit = 0u32;
    let mut ai_shortest_nonvoid = u32::MAX;
    for suit in Suit::ALL {
        let count = ai_hand.cards_of_suit(suit).count();
        if count == 0 {
            ai_void_count += 1;
        } else {
            if count > ai_longest_suit {
                ai_longest_suit = count;
            }
            if count < ai_shortest_nonvoid {
                ai_shortest_nonvoid = count;
            }
        }
    }
    if ai_shortest_nonvoid == u32::MAX {
        ai_shortest_nonvoid = 0;
    }

    let mut ai_top_card_count = 0u32;
    let mut ai_top2_card_count = 0u32;
    for suit in Suit::ALL {
        let suit_remaining = all_remaining.cards_of_suit(suit);
        if suit_remaining.is_empty() {
            continue;
        }
        // cards() iterates low→high, so track the last two seen
        let mut highest = None;
        let mut second_highest = None;
        for card in suit_remaining.cards() {
            second_highest = highest;
            highest = Some(card);
        }
        if let Some(h) = highest {
            if ai_hand.contains(h) {
                ai_top_card_count += 1;
                ai_top2_card_count += 1;
            } else if let Some(sh) = second_highest {
                if ai_hand.contains(sh) {
                    ai_top2_card_count += 1;
                }
            }
        }
    }

    let mut opp_void_count = 0u32;
    for p in PlayerIndex::all() {
        if p == ai_player {
            continue;
        }
        for suit in Suit::ALL {
            if !state.hands[p.index()].has_suit(suit) {
                opp_void_count += 1;
            }
        }
    }

    let ai_has_lead = (state.current_player == ai_player) as i32 as f64;

    // ── Interaction terms ───────────────────────────────────────────────
    let ai_has_qs_x_opp_voids = ai_has_qs * opp_void_count as f64;
    let ai_has_as_x_qs_in_play = ai_has_as * (1.0 - qs_already_played);
    let ai_has_ks_x_qs_in_play = ai_has_ks * (1.0 - qs_already_played);
    let ai_top_cards_x_hearts_remaining = ai_top_card_count as f64 * hearts_remaining;
    let ai_has_lead_x_top_card_count = ai_has_lead * ai_top_card_count as f64;

    // ── Moon indicators ─────────────────────────────────────────────────
    let ai_took_all_penalties = state.could_shoot_moon(ai_player) as i32 as f64;
    let opp_took_all_penalties = PlayerIndex::all()
        .iter()
        .any(|&p| p != ai_player && state.could_shoot_moon(p)) as i32 as f64;

    [
        ai_points_taken,                   // 0
        opp_max_points as f64,             // 1
        opp_total_points as f64,           // 2
        hearts_remaining,                  // 3
        ai_has_qs,                         // 4
        ai_has_qs_exposed,                 // 5
        ai_has_qs_protected,               // 6
        ai_has_as,                         // 7
        ai_has_ks,                         // 8
        qs_already_played,                 // 9
        ai_spade_count,                    // 10
        ai_heart_count,                    // 11
        ai_top_hearts,                     // 12
        ai_void_hearts,                    // 13
        hearts_in_play,                    // 14
        ai_void_count as f64,              // 15
        ai_top_card_count as f64,          // 16
        ai_top2_card_count as f64,         // 17
        opp_void_count as f64,             // 18
        ai_longest_suit as f64,            // 19
        ai_shortest_nonvoid as f64,        // 20
        ai_has_lead,                       // 21
        ai_has_qs_x_opp_voids,            // 22
        ai_has_as_x_qs_in_play,           // 23
        ai_has_ks_x_qs_in_play,           // 24
        ai_top_cards_x_hearts_remaining,  // 25
        ai_has_lead_x_top_card_count,     // 26
        ai_took_all_penalties,             // 27
        opp_took_all_penalties,            // 28
    ]
}

/// Evaluate a position using the linear regression model.
/// Returns estimated AI player score (lower is better for AI).
pub fn eval_position(state: &GameState, ai_player: PlayerIndex) -> f64 {
    let features = extract_features(state, ai_player);
    let mut score = EVAL_WEIGHTS[0]; // bias
    for i in 0..FEATURE_COUNT {
        score += EVAL_WEIGHTS[i + 1] * features[i];
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::DeckConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn feature_count_matches() {
        assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
        assert_eq!(EVAL_WEIGHTS.len(), FEATURE_COUNT + 1);
    }

    #[test]
    fn features_on_fresh_full_deal() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Full.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Full);
        let ai = state.current_player;
        let features = extract_features(&state, ai);

        // At game start: no points taken
        assert_eq!(features[0], 0.0); // ai_points_taken
        assert_eq!(features[1], 0.0); // opp_max_points
        assert_eq!(features[2], 0.0); // opp_total_points

        // All 13 hearts in play
        assert_eq!(features[3], 13.0); // hearts_remaining
        assert_eq!(features[14], 13.0); // hearts_in_play

        // Qs not played yet
        assert_eq!(features[9], 0.0); // qs_already_played

        // AI is leading
        assert_eq!(features[21], 1.0); // ai_has_lead

        // No moon shooting at start (no points taken)
        assert_eq!(features[27], 0.0); // ai_took_all_penalties
        assert_eq!(features[28], 0.0); // opp_took_all_penalties
    }

    #[test]
    fn eval_returns_reasonable_score() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Full.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Full);
        let ai = state.current_player;

        let score = eval_position(&state, ai);
        // Score should be in a reasonable range for Hearts (0-26 typical)
        assert!(score > -10.0 && score < 40.0, "eval score {} out of range", score);
    }

    #[test]
    fn features_range_on_28_card_positions() {
        // Play forward to 28 cards and verify feature ranges
        for seed in 0..50 {
            let mut rng = StdRng::seed_from_u64(seed);
            let hands = DeckConfig::Full.deal(&mut rng);
            let mut state = GameState::new_with_deal(hands, DeckConfig::Full);

            // Play 6 tricks (24 cards) — always pick first legal move
            let mut cards_played = 0u32;
            while cards_played < 24 {
                let legal = state.legal_moves();
                let card = legal.cards().next().unwrap();
                state.play_card(card);
                cards_played += 1;
            }

            let cards_remaining: u32 = state.hands.iter().map(|h| h.count()).sum();
            assert_eq!(cards_remaining, 28);
            assert!(state.current_trick.is_empty());

            let ai = state.current_player;
            let f = extract_features(&state, ai);

            // Points range
            assert!(f[0] >= 0.0 && f[0] <= 26.0, "ai_points out of range: {}", f[0]);
            assert!(f[1] >= 0.0 && f[1] <= 26.0, "opp_max out of range: {}", f[1]);
            assert!(f[2] >= 0.0 && f[2] <= 26.0, "opp_total out of range: {}", f[2]);

            // Hearts remaining: 0..13
            assert!(f[3] >= 0.0 && f[3] <= 13.0);
            assert_eq!(f[3], f[14]); // hearts_remaining == hearts_in_play

            // Binary features: 0 or 1
            for &idx in &[4, 5, 6, 7, 8, 9, 13, 21, 27, 28] {
                assert!(f[idx] == 0.0 || f[idx] == 1.0, "feature {} = {} not binary", idx, f[idx]);
            }

            // Count features: non-negative
            assert!(f[10] >= 0.0); // ai_spade_count
            assert!(f[11] >= 0.0); // ai_heart_count
            assert!(f[12] >= 0.0); // ai_top_hearts
            assert!(f[15] >= 0.0 && f[15] <= 3.0); // ai_void_count
            assert!(f[16] >= 0.0 && f[16] <= 4.0); // ai_top_card_count
            assert!(f[17] >= 0.0 && f[17] <= 4.0); // ai_top2_card_count
            assert!(f[18] >= 0.0 && f[18] <= 12.0); // opp_void_count (3 opps × 4 suits)
            assert!(f[19] >= 1.0); // ai_longest_suit (must have at least 1 card)
        }
    }

    #[test]
    fn qs_features_consistent() {
        let mut rng = StdRng::seed_from_u64(42);
        let hands = DeckConfig::Full.deal(&mut rng);
        let state = GameState::new_with_deal(hands, DeckConfig::Full);

        let qs = Card::new(Suit::Spades, Rank::Queen);
        for p in PlayerIndex::all() {
            let f = extract_features(&state, p);
            let holds_qs = state.hands[p.index()].contains(qs);
            assert_eq!(f[4], holds_qs as i32 as f64);

            // If player doesn't have Qs, exposed and protected must be 0
            if !holds_qs {
                assert_eq!(f[5], 0.0); // exposed
                assert_eq!(f[6], 0.0); // protected
            }
            // exposed and protected are mutually exclusive
            assert!(!(f[5] == 1.0 && f[6] == 1.0));
        }
    }
}
