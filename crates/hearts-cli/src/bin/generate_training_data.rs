use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use hearts_core::bots::rule_bot::RuleBot;
use hearts_core::card_set::CardSet;
use hearts_core::deck::DeckConfig;
use hearts_core::game::Player;
use hearts_core::game_state::GameState;
use hearts_core::solver::eval::{extract_features, FEATURE_COUNT, FEATURE_NAMES};
use hearts_core::solver::paranoid::paranoid_solve;
use hearts_core::trick::PlayerIndex;
use hearts_core::types::{Card, Rank, Suit, ALL_CARDS};

#[derive(Parser)]
#[command(about = "Generate training data for regression eval")]
struct Args {
    /// Number of samples (positions) to generate
    #[arg(long, default_value = "1000")]
    samples: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output CSV path
    #[arg(long, default_value = "regression/data/training_data.csv")]
    output: String,

    /// Include raw card position columns (208 card + 6 scalar = 214 extra columns)
    #[arg(long)]
    raw_cards: bool,
}

// ── Position generation ─────────────────────────────────────────────────

fn duck_choose(state: &GameState, legal: CardSet, _rng: &mut impl Rng) -> Card {
    if legal.count() == 1 {
        return legal.cards().next().unwrap();
    }

    let qs = Card::new(Suit::Spades, Rank::Queen);

    if state.current_trick.is_empty() {
        let safe: Vec<Card> = legal
            .cards()
            .filter(|c| c.suit() != Suit::Hearts && *c != qs)
            .collect();
        if !safe.is_empty() {
            return *safe.iter().min_by_key(|c| c.rank() as u8).unwrap();
        }
        return legal.cards().min_by_key(|c| c.rank() as u8).unwrap();
    }

    let led_suit = state.current_trick.led_suit().unwrap();
    let in_suit = legal.cards_of_suit(led_suit);

    if !in_suit.is_empty() {
        let current_high = state
            .current_trick
            .played_cards()
            .iter()
            .filter(|(_, c)| c.suit() == led_suit)
            .map(|(_, c)| c.rank())
            .max();

        if let Some(high_rank) = current_high {
            let mut duck: Option<Card> = None;
            for card in in_suit.cards() {
                if card.rank() < high_rank {
                    if duck.map_or(true, |d| card.rank() > d.rank()) {
                        duck = Some(card);
                    }
                }
            }
            if let Some(d) = duck {
                return d;
            }
        }
        return in_suit.cards().min_by_key(|c| c.rank() as u8).unwrap();
    }

    if legal.contains(qs) {
        return qs;
    }
    let hearts = legal.cards_of_suit(Suit::Hearts);
    if !hearts.is_empty() {
        return hearts.cards().max_by_key(|c| c.rank() as u8).unwrap();
    }
    legal
        .cards()
        .max_by_key(|c| (c.suit() as u8, c.rank() as u8))
        .unwrap()
}

fn mixed_choose(
    state: &GameState,
    legal: CardSet,
    rng: &mut StdRng,
    rule_bot: &mut RuleBot,
) -> Card {
    let roll: u32 = rng.gen_range(0..100);
    if roll < 30 {
        let count = legal.count() as usize;
        let idx = rng.gen_range(0..count);
        legal.cards().nth(idx).unwrap()
    } else if roll < 65 {
        duck_choose(state, legal, rng)
    } else {
        rule_bot.choose_card(state, legal)
    }
}

fn play_to_28_cards(hands: [CardSet; 4], rng: &mut StdRng) -> Option<GameState> {
    let mut state = GameState::new_with_deal(hands, DeckConfig::Full);
    let mut rule_bot = RuleBot::new();

    for _ in 0..24 {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return None;
        }
        let card = mixed_choose(&state, legal, rng, &mut rule_bot);
        state.play_card(card);
    }

    let cards_remaining: u32 = state.hands.iter().map(|h| h.count()).sum();
    if cards_remaining != 28 || !state.current_trick.is_empty() {
        return None;
    }

    Some(state)
}

// ── Raw card data extraction ────────────────────────────────────────────

const RAW_CARD_COLS: usize = 208; // 52 cards × 4 holders
const RAW_SCALAR_COLS: usize = 6;
const RAW_TOTAL: usize = RAW_CARD_COLS + RAW_SCALAR_COLS; // 214

fn rank_char(r: Rank) -> char {
    match r {
        Rank::Two => '2',
        Rank::Three => '3',
        Rank::Four => '4',
        Rank::Five => '5',
        Rank::Six => '6',
        Rank::Seven => '7',
        Rank::Eight => '8',
        Rank::Nine => '9',
        Rank::Ten => 'T',
        Rank::Jack => 'J',
        Rank::Queen => 'Q',
        Rank::King => 'K',
        Rank::Ace => 'A',
    }
}

fn suit_char(s: Suit) -> char {
    match s {
        Suit::Clubs => 'c',
        Suit::Diamonds => 'd',
        Suit::Spades => 's',
        Suit::Hearts => 'h',
    }
}

/// Generate header names for the 214 raw columns.
fn raw_card_header() -> Vec<String> {
    let mut names = Vec::with_capacity(RAW_TOTAL);
    for card in ALL_CARDS.iter() {
        let rc = rank_char(card.rank());
        let sc = suit_char(card.suit());
        names.push(format!("card_{}{}_ai", rc, sc));
        names.push(format!("card_{}{}_opp1", rc, sc));
        names.push(format!("card_{}{}_opp2", rc, sc));
        names.push(format!("card_{}{}_opp3", rc, sc));
    }
    names.push("ai_points_taken_raw".to_string());
    names.push("opp1_points_taken".to_string());
    names.push("opp2_points_taken".to_string());
    names.push("opp3_points_taken".to_string());
    names.push("ai_has_lead_raw".to_string());
    names.push("hearts_broken".to_string());
    names
}

/// Map opponent player indices relative to AI (player index order, skipping AI).
fn opponent_indices(ai: PlayerIndex) -> [usize; 3] {
    let mut opps = [0usize; 3];
    let mut k = 0;
    for p in PlayerIndex::all() {
        if p != ai {
            opps[k] = p.index();
            k += 1;
        }
    }
    opps
}

/// Extract 214 raw values: 208 card binary + 6 scalars.
fn extract_raw_cards(state: &GameState, ai: PlayerIndex) -> Vec<u8> {
    let mut raw = vec![0u8; RAW_TOTAL];
    let opps = opponent_indices(ai);

    // 208 card columns: for each card, 4 binary columns (ai, opp1, opp2, opp3)
    for (card_idx, card) in ALL_CARDS.iter().enumerate() {
        let base = card_idx * 4;
        // Check if card is in any hand (not played)
        if state.hands[ai.index()].contains(*card) {
            raw[base] = 1; // ai_holds
        } else if state.hands[opps[0]].contains(*card) {
            raw[base + 1] = 1; // opp1_holds
        } else if state.hands[opps[1]].contains(*card) {
            raw[base + 2] = 1; // opp2_holds
        } else if state.hands[opps[2]].contains(*card) {
            raw[base + 3] = 1; // opp3_holds
        }
        // If card was already played, all 4 stay 0
    }

    // 6 scalar columns
    let scalar_base = RAW_CARD_COLS;
    raw[scalar_base] = state.points_taken[ai.index()] as u8;
    raw[scalar_base + 1] = state.points_taken[opps[0]] as u8;
    raw[scalar_base + 2] = state.points_taken[opps[1]] as u8;
    raw[scalar_base + 3] = state.points_taken[opps[2]] as u8;
    raw[scalar_base + 4] = (state.current_player == ai) as u8;
    raw[scalar_base + 5] = state.hearts_broken as u8;

    raw
}

// ── Sample types and formatting ─────────────────────────────────────────

struct Sample {
    raw_cards: Option<Vec<u8>>,
    features: [f64; FEATURE_COUNT],
    score: i32,
}

fn format_row(sample: &Sample) -> String {
    let mut row = String::new();

    // Raw card columns first (if present)
    if let Some(ref raw) = sample.raw_cards {
        for &v in raw.iter() {
            write!(row, "{},", v).unwrap();
        }
    }

    // Existing features
    for (i, &f) in sample.features.iter().enumerate() {
        if i > 0 {
            row.push(',');
        }
        if f == f.floor() && f.abs() < 1e9 {
            write!(row, "{}", f as i64).unwrap();
        } else {
            write!(row, "{:.6}", f).unwrap();
        }
    }

    // Score
    write!(row, ",{}", sample.score).unwrap();
    row
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        fs::create_dir_all(parent).expect("failed to create output directory");
    }

    // Pre-generate positions
    eprintln!("Generating {} positions...", args.samples);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut positions: Vec<(GameState, u64)> = Vec::with_capacity(args.samples);

    while positions.len() < args.samples {
        let hands = DeckConfig::Full.deal(&mut rng);
        let play_seed = rng.gen::<u64>();
        let mut play_rng = StdRng::seed_from_u64(play_seed);
        if let Some(state) = play_to_28_cards(hands, &mut play_rng) {
            let solve_seed = rng.gen::<u64>();
            positions.push((state, solve_seed));
        }
    }
    eprintln!("Positions ready. Solving in parallel...");

    let start = Instant::now();
    let done = AtomicUsize::new(0);
    let total = positions.len();
    let raw_cards_flag = args.raw_cards;

    let samples: Vec<Sample> = positions
        .par_iter()
        .map(|(state, _seed)| {
            let ai_player = state.current_player;
            let mut solve_state = state.clone();
            let score = paranoid_solve(&mut solve_state, ai_player);
            let features = extract_features(state, ai_player);
            let raw_cards = if raw_cards_flag {
                Some(extract_raw_cards(state, ai_player))
            } else {
                None
            };

            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 10 == 0 || n == total {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = n as f64 / elapsed;
                let eta = if rate > 0.0 {
                    (total - n) as f64 / rate
                } else {
                    0.0
                };
                let pct = 100.0 * n as f64 / total as f64;
                let bar_len = 30;
                let filled = (bar_len as f64 * pct / 100.0) as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(bar_len - filled);
                eprint!(
                    "\r  {} {:.0}% ({}/{}) {:.1}/s ETA {:.0}s   ",
                    bar, pct, n, total, rate, eta
                );
            }

            Sample {
                raw_cards,
                features,
                score,
            }
        })
        .collect();

    eprintln!();

    // Write CSV
    let file = File::create(&args.output).expect("failed to create output file");
    let mut writer = BufWriter::new(file);

    // Header
    let mut header_parts: Vec<String> = Vec::new();
    if args.raw_cards {
        header_parts.extend(raw_card_header());
    }
    for name in FEATURE_NAMES.iter() {
        header_parts.push(name.to_string());
    }
    header_parts.push("score".to_string());
    writeln!(writer, "{}", header_parts.join(",")).unwrap();

    for sample in &samples {
        writeln!(writer, "{}", format_row(sample)).unwrap();
    }
    writer.flush().unwrap();

    let n_cols = if args.raw_cards {
        RAW_TOTAL + FEATURE_COUNT + 1
    } else {
        FEATURE_COUNT + 1
    };
    let elapsed = start.elapsed();
    eprintln!(
        "Done: {} samples, {} columns in {:.1}s ({:.1} samples/sec)",
        samples.len(),
        n_cols,
        elapsed.as_secs_f64(),
        samples.len() as f64 / elapsed.as_secs_f64()
    );
    eprintln!("Output: {}", args.output);
}
