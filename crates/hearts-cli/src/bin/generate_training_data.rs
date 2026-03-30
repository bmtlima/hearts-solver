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
use hearts_core::types::{Card, Rank, Suit};

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
}

/// Simple duck heuristic: play to avoid taking tricks.
fn duck_choose(state: &GameState, legal: CardSet, _rng: &mut impl Rng) -> Card {
    if legal.count() == 1 {
        return legal.cards().next().unwrap();
    }

    let qs = Card::new(Suit::Spades, Rank::Queen);

    if state.current_trick.is_empty() {
        // Leading: play lowest non-heart, non-Qs card
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

    // Void in led suit: dump Qs > highest heart > highest card
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

/// Pick a card using mixed policy.
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

/// Play forward exactly 6 tricks from a full deal.
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

struct Sample {
    features: [f64; FEATURE_COUNT],
    score: i32,
}

fn format_row(sample: &Sample) -> String {
    let mut row = String::new();
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
    write!(row, ",{}", sample.score).unwrap();
    row
}

fn main() {
    let args = Args::parse();

    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        fs::create_dir_all(parent).expect("failed to create output directory");
    }

    // Pre-generate all positions sequentially (fast, needs RNG)
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

    // Solve all positions in parallel (expensive part)
    let start = Instant::now();
    let done = AtomicUsize::new(0);
    let total = positions.len();

    let samples: Vec<Sample> = positions
        .par_iter()
        .map(|(state, _seed)| {
            let ai_player = state.current_player;
            let mut solve_state = state.clone();
            let score = paranoid_solve(&mut solve_state, ai_player);
            let features = extract_features(state, ai_player);

            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 10 == 0 || n == total {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = n as f64 / elapsed;
                let eta = if rate > 0.0 { (total - n) as f64 / rate } else { 0.0 };
                let pct = 100.0 * n as f64 / total as f64;
                let bar_len = 30;
                let filled = (bar_len as f64 * pct / 100.0) as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(bar_len - filled);
                eprint!(
                    "\r  {} {:.0}% ({}/{}) {:.1}/s ETA {:.0}s   ",
                    bar, pct, n, total, rate, eta
                );
            }

            Sample { features, score }
        })
        .collect();

    eprintln!(); // newline after progress bar

    // Write CSV
    let file = File::create(&args.output).expect("failed to create output file");
    let mut writer = BufWriter::new(file);

    let header: Vec<&str> = FEATURE_NAMES
        .iter()
        .copied()
        .chain(std::iter::once("score"))
        .collect();
    writeln!(writer, "{}", header.join(",")).unwrap();

    for sample in &samples {
        writeln!(writer, "{}", format_row(sample)).unwrap();
    }
    writer.flush().unwrap();

    let elapsed = start.elapsed();
    eprintln!(
        "Done: {} samples in {:.1}s ({:.1} samples/sec)",
        samples.len(),
        elapsed.as_secs_f64(),
        samples.len() as f64 / elapsed.as_secs_f64()
    );
    eprintln!("Output: {}", args.output);
}
