use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use hearts_core::bots::rule_bot::RuleBot;
use hearts_core::deck::DeckConfig;
use hearts_core::game::Player;
use hearts_core::game_state::GameState;
use hearts_core::solver::paranoid::{paranoid_solve, paranoid_solve_depth_limited};

#[derive(Parser)]
#[command(about = "Benchmark depth-limited vs full paranoid solve")]
struct Args {
    /// Eval cutoff in cards remaining
    #[arg(long, default_value = "28")]
    cutoff: u32,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Number of 32-card positions (full solve comparison)
    #[arg(long, default_value = "100")]
    n32: usize,

    /// Number of 36-card positions (depth-limited only)
    #[arg(long, default_value = "50")]
    n36: usize,

    /// Number of 40-card positions (depth-limited only)
    #[arg(long, default_value = "20")]
    n40: usize,
}

/// Play forward from a Full deal to reach a state with `target_cards` remaining.
fn play_to_cards_remaining(
    hands: [hearts_core::card_set::CardSet; 4],
    target_cards: u32,
    _rng: &mut StdRng,
) -> Option<GameState> {
    let mut state = GameState::new_with_deal(hands, DeckConfig::Full);
    let mut rule_bot = RuleBot::new();
    let cards_to_play = 52 - target_cards;

    for _ in 0..cards_to_play {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return None;
        }
        let card = rule_bot.choose_card(&state, legal);
        state.play_card(card);
    }

    let remaining: u32 = state.hands.iter().map(|h| h.count()).sum();
    if remaining != target_cards || !state.current_trick.is_empty() {
        return None;
    }

    Some(state)
}

struct BenchResult {
    full_time_ms: Option<f64>,
    dl_time_ms: f64,
    full_score: Option<i32>,
    dl_score: f64,
}

fn bench_position(state: &GameState, cutoff: u32, run_full: bool) -> BenchResult {
    let ai = state.current_player;

    // Depth-limited solve
    let start = Instant::now();
    let mut dl_state = state.clone();
    let dl_score = paranoid_solve_depth_limited(&mut dl_state, ai, cutoff);
    let dl_time = start.elapsed();

    // Full solve (optional)
    let (full_time, full_score) = if run_full {
        let start = Instant::now();
        let mut full_state = state.clone();
        let score = paranoid_solve(&mut full_state, ai);
        let time = start.elapsed();
        (Some(time.as_secs_f64() * 1000.0), Some(score))
    } else {
        (None, None)
    };

    BenchResult {
        full_time_ms: full_time,
        dl_time_ms: dl_time.as_secs_f64() * 1000.0,
        full_score,
        dl_score,
    }
}

fn print_stats(label: &str, results: &[BenchResult]) {
    let n = results.len();
    if n == 0 {
        return;
    }

    println!("\n=== {} ({} positions) ===", label, n);

    // DL timing
    let mut dl_times: Vec<f64> = results.iter().map(|r| r.dl_time_ms).collect();
    dl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dl_mean = dl_times.iter().sum::<f64>() / n as f64;
    let dl_median = dl_times[n / 2];
    let dl_max = dl_times.last().unwrap();
    println!(
        "  Depth-limited time: mean={:.2}ms, median={:.2}ms, max={:.2}ms",
        dl_mean, dl_median, dl_max
    );

    // Full timing (if available)
    let full_times: Vec<f64> = results.iter().filter_map(|r| r.full_time_ms).collect();
    if !full_times.is_empty() {
        let ft_mean = full_times.iter().sum::<f64>() / full_times.len() as f64;
        let mut ft_sorted = full_times.clone();
        ft_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ft_median = ft_sorted[ft_sorted.len() / 2];
        let speedup = ft_mean / dl_mean;
        println!(
            "  Full solve time:    mean={:.0}ms, median={:.0}ms",
            ft_mean, ft_median
        );
        println!("  Speedup:            {:.0}x", speedup);
    }

    // Score comparison (if full solve available)
    let diffs: Vec<f64> = results
        .iter()
        .filter_map(|r| r.full_score.map(|fs| (r.dl_score - fs as f64).abs()))
        .collect();
    if !diffs.is_empty() {
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let mut sorted_diffs = diffs.clone();
        sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_diff = sorted_diffs[sorted_diffs.len() / 2];
        let max_diff = sorted_diffs.last().unwrap();
        let variance =
            diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;
        let std_diff = variance.sqrt();
        println!(
            "  Score diff (|DL - full|): mean={:.2}, median={:.2}, max={:.2}, std={:.2}",
            mean_diff, median_diff, max_diff, std_diff
        );
    }

    // DL score distribution
    let mut dl_scores: Vec<f64> = results.iter().map(|r| r.dl_score).collect();
    dl_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dl_mean_score = dl_scores.iter().sum::<f64>() / n as f64;
    println!(
        "  DL score range: [{:.1}, {:.1}], mean={:.2}",
        dl_scores.first().unwrap(),
        dl_scores.last().unwrap(),
        dl_mean_score
    );
}

fn main() {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);

    println!("Depth-limited solver benchmark (cutoff={})", args.cutoff);

    // ── 32-card positions ───────────────────────────────────────────────
    if args.n32 > 0 {
        eprintln!("Generating {} positions at 32 cards...", args.n32);
        let mut results = Vec::new();
        while results.len() < args.n32 {
            let hands = DeckConfig::Full.deal(&mut rng);
            let play_seed = rng.gen::<u64>();
            let mut play_rng = StdRng::seed_from_u64(play_seed);
            if let Some(state) = play_to_cards_remaining(hands, 32, &mut play_rng) {
                let r = bench_position(&state, args.cutoff, true);
                if (results.len() + 1) % 10 == 0 {
                    eprintln!(
                        "  {}/{} (full={:.0}ms, dl={:.2}ms)",
                        results.len() + 1,
                        args.n32,
                        r.full_time_ms.unwrap_or(0.0),
                        r.dl_time_ms
                    );
                }
                results.push(r);
            }
        }
        print_stats("32 cards (vs full solve)", &results);
    }

    // ── 36-card positions ───────────────────────────────────────────────
    if args.n36 > 0 {
        eprintln!("Generating {} positions at 36 cards...", args.n36);
        let mut results = Vec::new();
        while results.len() < args.n36 {
            let hands = DeckConfig::Full.deal(&mut rng);
            let play_seed = rng.gen::<u64>();
            let mut play_rng = StdRng::seed_from_u64(play_seed);
            if let Some(state) = play_to_cards_remaining(hands, 36, &mut play_rng) {
                let r = bench_position(&state, args.cutoff, false);
                results.push(r);
            }
        }
        print_stats("36 cards (DL only)", &results);
    }

    // ── 40-card positions ───────────────────────────────────────────────
    if args.n40 > 0 {
        eprintln!("Generating {} positions at 40 cards...", args.n40);
        let mut results = Vec::new();
        while results.len() < args.n40 {
            let hands = DeckConfig::Full.deal(&mut rng);
            let play_seed = rng.gen::<u64>();
            let mut play_rng = StdRng::seed_from_u64(play_seed);
            if let Some(state) = play_to_cards_remaining(hands, 40, &mut play_rng) {
                let r = bench_position(&state, args.cutoff, false);
                results.push(r);
            }
        }
        print_stats("40 cards (DL only)", &results);
    }

    println!("\nDone.");
}
