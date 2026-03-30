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
use hearts_core::solver::paranoid::{paranoid_best_move, paranoid_best_move_depth_limited, paranoid_solve};
use hearts_core::types::{Card, Rank, Suit};

#[derive(Parser)]
#[command(about = "Test move agreement between exact solver and regression eval")]
struct Args {
    /// Number of positions to test
    #[arg(long, default_value = "500")]
    samples: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,
}

// ── Position generation (same as generate_training_data) ────────────────

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

// ── Agreement test ──────────────────────────────────────────────────────

struct AgreementResult {
    agreed: bool,
    regret: f64, // 0.0 if agreed, positive if DL picked a worse move
    exact_score: i32,
    n_legal: u32,
}

fn main() {
    let args = Args::parse();

    // Generate positions
    eprintln!("Generating {} positions...", args.samples);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut positions: Vec<GameState> = Vec::with_capacity(args.samples);

    while positions.len() < args.samples {
        let hands = DeckConfig::Full.deal(&mut rng);
        let play_seed = rng.gen::<u64>();
        let mut play_rng = StdRng::seed_from_u64(play_seed);
        if let Some(state) = play_to_28_cards(hands, &mut play_rng) {
            positions.push(state);
        }
    }
    eprintln!("Positions ready. Testing agreement in parallel...");

    let start = Instant::now();
    let done = AtomicUsize::new(0);
    let total = positions.len();

    let results: Vec<AgreementResult> = positions
        .par_iter()
        .map(|state| {
            let ai = state.current_player;
            let n_legal = state.legal_moves().count();

            // Exact solver best move
            let mut exact_state = state.clone();
            let (exact_card, exact_score) = paranoid_best_move(&mut exact_state, ai);

            // Depth-limited (regression-only) best move
            let mut dl_state = state.clone();
            let (dl_card, _dl_score) = paranoid_best_move_depth_limited(&mut dl_state, ai, 28);

            let agreed = exact_card == dl_card;

            // Compute regret if disagreed
            let regret = if agreed {
                0.0
            } else {
                // Evaluate dl_card with exact solver
                let mut regret_state = state.clone();
                let undo = regret_state.play_card_with_undo(dl_card);
                let dl_card_exact_score = paranoid_solve(&mut regret_state, ai);
                regret_state.undo_card(&undo);
                (dl_card_exact_score - exact_score) as f64
            };

            // Progress bar
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

            AgreementResult {
                agreed,
                regret,
                exact_score,
                n_legal,
            }
        })
        .collect();

    eprintln!();

    // ── Summary ─────────────────────────────────────────────────────────
    let elapsed = start.elapsed();
    let n_agreed = results.iter().filter(|r| r.agreed).count();
    let n_total = results.len();
    let n_forced = results.iter().filter(|r| r.n_legal == 1).count();
    let n_choice = n_total - n_forced;
    let n_agreed_choice = results
        .iter()
        .filter(|r| r.agreed && r.n_legal > 1)
        .count();

    println!("\n=== Move Agreement Test ({:.1}s) ===", elapsed.as_secs_f64());
    println!("  Positions tested: {}", n_total);
    println!(
        "  Forced moves (1 legal): {} ({:.1}%)",
        n_forced,
        100.0 * n_forced as f64 / n_total as f64
    );
    println!(
        "  Agreement (all):    {}/{} ({:.1}%)",
        n_agreed,
        n_total,
        100.0 * n_agreed as f64 / n_total as f64
    );
    println!(
        "  Agreement (choice): {}/{} ({:.1}%)",
        n_agreed_choice,
        n_choice,
        100.0 * n_agreed_choice as f64 / n_choice as f64
    );

    // Regret stats (only for disagreements with >1 legal move)
    let regrets: Vec<f64> = results
        .iter()
        .filter(|r| !r.agreed && r.n_legal > 1)
        .map(|r| r.regret)
        .collect();

    if !regrets.is_empty() {
        let mean_regret = regrets.iter().sum::<f64>() / regrets.len() as f64;
        let mut sorted = regrets.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_regret = sorted[sorted.len() / 2];
        let max_regret = sorted.last().unwrap();
        let n_zero_regret = regrets.iter().filter(|&&r| r == 0.0).count();

        println!("\n=== Regret (when disagreed, {} cases) ===", regrets.len());
        println!("  Mean:   {:.2} points", mean_regret);
        println!("  Median: {:.1} points", median_regret);
        println!("  Max:    {:.1} points", max_regret);
        println!(
            "  Zero regret (different move, same score): {} ({:.1}%)",
            n_zero_regret,
            100.0 * n_zero_regret as f64 / regrets.len() as f64
        );
    }

    // Score distribution of disagreements
    let disagree_scores: Vec<i32> = results
        .iter()
        .filter(|r| !r.agreed && r.n_legal > 1)
        .map(|r| r.exact_score)
        .collect();
    if !disagree_scores.is_empty() {
        let moon_disagree = disagree_scores
            .iter()
            .filter(|&&s| s == 0 || s == 26)
            .count();
        println!(
            "  Disagreements at moon scores (0 or 26): {} ({:.1}%)",
            moon_disagree,
            100.0 * moon_disagree as f64 / disagree_scores.len() as f64
        );
    }
}
