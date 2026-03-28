use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;

use hearts_core::bots::random_bot::RandomBot;
use hearts_core::bots::rule_bot::RuleBot;
use hearts_core::bots::solver_bot::SolverBot;
use hearts_core::deck::DeckConfig;
use hearts_core::game::Player;
use hearts_core::replay;
use hearts_core::stats;

#[derive(Parser)]
#[command(name = "hearts-cli", about = "Hearts card game AI runner")]
struct Args {
    /// Number of games to play
    #[arg(long, default_value = "100")]
    games: usize,

    /// Deck size: tiny, small, medium, full
    #[arg(long, default_value = "full")]
    deck: String,

    /// Comma-separated player types for seats 0-3 (random, rule)
    #[arg(long, default_value = "rule,rule,rule,rule")]
    players: String,

    /// Base random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Generate an HTML replay of a single game to this file path
    #[arg(long)]
    replay: Option<String>,
}

fn parse_deck(s: &str) -> DeckConfig {
    match s.to_lowercase().as_str() {
        "tiny" => DeckConfig::Tiny,
        "small" => DeckConfig::Small,
        "medium" => DeckConfig::Medium,
        "full" => DeckConfig::Full,
        _ => {
            eprintln!("Unknown deck '{}', using full", s);
            DeckConfig::Full
        }
    }
}

fn make_player(kind: &str, seed: u64) -> Box<dyn Player> {
    match kind.trim().to_lowercase().as_str() {
        "random" => Box::new(RandomBot::new(StdRng::seed_from_u64(seed))),
        "rule" => Box::new(RuleBot::new()),
        "solver" => Box::new(SolverBot::new()),
        _ => {
            eprintln!("Unknown player type '{}', using rule", kind);
            Box::new(RuleBot::new())
        }
    }
}

fn main() {
    let args = Args::parse();
    let deck_config = parse_deck(&args.deck);
    let player_types: Vec<String> = args.players.split(',').map(|s| s.trim().to_string()).collect();

    if player_types.len() != 4 {
        eprintln!("Need exactly 4 player types, got {}", player_types.len());
        std::process::exit(1);
    }

    if let Some(ref path) = args.replay {
        // Single-game HTML replay mode
        let mut rng = StdRng::seed_from_u64(args.seed);
        let hands = deck_config.deal(&mut rng);
        let mut players: [Box<dyn Player>; 4] = [
            make_player(&player_types[0], args.seed * 4),
            make_player(&player_types[1], args.seed * 4 + 1),
            make_player(&player_types[2], args.seed * 4 + 2),
            make_player(&player_types[3], args.seed * 4 + 3),
        ];
        let labels = [
            player_types[0].clone(),
            player_types[1].clone(),
            player_types[2].clone(),
            player_types[3].clone(),
        ];
        let game_replay = replay::collect_replay(deck_config, hands, &mut players, labels, args.seed);
        let html = replay::render_html(&game_replay);
        std::fs::write(path, html).expect("failed to write replay file");
        println!("Replay written to {}", path);
        return;
    }

    let labels: [&str; 4] = [
        &player_types[0],
        &player_types[1],
        &player_types[2],
        &player_types[3],
    ];

    println!(
        "Running {} games on {:?} deck: {} vs {} vs {} vs {}",
        args.games, deck_config, labels[0], labels[1], labels[2], labels[3]
    );
    println!();

    let pt = player_types.clone();
    let base = args.seed;
    let game_stats = stats::run_batch(deck_config, args.games, move |seed| {
        [
            make_player(&pt[0], base + seed * 4),
            make_player(&pt[1], base + seed * 4 + 1),
            make_player(&pt[2], base + seed * 4 + 2),
            make_player(&pt[3], base + seed * 4 + 3),
        ]
    }, args.seed);

    game_stats.print_summary(&labels);
}
