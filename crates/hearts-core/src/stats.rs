use crate::deck::DeckConfig;
use crate::game::{GameRunner, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Accumulated statistics over multiple games.
#[derive(Debug, Clone)]
pub struct GameStats {
    pub games_played: usize,
    pub total_scores: [i64; 4],
    pub moon_shoots: [usize; 4],
}

impl GameStats {
    pub fn new() -> Self {
        GameStats {
            games_played: 0,
            total_scores: [0; 4],
            moon_shoots: [0; 4],
        }
    }

    pub fn record(&mut self, scores: [i32; 4], deck_config: DeckConfig) {
        self.games_played += 1;
        for i in 0..4 {
            self.total_scores[i] += scores[i] as i64;
        }
        // Detect moon shot: one player got 0 while others got deck_points
        let dp = deck_config.total_points();
        for i in 0..4 {
            if scores[i] == 0 && scores[(i + 1) % 4] == dp {
                self.moon_shoots[i] += 1;
            }
        }
    }

    pub fn avg_score(&self, player: usize) -> f64 {
        if self.games_played == 0 {
            return 0.0;
        }
        self.total_scores[player] as f64 / self.games_played as f64
    }

    pub fn print_summary(&self, labels: &[&str; 4]) {
        println!("Games played: {}", self.games_played);
        println!("{:<12} {:>8} {:>8} {:>6}", "Player", "Avg", "Total", "Moons");
        println!("{}", "-".repeat(38));
        for i in 0..4 {
            println!(
                "{:<12} {:>8.2} {:>8} {:>6}",
                labels[i],
                self.avg_score(i),
                self.total_scores[i],
                self.moon_shoots[i],
            );
        }
    }
}

/// Run a batch of games and collect stats.
pub fn run_batch(
    deck_config: DeckConfig,
    num_games: usize,
    mut make_players: impl FnMut(u64) -> [Box<dyn Player>; 4],
    base_seed: u64,
) -> GameStats {
    let mut stats = GameStats::new();
    for i in 0..num_games {
        let seed = base_seed + i as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let players = make_players(seed);
        let mut runner = GameRunner::new_with_deal(deck_config, players, &mut rng);
        let scores = runner.play_game();
        stats.record(scores, deck_config);
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::FirstLegalBot;

    #[test]
    fn stats_accumulate() {
        let mut stats = GameStats::new();
        stats.record([5, 10, 8, 3], DeckConfig::Full);
        stats.record([3, 7, 6, 10], DeckConfig::Full);
        assert_eq!(stats.games_played, 2);
        assert_eq!(stats.total_scores, [8, 17, 14, 13]);
        assert!((stats.avg_score(0) - 4.0).abs() < 0.01);
    }

    #[test]
    fn batch_runner_works() {
        let stats = run_batch(
            DeckConfig::Tiny,
            100,
            |_| {
                [
                    Box::new(FirstLegalBot),
                    Box::new(FirstLegalBot),
                    Box::new(FirstLegalBot),
                    Box::new(FirstLegalBot),
                ]
            },
            0,
        );
        assert_eq!(stats.games_played, 100);
    }
}
