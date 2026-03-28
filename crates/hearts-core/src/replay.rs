use serde::Serialize;

use crate::card_set::CardSet;
use crate::deck::DeckConfig;
use crate::game::Player;
use crate::game_state::GameState;

#[derive(Serialize)]
pub struct ReplayEvent {
    pub player: usize,
    pub card: String,
    pub legal_moves: Vec<String>,
}

#[derive(Serialize)]
pub struct ReplayTrick {
    pub leader: usize,
    pub plays: Vec<ReplayEvent>,
    pub winner: usize,
    pub points: i32,
}

#[derive(Serialize)]
pub struct GameReplay {
    pub deck_config: String,
    pub player_labels: [String; 4],
    pub initial_hands: [Vec<String>; 4],
    pub tricks: Vec<ReplayTrick>,
    pub final_scores: [i32; 4],
    pub moon_shooter: Option<usize>,
    pub total_points: i32,
    pub seed: u64,
}

fn cardset_to_strings(cs: CardSet) -> Vec<String> {
    cs.cards().map(|c| format!("{}", c)).collect()
}

/// Play a full game and record every move for replay.
pub fn collect_replay(
    deck_config: DeckConfig,
    hands: [CardSet; 4],
    players: &mut [Box<dyn Player>; 4],
    player_labels: [String; 4],
    seed: u64,
) -> GameReplay {
    let mut state = GameState::new_with_deal(hands, deck_config);

    let initial_hands = [
        cardset_to_strings(state.hands[0]),
        cardset_to_strings(state.hands[1]),
        cardset_to_strings(state.hands[2]),
        cardset_to_strings(state.hands[3]),
    ];

    let mut tricks = Vec::new();
    let mut current_plays = Vec::new();
    let mut current_leader = state.current_player.index();

    while !state.is_game_over() {
        let legal = state.legal_moves();
        let idx = state.current_player.index();
        let card = players[idx].choose_card(&state, legal);

        current_plays.push(ReplayEvent {
            player: idx,
            card: format!("{}", card),
            legal_moves: cardset_to_strings(legal),
        });

        if let Some(result) = state.play_card(card) {
            tricks.push(ReplayTrick {
                leader: current_leader,
                plays: std::mem::take(&mut current_plays),
                winner: result.winner.index(),
                points: result.points,
            });
            current_leader = result.winner.index();
        }
    }

    let final_scores = state.final_scores();
    let moon_shooter = state.moon_shooter().map(|p| p.index());

    GameReplay {
        deck_config: format!("{:?}", deck_config).to_lowercase(),
        player_labels,
        initial_hands,
        tricks,
        final_scores,
        moon_shooter,
        total_points: deck_config.total_points(),
        seed,
    }
}

/// Render a GameReplay as a self-contained HTML file.
pub fn render_html(replay: &GameReplay) -> String {
    let json = serde_json::to_string(replay).unwrap();
    let template = include_str!("replay_template.html");
    template.replace("\"__GAME_DATA__\"", &json)
}
