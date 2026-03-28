use crate::card_set::CardSet;
use crate::game::Player;
use crate::game_state::GameState;
use crate::solver::paranoid::paranoid_best_move;
use crate::types::Card;

/// Bot that plays optimally using the paranoid (double-dummy) solver.
/// Only practical on Tiny/Small decks.
pub struct SolverBot;

impl SolverBot {
    pub fn new() -> Self {
        SolverBot
    }
}

impl Player for SolverBot {
    fn choose_card(&mut self, state: &GameState, legal_moves: CardSet) -> Card {
        if legal_moves.count() == 1 {
            return legal_moves.cards().next().unwrap();
        }
        let mut state = state.clone();
        let ai = state.current_player;
        let (card, _score) = paranoid_best_move(&mut state, ai);
        card
    }
}
