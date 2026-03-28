# Hearts Solver

State-of-the-art Hearts AI using Pluribus/Libratus-style architecture adapted for trick-taking games.

## Project Structure

```
crates/
  hearts-core/    # Library: game engine, solvers, bots, search
  hearts-cli/     # Binary: CLI runner for games and benchmarks
docs/
  plan.md         # Design document (architecture, algorithms)
  step-by-step.md # Incremental implementation plan (MAIN PLAN - READ THIS)
```

## Language & Build

- **Rust** (Cargo workspace, edition 2021)
- `cargo build` / `cargo test` / `cargo run -p hearts-cli`
- Dependencies: `rand` (0.8)

## Architecture

- **CardSet** (`card_set.rs`): u64 bitboard — the universal set type for cards. Never use `Vec<Card>` for hands.
- **DeckConfig** (`deck.rs`): Tiny (3/player), Small (5), Medium (8), Full (13). Reduced decks strip low ranks first, always keep Qs/Ks/As.
- **GameState** (`game_state.rs`): Full game state with legal move generation. Supports play/undo for solver efficiency.
- **Player trait** (`game.rs`): `fn choose_card(&mut self, state: &GameState, legal_moves: CardSet) -> Card`

## Code Routing

All source lives under `crates/hearts-core/src/`:

| Area | Files | What goes here |
|------|-------|----------------|
| Types & primitives | `types.rs`, `card_set.rs`, `trick.rs` | Card, Suit, Rank, CardSet bitboard, PlayerIndex, Trick |
| Game engine | `deck.rs`, `game_state.rs`, `game.rs` | Deck configs, state + legal moves + scoring, Player trait + GameRunner |
| Bots | `bots/` | RandomBot, RuleBot, PIMCBot, AlphaMuBot |
| Solvers | `solver/` | Brute-force, alpha-beta DD solver, transposition table |
| Belief | `belief/` | Observation model, constraint sampler, Bayesian tracker |
| Search | `search/` | PIMC, Alpha-Mu |
| Stats | `stats.rs` | Game statistics, batch runner |

CLI lives at `crates/hearts-cli/src/main.rs`.

## Key Conventions

- All randomness via explicit `&mut impl Rng` parameters — no global RNG
- Seeded RNG (`StdRng::seed_from_u64`) in all tests for reproducibility
- DeckConfig is a parameter, not global — allows mixing deck sizes in one binary
- Moon shooting uses deck's total points (not hardcoded 26) for reduced deck support
- Brute-force solver on Tiny decks = correctness oracle for all solver optimizations

## Testing

- `cargo test -p hearts-core` runs all unit tests
- Every solver optimization must match brute-force on 100 random Tiny deals
- Score invariant: `sum(final_scores) == deck_points` (normal) or `deck_points * 3` (moon shot)

## Auto-Update Memory (MANDATORY)

**Update `.claude/memory/` files AS YOU GO, not at the end.**

| Trigger | File |
|---------|------|
| User shares a fact about themselves | `memory-profile.md` |
| User states a preference | `memory-preferences.md` |
| A decision is made | `memory-decisions.md` (with date) |
| Completing substantive work | `memory-sessions.md` |

**DO NOT ASK. Just update when you learn something.**

## Solver Design Notes

- **Two solvers:** Max^n (`maxn.rs`) is correct but slow. Paranoid (`paranoid.rs`) has full alpha-beta pruning, 6x faster, but pessimistic. Use Max^n for correctness testing, Paranoid for PIMC.
- Standard 2-player negamax does NOT work for 4-player Hearts. Max^n uses plain minimax per player.
- Move ordering + zero-score pruning is unsafe in Max^n (causes incorrect results).
- Paranoid TT with bound types is deferred — alpha-beta pruning alone provides the main speedup.
- All solvers take `&mut GameState` and use undo-based traversal (no cloning per node).
