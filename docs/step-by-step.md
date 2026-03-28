# Hearts Solver — Step-by-Step Implementation Plan

Language: **Rust** (Cargo workspace). Each step is small, testable, and builds on the previous one.

---

## Phase 0: Project Scaffolding

### Step 0.1 — Initialize Cargo workspace

**Implement:**
- Create a Cargo workspace with two crates:
  - `hearts-core` (library) — all game logic, solvers, bots
  - `hearts-cli` (binary) — CLI runner for games and benchmarks
- Layout:
  ```
  hearts-solver/
    Cargo.toml          (workspace root)
    crates/
      hearts-core/
        Cargo.toml
        src/lib.rs
      hearts-cli/
        Cargo.toml
        src/main.rs
  ```
- Update `.gitignore` to include `/target`

**Test:**
- `cargo build` succeeds
- `cargo test` succeeds (vacuously)
- `cargo run -p hearts-cli` prints "Hearts Solver"

**Done when:** Workspace compiles and binary runs.

---

## Phase 1: Game Engine Foundation

### Step 1.1 — Card types (Suit, Rank, Card)

**Implement** in `hearts-core/src/types.rs`:
- `Suit` enum: Clubs, Diamonds, Spades, Hearts
- `Rank` enum: Two through Ace (internal values 0..12 for bitboard indexing)
- `Card` struct with: `new(suit, rank)`, `suit()`, `rank()`, `bit_index() -> u8`, `point_value() -> i32` (Hearts=1, Q-Spades=13, else 0), `Display` impl
- `const ALL_CARDS: [Card; 52]`
- Derive `Clone, Copy, PartialEq, Eq, Hash, Debug, Ord, PartialOrd`

**Test:**
- [ ] `Card::new(Spades, Queen).point_value() == 13`
- [ ] All 13 hearts have `point_value() == 1`
- [ ] Non-point cards return `point_value() == 0`
- [ ] `ALL_CARDS.len() == 52`, all unique
- [ ] `bit_index()` returns 0 for Two, 12 for Ace
- [ ] Display formatting is readable (e.g., "Qs", "2c", "Ah")

**Done when:** `cargo test -p hearts-core` — all card type tests pass.

---

### Step 1.2 — Bitboard hand representation (CardSet)

**Implement** in `hearts-core/src/card_set.rs`:

`CardSet` wraps a `u64`. Bits 0..12 = Clubs, 13..25 = Diamonds, 26..38 = Spades, 39..51 = Hearts.

Methods:
- `empty()`, `full()` (all 52), `from_cards(iter)`
- `insert(card)`, `remove(card)`, `contains(card) -> bool`
- `count() -> u32` (popcount), `is_empty() -> bool`
- `cards() -> impl Iterator<Item = Card>`
- `suit_mask(suit) -> u16`, `cards_of_suit(suit) -> CardSet`, `has_suit(suit) -> bool`
- `union`, `intersection`, `difference`
- Implement `BitOr`, `BitAnd`, `Sub` operators
- `Display` impl (cards grouped by suit)

**Test:**
- [ ] Insert 5 cards → `count() == 5`, `contains` each
- [ ] `full().count() == 52`, `empty().count() == 0`
- [ ] Union of disjoint sets has sum of counts
- [ ] Intersection of disjoint sets is empty
- [ ] `cards_of_suit(Hearts)` on a mixed set returns only hearts
- [ ] Remove a card → `contains() == false`, count decremented
- [ ] Iterator yields exactly the inserted cards

**Done when:** All CardSet unit tests pass. Bitwise operations are correct.

---

### Step 1.3 — Deck configurations (Tiny, Small, Medium, Full)

**Implement** in `hearts-core/src/deck.rs`:

```rust
pub enum DeckConfig { Tiny, Small, Medium, Full }
```

Each config provides:
- `cards_per_player() -> usize` (3, 5, 8, 13)
- `total_cards() -> usize` (12, 20, 32, 52)
- `deck_cards() -> CardSet` — the subset of cards in play

Card selection rules:
- Always keep Q-S, K-S, A-S
- Strip low-rank cards first, keep hearts as long as possible
- Explicitly define each deck as a constant (no runtime logic surprises)

Also provide:
- `deal(&self, rng) -> [CardSet; 4]` — shuffle and deal
- `first_leader(&self) -> Card` — lowest club in the deck (2c for Full)

**Test:**
- [ ] `Full.deck_cards().count() == 52`
- [ ] `Tiny.deck_cards().count() == 12`, `Small == 20`, `Medium == 32`
- [ ] Every config's deck contains Qs, Ks, As
- [ ] Every config's deck contains at least some hearts
- [ ] `deal()` gives `cards_per_player()` cards to each of 4 players
- [ ] Union of all dealt hands equals the deck
- [ ] Seeded RNG produces deterministic deals
- [ ] `first_leader()` returns a card present in the deck

**Done when:** All deck configuration tests pass.

---

### Step 1.4 — Player index and trick types

**Implement** in `hearts-core/src/trick.rs`:

- `PlayerIndex`: newtype over `u8` (0..3), with `next()` (wraps 3→0), `all() -> [0,1,2,3]`
- `Trick` struct:
  - `leader: PlayerIndex`
  - Cards played (fixed-size array + count, or small array)
  - `led_suit() -> Option<Suit>`
  - `is_complete() -> bool` (4 cards)
  - `winner() -> (PlayerIndex, Card)` — highest card of the led suit
  - `points() -> i32` — sum of point values
  - `play(player, card)` — add a card

**Test:**
- [ ] Trick with {5h, 3h, Kh, Jh} led Hearts → winner is Kh's player
- [ ] Off-suit cards don't win (highest of led suit only)
- [ ] Points: trick with Qs + 2 hearts = 15
- [ ] Trick with no point cards = 0 points
- [ ] `PlayerIndex::next()`: 0→1, 1→2, 2→3, 3→0
- [ ] `is_complete()` false for 0..3 cards, true for 4

**Done when:** Trick resolution logic handles all edge cases.

---

### Step 1.5 — Game state and legal move generation

**Implement** in `hearts-core/src/game_state.rs`:

`GameState` struct:
- `hands: [CardSet; 4]`, `current_trick: Trick`, `points_taken: [i32; 4]`
- `hearts_broken: bool`, `current_player: PlayerIndex`
- `deck_config: DeckConfig`, `trick_number: usize`, `cards_played: CardSet`

Key methods:
- `new(hands, deck_config, first_leader) -> Self`
- `legal_moves(&self) -> CardSet` — applies all Hearts rules:
  1. Must follow led suit if possible
  2. First trick: no hearts or Qs (unless forced)
  3. Cannot lead hearts until broken (unless forced)
- `play_card(&mut self, card) -> Option<TrickResult>` — advance state, return result if trick complete
- `is_game_over(&self) -> bool`
- `final_scores(&self) -> [i32; 4]`

**Test:**
- [ ] Follow suit: has clubs + hearts, clubs led → legal moves = clubs only
- [ ] Void: no clubs, clubs led → legal moves = entire hand
- [ ] First trick: hearts and Qs not legal (unless forced)
- [ ] First trick forced: only hearts + Qs in hand → all legal
- [ ] Hearts not broken: can't lead hearts (unless only hearts remain)
- [ ] Hearts broken flag updates when a heart is played
- [ ] After trick, current_player = winner
- [ ] Points accumulate correctly
- [ ] Play a complete Tiny game with predetermined hands → verify scores

**Done when:** Full game plays start to finish with correct legal moves and scoring.

---

### Step 1.6 — Moon shooting in scoring

**Implement** in `game_state.rs`:

- `final_scores()` detects moon shot: if one player took all 26 points → they get 0, others get 26
- `moon_shooter() -> Option<PlayerIndex>`
- `could_shoot_moon(player) -> bool` — mid-game check

**Test:**
- [ ] Player takes all 26 points → scores = [0, 26, 26, 26]
- [ ] Player takes 25 points, another takes 1 → normal scoring
- [ ] `moon_shooter()` returns correct player or None
- [ ] `could_shoot_moon()` true when player has all points so far

**Done when:** Moon shooting correctly detected and scored.

---

### Step 1.7 — Game simulation loop

**Implement** in `hearts-core/src/game.rs`:

```rust
pub trait Player {
    fn choose_card(&mut self, state: &GameState, legal_moves: CardSet) -> Card;
}

pub struct GameRunner { state: GameState, players: [Box<dyn Player>; 4] }
```

Methods: `new(players, deck_config, rng)`, `play_game() -> [i32; 4]`

**Test:**
- [ ] 4 "first legal move" bots on Tiny → game completes, scores sum to 26 or 78
- [ ] Run 100 games with random seeds on Tiny → all complete
- [ ] Player trait works with different bot types

**Done when:** Complete game runs programmatically with pluggable players.

---

## Phase 2: Heuristic Bots

### Step 2.1 — Random-legal bot

**Implement** in `hearts-core/src/bots/random_bot.rs`:

`RandomBot` picks uniformly at random from `legal_moves`.

**Test:**
- [ ] Over 1000 calls with 5 legal moves, each is chosen at least once
- [ ] 1000 Tiny games with 4 RandomBots → all complete
- [ ] Average score per player over 1000 Full games ≈ 6.5

**Done when:** RandomBot works and produces valid games.

---

### Step 2.2 — Rule-based bot

**Implement** in `hearts-core/src/bots/rule_bot.rs`:

Priority rules:
- **Following suit:** dump Qs if possible; duck under highest card; play lowest if can't duck
- **Void (sloughing):** dump Qs → highest heart → highest card in longest suit
- **Leading:** no hearts unless broken; lead lowest in shortest non-heart suit
- Moon-shoot defense: if opponent has many points, try to give them a point card to break the shoot

**Test:**
- [ ] Can dump Qs on someone else's trick → does so
- [ ] Void in led suit + holding Qs → plays Qs
- [ ] Following suit + can duck → ducks with highest safe card
- [ ] Leading → doesn't lead hearts if unbroken
- [ ] 1000 Full games (4 RuleBots) → all complete, scores valid
- [ ] **RuleBot vs RandomBot (2 each), 1000 games → RuleBot averages fewer points**

**Done when:** RuleBot beats RandomBot consistently.

---

### Step 2.3 — Statistics collection and CLI runner

**Implement:**
- `hearts-core/src/stats.rs`: `GameStats` for per-player totals, averages, moon shoots. `BatchRunner` for N games.
- `hearts-cli/src/main.rs`: CLI with `clap` — args: `--games`, `--deck`, `--players` (e.g. `random,random,rule,rule`)

**Test:**
- [ ] `GameStats` accumulates correctly
- [ ] `cargo run -p hearts-cli -- --games 100 --deck tiny --players random,random,random,random` works
- [ ] `--players rule,rule,random,random` shows RuleBot outperforming RandomBot

**Done when:** CLI runs game batches and reports stats.

---

## Phase 3: Double-Dummy Solver

### Step 3.1 — Brute-force solver (oracle for Tiny games)

**Implement** in `hearts-core/src/solver/brute_force.rs`:

Exhaustive minimax — no pruning, no TT. Each player minimizes own score. Only practical for Tiny (3 cards/player). This is the **correctness oracle** for all future solvers.

**Test:**
- [ ] Hand-craft 5+ Tiny deals with known optimal scores → brute force matches
- [ ] Scores always sum to 26 or 78
- [ ] Completes in <100ms on Tiny

**Done when:** Brute-force produces correct results on all hand-crafted cases.

---

### Step 3.2 — Alpha-beta double-dummy solver

**Implement** in `hearts-core/src/solver/alpha_beta.rs`:

Alpha-beta minimax for perfect-information Hearts. Each player minimizes own score. No TT, no move ordering yet.

**Test:**
- [ ] **100 random Tiny deals: `alpha_beta == brute_force` on every single one** (this is the most important test)
- [ ] Same hand-crafted cases from 3.1
- [ ] Faster than brute force on Tiny

**Done when:** 100% agreement with brute force. Zero discrepancies.

---

### Step 3.3 — Transposition table

**Implement** in `hearts-core/src/solver/transposition.rs`:

Zobrist hashing: pre-computed random u64 per (player, card), XOR together. Fixed-size hash map with always-replace policy.

Wire into `alpha_beta.rs`.

**Test:**
- [ ] **100 random Tiny deals still match brute force exactly**
- [ ] Benchmark: alpha-beta+TT vs alpha-beta on Small → measurable speedup
- [ ] TT probe returns None on first access, correct entry after store

**Done when:** TT integrated, correctness preserved, speedup measured.

---

### Step 3.4 — Move ordering

**Implement** in `alpha_beta.rs`:

Sort legal moves before exploring: dump Qs first → dump point cards when voiding → duck under high cards → lead short suits → low cards first when following.

**Test:**
- [ ] **100 Tiny deals still match brute force**
- [ ] Benchmark on Small/Medium: fewer nodes evaluated than without ordering
- [ ] No correctness change (ordering is heuristic only)

**Done when:** Correctness preserved, node count reduced.

---

### Step 3.5 — Performance benchmarks and undo optimization

**Implement:**
- `hearts-core/benches/solver_bench.rs` using `criterion`: benchmark DD solve per deck size
- Add `undo_card()` to `GameState` for allocation-free backtracking
- Use undo-based traversal in alpha-beta (instead of cloning state per node)

**Targets:**
| Deck | Target |
|------|--------|
| Tiny | < 0.1ms |
| Small | < 1ms |
| Medium | < 10ms |
| Full | < 100ms (stretch: <10ms) |

**Test:**
- [ ] Undo-based solver still matches brute force on Tiny
- [ ] Benchmarks run and produce timing data
- [ ] Profile: no unexpected allocations in hot loop

**Done when:** Benchmarks exist. Performance targets met (or documented gap with optimization plan).

---

### Step 3.6 — Moon shooting in the solver

**Implement** in `alpha_beta.rs`:

During search, detect if a player has taken all point cards so far and can win the rest. Adjust the minimax objective: shooter maximizes points, others try to block.

**Test:**
- [ ] Deal where moon shot is optimal → solver finds shooting line
- [ ] Deal where one player can almost shoot but another can block → solver finds block
- [ ] **100 Tiny deals still match brute force** (brute force already handles moon correctly)

**Done when:** Solver finds both shooting and blocking lines correctly.

---

## Phase 4: Belief Tracking and PIMC

### Step 4.1 — Observation model and information set

**Implement** in `hearts-core/src/belief/observation.rs`:

`Observation` — what a single player can see:
- `my_hand: CardSet`, `cards_played: CardSet`
- `voids: [[bool; 4]; 4]` — known voids from failed suit-follows
- `trick_history`, `current_trick`

`Observation::from_game_state(state, viewer)` — extract viewer's knowledge.
`unknown_cards(obs) -> CardSet` — cards not in my hand and not yet played.

**Test:**
- [ ] Fresh deal: observation shows only my hand, no cards played
- [ ] After 2 tricks: `cards_played` has 8 cards
- [ ] Player fails to follow clubs → `voids[player][Clubs]` is true
- [ ] `unknown_cards().count()` = sum of opponents' hand sizes

**Done when:** Observation correctly captures public + private info.

---

### Step 4.2 — Constraint-based world sampler

**Implement** in `hearts-core/src/belief/sampler.rs`:

`sample_world(obs, rng) -> [CardSet; 4]`: distribute unknown cards to 3 opponents respecting void constraints and hand sizes.

Algorithm: partition unknowns by suit, assign only to non-void players.

**Test:**
- [ ] Sampled world respects void constraints
- [ ] Correct hand sizes for each player
- [ ] Union of all hands = my_hand + unknown_cards
- [ ] Viewer's hand unchanged
- [ ] Over 10000 samples with no constraints: cards distributed roughly equally
- [ ] With void constraint: voided player never has that suit's cards
- [ ] 500 samples in <10ms

**Done when:** Sampler generates valid, constraint-respecting worlds efficiently.

---

### Step 4.3 — Basic PIMC search

**Implement** in `hearts-core/src/search/pimc.rs`:

`pimc_choose(state, player, n_worlds, rng) -> Card`:
1. Get observation, compute legal moves
2. Sample `n_worlds` consistent worlds
3. For each world, for each legal move: DD solve → record player's score
4. Return move with lowest average score

Create `PIMCBot` implementing `Player`.

**Test:**
- [ ] Tiny deal with obvious best move (can dump Qs) → PIMC finds it
- [ ] PIMC (n=50) vs 3 RandomBots, Tiny, 100 games → PIMC averages fewer points
- [ ] PIMC (n=50) vs 3 RuleBots, Small, 100 games → competitive or better
- [ ] PIMC decision on Tiny with n=100 in <1 second

**Done when:** PIMC bot outperforms RandomBot. Acceptable performance on Tiny/Small.

---

### Step 4.4 — PIMC parallelism and optimization

**Implement:**
- Use `rayon` to parallelize DD solves across sampled worlds
- Early termination: stop if one move dominates after N/2 samples
- Pre-allocate TT per thread, clear between worlds

**Test:**
- [ ] Benchmark: parallel vs sequential on Small deck → speedup
- [ ] Same seeds produce same move choices (deterministic parallel)
- [ ] PIMC n=200 on Medium in <5 seconds per decision

**Done when:** Parallel PIMC is faster. Acceptable on Medium deck.

---

## Phase 5: Alpha-Mu Search

### Step 5.1 — Alpha-Mu search (true algorithm)

**Implement** in `hearts-core/src/search/alpha_mu.rs`:

**Key insight from design doc:** Alpha-Mu is NOT aggregation over independent DD solves. It is a single search tree where each node operates on a **vector of worlds simultaneously**. The current player must choose one move that applies to ALL worlds. Opponents may respond differently per world.

Practical approach: one-level Alpha-Mu at the root (your move is the same across all worlds), then fall back to independent DD solves for opponent responses.

Create `AlphaMuBot`.

**Test:**
- [ ] Construct a strategy-fusion test case where PIMC picks the wrong move but Alpha-Mu picks the right one
- [ ] Alpha-Mu vs PIMC (2 each), Small, 500 games → Alpha-Mu at least as good
- [ ] Alpha-Mu's move is never dominated (not worse in ALL worlds than another move)

**Done when:** Alpha-Mu produces decisions. Avoids strategy fusion in constructed cases.

---

### Step 5.2 — Softened Alpha-Mu (probability-weighted)

**Implement** in `alpha_mu.rs`:

Instead of pure worst-case: weight each world by probability. Within each partition (grouped by which move is optimal), use worst-case score. Middle ground between PIMC (too optimistic) and pure Alpha-Mu (too conservative).

**Test:**
- [ ] Uniform probabilities → degenerates to regular Alpha-Mu
- [ ] Softened vs pure Alpha-Mu, Medium, 500 games → softened at least as good
- [ ] Parameter sweep: different softening values, measure average score

**Done when:** Softened Alpha-Mu implemented and competitive.

---

## Phase 6: Full Game and Polish

### Step 6.1 — Scale to full 52-card deck

**Implement:**
- Run all bots on Full deck
- Profile and optimize: DD solver must hit <10ms on Full
- Tune: n_worlds, TT size, move ordering

**Test:**
- [ ] DD solve on Full in <10ms (mean)
- [ ] 100 Full games with Alpha-Mu bot complete without timeout
- [ ] Alpha-Mu vs RuleBot, 1000 Full games → Alpha-Mu wins convincingly

**Done when:** Full system works on standard 52-card Hearts.

---

### Step 6.2 — Enhanced belief tracking (soft signals)

**Implement** in `hearts-core/src/belief/bayesian.rs`:

Bayesian updates using play patterns as soft signals:
- Leading low heart → less likely shooting moon
- Playing high when could duck → adjust beliefs
- P(hand | observations) ∝ P(observations | hand) * P(hand)

Use RuleBot's policy as the likelihood model.

**Test:**
- [ ] Failed suit-follow → P(has that suit) = 0
- [ ] Soft signals shift distributions in expected direction
- [ ] Belief-enhanced PIMC vs constraint-only PIMC, 1000 Full games → at least as good

**Done when:** More accurate world samples. Measurable improvement.

---

### Step 6.3 — Endgame solver

**Implement** in `hearts-core/src/search/endgame.rs`:

For last K tricks (K=4-6), if possible opponent configurations are below a threshold, enumerate all and solve exactly instead of sampling.

**Test:**
- [ ] Endgame solver agrees with brute force on Tiny
- [ ] PIMC + endgame vs pure PIMC, 1000 Full games → improvement
- [ ] Last 4 tricks with known hands solved in <1ms

**Done when:** Endgame solver correct and improves late-game play.

---

### Step 6.4 — Comprehensive evaluation suite

**Implement:**
- Tournament mode: all bot combinations, round-robin
- Statistical significance testing (5000+ games)
- Output: tables of win rates, scores, moon shoots, decision times

CLI: `cargo run -p hearts-cli -- tournament --games 5000 --deck full`

**Test:**
- [ ] Consistent rankings across runs (same seed)
- [ ] Expected ranking: Alpha-Mu > PIMC > RuleBot > RandomBot

**Done when:** Rigorous evaluation framework exists.

---

## Dependency Graph

```
0.1 Cargo setup
 └─ 1.1 Card types
     └─ 1.2 CardSet (bitboard)
         ├─ 1.3 Deck configs
         └─ 1.4 Trick types
             └─ 1.5 GameState + legal moves
                 └─ 1.6 Moon scoring
                     └─ 1.7 Game loop (Player trait)
                         ├─ 2.1 RandomBot
                         │   └─ 2.2 RuleBot
                         │       └─ 2.3 Stats + CLI
                         └─ 3.1 Brute-force solver
                             └─ 3.2 Alpha-beta solver
                                 └─ 3.3 Transposition table
                                     └─ 3.4 Move ordering
                                         └─ 3.5 Benchmarks + undo
                                             └─ 3.6 Moon in solver
                                                 ├─ 4.1 Observation model
                                                 │   └─ 4.2 World sampler
                                                 │       └─ 4.3 Basic PIMC
                                                 │           └─ 4.4 PIMC parallel
                                                 │               └─ 5.1 Alpha-Mu
                                                 │                   └─ 5.2 Softened Alpha-Mu
                                                 └─ 6.1 Full deck scaling
                                                     └─ 6.2 Enhanced beliefs
                                                         └─ 6.3 Endgame solver
                                                             └─ 6.4 Evaluation suite
```

## Key Design Decisions

1. **`CardSet` (u64 bitboard) is the universal set type.** Never use `Vec<Card>` for hands. `legal_moves()` returns `CardSet`.
2. **`GameState` supports both clone and play/undo.** Game runner uses clone; DD solver uses undo for speed.
3. **`Player` trait is synchronous.** Parallelism is internal to bot implementations.
4. **`DeckConfig` is a parameter, not global.** Mix deck sizes in one test binary.
5. **All randomness via explicit `rng` parameters.** Seeded RNG = reproducible tests.
6. **Brute-force solver = oracle.** Every solver optimization is validated against it on Tiny decks.
