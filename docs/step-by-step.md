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
- `deck_cards() -> CardSet` — the subset of cards in play, defined as explicit constants
- `total_points() -> i32` — sum of `point_value()` for all cards in `deck_cards()`

Card selection rules:
- Always keep Q♠, K♠, A♠
- Strip low-rank cards first, keep hearts as long as possible
- Explicitly define each deck as a constant (no runtime logic surprises)

Also provide:
- `deal(&self, rng) -> [CardSet; 4]` — shuffle and deal
- `first_leader(&self) -> Card` — lowest club in the deck (2c for Full)

**Test:**
- [ ] `Full.deck_cards().count() == 52`, `Full.total_points() == 26`
- [ ] `Tiny.deck_cards().count() == 12`, `Small == 20`, `Medium == 32`
- [ ] Each config's `total_points()` equals the sum of `point_value()` over its `deck_cards()`
- [ ] Every config's deck contains Q♠, K♠, A♠
- [ ] Every config's deck contains at least some hearts
- [ ] `deal()` gives `cards_per_player()` cards to each of 4 players
- [ ] Union of all dealt hands equals the deck
- [ ] Seeded RNG produces deterministic deals
- [ ] `first_leader()` returns a card present in the deck

**Done when:** All deck configuration tests pass. Point totals are explicit and verified.

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
- `TrickResult` struct (returned when a trick completes):
  - `winner: PlayerIndex`
  - `points: i32`
  - `cards: [(PlayerIndex, Card); 4]` — who played what (needed for void inference in Phase 4)

**Test:**
- [ ] Trick with {5h, 3h, Kh, Jh} led Hearts → winner is Kh's player
- [ ] Off-suit cards don't win (highest of led suit only)
- [ ] Points: trick with Q♠ + 2 hearts = 15
- [ ] Trick with no point cards = 0 points
- [ ] `PlayerIndex::next()`: 0→1, 1→2, 2→3, 3→0
- [ ] `is_complete()` false for 0..3 cards, true for 4
- [ ] `TrickResult` contains all 4 (player, card) pairs in play order

**Done when:** Trick resolution logic handles all edge cases. TrickResult captures full trick information.

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
  2. First trick: no hearts or Q♠ (unless forced)
  3. Cannot lead hearts until broken (unless forced)
- `play_card(&mut self, card) -> Option<TrickResult>` — advance state, return TrickResult if trick complete
- `is_game_over(&self) -> bool`
- `final_scores(&self) -> [i32; 4]`

**Test:**
- [ ] Follow suit: has clubs + hearts, clubs led → legal moves = clubs only
- [ ] Void: no clubs, clubs led → legal moves = entire hand
- [ ] First trick: hearts and Q♠ not legal (unless forced)
- [ ] First trick forced: only hearts + Q♠ in hand → all legal
- [ ] Hearts not broken: can't lead hearts (unless only hearts remain)
- [ ] Hearts broken flag updates when a heart is played
- [ ] After trick, current_player = winner
- [ ] Points accumulate correctly
- [ ] Play a complete Tiny game with predetermined hands → verify scores

**Done when:** Full game plays start to finish with correct legal moves and scoring.

---

### Step 1.6 — Moon shooting in scoring

**Implement** in `game_state.rs`:

- `final_scores()` detects moon shot: if one player took all points → they get 0, others get `config.total_points()`
- `moon_shooter() -> Option<PlayerIndex>`
- `could_shoot_moon(player) -> bool` — mid-game check

**Test:**
- [ ] Player takes all point cards → scores = `[0, T, T, T]` where `T = config.total_points()`
- [ ] Player takes all but 1 point, another takes 1 → normal scoring
- [ ] `moon_shooter()` returns correct player or None
- [ ] `could_shoot_moon()` true when player has all points so far
- [ ] Works correctly on each deck config (Tiny through Full have different total_points)

**Done when:** Moon shooting correctly detected and scored across all deck sizes.

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
- [ ] 4 "first legal move" bots on Tiny → game completes, scores sum to `config.total_points()` or `config.total_points() * 3`
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
- **Following suit:** dump Q♠ if possible; duck under highest card; play lowest if can't duck
- **Void (sloughing):** dump Q♠ → highest heart → highest card in longest suit
- **Leading:** no hearts unless broken; lead lowest in shortest non-heart suit
- Moon-shoot defense: if opponent has many points, try to give them a point card to break the shoot

**Test:**
- [ ] Can dump Q♠ on someone else's trick → does so
- [ ] Void in led suit + holding Q♠ → plays Q♠
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

## Phase 3: Double-Dummy Solvers

### Step 3.1 — Brute-force solver (oracle for Tiny games)

**Implement** in `hearts-core/src/solver/brute_force.rs`:

Exhaustive minimax — no pruning, no TT. Each player minimizes own score (Max^n). Only practical for Tiny (3 cards/player). This is the **correctness oracle** for all future solvers.

**Test:**
- [ ] Hand-craft 5+ Tiny deals with known optimal scores → brute force matches
- [ ] Scores always sum to `config.total_points()` or `config.total_points() * 3`
- [ ] Completes in <100ms on Tiny

**Done when:** Brute-force produces correct results on all hand-crafted cases.

---

### Step 3.2 — Max^n solver

**Implement** in `hearts-core/src/solver/maxn.rs`:

Max^n search: each player minimizes their own score. Note that Max^n does NOT support standard alpha-beta pruning bounds across players — only weak pruning is possible (a player's score cannot exceed `total_points`, and known lower bounds on one player's score provide weak upper bounds on others). This is the **correct** solver for multiplayer Hearts, but it will be too slow for large decks under PIMC volume.

**Test:**
- [ ] **100 random Tiny deals: `maxn == brute_force` on every single one** (most important test)
- [ ] Same hand-crafted cases from 3.1
- [ ] Faster than brute force on Tiny

**Done when:** 100% agreement with brute force. Zero discrepancies.

---

### Step 3.3 — Paranoid solver

**Implement** in `hearts-core/src/solver/paranoid.rs`:

Paranoid search: treats the game as 2-player zero-sum. The AI (one player) minimizes its own score; the 3 opponents form a coalition that maximizes the AI's score with shared perfect information. This enables standard alpha-beta pruning with full α/β bounds, giving massive speedup over Max^n.

Tradeoff: paranoid search is pessimistic — it assumes opponents coordinate perfectly against you, which they don't in reality. Results will differ from Max^n/brute-force.

Interface: `paranoid_solve(state, ai_player) -> (Card, i32)` — returns best move and score for the specified AI player.

**Test:**
- [ ] Produces valid scores (0 ≤ ai_score ≤ `config.total_points()`)
- [ ] **100 random Tiny deals: paranoid score ≥ Max^n score for the AI player** (paranoid is pessimistic — it thinks you'll do at least as badly as Max^n says)
- [ ] Benchmark: paranoid vs Max^n on Small deck → paranoid is significantly faster
- [ ] Hand-crafted deal where opponents can obviously coordinate → paranoid finds the correct worst-case
- [ ] Hand-crafted deal where paranoid is overly conservative → document the gap vs Max^n

**Done when:** Paranoid solver correct (pessimistic bound verified), measurably faster than Max^n.

---

### Step 3.4 — Transposition table

**Implement** in `hearts-core/src/solver/transposition.rs`:

Zobrist hashing: pre-computed random u64 per (player, card), plus components for trick leader and cards in current trick. Fixed-size hash map with always-replace policy.

Hash key must include:
1. Exact cards remaining in each player's hand
2. Trick leader (PlayerIndex)
3. Cards currently played in the active incomplete trick
4. Points-taken state — at minimum, a hash of which players have taken any points (4 bits, captures moon-relevant state compactly). Full points vector is more correct but reduces TT hit rate.

Wire into both `maxn.rs` and `paranoid.rs`.

**Test:**
- [ ] **100 random Tiny deals: Max^n+TT still matches brute force exactly**
- [ ] **100 random Tiny deals: paranoid+TT matches paranoid without TT**
- [ ] Benchmark: solvers with TT vs without TT on Small → measurable speedup
- [ ] TT probe returns None on first access, correct entry after store
- [ ] Two states differing only in points-taken produce different hashes

**Done when:** TT integrated into both solvers, correctness preserved, speedup measured.

---

### Step 3.5 — Move ordering

**Implement** in both `maxn.rs` and `paranoid.rs`:

Sort legal moves before exploring: dump Q♠ first → dump point cards when voiding → duck under high cards → lead short suits → low cards first when following.

**Test:**
- [ ] **100 Tiny deals: Max^n still matches brute force**
- [ ] Benchmark on Small/Medium: fewer nodes evaluated than without ordering
- [ ] No correctness change (ordering is heuristic only)

**Done when:** Correctness preserved, node count reduced.

---

### Step 3.6 — Undo optimization and performance benchmarks

**Implement:**
- Add `undo_card()` to `GameState` for allocation-free backtracking:
  - Restores `trick_history`, decrements `points_taken`
  - Correctly toggles `hearts_broken` back to `false` if the undone card was the first heart played
  - Restores `current_player`, `trick_number`, `cards_played`
- Use undo-based traversal in both solvers (instead of cloning state per node)
- `hearts-core/benches/solver_bench.rs` using `criterion`: benchmark DD solve per deck size, per solver variant

**Top-down performance budgeting:**

The real target is PIMC decision time. Budget: ~2 seconds per decision, 200 sampled worlds, ~8 legal moves average = ~1600 solves per decision → **~1.25ms per solve**.

| Deck | Max^n Target | Paranoid Target | PIMC-viable? |
|------|-------------|-----------------|--------------|
| Tiny | < 0.01ms | < 0.01ms | Both |
| Small | < 1ms | < 0.1ms | Both |
| Medium | < 50ms | < 5ms | Paranoid only |
| Full | Likely too slow | < 10ms (stretch: <1.25ms) | Paranoid maybe; may need endgame split |

Add a PIMC-volume benchmark: 1600 sequential solves of the same position, report total wall time. This is the real performance gate.

**Test:**
- [ ] Undo-based Max^n still matches brute force on Tiny
- [ ] Undo-based paranoid matches non-undo paranoid on Tiny
- [ ] Benchmarks run and produce timing data
- [ ] Profile: no unexpected allocations in hot loop
- [ ] Document which solver + deck combinations are PIMC-viable

**Done when:** Benchmarks exist. Performance gaps documented. Clear plan for which solver to use at each deck size.

---

### Step 3.7 — Moon shooting in the solvers

**Implement** in both `maxn.rs` and `paranoid.rs`:

During search, detect if a player has taken all point cards so far and can win the rest. Adjust the minimax objective: shooter maximizes points, others try to block.

In paranoid mode: if the AI could shoot, it switches to maximizing its own points. If an opponent could shoot, the coalition allows it only if it hurts the AI (opponents get `total_points` each).

**Test:**
- [ ] Deal where moon shot is optimal → Max^n solver finds shooting line
- [ ] Deal where one player can almost shoot but another can block → solver finds block
- [ ] **100 Tiny deals: Max^n still matches brute force** (brute force already handles moon correctly)
- [ ] Paranoid solver handles moon correctly from the AI's perspective

**Done when:** Both solvers find shooting and blocking lines correctly.

---

## Phase 4: Belief Tracking and PIMC

### Step 4.1 — Observation model and information set

**Implement** in `hearts-core/src/belief/observation.rs`:

`Observation` — what a single player can see:
- `my_hand: CardSet`, `cards_played: CardSet`
- `voids: [[bool; 4]; 4]` — known voids from failed suit-follows (derived from TrickResult history)
- `trick_history: Vec<TrickResult>`, `current_trick`

`Observation::from_game_state(state, viewer)` — extract viewer's knowledge.
`unknown_cards(obs) -> CardSet` — cards not in my hand and not yet played.

Void inference: iterate over `trick_history`, for each `TrickResult` check if any player played off-suit when the led suit was different → mark void.

**Test:**
- [ ] Fresh deal: observation shows only my hand, no cards played
- [ ] After 2 tricks: `cards_played` has 8 cards
- [ ] Player fails to follow clubs → `voids[player][Clubs]` is true
- [ ] `unknown_cards().count()` = sum of opponents' hand sizes
- [ ] Void inference correctly derived from TrickResult's `cards` field

**Done when:** Observation correctly captures public + private info.

---

### Step 4.2 — Constraint-based world sampler

**Implement** in `hearts-core/src/belief/sampler.rs`:

`sample_world(obs, rng) -> [CardSet; 4]`: distribute unknown cards to 3 opponents respecting void constraints and hand sizes.

Algorithm: partition unknowns by suit, assign only to non-void players. Use rejection sampling or deal-and-check for simplicity first.

**Test:**
- [ ] Sampled world respects void constraints
- [ ] Correct hand sizes for each player
- [ ] Union of all hands = deck_cards
- [ ] Viewer's hand unchanged
- [ ] Over 10000 samples with no constraints: cards distributed roughly equally
- [ ] With void constraint: voided player never has that suit's cards
- [ ] 500 samples in <10ms

**Done when:** Sampler generates valid, constraint-respecting worlds efficiently.

---

### Step 4.3 — Basic PIMC search

**Implement** in `hearts-core/src/search/pimc.rs`:

`pimc_choose(state, player, n_worlds, solver, rng) -> Card`:
1. Get observation, compute legal moves
2. Sample `n_worlds` consistent worlds
3. For each world, for each legal move: DD solve → record player's score
4. Return move with lowest average score

The `solver` parameter selects Max^n or paranoid. Use whichever is PIMC-viable for the current deck size (from Step 3.6 benchmarks). On Full, this will almost certainly be paranoid.

Create `PIMCBot` implementing `Player`.

**Test:**
- [ ] Tiny deal with obvious best move (can dump Q♠) → PIMC finds it
- [ ] PIMC (n=50) vs 3 RandomBots, Tiny, 500 games → PIMC averages fewer points
- [ ] PIMC (n=50) vs 3 RuleBots, Small, 500 games → competitive or better
- [ ] PIMC decision on Tiny with n=100 in <1 second

**Done when:** PIMC bot outperforms RandomBot. Acceptable performance on Tiny/Small.

---

### Step 4.4 — PIMC parallelism and optimization

**Implement:**
- Use `rayon` to parallelize DD solves across sampled worlds
- Pre-generate all N sampled worlds sequentially (deterministic with seed), then solve in parallel (order-independent since we're collecting scores per move)
- Early termination: stop if one move dominates after N/2 samples
- Pre-allocate TT per thread, clear between worlds

**Test:**
- [ ] Benchmark: parallel vs sequential on Small deck → speedup
- [ ] Same seeds produce same sampled worlds (sampling is deterministic; solve order doesn't affect results)
- [ ] PIMC n=200 on Medium in <5 seconds per decision

**Done when:** Parallel PIMC is faster. Acceptable on Medium deck.

---

## Phase 5: Alpha-Mu Search

### Step 5.1 — Alpha-Mu aggregation

**Implement** in `hearts-core/src/search/alpha_mu.rs`:

Alpha-Mu replaces PIMC's averaging with worst-case-over-partitions. Same PIMC pipeline (sample worlds → DD solve each), but different aggregation:

1. For each sampled world, solve all legal moves → get score per (world, move).
2. Group worlds by which move is optimal → partitions.
3. For each candidate move, compute its worst-case score across partitions.
4. Pick the move with the best (lowest) worst-case.

This directly addresses PIMC's strategy fusion problem: it avoids picking a move that's usually good but sometimes catastrophic.

Create `AlphaMuBot`.

**Test:**
- [ ] Construct a strategy-fusion test case where PIMC picks the wrong move but Alpha-Mu picks the right one (e.g., move A is optimal in 60% of worlds but catastrophic in 40%; move B is slightly suboptimal in 60% but safe in 40%)
- [ ] Alpha-Mu vs PIMC (2 each), Small, 2000 games → Alpha-Mu at least as good
- [ ] Alpha-Mu's move is never dominated (not worse in ALL worlds than another move)
- [ ] Falls back to PIMC behavior when all worlds agree on the best move

**Done when:** Alpha-Mu produces decisions. Avoids strategy fusion in constructed cases.

---

### Step 5.2 — Softened Alpha-Mu (probability-weighted)

**Implement** in `alpha_mu.rs`:

Instead of pure worst-case over partitions: weight each partition by probability (from the belief tracker). Within each partition, still use worst-case score. This is a middle ground between PIMC (pure average, too optimistic) and pure Alpha-Mu (pure worst-case, too conservative).

Note: when using the paranoid solver underneath, the combination of paranoid (pessimistic opponent model) + pure Alpha-Mu (pessimistic aggregation) compounds conservatism. Softened Alpha-Mu counteracts this.

Softening parameter `λ ∈ [0, 1]`: `λ=0` is pure average (PIMC), `λ=1` is pure worst-case (Alpha-Mu).

**Test:**
- [ ] `λ=1` → identical to pure Alpha-Mu
- [ ] `λ=0` → identical to PIMC averaging
- [ ] Softened vs pure Alpha-Mu, Medium, 2000 games → softened at least as good
- [ ] Parameter sweep: `λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`, measure average score over 2000 games each
- [ ] Document optimal λ range for each deck size

**Done when:** Softened Alpha-Mu implemented. Optimal λ range identified.

---

## Phase 6: Full Game and Polish

### Step 6.1 — Scale to full 52-card deck

**Implement:**
- Run all bots on Full deck
- Profile and optimize DD solver:
  - If paranoid solver hits <1.25ms on Full → PIMC-viable, proceed
  - If not → implement endgame split: exact solve last K tricks (K=4-6), heuristic evaluation for earlier tricks. Move this up from Step 6.3 if needed.
- Tune: n_worlds, TT size, move ordering, λ

**Test:**
- [ ] Paranoid DD solve on Full: measure p50/p95/p99 latency
- [ ] PIMC decision (n=200) on Full: total wall time <5 seconds (stretch: <2 seconds)
- [ ] 100 Full games with Alpha-Mu bot complete without timeout
- [ ] Alpha-Mu vs RuleBot, 2000 Full games → Alpha-Mu wins convincingly

**Done when:** Full system works on standard 52-card Hearts within time budgets.

---

### Step 6.2 — Enhanced belief tracking (soft signals)

**Implement** in `hearts-core/src/belief/bayesian.rs`:

Bayesian updates using play patterns as soft signals:
- Leading low heart → less likely shooting moon
- Playing high when could duck → adjust beliefs
- P(hand | observations) ∝ P(observations | hand) × P(hand)

Use RuleBot's policy as the likelihood model: given a hand and game state, what's the probability the RuleBot would play the card that was actually played?

**Test:**
- [ ] Failed suit-follow → P(has that suit) = 0
- [ ] Soft signals shift distributions in expected direction
- [ ] Belief-enhanced PIMC vs constraint-only PIMC, 2000 Full games → at least as good

**Done when:** More accurate world samples. Measurable improvement.

---

### Step 6.3 — Endgame solver

**Implement** in `hearts-core/src/search/endgame.rs`:

For last K tricks (K=4-6), if possible opponent configurations are below a threshold, enumerate all consistent deals and solve exactly instead of sampling. This eliminates sampling noise in the endgame where information is nearly perfect.

Note: if this was pulled forward to Step 6.1 for performance reasons, expand it here with further optimization.

**Test:**
- [ ] Endgame solver agrees with brute force on Tiny (where the whole game is the "endgame")
- [ ] PIMC + endgame vs pure PIMC, 2000 Full games → improvement
- [ ] Last 4 tricks with known hands solved in <1ms

**Done when:** Endgame solver correct and improves late-game play.

---

### Step 6.4 — Comprehensive evaluation suite

**Implement:**
- Tournament mode: all bot combinations, round-robin
- Statistical significance testing (5000+ games per matchup, report confidence intervals on score differences)
- Output: tables of win rates, scores, moon shoots, decision times

CLI: `cargo run -p hearts-cli -- tournament --games 5000 --deck full`

**Test:**
- [ ] Consistent rankings across runs (same seed)
- [ ] Expected ranking: Alpha-Mu > PIMC > RuleBot > RandomBot
- [ ] Confidence intervals are tight enough to distinguish adjacent bots

**Done when:** Rigorous evaluation framework exists.

---

## Dependency Graph

```
0.1 Cargo setup
 └─ 1.1 Card types
     └─ 1.2 CardSet (bitboard)
         ├─ 1.3 Deck configs (with total_points)
         └─ 1.4 Trick types (with TrickResult)
             └─ 1.5 GameState + legal moves
                 └─ 1.6 Moon scoring
                     └─ 1.7 Game loop (Player trait)
                         ├─ 2.1 RandomBot
                         │   └─ 2.2 RuleBot
                         │       └─ 2.3 Stats + CLI
                         └─ 3.1 Brute-force solver (Max^n oracle)
                             ├─ 3.2 Max^n solver
                             │   └─ 3.3 Paranoid solver
                             │       └─ 3.4 Transposition table (both solvers)
                             │           └─ 3.5 Move ordering (both solvers)
                             │               └─ 3.6 Undo + benchmarks
                             │                   └─ 3.7 Moon in solvers
                             │                       ├─ 4.1 Observation model
                             │                       │   └─ 4.2 World sampler
                             │                       │       └─ 4.3 Basic PIMC
                             │                       │           └─ 4.4 PIMC parallel
                             │                       │               └─ 5.1 Alpha-Mu
                             │                       │                   └─ 5.2 Softened Alpha-Mu
                             │                       └─ 6.1 Full deck scaling
                             │                           └─ 6.2 Enhanced beliefs
                             │                               └─ 6.3 Endgame solver
                             │                                   └─ 6.4 Evaluation suite
```

## Key Design Decisions

1. **`CardSet` (u64 bitboard) is the universal set type.** Never use `Vec<Card>` for hands. `legal_moves()` returns `CardSet`.
2. **`GameState` supports both clone and play/undo.** Game runner uses clone; DD solvers use undo for speed.
3. **Two solver variants exist for different purposes.** Max^n is correct (matches brute force) but slow. Paranoid is fast but pessimistic. PIMC uses whichever is viable for the deck size.
4. **`Player` trait is synchronous.** Parallelism is internal to bot implementations.
5. **`DeckConfig` is a parameter, not global.** Mix deck sizes in one test binary.
6. **All randomness via explicit `rng` parameters.** Seeded RNG = reproducible tests.
7. **Brute-force solver = oracle.** Every solver optimization is validated against it on Tiny decks.
8. **Performance targets are derived top-down** from PIMC decision budget, not set per-solve in isolation.