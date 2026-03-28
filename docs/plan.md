# Hearts Solver — Design Document

## 1. Goals & Scope

Build a state-of-the-art Hearts AI, drawing from modern poker solver architecture (Pluribus/Libratus style) adapted to the structure of trick-taking games. No card passing phase.

**Development strategy:** start with a reduced deck (fewer cards per player) against rule-based heuristic bots, then scale up to a full 52-card game and self-play.

---

## 2. Why Hearts Is Hard (and Where It Differs from Poker)

Hearts sits in an interesting spot. Unlike poker, there's no betting, so there's no explicit bluffing or bet-sizing abstraction problem. But unlike fully cooperative trick-taking (Bridge with a partner), it's a 4-player competitive game where the objectives of all players interact in complex ways.

Key challenges:

- **4 players.** CFR has no Nash equilibrium convergence guarantee beyond 2-player zero-sum. Pluribus dealt with this by computing an approximate blueprint via MCCFR and then refining with real-time search — we should follow a similar path.
- **Shooting the moon.** The payoff function is discontinuous: taking all 26 points flips the outcome entirely. This creates a non-linear strategic dynamic where a player who's "losing" (accumulating hearts) might suddenly become dangerous. Any search or evaluation must handle this.
- **Card play is tightly constrained.** You must follow suit. This drastically reduces the branching factor compared to poker's arbitrary bet sizing, which is good — less need for action abstraction.
- **Imperfect information is "simple."** The only hidden information is which cards are in which opponent's hand. No private betting history, no ranges. This makes belief tracking tractable and double-dummy solving very effective.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Game Engine                     │
│   (rules, reduced deck configs, state mgmt)     │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐       ┌─────────────────────┐
│ Belief Tracker│       │  Heuristic Bots     │
│ (Bayesian     │       │  (baseline opponents│
│  card infer.) │       │   for dev/testing)  │
└───────┬───────┘       └─────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│           Real-Time Search                │
│                                           │
│  ┌─────────────┐    ┌──────────────────┐  │
│  │    PIMC      │    │   Alpha-Mu       │  │
│  │  Sampling    │───▶│   Aggregation    │  │
│  └──────┬──────┘    └──────────────────┘  │
│         │                                 │
│         ▼                                 │
│  ┌──────────────┐                         │
│  │ Double-Dummy │                         │
│  │   Solver     │                         │
│  └──────────────┘                         │
└───────────────────┬───────────────────────┘
                    │
                    ▼  (later phases)
┌───────────────────────────────────────────┐
│         Learned Components                │
│                                           │
│  ┌────────────────┐  ┌─────────────────┐  │
│  │ Rollout Policy │  │ Value Network   │  │
│  │ (guided sims)  │  │ (skip search    │  │
│  │                │  │  in clear spots)│  │
│  └────────────────┘  └─────────────────┘  │
└───────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Game Engine

**Responsibilities:** State representation, legal move generation, trick resolution, scoring (including moon detection), and — critically — support for configurable reduced decks.

**Reduced deck configurations** for development:

| Config   | Cards/Player | Total | Purpose                           |
|----------|-------------|-------|-----------------------------------|
| Tiny     | 3           | 12    | Unit tests, algorithm debugging   |
| Small    | 5           | 20    | Fast iteration on search/belief   |
| Medium   | 8           | 32    | Meaningful strategic complexity    |
| Full     | 13          | 52    | Final target                      |

For reduced decks, strip low cards first (keep face cards and hearts to preserve strategic structure). Always keep Q♠ and A♠, K♠. Ensure each suit has enough cards to make voiding possible but not trivial. The 2♣ lead rule can be relaxed for tiny/small configs — just use lowest club or arbitrary lead.

**State representation:** Bitboard per player per suit (uint16 per suit is enough for 13 ranks). Trick history as a compact log. This keeps states small for hashing and fast for legal move generation.

**Key design decision:** The engine should expose a `compatible_worlds(info_set) -> Iterator[State]` method that generates complete deals consistent with a player's observation. This is the foundation for both PIMC sampling and belief tracking.

### 4.2 Belief Tracker

The belief tracker maintains a probability distribution over opponent card holdings, conditioned on all observed play. This is the single biggest differentiator between a mediocre and strong Hearts AI.

**What signals to incorporate:**

- **Hard constraints.** If a player didn't follow suit, they're void in that suit — eliminate all cards of that suit from their possible holdings. This alone is powerful.
- **Played cards.** Trivially remove from all distributions.
- **Card counting.** Track how many unknown cards remain per suit per player. Combined with void inference, this narrows distributions fast.
- **Soft signals (later phase).** Play patterns leak information. A player who leads a low heart is probably not trying to shoot the moon. A player who ducks under the Q♠ when they could have played higher might be void-hunting. These are harder to quantify but matter.

**Implementation:** Maintain a per-player, per-card probability matrix (or constraint set). For sampling, use rejection sampling or a constraint-satisfaction sampler that deals remaining cards to opponents consistent with all known constraints. Start with hard constraints only — this already beats uniform sampling dramatically.

**Connection to poker:** This is analogous to opponent range tracking in poker. The difference is that in poker, you update ranges based on betting actions (which are strategic signals). In Hearts, you update based on card play, which is a mix of forced moves and strategic choices. The forced-move component (must follow suit) gives you much cleaner inference than poker betting.

### 4.3 Double-Dummy Solver

A "double-dummy" solver plays out a fully-determined trick-taking game (all hands visible) optimally. This is the workhorse inside PIMC — for each sampled world, you solve it double-dummy to get the value of each candidate move.

**For Hearts specifically:** Unlike Bridge (where you maximize tricks), Hearts minimizes points taken (or maximizes if shooting). The solver needs to handle the Q♠ (13 points) and hearts (1 point each), plus the moon-shoot discontinuity.

**Algorithm:** Alpha-beta with the following enhancements:

- **Transposition table** keyed on (remaining cards per player, trick leader, cards played in current trick). Bitboard representation makes hashing cheap.
- **Move ordering.** In Hearts, good heuristics are: play Q♠ on a trick you won't win, duck under high cards in danger suits, void short suits early. Good ordering dramatically improves pruning.
- **Moon-shoot awareness.** The solver must track whether any single player has taken all point cards so far, and adjust the objective accordingly. This makes the evaluation non-trivial — you can't just minimize your own points greedily because sometimes you need to prevent an opponent from shooting.

**Performance target:** For a full 13-card game, double-dummy solves should complete in <10ms. This is very achievable with alpha-beta + transposition table on modern hardware — Bridge double-dummy solvers (which are harder due to the 2-partnership structure) routinely hit this. For reduced games, it'll be sub-millisecond.

### 4.4 PIMC + Alpha-Mu Search

This is the core decision-making loop. At each decision point:

**PIMC layer:**
1. Sample N worlds from the belief tracker (N = 100–500, tunable).
2. For each world, run the double-dummy solver for every legal move, producing a score vector: `scores[world][move] -> points_taken`.

**Alpha-mu aggregation:**
3. For each move, find which worlds it's optimal in. This partitions the N worlds.
4. For each move, compute its worst-case score across the partitions where it's *not* optimal.
5. Choose the move with the best worst-case.

**Softened alpha-mu (recommended):** Pure worst-case is too conservative for Hearts, where some unlikely worlds shouldn't drive decisions. Weight each partition by its probability (estimated from the belief tracker). This becomes: `value(move) = Σ_partition P(partition) * score(move, partition)`, but where `score(move, partition)` uses the worst sample in that partition rather than the average. This is a middle ground between pure averaging (PIMC) and pure worst-case (alpha-mu).

**Moon-shoot integration:** When evaluating moves, the solver must consider not just your own score but also whether a move enables/prevents an opponent's moon shot. In the double-dummy solve, if an opponent can shoot, their score becomes -26 (or +26 to everyone else). This needs special-case logic in the solver's evaluation.

**Practical concern — speed budget:** At N=200 worlds with 10 legal moves, you need 2000 double-dummy solves per decision. At <10ms each, that's 20 seconds for a full game decision. For reduced games, much faster. If this is too slow, you can:
- Reduce N for early-game decisions (less information, diminishing returns on more samples).
- Use a lightweight evaluation function instead of full double-dummy for rollouts, reserving exact solving for the last few tricks (endgame solving, exactly as Libratus does for poker).
- Parallelize — double-dummy solves across sampled worlds are embarrassingly parallel.

### 4.5 Heuristic Bots (Baseline Opponents)

For development and testing, you need bots that play reasonably but aren't perfect. Levels:

**Random-legal bot:** Plays a uniformly random legal move. Useful only for smoke tests.

**Rule-based bot (primary development target):**
- Avoid taking points when possible (duck under high cards, slough hearts on off-suit tricks).
- Play Q♠ on a trick led by someone else if possible.
- If holding Q♠ and can't safely dump it, try to void spades.
- Lead low cards in non-heart suits to probe for voids.
- Don't lead hearts until broken (or forced).
- Simple moon-shoot detection: if an opponent has taken many points, play to give them a point card to break the shoot.

**Stronger heuristic bot (stretch):**
- Track which cards have been played, count remaining cards per suit.
- Estimate void probabilities for opponents.
- Simple lookahead (1-trick greedy evaluation).

These bots serve two purposes: opponents during development, and later a component of opponent modeling (in Pluribus style, where you assume opponents play one of K strategies and search against that mixture).

### 4.6 Learned Components (Phase 2+)

Once the search pipeline works, neural networks can improve two bottlenecks:

**Rollout policy network:** Instead of full double-dummy solves from the root, solve only the last K tricks exactly and use a policy network to play out earlier tricks. This is analogous to how AlphaGo replaced random rollouts with a value net. Train via supervised learning on double-dummy solutions, or via self-play. Input: current trick state + hand + visible cards + belief summary. Output: probability distribution over legal moves.

**Value network:** Given a game state (visible information + belief summary), estimate expected points taken without any search. Used for fast pruning: if a move is clearly terrible according to the value net, skip the expensive double-dummy analysis. Also useful for real-time play under time pressure.

**Training approach (Pluribus-inspired):**
1. Start with MCCFR or policy gradient self-play on the reduced game to get a blueprint strategy.
2. Use the blueprint as the rollout policy inside PIMC.
3. At game time, run real-time search (PIMC + alpha-mu) to improve on the blueprint for the specific situation.
4. Pluribus's key insight: during real-time search, don't assume opponents play the Nash/blueprint strategy. Instead, assume they play one of K candidate strategies (e.g., blueprint, risk-averse variant, aggressive variant) and search against the worst case. This transfers directly to Hearts.

---

## 5. Moon Shooting

Moon shooting deserves its own section because it fundamentally changes the game's structure.

**Detection during search:** At every node in the double-dummy search, check if any player has taken all point cards so far and all remaining point cards are winnable by them. If so, the game flips: that player scores 0 and everyone else gets 26 (or equivalently, that player gets -26 in a minimize-your-score framing).

**Strategic implications:**
- When you detect an opponent might be shooting (they've taken many point cards and haven't been stopped), you must actively try to take a point card to break the shoot — even if it costs you some points.
- When you're in a position to shoot, the solver should recognize this and shift to a maximize-points objective.
- PIMC handles this somewhat naturally: in worlds where you have cards that let you shoot, the double-dummy solver will find the shoot line. But alpha-mu's worst-case reasoning is important here — you don't want to commit to a shooting line that only works in 60% of worlds.

**Heuristic for early detection:** Track a "moon threat" score for each player based on how many point cards they've taken and how many tricks remain. If the threat exceeds a threshold, inject a "break the shoot" bias into move ordering and evaluation.

---

## 6. Development Phases

### Phase 1: Foundation
- [ ] Game engine with configurable deck sizes
- [ ] State representation (bitboards)
- [ ] Legal move generation, trick resolution, scoring
- [ ] Random-legal and rule-based bots
- [ ] Game simulation loop, basic statistics collection

### Phase 2: Core Solver
- [ ] Double-dummy solver (alpha-beta + transposition table)
- [ ] Double-dummy solver correctness tests (verify against brute force on tiny games)
- [ ] Belief tracker (hard constraints: void inference, card counting)
- [ ] PIMC with uniform sampling
- [ ] Benchmark: PIMC vs heuristic bots on small deck

### Phase 3: Search Improvements
- [ ] Alpha-mu aggregation on top of PIMC
- [ ] Softened alpha-mu (probability-weighted worst-case)
- [ ] Belief-informed sampling (replace uniform with constraint-aware)
- [ ] Benchmark: alpha-mu vs plain PIMC on medium deck
- [ ] Performance profiling + optimization of double-dummy solver

### Phase 4: Moon Shooting
- [ ] Moon-shoot logic in double-dummy solver
- [ ] Moon-shoot detection/prevention heuristics
- [ ] Benchmark: moon-shoot games specifically

### Phase 5: Scale to Full Game
- [ ] Full 52-card deck testing
- [ ] Performance optimization (parallelism, endgame tablebases if feasible)
- [ ] Tune sampling count, alpha-mu softening, move ordering

### Phase 6: Learning (Optional, Major Effort)
- [ ] Self-play infrastructure
- [ ] Policy network (supervised on double-dummy solutions)
- [ ] Value network
- [ ] Blueprint strategy via self-play
- [ ] Real-time search with blueprint rollouts

---

## 7. Evaluation & Metrics

**Primary metric:** Average score per hand over 10,000+ hands against opponent pool.

**Opponent pool:**
- Rule-based bots (sanity check — should dominate)
- Copies of the solver itself at different strength settings (reduced samples, no alpha-mu, etc.)
- Eventually: other Hearts programs if available

**Diagnostic metrics:**
- Decision time per move (track percentile distribution, not just average)
- Double-dummy solve time distribution
- Belief tracker accuracy: after the game, compare inferred distributions to actual hands. Measure log-likelihood of the true deal under the belief model.
- Moon-shoot rate: how often does the AI successfully shoot? How often does it fail to prevent an opponent's shoot?
- Score breakdown by game phase (early/mid/late tricks)

**A/B testing framework:** When testing a change (e.g., adding alpha-mu on top of PIMC), play the new version against the old version in a 4-player setup (2 copies of each). Run enough games for statistical significance. Hearts has high variance per hand, so you need large sample sizes — 5,000+ hands minimum for detecting moderate improvements.

---

## 8. Technical Decisions

**Language:** The engine, solver, and search should be in a compiled language (Rust or C++) for speed. The double-dummy solver's performance is critical-path. Python is fine for orchestration, training, and analysis, but not for the inner loop.

**Parallelism:** Each PIMC sample (determinize → double-dummy solve) is independent. Parallelize across samples trivially. On a modern machine with 8+ cores, this gives near-linear speedup for the PIMC layer.

**Testing:** The double-dummy solver is the most bug-prone component. Test it exhaustively on tiny games where brute-force enumeration is feasible. Any bug here silently corrupts all downstream decisions.

----

## 9. For now

Reality Check: $\alpha\mu$ is a Search, Not an AggregatorThe Misconception: In Section 4.4, you describe $\alpha\mu$ as an aggregation step that happens after you run independent double-dummy (DD) solves for each world.The Reality: If you solve the worlds independently from the bottom up and then try to aggregate the root scores, you are still fundamentally doing PIMC and will suffer from Strategy Fusion. The AI will still "cheat" by assuming it knows the cards in the sub-trees.True $\alpha\mu$ algorithm is a specialized minimax search tree where the nodes themselves are vectors of worlds (the belief state), not single game states.You do not run $N$ independent DD searches.Instead, you run one search tree. At every decision node, the algorithm forces the engine to choose a single legal move that applies to the entire vector of consistent worlds simultaneously.This is computationally heavier than standard PIMC because you cannot easily prune a branch unless it is worse across the entire vector of worlds.Recommendation: Start with vanilla PIMC and average the scores. Accept the strategy fusion initially just to get the pipeline working. When you implement $\alpha\mu$, realize you will be writing a completely new search function, not just a wrapper around your DD solver.