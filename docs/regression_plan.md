# Hearts Solver: Linear Regression Eval — Implementation Plan

## Context

I have a Hearts card game engine in Rust at `crates/hearts-core/`. The solver is at `crates/hearts-core/src/solver/paranoid.rs`. It uses alpha-beta search with the paranoid assumption (AI vs 3 opponents cooperating) and solves to terminal state.

The solver is too slow at high card counts. At 28 cards remaining it takes 127ms, at 32 it takes 16.4s, above that it's unusable. I need it under ~10ms for PIMC (Perfect Information Monte Carlo) which calls the solver ~150 times per decision.

The plan: train a linear regression model to approximate the solver's output at 28 cards remaining, then use it as a leaf evaluator when searching from higher card counts. The solver will search from 52 cards down to 28 remaining, call the regression instead of continuing to terminal, giving bounded search time.

## File Structure

All regression-related code should follow this layout:

```
hearts-solver/
  ├── regression/                                # All regression artifacts
  │   ├── train_eval.py                          # Python training script (Step 2)
  │   ├── data/                                  # Generated CSVs (gitignored)
  │   └── plots/                                 # Scatter plots, diagnostics
  ├── crates/
  │   ├── hearts-core/
  │   │   └── src/
  │   │       └── solver/
  │   │           ├── eval.rs                    # Feature extraction + dot product (Step 3)
  │   │           └── ...existing files
  │   └── hearts-cli/
  │       └── src/
  │           └── bin/
  │               ├── depth_bench.rs             # Existing
  │               └── generate_training_data.rs  # Data generation binary (Step 1)
```

Add `regression/data/` to `.gitignore`. The trained weights end up as a Rust constant in `solver/eval.rs`, so they're version controlled naturally.

## Step 1: Build a training data generator

Create `crates/hearts-cli/src/bin/generate_training_data.rs` that:

1. Generates random 28-card-remaining game states by playing forward from a full 52-card deal for exactly 6 tricks (24 cards played). Every generated position MUST be at the start of a new trick (no cards played to the current trick). Discard any state that ends up mid-trick.

   Use a mix of play policies to get diverse positions:
   - ~30% random legal moves
   - ~40% simple heuristic (duck when possible — play lowest card that doesn't win the trick; dump high hearts and Q♠ when void in the led suit; avoid taking Q♠)
   - ~40% rule-based 

   for duck and rule-based, see code in crates/hearts-core/src/bots/

2. For each generated 28-card state, run the existing paranoid solver to get the exact minimax value (points the AI player will take under paranoid assumption). This is Y.

3. Extract features (see feature list below) from the state. This is X.

4. Write rows to a CSV file at `regression/data/training_data.csv`: one row per sample, columns are features + label.

5. For now, enerate 1,000 samples. Print progress every 100. At 127ms per solve, this should take be fast. Allow the sample count to be configurable via command-line arg.

### Feature list to extract (for the AI player perspective):

**Already locked in (4 features):**
- `ai_points_taken`: points AI has taken so far
- `opp_max_points`: maximum points taken by any single opponent
- `opp_total_points`: total points taken by all opponents combined
- `hearts_remaining`: number of heart-point cards still in play

**Queen of Spades (7 features):**
- `ai_has_qs`: 1 if AI holds Q♠, else 0
- `ai_has_qs_exposed`: 1 if AI holds Q♠ AND has ≤2 other spades
- `ai_has_qs_protected`: 1 if AI holds Q♠ AND has ≥4 other spades
- `ai_has_as`: 1 if AI holds A♠
- `ai_has_ks`: 1 if AI holds K♠
- `qs_already_played`: 1 if Q♠ is already taken
- `ai_spade_count`: number of spades AI holds (excluding Q♠)

**Heart exposure (4 features):**
- `ai_heart_count`: number of hearts AI holds
- `ai_top_hearts`: number of hearts AI holds that are currently the highest remaining hearts (forced winners)
- `ai_void_hearts`: 1 if AI holds no hearts
- `hearts_in_play`: total number of heart cards still in any hand

**Suit control (7 features):**
- `ai_void_count`: number of suits where AI has no cards (0-3, since they must hold something)
- `ai_top_card_count`: across all 4 suits, how many suits does AI hold the highest remaining card
- `ai_top2_card_count`: across all 4 suits, count of suits where AI holds the highest OR second-highest remaining card
- `opp_void_count`: number of (opponent, suit) pairs where that opponent is void (0-9 possible)
- `ai_longest_suit`: length of AI's longest suit
- `ai_shortest_nonvoid_suit`: length of AI's shortest non-void suit (or 0 if void in 3 suits)
- `ai_has_lead`: 1 if AI is the leader for the next trick (the player who must play first)

**Interaction terms (5 features):**
- `ai_has_qs_x_opp_voids`: ai_has_qs × opp_void_count
- `ai_has_as_x_qs_in_play`: ai_has_as × (1 - qs_already_played)
- `ai_has_ks_x_qs_in_play`: ai_has_ks × (1 - qs_already_played)
- `ai_top_cards_x_hearts_remaining`: ai_top_card_count × hearts_remaining
- `ai_has_lead_x_top_card_count`: ai_has_lead × ai_top_card_count

**Moon indicators (2 features):**
- `ai_took_all_penalties`: 1 if AI has taken every penalty card that has been taken so far (and at least 1 penalty card has been taken). This detects potential AI moon shots.
- `opp_took_all_penalties`: 1 if any single opponent has taken every penalty card that has been taken so far (and at least 1 penalty card has been taken). This detects potential opponent moon shots.

That's ~30 features total. All should be cheap to compute from the game state.

## Step 2: Train the linear regression

Create `regression/train_eval.py` that:

1. Reads the CSV from `regression/data/training_data.csv`
2. Splits into 80% train, 20% test
3. Fits a linear regression (sklearn or just numpy least squares) on train
4. Reports R², MAE, and max error on test set
5. Prints the learned weights with feature names so we can sanity-check them
6. Also try Ridge regression with a few alpha values and report if it helps
7. Generate a scatter plot of predicted vs actual values on the test set. Save as `regression/plots/predicted_vs_actual.png`. Look at where the large errors cluster — if they're all moon-shooting positions, we'll handle moon as a special case later.
8. Output the weights as a Rust constant: a struct or array that can be pasted into `crates/hearts-core/src/solver/eval.rs`. Format it as valid Rust, e.g.:
```rust
pub const EVAL_WEIGHTS: [f64; 31] = [
    // bias
    3.14159,
    // ai_points_taken
    0.98765,
    // ... etc, one comment per weight with the feature name
];
```

## Step 3: Integrate the eval into the solver

1. Create `crates/hearts-core/src/solver/eval.rs` with a function `pub fn eval_position(state: &GameState, ai_player: PlayerIndex) -> f64` that computes the features from the game state and returns the dot product with the learned weights (bias + sum of weight_i * feature_i). Register it in `solver/mod.rs`.

2. Modify `paranoid_recursive` (or add a new entry point) so that the search uses the eval as a leaf evaluator. The cutoff rule is: **eval only at trick boundaries** — specifically, when `cards_remaining <= 28 AND the current trick is empty (no cards played to the current trick yet)`, call `eval_position` and return the result. If cards_remaining <= 28 but we're mid-trick, continue searching until the current trick completes, THEN eval. Keep the existing full-solve path available for comparison/testing.

3. IMPORTANT: Do NOT round the regression output to i32 prematurely. The regression returns f64, and alpha-beta pruning is more effective with real-valued scores (the granularity helps distinguish between positions). Either use f64 throughout the depth-limited search path, or scale to i32 by multiplying by 100 (centipawns-style). The full-solve path can remain integer-valued.

4. Add a benchmark/test that generates random game states and solves each one both ways (full solve vs depth-limited with regression). Report: time comparison and score difference distribution (mean, median, max, std). Run this at multiple starting card counts:
   - 1000 positions at 32 cards (4 plies of search before eval — validates basic accuracy)
   - 200 positions at 36 cards (8 plies — tests error compounding through more search)
   - 50 positions at 40 cards (12 plies — tests deeper search; fewer samples since these are slower)
   
   For 32-card positions, we have exact ground truth from the full solver (16.4s each, so 1000 positions ≈ 4.5 hours — run overnight if needed). For 36+ card positions where full solve is infeasible, just report the depth-limited solver's output and time — we can't compare to ground truth but we can verify the search completes in reasonable time and produces non-degenerate results.

## Step 4: Iterate if needed

After step 3, look at the R² and the score differences from the benchmark. If accuracy is poor (R² < 0.8 or median score difference > 2 points):

1. First diagnostic: look at the predicted-vs-actual scatter plot from step 2. If the big errors cluster on moon-shooting positions, handle moon as a special case: detect when all penalties so far belong to one player, and return a hardcoded value (0 for that player, 26 for others) instead of calling the regression. Then retrain the regression on non-moon positions only.

2. If errors are spread broadly, add more interaction features — pair-wise ANDs of the top 10 most important features (by absolute weight).

3. Consider training separate models for Q♠-capture prediction (binary: will AI take Q♠?) and hearts-taken prediction (integer: how many hearts will AI take?), then combining: predicted_score = P(Q♠) × 13 + predicted_hearts.

4. If still poor, increase training data to 200K samples.

5. Try moving the cutoff to 24 cards instead of 28 (more search, more accurate leaves, but slower per solve).

Do not jump to neural networks yet — exhaust the linear model's potential first.