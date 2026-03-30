# Regression Eval Training

Train evaluation functions to approximate the paranoid solver at 28 cards remaining.

## Generate Training Data

```bash
# Standard (35 hand-crafted features only)
cargo run --release --bin generate_training_data -- --samples 5000

# With raw card positions (214 extra columns for neural net training)
cargo run --release --bin generate_training_data -- --samples 50000 --raw-cards --seed 123 --output regression/data/training_data_50k.csv

# Control parallelism
RAYON_NUM_THREADS=16 cargo run --release --bin generate_training_data -- --samples 50000 --raw-cards
```

Output goes to `regression/data/` (gitignored). Each sample takes ~1s of solver time; parallelized across all cores via rayon.

## Train Models

```bash
# Linear regression on 35 features (default)
python regression/train_eval.py --filter-moon

# Combinatorial features (pairwise products)
python regression/train_eval.py --mode combinatorial --filter-moon

# Neural net on raw card positions (requires PyTorch)
python regression/train_eval.py --mode nn --input regression/data/training_data_50k.csv --filter-moon
```

## Python Dependencies

```bash
pip install torch scikit-learn matplotlib numpy
```

## EC2 Deployment

The data generator is pure Rust with rayon — compiles on Linux x86_64 with no changes.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and generate
git clone <repo> && cd hearts-solver
cargo run --release --bin generate_training_data -- --samples 50000 --raw-cards --output regression/data/training_data_50k.csv

# Train (install Python deps first)
pip install torch scikit-learn matplotlib numpy
python regression/train_eval.py --mode nn --input regression/data/training_data_50k.csv --filter-moon
```

Recommended instance: `c7a.8xlarge` (32 vCPUs) for ~4x speedup over M5 Pro.
