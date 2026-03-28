use criterion::{criterion_group, criterion_main, Criterion};
use hearts_core::deck::DeckConfig;
use hearts_core::game_state::GameState;
use hearts_core::solver::{brute_force, maxn, paranoid};
use hearts_core::solver::transposition::{TranspositionTable, ZobristKeys};
use hearts_core::solver::paranoid::ParanoidTT;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn bench_brute_force_tiny(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Tiny.deal(&mut rng);
    let state = GameState::new_with_deal(hands, DeckConfig::Tiny);

    c.bench_function("brute_force_tiny", |b| {
        b.iter(|| brute_force::brute_force_solve(&state))
    });
}

fn bench_maxn_tiny(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Tiny.deal(&mut rng);

    c.bench_function("maxn_tiny", |b| {
        b.iter(|| {
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            maxn::maxn_solve(&mut state)
        })
    });
}

fn bench_paranoid_tiny(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Tiny.deal(&mut rng);

    c.bench_function("paranoid_tiny", |b| {
        b.iter(|| {
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Tiny);
            let ai = state.current_player;
            paranoid::paranoid_solve(&mut state, ai)
        })
    });
}

fn bench_maxn_small(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Small.deal(&mut rng);

    c.bench_function("maxn_small", |b| {
        b.iter(|| {
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Small);
            maxn::maxn_solve(&mut state)
        })
    });
}

fn bench_paranoid_small(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Small.deal(&mut rng);

    c.bench_function("paranoid_small", |b| {
        b.iter(|| {
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Small);
            let ai = state.current_player;
            paranoid::paranoid_solve(&mut state, ai)
        })
    });
}

fn bench_paranoid_small_tt(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Small.deal(&mut rng);
    let keys = ZobristKeys::new(42);

    c.bench_function("paranoid_small_tt", |b| {
        let mut tt = ParanoidTT::new(20);
        b.iter(|| {
            tt.clear();
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Small);
            let ai = state.current_player;
            paranoid::paranoid_solve_with_tt(&mut state, ai, &mut tt, &keys)
        })
    });
}

fn bench_paranoid_medium(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Medium.deal(&mut rng);

    let mut group = c.benchmark_group("paranoid_medium");
    group.sample_size(10); // medium is slow, reduce samples

    group.bench_function("no_tt", |b| {
        b.iter(|| {
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Medium);
            let ai = state.current_player;
            paranoid::paranoid_solve(&mut state, ai)
        })
    });

    let keys = ZobristKeys::new(42);
    group.bench_function("with_tt", |b| {
        let mut tt = ParanoidTT::new(22);
        b.iter(|| {
            tt.clear();
            let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Medium);
            let ai = state.current_player;
            paranoid::paranoid_solve_with_tt(&mut state, ai, &mut tt, &keys)
        })
    });

    group.finish();
}

/// PIMC-volume benchmark: 1600 sequential solves (simulating 200 worlds × 8 moves).
/// This is the real performance gate for PIMC decisions.
fn bench_pimc_volume_small(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let hands = DeckConfig::Small.deal(&mut rng);

    let mut group = c.benchmark_group("pimc_volume_small");
    group.sample_size(10);

    group.bench_function("1600_paranoid_solves", |b| {
        b.iter(|| {
            let mut total = 0i32;
            for _ in 0..1600 {
                let mut state = GameState::new_with_deal(hands.clone(), DeckConfig::Small);
                let ai = state.current_player;
                total += paranoid::paranoid_solve(&mut state, ai);
            }
            total
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_brute_force_tiny,
    bench_maxn_tiny,
    bench_paranoid_tiny,
    bench_maxn_small,
    bench_paranoid_small,
    bench_paranoid_small_tt,
    bench_paranoid_medium,
    bench_pimc_volume_small,
);
criterion_main!(benches);
