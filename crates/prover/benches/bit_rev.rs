#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use stwo_prover::core::fields::m31::BaseField;

const LOG_SIZE: usize = 25;
const SIZE: usize = 1 << LOG_SIZE;

pub fn cpu_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::utils::bit_reverse;
    let data = (0..SIZE).map(BaseField::from).collect_vec();

    let bench_id = &format!("cpu bit_rev 2^{LOG_SIZE}");
    c.bench_function(bench_id, |b| {
        b.iter_batched(
            || data.clone(),
            |mut data| bit_reverse(&mut data),
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::backend::simd::bit_reverse::bit_reverse_m31;
    use stwo_prover::core::backend::simd::column::BaseFieldVec;
    let data = (0..SIZE).map(BaseField::from).collect::<BaseFieldVec>();

    let bench_id = &format!("simd bit_rev 2^{LOG_SIZE}");
    c.bench_function(bench_id, |b| {
        b.iter_batched(
            || data.data.clone(),
            |mut data| bit_reverse_m31(&mut data),
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_bit_rev, cpu_bit_rev);
criterion_main!(bit_rev);
