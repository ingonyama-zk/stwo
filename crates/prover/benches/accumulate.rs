#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use stwo_prover::core::air::accumulation::AccumulationOps;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::backend::cpu::CpuBackend;
use stwo_prover::core::backend::simd::SimdBackend;

const LOG_SIZE: usize = 28;
const SIZE: usize = 1 << LOG_SIZE;

pub fn cpu_accumulate(c: &mut Criterion) {


    let data = SecureColumnByCoords::<CpuBackend> {
        columns: std::array::from_fn(|i| vec![BaseField::from_u32_unchecked(i as u32); SIZE]),
    };
    let data2 = data.clone();
    let bench_id = format!("cpu accumulate SecureColumn 2^{LOG_SIZE}");
    c.bench_function(&bench_id, |b| {
        b.iter_batched(
            || data.clone(),
            |mut data| CpuBackend::accumulate(&mut data, &data2),
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_accumulate(c: &mut Criterion) {

    let cpu_col = SecureColumnByCoords::<CpuBackend> {
        columns: std::array::from_fn(|i| vec![BaseField::from_u32_unchecked(i as u32); SIZE]),
    };

    let columns = cpu_col.columns.map(|col| col.into_iter().collect());
    let data = SecureColumnByCoords::<SimdBackend> { columns };

    let data2 = data.clone();
    let bench_id = format!("simd accumulate SecureColumn 2^{LOG_SIZE}");
    c.bench_function(&bench_id, |b| {
        b.iter_batched(
            || data.clone(),
            |mut data| SimdBackend::accumulate(&mut data, &data2),
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_accumulate, cpu_accumulate);
criterion_main!(bit_rev);
