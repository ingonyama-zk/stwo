use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "icicle")]
use stwo_prover::core::backend::icicle::IcicleBackend;
use stwo_prover::core::backend::{Backend, ColumnOps, CpuBackend};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::fri::FriOps;
use stwo_prover::core::poly::circle::{CanonicCoset, PolyOps, SecureEvaluation};
use stwo_prover::core::poly::line::{LineDomain, LineEvaluation};
use stwo_prover::core::poly::twiddles::TwiddleTree;

fn folding_benchmark(c: &mut Criterion) {
    for log_size in 1..=20 {
        let coset = CanonicCoset::new(log_size + 1);
        let line_domain = LineDomain::new(coset.half_coset());
        let twiddles = CpuBackend::precompute_twiddles(line_domain.coset());
        let alpha = SecureField::from_u32_unchecked(2213980, 2213981, 2213982, 2213983);

        // fold line
        fn bench_fold_line<B: Backend + FriOps + ColumnOps<BaseField, Column = Vec<BaseField>>>(
            line_domain: LineDomain,
            c: &mut Criterion,
            log_size: u32,
            alpha: SecureField,
            twiddles: &TwiddleTree<CpuBackend>, // TODO: precompute on IcicleBackend?
            backend_descr: &str,
        ) {
            let evals = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords {
                    columns: std::array::from_fn(|i| {
                        vec![BaseField::from_u32_unchecked(i as u32); 1 << log_size]
                    }),
                },
            );

            c.bench_function(
                &format!("{} fold_line log2 = {}", backend_descr, log_size),
                |b| {
                    b.iter(|| {
                        black_box(B::fold_line(black_box(&evals), black_box(alpha), unsafe {
                            std::mem::transmute(&twiddles)
                        }));
                    })
                },
            );
        }

        bench_fold_line::<CpuBackend>(line_domain, c, log_size, alpha, &twiddles, "cpu");

        #[cfg(feature = "icicle")]
        bench_fold_line::<IcicleBackend>(line_domain, c, log_size, alpha, &twiddles, "icicle");

        // fold circle
        fn bench_fold_circle<B: Backend + FriOps + ColumnOps<BaseField>>(
            log_size: u32,
            c: &mut Criterion,
            twiddles: &TwiddleTree<CpuBackend>,
            backend_descr: &str,
        ) where
            SecureColumnByCoords<B>: FromIterator<SecureField>,
        {
            let values: Vec<SecureField> = (0..(1 << log_size))
                .map(|i| SecureField::from_u32_unchecked(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
                .collect();
            let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
            let circle_domain = CanonicCoset::new(log_size).circle_domain();
            let circle_eval =
                SecureEvaluation::new(circle_domain, values.iter().copied().collect());

            let line_domain = LineDomain::new(circle_domain.half_coset);
            let mut dst = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords::zeros(1 << (log_size - 1)),
            );

            c.bench_function(
                &format!(
                    "{} fold_circle_into_line log2 = {}",
                    backend_descr, log_size
                ),
                |b| {
                    b.iter(|| {
                        black_box(B::fold_circle_into_line(
                            black_box(&mut dst),
                            black_box(&circle_eval),
                            black_box(alpha),
                            unsafe { std::mem::transmute(&twiddles) },
                        ));
                    })
                },
            );
        }

        bench_fold_circle::<CpuBackend>(log_size, c, &twiddles, "cpu");

        #[cfg(feature = "icicle")]
        bench_fold_circle::<IcicleBackend>(log_size, c, &twiddles, "icicle");
    }
}

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
