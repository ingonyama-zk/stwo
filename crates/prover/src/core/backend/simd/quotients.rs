use itertools::{izip, zip_eq, Itertools};
use num_traits::Zero;

use super::column::SecureFieldVec;
use super::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::backend::cpu::quotients::{
    batch_random_coeffs, column_line_coeffs, QuotientConstants,
};
use crate::core::backend::{Col, Column};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumn, SECURE_EXTENSION_DEGREE};
use crate::core::fields::{ComplexConjugate, FieldOps};
use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, PolyOps, SecureEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::LOG_BLOWUP_FACTOR;
use crate::core::utils::{bit_reverse, bit_reverse_index};

impl QuotientOps for SimdBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        // Split the domain into a subdomain and a shift coset.
        // TODO(spapini): Move to the caller when Columns support slices.
        let (subdomain, mut subdomain_shifts) = domain.split(LOG_BLOWUP_FACTOR);

        assert!(subdomain.log_size() >= LOG_N_LANES);
        let mut values = SecureColumn::<Self>::zeros(subdomain.size());
        let quotient_constants = quotient_constants(sample_batches, random_coeff, subdomain);

        // TODO(spapini): bit reverse iterator.
        for vec_row in 0..1 << (subdomain.log_size() - LOG_N_LANES) {
            // TODO(spapini): Optimize this, for the small number of columns case.
            let points = std::array::from_fn(|i| {
                subdomain.at(bit_reverse_index(
                    (vec_row << LOG_N_LANES) + i,
                    subdomain.log_size(),
                ))
            });
            let domain_points_x = PackedBaseField::from_array(points.map(|p| p.x));
            let domain_points_y = PackedBaseField::from_array(points.map(|p| p.y));
            let row_accumulator = accumulate_row_quotients(
                sample_batches,
                columns,
                &quotient_constants,
                vec_row,
                (domain_points_x, domain_points_y),
            );
            unsafe { values.set_packed(vec_row, row_accumulator) };
        }

        // Extend the evaluation to the full domain.
        let mut extended_eval = SecureColumn::<Self>::zeros(domain.size());

        let mut i = 0;
        let values = values.columns;
        let twiddles = SimdBackend::precompute_twiddles(subdomain.half_coset);
        let subeval_polys = values.map(|c| {
            i += 1;
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(subdomain, c)
                .interpolate_with_twiddles(&twiddles)
        });

        // Since we traverse the domain in bit-reversed order, we need bit-reverse the shifts.
        // To see why, consider the index of a point in the natural order of the domain
        // (least to most):
        //   b0 b1 b2 b3 b4 b5
        // b0 adds P, b1 adds 2P, etc.. (b5 is special and flips the sign of the point).
        // Splitting the domain to 4 parts yields:
        //   subdomain: b2 b3 b4 b5, shifts: b0 b1.
        // b2 b3 b4 b5 is indeed a circle domain, with a bigger jump.
        // Traversing the domain in bit-reversed order, after we finish with b5, b4, b3, b2,
        // we need to change b1 and then b0. This is the bit reverse of the shift b0 b1.
        bit_reverse(&mut subdomain_shifts);

        // TODO(spapini): Try to optimize out all these copies.
        for (ci, &c) in subdomain_shifts.iter().enumerate() {
            let subdomain = subdomain.shift(c);

            let twiddles = SimdBackend::precompute_twiddles(subdomain.half_coset);
            #[allow(clippy::needless_range_loop)]
            for i in 0..SECURE_EXTENSION_DEGREE {
                // Sanity check.
                let eval = subeval_polys[i].evaluate_with_twiddles(subdomain, &twiddles);
                extended_eval.columns[i].data[(ci * eval.data.len())..((ci + 1) * eval.data.len())]
                    .copy_from_slice(&eval.data);
            }
        }

        SecureEvaluation {
            domain,
            values: extended_eval,
        }
    }
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    quotient_constants: &QuotientConstants<SimdBackend>,
    vec_row: usize,
    domain_point_vec: (PackedBaseField, PackedBaseField),
) -> PackedSecureField {
    let mut row_accumulator = PackedSecureField::zero();
    for (sample_batch, line_coeffs, batch_coeff, denominator_inverses) in izip!(
        sample_batches,
        &quotient_constants.line_coeffs,
        &quotient_constants.batch_random_coeffs,
        &quotient_constants.denominator_inverses
    ) {
        let mut numerator = PackedSecureField::zero();
        for ((column_index, _), (a, b, c)) in zip_eq(&sample_batch.columns_and_values, line_coeffs)
        {
            let column = &columns[*column_index];
            let value = PackedSecureField::broadcast(*c) * column.data[vec_row];
            // The numerator is a line equation passing through
            //   (sample_point.y, sample_value), (conj(sample_point), conj(sample_value))
            // evaluated at (domain_point.y, value).
            // When substituting a polynomial in this line equation, we get a polynomial with a root
            // at sample_point and conj(sample_point) if the original polynomial had the values
            // sample_value and conj(sample_value) at these points.
            // TODO(AlonH): Use single point vanishing to save a multiplication.
            let linear_term = PackedSecureField::broadcast(*a) * domain_point_vec.1
                + PackedSecureField::broadcast(*b);
            numerator += value - linear_term;
        }

        row_accumulator = row_accumulator * PackedSecureField::broadcast(*batch_coeff)
            + numerator * denominator_inverses.data[vec_row];
    }
    row_accumulator
}

/// Pair vanishing for the packed representation of the points. See
/// [crate::core::constraints::pair_vanishing] for more details.
fn packed_pair_vanishing(
    excluded0: CirclePoint<SecureField>,
    excluded1: CirclePoint<SecureField>,
    packed_p: (PackedBaseField, PackedBaseField),
) -> PackedSecureField {
    PackedSecureField::broadcast(excluded0.y - excluded1.y) * packed_p.0
        + PackedSecureField::broadcast(excluded1.x - excluded0.x) * packed_p.1
        + PackedSecureField::broadcast(excluded0.x * excluded1.y - excluded0.y * excluded1.x)
}

fn denominator_inverses(
    sample_batches: &[ColumnSampleBatch],
    domain: CircleDomain,
) -> Vec<Col<SimdBackend, SecureField>> {
    let flat_denominators: SecureFieldVec = sample_batches
        .iter()
        .flat_map(|sample_batch| {
            (0..(1 << (domain.log_size() - LOG_N_LANES)))
                .map(|vec_row| {
                    // TODO(spapini): Optimize this, for the small number of columns case.
                    let points = std::array::from_fn(|i| {
                        domain.at(bit_reverse_index(
                            (vec_row << LOG_N_LANES) + i,
                            domain.log_size(),
                        ))
                    });
                    let domain_points_x = PackedBaseField::from_array(points.map(|p| p.x));
                    let domain_points_y = PackedBaseField::from_array(points.map(|p| p.y));
                    let domain_point_vec = (domain_points_x, domain_points_y);
                    packed_pair_vanishing(
                        sample_batch.point,
                        sample_batch.point.complex_conjugate(),
                        domain_point_vec,
                    )
                })
                .collect_vec()
        })
        .collect();

    let mut flat_denominator_inverses = SecureFieldVec::zeros(flat_denominators.len());
    <SimdBackend as FieldOps<SecureField>>::batch_inverse(
        &flat_denominators,
        &mut flat_denominator_inverses,
    );

    flat_denominator_inverses
        .data
        .chunks(domain.size() / N_LANES)
        .map(|denominator_inverses| denominator_inverses.iter().copied().collect())
        .collect()
}

fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
    domain: CircleDomain,
) -> QuotientConstants<SimdBackend> {
    let line_coeffs = column_line_coeffs(sample_batches, random_coeff);
    let batch_random_coeffs = batch_random_coeffs(sample_batches, random_coeff);
    let denominator_inverses = denominator_inverses(sample_batches, domain);
    QuotientConstants {
        line_coeffs,
        batch_random_coeffs,
        denominator_inverses,
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::simd::column::BaseFieldVec;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::BaseField;
    use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::LOG_BLOWUP_FACTOR;
    use crate::qm31;

    #[test]
    fn test_accumulate_quotients() {
        const LOG_SIZE: u32 = 8;
        let small_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let e0: BaseFieldVec = (0..small_domain.size()).map(BaseField::from).collect();
        let e1: BaseFieldVec = (0..small_domain.size())
            .map(|i| BaseField::from(2 * i))
            .collect();
        let polys = vec![
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(small_domain, e0)
                .interpolate(),
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(small_domain, e1)
                .interpolate(),
        ];
        let columns = vec![polys[0].evaluate(domain), polys[1].evaluate(domain)];
        let random_coeff = qm31!(1, 2, 3, 4);
        let a = polys[0].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let b = polys[1].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let samples = vec![ColumnSampleBatch {
            point: SECURE_FIELD_CIRCLE_GEN,
            columns_and_values: vec![(0, a), (1, b)],
        }];
        let cpu_columns = columns
            .iter()
            .map(|c| {
                CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            })
            .collect::<Vec<_>>();
        let cpu_result = CpuBackend::accumulate_quotients(
            domain,
            &cpu_columns.iter().collect_vec(),
            random_coeff,
            &samples,
        )
        .values
        .to_vec();

        let res = SimdBackend::accumulate_quotients(
            domain,
            &columns.iter().collect_vec(),
            random_coeff,
            &samples,
        )
        .values
        .to_vec();

        assert_eq!(res, cpu_result);
    }
}
