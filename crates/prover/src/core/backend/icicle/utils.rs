use crate::core::fields::{ExtensionOf, Field};

/// Folds values recursively in `O(n)` by a hierarchical application of folding factors.
///
/// i.e. folding `n = 8` values with `folding_factors = [x, y, z]`:
///
/// ```text
///               n2=n1+x*n2
///           /               \
///     n1=n3+y*n4          n2=n5+y*n6
///      /      \            /      \
/// n3=a+z*b  n4=c+z*d  n5=e+z*f  n6=g+z*h
///   /  \      /  \      /  \      /  \
///  a    b    c    d    e    f    g    h
/// ```
///
/// # Panics
///
/// Panics if the number of values is not a power of two or if an incorrect number of of folding
/// factors is provided.
// TODO(Andrew): Can be made to run >10x faster by unrolling lower layers of recursion
pub fn fold<F: Field, E: ExtensionOf<F>>(values: &[F], folding_factors: &[E]) -> E {
    let n = values.len();
    assert_eq!(n, 1 << folding_factors.len());
    if n == 1 {
        let res: E = values[0].into();
        return res;
    }
    let (lhs_values, rhs_values) = values.split_at(n / 2);
    let (folding_factor, folding_factors) = folding_factors.split_first().unwrap();
    let lhs_val = fold(lhs_values, folding_factors);
    let rhs_val = fold(rhs_values, folding_factors);
    // println!(
    //     "n={:?} lhs_val{:?} + rhs_val{:?} x folding_factor: {:?}",
    //     n, lhs_val, rhs_val, *folding_factor
    // );
    let res = lhs_val + rhs_val * *folding_factor;
    // println!("res = {:?}; ", res);
    res
}

pub fn fold_gpu<F: Field, E: ExtensionOf<F>>(values: &[F], folding_factors: &[E]) -> E {
    let n = values.len();
    assert_eq!(n, 1 << folding_factors.len());
    if n == 1 {
        return values[0].into();
    }
    let (lhs_values, rhs_values) = values.split_at(n / 2);
    let (folding_factor, folding_factors) = folding_factors.split_first().unwrap();
    let lhs_val = fold(lhs_values, folding_factors);
    let rhs_val = fold(rhs_values, folding_factors);
    lhs_val + rhs_val * *folding_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    #[test]
    fn test_fold_works() {
        // Example input: power-of-two values and appropriate folding factors
        let values = vec![
            M31(1),
            M31(2),
            M31(3),
            M31(4),
            M31(5),
            M31(6),
            M31(7),
            M31(8),
        ];
        let folding_factors = vec![
            QM31::from_u32_unchecked(2, 0, 0, 0),
            QM31::from_u32_unchecked(3, 0, 0, 0),
            QM31::from_u32_unchecked(4, 0, 0, 0),
        ];
        let result = fold(&values, &folding_factors);

        let expected = QM31::from_u32_unchecked(358, 0, 0, 0);
        assert_eq!(result, expected, "Result for simple folding is incorrect");

        // Set the desired length for folding_factors
        let folding_factors_length = 20; // Example length
        let values_length = 1 << folding_factors_length; // 2^folding_factors_length

        // Initialize the `values` vector
        let mut values: Vec<M31> = Vec::with_capacity(values_length);
        use rayon::iter::IntoParallelIterator;
        use rayon::prelude::*;

        let values: Vec<M31> = (1..=values_length)
            .into_par_iter()
            .map(|i| M31(i as u32))
            .collect();

        // Initialize the `folding_factors` vector
        let mut folding_factors = Vec::with_capacity(folding_factors_length);
        for i in 2..(2 + folding_factors_length) {
            folding_factors.push(QM31::from_u32_unchecked(i as u32, 0, 0, 0));
        }
        let time = std::time::Instant::now();
        let result = fold(&values, &folding_factors);
        let elapsed = time.elapsed();
        println!(
            "Elapsed time for 2^{}: {:?}",
            folding_factors_length, elapsed
        );

        let expected = QM31::from_u32_unchecked(223550878, 0, 0, 0);
        assert_eq!(result, expected, "Result for large folding is incorrect");
    }
}
