use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::{Backend, BackendForChannel, ColumnOps};
use crate::core::air::accumulation::AccumulationOps;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::{Field, FieldOps};
use crate::core::fri::FriOps;
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::mle::MleOps;
use crate::core::pcs::quotients::QuotientOps;
use crate::core::poly::circle::PolyOps;
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::ops::MerkleOps;
use crate::core::vcs::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};

#[derive(Copy, Clone, Debug, Deserialize, Serialize, Default)]
pub struct IcicleBackend;

impl Backend for IcicleBackend {}

// stwo/crates/prover/src/core/backend/cpu/lookups/gkr.rs
impl GkrOps for IcicleBackend {
    fn gen_eq_evals(
        y: &[crate::core::fields::qm31::SecureField],
        v: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::mle::Mle<Self, crate::core::fields::qm31::SecureField> {
        todo!()
    }

    fn next_layer(
        layer: &crate::core::lookups::gkr_prover::Layer<Self>,
    ) -> crate::core::lookups::gkr_prover::Layer<Self> {
        todo!()
    }

    fn sum_as_poly_in_first_variable(
        h: &crate::core::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        claim: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::utils::UnivariatePoly<crate::core::fields::qm31::SecureField> {
        todo!()
    }
}

impl MleOps<BaseField> for IcicleBackend {
    fn fix_first_variable(
        mle: crate::core::lookups::mle::Mle<Self, BaseField>,
        assignment: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::mle::Mle<Self, crate::core::fields::qm31::SecureField>
    where
        Self: MleOps<crate::core::fields::qm31::SecureField>,
    {
        todo!()
    }
}
impl MleOps<SecureField> for IcicleBackend {
    fn fix_first_variable(
        mle: crate::core::lookups::mle::Mle<Self, SecureField>,
        assignment: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::mle::Mle<Self, crate::core::fields::qm31::SecureField>
    where
        Self: MleOps<crate::core::fields::qm31::SecureField>,
    {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/accumulation.rs
impl AccumulationOps for IcicleBackend {
    fn accumulate(
        column: &mut SecureColumnByCoords<Self>,
        other: &mut SecureColumnByCoords<Self>,
    ) {
        todo!()
    }

    #[cfg(feature = "icicle_poc")]
    fn confirm(column: &mut SecureColumnByCoords<Self>) {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/blake2s.rs
impl MerkleOps<Blake2sMerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<
            &super::Col<Self, <Blake2sMerkleHasher as crate::core::vcs::ops::MerkleHasher>::Hash>,
        >,
        columns: &[&super::Col<Self, crate::core::fields::m31::BaseField>],
    ) -> super::Col<Self, <Blake2sMerkleHasher as crate::core::vcs::ops::MerkleHasher>::Hash> {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/circle.rs
impl PolyOps for IcicleBackend {
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        coset: crate::core::poly::circle::CanonicCoset,
        values: super::Col<Self, crate::core::fields::m31::BaseField>,
    ) -> crate::core::poly::circle::CircleEvaluation<
        Self,
        crate::core::fields::m31::BaseField,
        crate::core::poly::BitReversedOrder,
    > {
        todo!()
    }

    fn interpolate(
        eval: crate::core::poly::circle::CircleEvaluation<
            Self,
            crate::core::fields::m31::BaseField,
            crate::core::poly::BitReversedOrder,
        >,
        itwiddles: &crate::core::poly::twiddles::TwiddleTree<Self>,
    ) -> crate::core::poly::circle::CirclePoly<Self> {
        todo!()
    }

    fn eval_at_point(
        poly: &crate::core::poly::circle::CirclePoly<Self>,
        point: crate::core::circle::CirclePoint<SecureField>,
    ) -> SecureField {
        todo!()
    }

    fn extend(
        poly: &crate::core::poly::circle::CirclePoly<Self>,
        log_size: u32,
    ) -> crate::core::poly::circle::CirclePoly<Self> {
        todo!()
    }

    fn evaluate(
        poly: &crate::core::poly::circle::CirclePoly<Self>,
        domain: crate::core::poly::circle::CircleDomain,
        twiddles: &crate::core::poly::twiddles::TwiddleTree<Self>,
    ) -> crate::core::poly::circle::CircleEvaluation<
        Self,
        crate::core::fields::m31::BaseField,
        crate::core::poly::BitReversedOrder,
    > {
        todo!()
    }

    fn precompute_twiddles(
        coset: crate::core::circle::Coset,
    ) -> crate::core::poly::twiddles::TwiddleTree<Self> {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/fri.rs
impl FriOps for IcicleBackend {
    fn fold_line(
        eval: &crate::core::poly::line::LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &crate::core::poly::twiddles::TwiddleTree<Self>,
    ) -> crate::core::poly::line::LineEvaluation<Self> {
        todo!()
    }

    fn fold_circle_into_line(
        dst: &mut crate::core::poly::line::LineEvaluation<Self>,
        src: &crate::core::poly::circle::SecureEvaluation<Self>,
        alpha: SecureField,
        twiddles: &crate::core::poly::twiddles::TwiddleTree<Self>,
    ) {
        todo!()
    }

    fn decompose(
        eval: &crate::core::poly::circle::SecureEvaluation<Self>,
    ) -> (
        crate::core::poly::circle::SecureEvaluation<Self>,
        SecureField,
    ) {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/grind.rs
impl<C: Channel> GrindOps<C> for IcicleBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/mod.rs
// impl Backend for IcicleBackend {}

impl BackendForChannel<Blake2sMerkleChannel> for IcicleBackend {}
impl BackendForChannel<Poseidon252MerkleChannel> for IcicleBackend {}
impl<T: Debug + Clone + Default> ColumnOps<T> for IcicleBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
    }
}
impl<F: Field> FieldOps<F> for IcicleBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/quotients.rs
impl QuotientOps for IcicleBackend {
    fn accumulate_quotients(
        domain: crate::core::poly::circle::CircleDomain,
        columns: &[&crate::core::poly::circle::CircleEvaluation<
            Self,
            crate::core::fields::m31::BaseField,
            crate::core::poly::BitReversedOrder,
        >],
        random_coeff: SecureField,
        sample_batches: &[crate::core::pcs::quotients::ColumnSampleBatch],
    ) -> crate::core::poly::circle::SecureEvaluation<Self> {
        todo!()
    }
}

// stwo/crates/prover/src/core/vcs/poseidon252_merkle.rs
impl MerkleOps<Poseidon252MerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<
            &super::Col<
                Self,
                <Poseidon252MerkleHasher as crate::core::vcs::ops::MerkleHasher>::Hash,
            >,
        >,
        columns: &[&super::Col<Self, crate::core::fields::m31::BaseField>],
    ) -> super::Col<Self, <Poseidon252MerkleHasher as crate::core::vcs::ops::MerkleHasher>::Hash>
    {
        todo!()
    }
}
