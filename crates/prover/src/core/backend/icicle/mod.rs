// IcicleBackend amalgamation
// TODO: move to separate files
use core::fmt::Debug;
use std::ffi::c_void;
use std::mem::{size_of_val, transmute};

use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
use serde::{Deserialize, Serialize};
use twiddles::TwiddleTree;

use super::{
    Backend, BackendForChannel, BaseField, Col, ColumnOps, CpuBackend, PolyOps, QuotientOps,
};
use crate::core::air::accumulation::AccumulationOps;
use crate::core::channel::Channel;
use crate::core::circle::{self, CirclePoint, Coset};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::{Field, FieldOps};
use crate::core::fri::FriOps;
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::mle::MleOps;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, SecureEvaluation,
};
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::{twiddles, BitReversedOrder};
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
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
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
        use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice};

        let cfg = VecOpsConfig::default();

        unsafe {
            let limbs_count: usize = size_of_val(&column.columns[0]) / 4;
            use std::slice;

            use icicle_core::traits::FieldImpl;
            use icicle_core::vec_ops::VecOps;
            use icicle_cuda_runtime::device::get_device_from_pointer;
            use icicle_cuda_runtime::memory::{DeviceSlice, HostSlice};
            use icicle_m31::field::{QuarticExtensionField, ScalarField};

            let mut a_ptr = column as *mut _ as *mut c_void;
            let mut d_a_slice;
            let n = column.columns[0].len();
            let secure_degree = column.columns.len();

            let column: &mut SecureColumnByCoords<IcicleBackend> = transmute(column);
            let other = transmute(other);

            let is_a_on_host = get_device_from_pointer(a_ptr).unwrap() == 18446744073709551614;
            let mut col_a;
            if is_a_on_host {
                col_a = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
                d_a_slice = &mut col_a[..];
                SecureColumnByCoords::convert_to_icicle(column, d_a_slice);
            } else {
                let mut v_ptr = a_ptr as *mut QuarticExtensionField;
                let rr = unsafe { slice::from_raw_parts_mut(v_ptr, n) };
                d_a_slice = DeviceSlice::from_mut_slice(rr);
            }
            let b_ptr = other as *const _ as *const c_void;
            let mut d_b_slice;
            let mut col_b;
            if get_device_from_pointer(b_ptr).unwrap() == 18446744073709551614 {
                col_b = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
                d_b_slice = &mut col_b[..];
                SecureColumnByCoords::convert_to_icicle(other, d_b_slice);
            } else {
                let mut v_ptr = b_ptr as *mut QuarticExtensionField;
                let rr = unsafe { slice::from_raw_parts_mut(v_ptr, n) };
                d_b_slice = DeviceSlice::from_mut_slice(rr);
            }

            accumulate_scalars(d_a_slice, d_b_slice, &cfg);

            let mut v_ptr = d_a_slice.as_mut_ptr() as *mut _;
            let d_slice = unsafe { slice::from_raw_parts_mut(v_ptr, secure_degree * n) };
            let d_a_slice = DeviceSlice::from_mut_slice(d_slice);
            SecureColumnByCoords::convert_from_icicle(column, d_a_slice);
        }
    }

    // fn confirm(column: &mut SecureColumnByCoords<Self>) {
    //     column.convert_from_icicle(); // TODO: won't be necessary here on each call, only send
    // back                                   // to stwo core
    // }
}

// stwo/crates/prover/src/core/backend/cpu/blake2s.rs
impl MerkleOps<Blake2sMerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash> {
        // todo!()
        <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        )
    }
}

// stwo/crates/prover/src/core/backend/cpu/circle.rs
impl PolyOps for IcicleBackend {
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // todo!()
        unsafe { transmute(CpuBackend::new_canonical_ordered(coset, values)) }
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        // todo!()
        unsafe {
            transmute(CpuBackend::interpolate(
                transmute(eval),
                transmute(itwiddles),
            ))
        }
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        // todo!()
        unsafe { CpuBackend::eval_at_point(transmute(poly), point) }
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        // todo!()
        unsafe { transmute(CpuBackend::extend(transmute(poly), log_size)) }
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // todo!()
        unsafe {
            transmute(CpuBackend::evaluate(
                transmute(poly),
                domain,
                transmute(twiddles),
            ))
        }
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // todo!()
        unsafe { transmute(CpuBackend::precompute_twiddles(coset)) }
    }
}

// stwo/crates/prover/src/core/backend/cpu/fri.rs
impl FriOps for IcicleBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        todo!()
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        todo!()
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/grind.rs
impl<C: Channel> GrindOps<C> for IcicleBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        // todo!()
        CpuBackend::grind(channel, pow_bits)
    }
}

// stwo/crates/prover/src/core/backend/cpu/mod.rs
// impl Backend for IcicleBackend {}

impl BackendForChannel<Blake2sMerkleChannel> for IcicleBackend {}
impl BackendForChannel<Poseidon252MerkleChannel> for IcicleBackend {}
impl<T: Debug + Clone + Default> ColumnOps<T> for IcicleBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        // todo!()
        CpuBackend::bit_reverse_column(column)
    }
}
impl<F: Field> FieldOps<F> for IcicleBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        // todo!()
        CpuBackend::batch_inverse(column, dst)
    }
}

// stwo/crates/prover/src/core/backend/cpu/quotients.rs
impl QuotientOps for IcicleBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        todo!()
    }
}

// stwo/crates/prover/src/core/vcs/poseidon252_merkle.rs
impl MerkleOps<Poseidon252MerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash> {
        // todo!()

        <CpuBackend as MerkleOps<Poseidon252MerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        )
    }
}

use std::ptr::{self, slice_from_raw_parts, slice_from_raw_parts_mut};

use icicle_core::ntt::FieldImpl;
use icicle_core::vec_ops::{stwo_convert, transpose_matrix};
use icicle_cuda_runtime::device::get_device_from_pointer;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;
use icicle_m31::field::{QuarticExtensionField, ScalarField};

impl SecureColumnByCoords<IcicleBackend> {
    pub fn convert_to_icicle(input: &Self, d_output: &mut DeviceSlice<QuarticExtensionField>) {
        let zero = ScalarField::zero();

        let n = input.columns[0].len();
        let secure_degree = input.columns.len();

        let cfg = VecOpsConfig::default();

        let a: &[u32] = unsafe { transmute(input.columns[0].as_slice()) };
        let b: &[u32] = unsafe { transmute(input.columns[1].as_slice()) };
        let c: &[u32] = unsafe { transmute(input.columns[2].as_slice()) };
        let d: &[u32] = unsafe { transmute(input.columns[3].as_slice()) };

        let a = HostSlice::from_slice(&a);
        let b = HostSlice::from_slice(&b);
        let c = HostSlice::from_slice(&c);
        let d = HostSlice::from_slice(&d);

        stwo_convert(a, b, c, d, d_output);
    }

    pub fn convert_from_icicle(input: &mut Self, d_input: &mut DeviceSlice<ScalarField>) {
        let zero = ScalarField::zero();

        let n = input.columns[0].len();
        let secure_degree = input.columns.len();
        let mut intermediate_host = vec![zero; secure_degree * n];

        let cfg = VecOpsConfig::default();

        let mut result_tr: DeviceVec<ScalarField> =
            DeviceVec::cuda_malloc(secure_degree * n).unwrap();

        transpose_matrix(
            d_input,
            secure_degree as u32,
            n as u32,
            &mut result_tr[..],
            &DeviceContext::default(),
            true,
            false,
        )
        .unwrap();

        let mut res_host = HostSlice::from_mut_slice(&mut intermediate_host[..]);
        result_tr.copy_to_host(res_host).unwrap();

        use crate::core::fields::m31::M31;

        let res: Vec<M31> = unsafe { transmute(intermediate_host) };

        // Assign the sub-slices to the column
        for i in 0..secure_degree {
            let start = i * n;
            let end = start + n;

            input.columns[i].truncate(0);
            input.columns[i].extend_from_slice(&res[start..end]);
        }
    }
}

// impl<T> HostOrDeviceSlice<T> for SecureColumnByCoords<IcicleBackend> {
//     fn is_on_device(&self) -> bool {
//         self.is_transposed && !self.device_data.is_null()
//     }

//     fn device_id(&self) -> Option<usize> {
//         Some(
//             get_device_from_pointer(unsafe { self.device_data as *const ::std::os::raw::c_void })
//                 .expect("Invalid pointer. Maybe host pointer was used here?"),
//         )
//     }

//     unsafe fn as_ptr(&self) -> *const T {
//         self.columns.as_ptr() as _
//     }

//     unsafe fn as_mut_ptr(&mut self) -> *mut T {
//         self.columns.as_mut_ptr() as _
//     }

//     fn len(&self) -> usize {
//         self.columns[0].len() * self.columns.len()
//     }

//     fn is_empty(&self) -> bool {
//         self.len() == 0
//     }
// }
