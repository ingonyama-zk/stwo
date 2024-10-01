// IcicleBackend amalgamation
// TODO: move to separate files
use core::fmt::Debug;

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
        let cfg = VecOpsConfig::default();
        other.convert_to_icicle(); // TODO: required on all calls or? make automated/semi- like when needed, or assume data is
                                   // already converted
        column.convert_to_icicle();
        accumulate_scalars(
            column.as_icicle_ext_slice_mut(),
            other.as_icicle_ext_slice_mut(),
            &cfg,
        )
        .unwrap();
    }

    // fn confirm(column: &mut SecureColumnByCoords<Self>) {
    //     column.convert_from_icicle(); // TODO: won't be necessary here on each call, only send back
    //                                   // to stwo core
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
        CpuBackend::commit_on_layer(log_size, prev_layer, columns)
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
        CpuBackend::new_canonical_ordered(coset, values)
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        // todo!()
        CpuBackend::interpolate(eval, itwiddles)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        // todo!()
        CpuBackend::eval_at_point(poly, point)
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        // todo!()
        CpuBackend::extend(poly, log_size)
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // todo!()
        CpuBackend::evaluate(poly, domain, twiddles)
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // todo!()
        CpuBackend::precompute_twiddles(coset)
    }
}

// stwo/crates/prover/src/core/backend/cpu/fri.rs
impl FriOps for IcicleBackend {
    
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

}

// stwo/crates/prover/src/core/vcs/poseidon252_merkle.rs
impl MerkleOps<Poseidon252MerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash> {
        // todo!()
        CpuBackend::commit_on_layer(log_size, prev_layer, columns)
    }
}

//////
use std::mem::{transmute, ManuallyDrop};
use std::ptr::{self, slice_from_raw_parts, slice_from_raw_parts_mut};

use icicle_core::ntt::FieldImpl;
use icicle_core::vec_ops::{stwo_convert, transpose_matrix};
use icicle_cuda_runtime::device::get_device_from_pointer;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;
use icicle_m31::field::{QuarticExtensionField, ScalarField};

impl SecureColumnByCoords<IcicleBackend> {
    // TODO: implement geneics
    pub fn as_icicle_slice_mut(&self) -> &mut DeviceSlice<ScalarField> {
        self.as_icicle_device_slice_mut::<ScalarField>()
    }

    pub fn as_icicle_ext_slice_mut(&self) -> &mut DeviceSlice<QuarticExtensionField> {
        self.as_icicle_device_slice_mut::<QuarticExtensionField>()
    }

    pub fn as_icicle_device_slice_mut<T>(&self) -> &mut DeviceSlice<T> {
        unsafe {
            DeviceSlice::from_mut_slice(&mut *slice_from_raw_parts_mut(
                self.device_data as *mut T,
                self.len(),
            ))
        }
    }

    pub fn convert_to_icicle(&mut self) {
        if !self.is_transposed {
            let zero = ScalarField::zero();

            let n = self.columns[0].len();
            let mut intermediate_host = vec![zero; 4 * n];

            // use std::ptr::from_raw_parts;
            use crate::core::fields::qm31::SecureField;
            let cfg = VecOpsConfig::default();

            let a: &[u32] = unsafe { transmute(self.columns[0].as_slice()) };
            let b: &[u32] = unsafe { transmute(self.columns[1].as_slice()) };
            let c: &[u32] = unsafe { transmute(self.columns[2].as_slice()) };
            let d: &[u32] = unsafe { transmute(self.columns[3].as_slice()) };

            let a = HostSlice::from_slice(&a);
            let b = HostSlice::from_slice(&b);
            let c = HostSlice::from_slice(&c);
            let d = HostSlice::from_slice(&d);

            let mut col_a = ManuallyDrop::new(DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap());

            stwo_convert(a, b, c, d, &mut col_a[..]);

            self.device_data = unsafe { col_a.as_mut_ptr() } as _;
            self.is_transposed = true;
        }
    }

    pub fn convert_from_icicle(&mut self) {
        if self.is_transposed {
            assert!(!self.device_data.is_null());
            let zero = ScalarField::zero();

            let n = self.columns[0].len();
            let secure_degree = self.columns.len();
            let mut intermediate_host = vec![zero; secure_degree * n];

            use crate::core::fields::qm31::SecureField;
            let cfg = VecOpsConfig::default();

            let mut red_u32_d = self.as_icicle_slice_mut();

            let mut result_tr: DeviceVec<ScalarField> =
                DeviceVec::cuda_malloc(secure_degree * n).unwrap();

            transpose_matrix(
                red_u32_d,
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

                self.columns[i].truncate(0);
                self.columns[i].extend_from_slice(&res[start..end]);
            }
            self.is_transposed = false;

            unsafe { self.as_icicle_slice_mut().cuda_free().unwrap() }
        }
    }
}

impl<T> HostOrDeviceSlice<T> for SecureColumnByCoords<IcicleBackend> {
    fn is_on_device(&self) -> bool {
        self.is_transposed && !self.device_data.is_null()
    }

    fn device_id(&self) -> Option<usize> {
        Some(
            get_device_from_pointer(unsafe { self.device_data as *const ::std::os::raw::c_void })
                .expect("Invalid pointer. Maybe host pointer was used here?"),
        )
    }

    unsafe fn as_ptr(&self) -> *const T {
        self.columns.as_ptr() as _
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.columns.as_mut_ptr() as _
    }

    fn len(&self) -> usize {
        self.columns[0].len() * self.columns.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
