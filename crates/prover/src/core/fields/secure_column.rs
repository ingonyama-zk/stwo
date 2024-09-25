use std::array;
use std::iter::zip;

use super::m31::BaseField;
use super::qm31::SecureField;
use super::{ExtensionOf, FieldOps};
use crate::core::backend::{Col, Column, ColumnOps, CpuBackend};

pub const SECURE_EXTENSION_DEGREE: usize =
    <SecureField as ExtensionOf<BaseField>>::EXTENSION_DEGREE;

/// A column major array of `SECURE_EXTENSION_DEGREE` base field columns, that represents a column
/// of secure field element coordinates.
#[derive(Clone, Debug)]
// #[repr(C, align(8))]
pub struct SecureColumnByCoords<B: FieldOps<BaseField>> {
    pub columns: [Col<B, BaseField>; SECURE_EXTENSION_DEGREE],
}

impl SecureColumnByCoords<CpuBackend> {
    // TODO(first): Remove.
    pub fn to_vec(&self) -> Vec<SecureField> {
        (0..self.len()).map(|i| self.at(i)).collect()
    }
}
impl<B: FieldOps<BaseField>> SecureColumnByCoords<B> {
    pub fn at(&self, index: usize) -> SecureField {
        SecureField::from_m31_array(std::array::from_fn(|i| self.columns[i].at(index)))
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            columns: std::array::from_fn(|_| Col::<B, BaseField>::zeros(len)),
        }
    }

    /// # Safety
    pub unsafe fn uninitialized(len: usize) -> Self {
        Self {
            columns: std::array::from_fn(|_| Col::<B, BaseField>::uninitialized(len)),
        }
    }

    pub fn len(&self) -> usize {
        self.columns[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns[0].is_empty()
    }

    pub fn to_cpu(&self) -> SecureColumnByCoords<CpuBackend> {
        SecureColumnByCoords {
            columns: self.columns.clone().map(|c| c.to_cpu()),
        }
    }

    pub fn set(&mut self, index: usize, value: SecureField) {
        let values = value.to_m31_array();
        #[allow(clippy::needless_range_loop)]
        for i in 0..SECURE_EXTENSION_DEGREE {
            self.columns[i].set(index, values[i]);
        }
    }
}

pub struct SecureColumnByCoordsIter<'a> {
    column: &'a SecureColumnByCoords<CpuBackend>,
    index: usize,
}
impl Iterator for SecureColumnByCoordsIter<'_> {
    type Item = SecureField;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.column.len() {
            let value = self.column.at(self.index);
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
}
impl<'a> IntoIterator for &'a SecureColumnByCoords<CpuBackend> {
    type Item = SecureField;
    type IntoIter = SecureColumnByCoordsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SecureColumnByCoordsIter {
            column: self,
            index: 0,
        }
    }
}
impl FromIterator<SecureField> for SecureColumnByCoords<CpuBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let values = iter.into_iter();
        let (lower_bound, _) = values.size_hint();
        let mut columns = array::from_fn(|_| Vec::with_capacity(lower_bound));

        for value in values {
            let coords = value.to_m31_array();
            zip(&mut columns, coords).for_each(|(col, coord)| col.push(coord));
        }

        SecureColumnByCoords { columns }
    }
}
impl From<SecureColumnByCoords<CpuBackend>> for Vec<SecureField> {
    fn from(column: SecureColumnByCoords<CpuBackend>) -> Self {
        column.into_iter().collect()
    }
}

#[cfg(feature = "icicle_poc")]
mod icicle_poc {
    use std::mem::{transmute, ManuallyDrop};
    use std::ptr::{self, slice_from_raw_parts, slice_from_raw_parts_mut};

    use icicle_core::ntt::FieldImpl;
    use icicle_core::vec_ops::{accumulate_scalars, stwo_convert, transpose_matrix, VecOpsConfig};
    use icicle_cuda_runtime::device::get_device_from_pointer;
    use icicle_cuda_runtime::device_context::DeviceContext;
    use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
    use icicle_cuda_runtime::stream::CudaStream;
    use icicle_m31::field::{ExtensionField, ScalarField};

    use super::SecureColumnByCoords;
    use crate::core::backend::simd::column;
    use crate::core::backend::CpuBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldOps;

    impl SecureColumnByCoords<CpuBackend> {
        pub fn convert_to_icicle(input: &Self, d_output: &mut DeviceSlice<ExtensionField>) {
            let zero = ScalarField::zero();

            let n = input.columns[0].len();
            let secure_degree = input.columns.len();
            let mut intermediate_host = vec![zero; secure_degree * n];

            use crate::core::fields::secure_column::SecureField;
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

            use crate::core::fields::secure_column::SecureField;
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
}
