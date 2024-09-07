use std::os::raw::c_void;
use std::ptr::slice_from_raw_parts_mut;

#[cfg(feature = "icicle_poc")]
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec};
#[cfg(feature = "icicle_poc")]
use icicle_m31::field::ScalarField;

use super::m31::{BaseField, M31};
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
    #[cfg(feature = "icicle_poc")]
    pub is_transposed: bool,
    #[cfg(feature = "icicle_poc")]
    pub device_data: *mut c_void,
}

impl<B: FieldOps<BaseField>> Default for SecureColumnByCoords<B>
where
    <B as ColumnOps<M31>>::Column: std::marker::Copy, // TODO: ??
{
    fn default() -> Self {
        Self {
            columns: [Col::<B, BaseField>::default(); SECURE_EXTENSION_DEGREE],
            #[cfg(feature = "icicle_poc")]
            is_transposed: false,
            #[cfg(feature = "icicle_poc")]
            device_data: std::ptr::null_mut(),
        }
    }
}

impl SecureColumnByCoords<CpuBackend> {
    // TODO(spapini): Remove when we no longer use CircleEvaluation<SecureField>.
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
            #[cfg(feature = "icicle_poc")]
            is_transposed: false,
            #[cfg(feature = "icicle_poc")]
            device_data: std::ptr::null_mut(),
        }
    }

    /// # Safety
    pub unsafe fn uninitialized(len: usize) -> Self {
        Self {
            columns: std::array::from_fn(|_| Col::<B, BaseField>::uninitialized(len)),
            #[cfg(feature = "icicle_poc")]
            is_transposed: false,
            #[cfg(feature = "icicle_poc")]
            device_data: std::ptr::null_mut(),
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
            #[cfg(feature = "icicle_poc")]
            is_transposed: false,
            #[cfg(feature = "icicle_poc")]
            device_data: std::ptr::null_mut(),
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
        let mut columns = std::array::from_fn(|_| vec![]);
        for value in iter.into_iter() {
            let vals = value.to_m31_array();
            for j in 0..SECURE_EXTENSION_DEGREE {
                columns[j].push(vals[j]);
            }
        }
        SecureColumnByCoords {
            columns,
            #[cfg(feature = "icicle_poc")]
            is_transposed: false,
            #[cfg(feature = "icicle_poc")]
            device_data: std::ptr::null_mut(),
        }
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
    use crate::core::fields::FieldOps;

    impl SecureColumnByCoords<CpuBackend> {
        // TODO: implement geneics
        pub fn as_icicle_slice_mut(&self) -> &mut DeviceSlice<ScalarField> {
            self.as_icicle_device_slice_mut::<ScalarField>()
        }

        pub fn as_icicle_ext_slice_mut(&self) -> &mut DeviceSlice<ExtensionField> {
            self.as_icicle_device_slice_mut::<ExtensionField>()
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
                use crate::core::SecureField;
                let cfg = VecOpsConfig::default();

                let a: &[u32] = unsafe { transmute(self.columns[0].as_slice()) };
                let b: &[u32] = unsafe { transmute(self.columns[1].as_slice()) };
                let c: &[u32] = unsafe { transmute(self.columns[2].as_slice()) };
                let d: &[u32] = unsafe { transmute(self.columns[3].as_slice()) };

                let a = HostSlice::from_slice(&a);
                let b = HostSlice::from_slice(&b);
                let c = HostSlice::from_slice(&c);
                let d = HostSlice::from_slice(&d);

                let mut col_a =
                    ManuallyDrop::new(DeviceVec::<ExtensionField>::cuda_malloc(n).unwrap());

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
                let secure_degree = self .columns.len();
                let mut intermediate_host = vec![zero; secure_degree * n];

                use crate::core::SecureField;
                let cfg = VecOpsConfig::default();

                let mut red_u32_d = self.as_icicle_slice_mut();

                let mut result_tr: DeviceVec<ScalarField> = DeviceVec::cuda_malloc(secure_degree * n).unwrap();

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

    impl<T> HostOrDeviceSlice<T> for SecureColumnByCoords<CpuBackend> {
        fn is_on_device(&self) -> bool {
            self.is_transposed && !self.device_data.is_null()
        }

        fn device_id(&self) -> Option<usize> {
            Some(
                get_device_from_pointer(unsafe {
                    self.device_data as *const ::std::os::raw::c_void
                })
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
}
