use std::mem::size_of_val;
use std::os::raw::c_void;

use super::CpuBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumnByCoords;

impl AccumulationOps for CpuBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        #[cfg(not(feature = "icicle_poc"))]
        for i in 0..column.len() {
            let res_coeff = column.at(i) + other.at(i);
            column.set(i, res_coeff);
        }

        #[cfg(feature = "icicle_poc")]
        {
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
                SecureColumnByCoords::<CpuBackend>::convert_from_icicle(column, d_a_slice);
            }
        }
    }
}
