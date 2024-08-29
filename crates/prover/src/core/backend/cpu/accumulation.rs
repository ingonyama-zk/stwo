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
            // icicle for now suboptimal due to conversion
            use std::mem::transmute;

            use icicle_core::ntt::FieldImpl;
            use icicle_core::vec_ops::{
                accumulate_scalars, stwo_convert, transpose_matrix, VecOpsConfig,
            };
            use icicle_cuda_runtime::device_context::DeviceContext;
            use icicle_cuda_runtime::memory::{
                DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice,
            };
            use icicle_cuda_runtime::stream::CudaStream;
            use icicle_m31::field::{ExtensionField, ScalarField};

            let zero = ScalarField::zero();

            let n = column.columns[0].len();
            let mut intermediate_host = vec![zero; 4 * n];

            // use std::ptr::from_raw_parts;
            use crate::core::SecureField;
            let cfg = VecOpsConfig::default();

            let a: &[u32] = unsafe { transmute(column.columns[0].as_slice()) };
            let b: &[u32] = unsafe { transmute(column.columns[1].as_slice()) };
            let c: &[u32] = unsafe { transmute(column.columns[2].as_slice()) };
            let d: &[u32] = unsafe { transmute(column.columns[3].as_slice()) };

            let a = HostSlice::from_slice(&a);
            let b = HostSlice::from_slice(&b);
            let c = HostSlice::from_slice(&c);
            let d = HostSlice::from_slice(&d);

            let stream1 = CudaStream::create().expect("Failed to create CUDA stream");
            let stream2 = CudaStream::create().expect("Failed to create CUDA stream");
            let stream3 = CudaStream::create().expect("Failed to create CUDA stream");
            let mut col_a: DeviceVec<ExtensionField> =
                DeviceVec::cuda_malloc_async(n, &stream1).unwrap();

            stream1.synchronize().unwrap();
            stream1.destroy().unwrap();
            stwo_convert(a, b, c, d, &mut col_a[..]);
            let mut col_b: DeviceVec<ExtensionField> =
                DeviceVec::cuda_malloc_async(n, &stream2).unwrap();
            let a: &[u32] = unsafe { transmute(other.columns[0].as_slice()) };
            let b: &[u32] = unsafe { transmute(other.columns[1].as_slice()) };
            let c: &[u32] = unsafe { transmute(other.columns[2].as_slice()) };
            let d: &[u32] = unsafe { transmute(other.columns[3].as_slice()) };

            let a = HostSlice::from_slice(&a);
            let b = HostSlice::from_slice(&b);
            let c = HostSlice::from_slice(&c);
            let d = HostSlice::from_slice(&d);

            stream2.synchronize().unwrap();
            stream2.destroy().unwrap();
            stwo_convert(a, b, c, d, &mut col_b[..]);
            accumulate_scalars(&mut col_a[..], &col_b[..], &cfg).unwrap();

            let red_u32_d: DeviceVec<ScalarField> = unsafe { transmute(col_a) };

            let on_device = true;
            let is_async = false;
            let mut result_tr: DeviceVec<ScalarField> =
                DeviceVec::cuda_malloc_async(4 * n, &stream3).unwrap();
            stream3.synchronize().unwrap();
            stream3.destroy().unwrap();
            transpose_matrix(
                &red_u32_d[..],
                4,
                n as u32,
                &mut result_tr[..],
                &DeviceContext::default(),
                on_device,
                is_async,
            )
            .unwrap();
            let res_host = HostSlice::from_mut_slice(&mut intermediate_host[..]);
            result_tr.copy_to_host(res_host).unwrap();

            use crate::core::fields::m31::M31;

            let res: Vec<M31> = unsafe { transmute(intermediate_host) };

            // Assign the sub-slices to the column
            column.columns[0].truncate(0);
            column.columns[0].extend_from_slice(&res[..n]);
            column.columns[1].truncate(0);
            column.columns[1].extend_from_slice(&res[n..2 * n]);
            column.columns[2].truncate(0);
            column.columns[2].extend_from_slice(&res[2 * n..3 * n]);
            column.columns[3].truncate(0);
            column.columns[3].extend_from_slice(&res[3 * n..]);
        }
        // panic!("Acc cpu");
    }
}
