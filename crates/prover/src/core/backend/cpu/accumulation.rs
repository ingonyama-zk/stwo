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
            use icicle_cuda_runtime::memory::{
                DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice,
            };
            use icicle_m31::field::{ExtensionField, ScalarField};

            // use std::ptr::from_raw_parts;
            use crate::core::SecureField;

            let mut a: Vec<ExtensionField> = vec![];
            let mut b: Vec<ExtensionField> = vec![];
            let len = column.len();
            for i in 0..len {
                // TODO: just for the sake of correctness check - perf optimisation can be done
                // without data conversion
                let ci = column.at(i);
                let oi = other.at(i);

                let aa = ci.to_m31_array();
                let bb = oi.to_m31_array();

                let aa: ExtensionField = unsafe { transmute(aa) };
                let bb: ExtensionField = unsafe { transmute(bb) };

                a.push(aa);
                b.push(bb);
            }

            let aa = HostSlice::from_mut_slice(&mut a);
            let bb = HostSlice::from_slice(&b);

            let cfg = VecOpsConfig::default();

            let a: &[u32] = unsafe { transmute(column.columns[0].as_slice()) };
            let b: &[u32] = unsafe { transmute(column.columns[1].as_slice()) };
            let c: &[u32] = unsafe { transmute(column.columns[2].as_slice()) };
            let d: &[u32] = unsafe { transmute(column.columns[3].as_slice()) };

            let n = column.columns[0].len();

            let mut result: DeviceVec<ExtensionField> = DeviceVec::cuda_malloc(n).unwrap();

            let a = HostSlice::from_slice(&a);
            let b = HostSlice::from_slice(&b);
            let c = HostSlice::from_slice(&c);
            let d = HostSlice::from_slice(&d);

            stwo_convert(a, b, c, d, &mut result[..]);

            accumulate_scalars(&mut result[..], bb, &cfg).unwrap();

            let red_u32_d: DeviceVec<ScalarField> = unsafe { transmute(result) };

            let mut result_tr: DeviceVec<ScalarField> = DeviceVec::cuda_malloc(4 * n).unwrap();

            let mut intermediate_host = vec![ScalarField::one(); 4 * n];

            let on_device = true;
            let is_async = false;
            // for now, columns batching only works with MixedRadix NTT
            use icicle_cuda_runtime::device_context::DeviceContext;
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
            column.columns[0] = res[..n].to_vec();
            column.columns[1] = res[n..2 * n].to_vec();
            column.columns[2] = res[2 * n..3 * n].to_vec();
            column.columns[3] = res[3 * n..].to_vec();
        }
        // panic!("Acc cpu");
    }
}
