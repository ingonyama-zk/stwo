use super::CpuBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumnByCoords;

impl AccumulationOps for CpuBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &mut SecureColumnByCoords<Self>) {
        #[cfg(not(feature = "icicle_poc"))]
        for i in 0..column.len() {
            let res_coeff = column.at(i) + other.at(i);
            column.set(i, res_coeff);
        }

        #[cfg(feature = "icicle_poc")]
        {
            use icicle_core::vec_ops::{
                accumulate_scalars, VecOpsConfig,
            };

            let cfg = VecOpsConfig::default();
            other.convert_to_icicle();
            column.convert_to_icicle();
            accumulate_scalars(column.as_icicle_ext_slice_mut(), other.as_icicle_ext_slice_mut(), &cfg).unwrap();

            // column.convert_from_icicle(); // TODO: on icicle backend conversion will happen only once on transfer to device and back when needed
        }
    }
    
    fn confirm(column: &mut SecureColumnByCoords<Self>) {
        column.convert_from_icicle();
    }
}
