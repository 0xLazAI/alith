pub use ort::{
    Result,
    execution_providers::{
        CANNExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider,
        CoreMLExecutionProvider, TensorRTExecutionProvider,
    },
    info, init, inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

pub mod present;

#[inline]
pub fn ort_init() -> ort::Result<()> {
    ort::init()
        .with_execution_providers([
            #[cfg(feature = "trt")]
            TensorRTExecutionProvider::default().build(),
            #[cfg(feature = "cuda")]
            CUDAExecutionProvider::default().build(),
        ])
        .commit()?;

    Ok(())
}
