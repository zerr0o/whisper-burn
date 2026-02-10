//! GGUF quantized inference on GPU.
//!
//! Provides GGUF file reading, Q4_0 GPU tensor storage, and a fused
//! dequant+matmul compute shader launched through Burn's custom kernel API.

pub mod linear;
pub mod loader;
pub mod op;
pub mod reader;
pub mod tensor;

#[cfg(test)]
mod tests;

pub use linear::Q4Linear;
pub use op::q4_matmul;
pub use reader::{GgmlDtype, GgufReader, GgufTensorInfo};
pub use tensor::Q4Tensor;
