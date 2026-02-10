//! Basic layers for Whisper: LayerNorm, GELU, FFN, Conv1D.

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use std::f32::consts::PI;

use crate::gguf::Q4Linear;

type B = Wgpu;

/// Layer normalization with weight and bias (eps=1e-5).
pub struct LayerNorm {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}

impl LayerNorm {
    pub fn new(weight: Tensor<B, 1>, bias: Tensor<B, 1>) -> Self {
        Self { weight, bias }
    }

    /// Apply layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let eps = 1e-5;
        // x shape: [B, T, C]
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normed = centered / (var + eps).sqrt();
        normed * self.weight.clone().unsqueeze::<3>() + self.bias.clone().unsqueeze::<3>()
    }
}

/// GELU activation function (approximate).
pub fn gelu(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = (2.0f32 / PI).sqrt();
    let x3 = x.clone().powf_scalar(3.0);
    let inner = (x.clone() + x3 * 0.044715) * sqrt_2_over_pi;
    x * 0.5 * (inner.tanh() + 1.0)
}

/// Feed-forward network: fc1 -> GELU -> fc2 (with biases).
pub struct Q4FFN {
    fc1: Q4Linear,
    fc2: Q4Linear,
}

impl Q4FFN {
    pub fn new(fc1: Q4Linear, fc2: Q4Linear) -> Self {
        Self { fc1, fc2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.fc1.forward(x);
        let h = gelu(h);
        self.fc2.forward(h)
    }
}

/// 1D convolution layer (F32 weights, not quantized).
pub struct Conv1D {
    /// Weight shape: [out_channels, in_channels, kernel_size]
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
}

impl Conv1D {
    pub fn new(weight: Tensor<B, 3>, bias: Tensor<B, 1>) -> Self {
        Self { weight, bias }
    }

    /// Forward pass for Conv1D.
    ///
    /// Input shape: [B, C_in, T]
    /// Output shape: [B, C_out, T_out]
    ///
    /// Uses stride=1, padding=1 by default (matching Whisper conv1).
    /// For conv2 (stride=2), use `forward_strided`.
    pub fn forward(&self, x: Tensor<B, 3>, padding: usize, stride: usize) -> Tensor<B, 3> {
        // Manual conv1d implementation using unfold-like approach
        let [batch, _in_ch, t_in] = x.dims();
        let [out_ch, in_ch, kernel_size] = self.weight.dims();

        let t_out = (t_in + 2 * padding - kernel_size) / stride + 1;

        // Pad input if needed
        let x = if padding > 0 {
            // Zero-pad on both sides along time dimension
            let device = x.device();
            let zeros_left: Tensor<B, 3> =
                Tensor::zeros([batch, in_ch, padding], &device);
            let zeros_right: Tensor<B, 3> =
                Tensor::zeros([batch, in_ch, padding], &device);
            Tensor::cat(vec![zeros_left, x, zeros_right], 2)
        } else {
            x
        };

        // Reshape weight to [out_ch, in_ch * kernel_size]
        let w_flat: Tensor<B, 2> = self.weight.clone().reshape([out_ch, in_ch * kernel_size]);

        // Extract patches and compute convolution via matmul
        let mut columns = Vec::with_capacity(t_out);
        for t in 0..t_out {
            let start = t * stride;
            // x[:, :, start..start+kernel_size] -> [B, in_ch, kernel_size]
            let patch = x.clone().slice([0..batch, 0..in_ch, start..start + kernel_size]);
            // Reshape to [B, in_ch * kernel_size]
            let patch_flat: Tensor<B, 2> = patch.reshape([batch, in_ch * kernel_size]);
            // Matmul: [B, in_ch*k] x [in_ch*k, out_ch] = [B, out_ch]
            let out = patch_flat.matmul(w_flat.clone().transpose());
            // Add bias
            let out = out + self.bias.clone().unsqueeze::<2>();
            // [B, out_ch] -> [B, out_ch, 1]
            columns.push(out.unsqueeze_dim(2));
        }

        // Concatenate along time dimension: [B, out_ch, t_out]
        Tensor::cat(columns, 2)
    }
}
