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

    /// Forward pass for Conv1D using im2col for efficient GPU execution.
    ///
    /// Input shape: [B, C_in, T]
    /// Output shape: [B, C_out, T_out]
    pub fn forward(&self, x: Tensor<B, 3>, padding: usize, stride: usize) -> Tensor<B, 3> {
        let [batch, _in_ch, t_in] = x.dims();
        let [out_ch, in_ch, kernel_size] = self.weight.dims();
        let t_out = (t_in + 2 * padding - kernel_size) / stride + 1;

        // Pad input if needed
        let x = if padding > 0 {
            let device = x.device();
            let zl: Tensor<B, 3> = Tensor::zeros([batch, in_ch, padding], &device);
            let zr: Tensor<B, 3> = Tensor::zeros([batch, in_ch, padding], &device);
            Tensor::cat(vec![zl, x, zr], 2)
        } else {
            x
        };

        // im2col: build [batch, kernel_size * in_ch, t_out] with one slice per kernel position
        let mut cols = Vec::with_capacity(kernel_size);
        for k in 0..kernel_size {
            if stride == 1 {
                // Simple contiguous slice: [batch, in_ch, t_out]
                let col = x.clone().slice([0..batch, 0..in_ch, k..k + t_out]);
                cols.push(col);
            } else {
                // Take every stride-th element via reshape trick
                let needed = t_out * stride;
                let shifted = x.clone().slice([0..batch, 0..in_ch, k..k + needed]);
                // [batch, in_ch, t_out * stride] -> [batch, in_ch, t_out, stride]
                // -> select [:,:,:,0] -> [batch, in_ch, t_out]
                let strided: Tensor<B, 3> = shifted
                    .reshape([batch, in_ch, t_out, stride])
                    .slice([0..batch, 0..in_ch, 0..t_out, 0..1])
                    .reshape([batch, in_ch, t_out]);
                cols.push(strided);
            }
        }

        // Cat along channel dim: [batch, kernel_size * in_ch, t_out]
        let col = Tensor::cat(cols, 1);
        // Transpose to [batch, t_out, kernel_size * in_ch]
        let col = col.swap_dims(1, 2);

        // Reorder weight [out_ch, in_ch, kernel_size] -> [out_ch, kernel_size, in_ch]
        // to match im2col memory layout (kernel-major grouping)
        let w: Tensor<B, 3> = self.weight.clone().swap_dims(1, 2);
        let w_flat: Tensor<B, 2> = w.reshape([out_ch, kernel_size * in_ch]);

        // Single large matmul: [batch, t_out, K*C_in] x [1, K*C_in, C_out] -> [batch, t_out, C_out]
        let w_3d: Tensor<B, 3> = w_flat.transpose().unsqueeze_dim(0);
        let out = col.matmul(w_3d);

        // Add bias: [C_out] broadcast to [batch, t_out, C_out]
        let out = out + self.bias.clone().unsqueeze::<3>();

        // Transpose to [batch, C_out, t_out]
        out.swap_dims(1, 2)
    }
}
