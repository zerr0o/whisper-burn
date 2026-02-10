//! Whisper audio encoder.
//!
//! Two Conv1D layers (with GELU) followed by sinusoidal positional embeddings
//! and a stack of transformer encoder blocks.

use burn::backend::Wgpu;
use burn::tensor::Tensor;

use super::attention::Q4MultiHeadAttention;
use super::layers::{gelu, Conv1D, LayerNorm, Q4FFN};

type B = Wgpu;

/// A single encoder transformer block.
pub struct EncoderBlock {
    attn_ln: LayerNorm,
    attn: Q4MultiHeadAttention,
    mlp_ln: LayerNorm,
    mlp: Q4FFN,
}

impl EncoderBlock {
    pub fn new(
        attn_ln: LayerNorm,
        attn: Q4MultiHeadAttention,
        mlp_ln: LayerNorm,
        mlp: Q4FFN,
    ) -> Self {
        Self {
            attn_ln,
            attn,
            mlp_ln,
            mlp,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm self-attention with residual
        let residual = x.clone();
        let x = self.attn_ln.forward(x);
        let x = self.attn.forward(x, false); // encoder uses full (non-causal) attention
        let x = residual + x;

        // Pre-norm FFN with residual
        let residual = x.clone();
        let x = self.mlp_ln.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

/// Whisper audio encoder.
pub struct WhisperEncoder {
    /// First conv: [128 -> 1280, kernel=3, padding=1]
    conv1: Conv1D,
    /// Second conv: [1280 -> 1280, kernel=3, stride=2, padding=1]
    conv2: Conv1D,
    /// Sinusoidal positional embedding [1500, 1280]
    positional_embedding: Tensor<B, 2>,
    /// Encoder transformer blocks
    blocks: Vec<EncoderBlock>,
    /// Final layer norm
    ln_post: LayerNorm,
}

impl WhisperEncoder {
    pub fn new(
        conv1: Conv1D,
        conv2: Conv1D,
        positional_embedding: Tensor<B, 2>,
        blocks: Vec<EncoderBlock>,
        ln_post: LayerNorm,
    ) -> Self {
        Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        }
    }

    /// Encode mel spectrogram to hidden states.
    ///
    /// Input: mel [B, 128, 3000] (128 mel bins, 3000 time frames for 30s audio)
    /// Output: [B, 1500, 1280] (1500 frames after stride-2 conv, 1280 dim)
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        // Conv1: [B, 128, 3000] -> [B, 1280, 3000]
        let x = self.conv1.forward(mel, 1, 1);
        let x = gelu(x);

        // Conv2 with stride 2: [B, 1280, 3000] -> [B, 1280, 1500]
        let x = self.conv2.forward(x, 1, 2);
        let x = gelu(x);

        // Transpose to [B, 1500, 1280] (sequence-first for attention)
        let x = x.swap_dims(1, 2);

        // Add positional embeddings
        let seq_len = x.dims()[1];
        let pos_embed = self
            .positional_embedding
            .clone()
            .slice([0..seq_len, 0..x.dims()[2]])
            .unsqueeze::<3>();
        let mut x = x + pos_embed;

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Final layer norm
        self.ln_post.forward(x)
    }
}
