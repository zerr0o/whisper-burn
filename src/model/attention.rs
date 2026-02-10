//! Multi-head attention and cross-attention for Whisper.

use burn::backend::Wgpu;
use burn::tensor::Tensor;

use crate::gguf::Q4Linear;

type B = Wgpu;

/// Multi-head self-attention (used in both encoder and decoder).
pub struct Q4MultiHeadAttention {
    query: Q4Linear,
    key: Q4Linear,
    value: Q4Linear,
    out: Q4Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Q4MultiHeadAttention {
    pub fn new(
        query: Q4Linear,
        key: Q4Linear,
        value: Q4Linear,
        out: Q4Linear,
        n_heads: usize,
    ) -> Self {
        // Infer head_dim from the output size of the query projection
        // For Whisper: 1280 / 20 = 64
        let head_dim = 64; // 1280 / 20
        Self {
            query,
            key,
            value,
            out,
            n_heads,
            head_dim,
        }
    }

    /// Forward pass for self-attention.
    ///
    /// Input shape: [B, T, D]
    /// Output shape: [B, T, D]
    ///
    /// If `causal` is true, applies a causal mask (for decoder self-attention).
    pub fn forward(&self, x: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();

        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let out = scaled_dot_product_attention(
            q, k, v, self.n_heads, self.head_dim, batch, seq_len, seq_len, causal,
        );

        self.out.forward(out)
    }

    /// Forward pass for decoder self-attention with KV cache.
    ///
    /// `x` shape: [B, 1, D] (single new token)
    /// `k_cache`, `v_cache`: accumulated keys/values from previous steps
    ///
    /// Returns (output, new_k, new_v) where new_k/new_v include the current step.
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        k_cache: Option<Tensor<B, 3>>,
        v_cache: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _seq_len, _d_model] = x.dims();

        let q = self.query.forward(x.clone());
        let k_new = self.key.forward(x.clone());
        let v_new = self.value.forward(x);

        // Concatenate with cache
        let (k, v) = match (k_cache, v_cache) {
            (Some(kc), Some(vc)) => {
                let k = Tensor::cat(vec![kc, k_new.clone()], 1);
                let v = Tensor::cat(vec![vc, v_new.clone()], 1);
                (k, v)
            }
            _ => (k_new.clone(), v_new.clone()),
        };

        let kv_len = k.dims()[1];
        let q_len = q.dims()[1];

        // No causal mask needed when using KV cache (single token query)
        let out = scaled_dot_product_attention(
            q, k.clone(), v.clone(), self.n_heads, self.head_dim, batch, q_len, kv_len, false,
        );

        let out = self.out.forward(out);
        (out, k, v)
    }
}

/// Cross-attention (decoder queries attend to encoder outputs).
pub struct Q4CrossAttention {
    query: Q4Linear,
    key: Q4Linear,
    value: Q4Linear,
    out: Q4Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Q4CrossAttention {
    pub fn new(
        query: Q4Linear,
        key: Q4Linear,
        value: Q4Linear,
        out: Q4Linear,
        n_heads: usize,
    ) -> Self {
        let head_dim = 64;
        Self {
            query,
            key,
            value,
            out,
            n_heads,
            head_dim,
        }
    }

    /// Forward pass for cross-attention.
    ///
    /// `x` shape: [B, T_q, D] (decoder hidden states)
    /// `encoder_out` shape: [B, T_kv, D] (encoder output)
    pub fn forward(&self, x: Tensor<B, 3>, encoder_out: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, q_len, _] = x.dims();
        let kv_len = encoder_out.dims()[1];

        let q = self.query.forward(x);
        let k = self.key.forward(encoder_out.clone());
        let v = self.value.forward(encoder_out);

        let out = scaled_dot_product_attention(
            q, k, v, self.n_heads, self.head_dim, batch, q_len, kv_len, false,
        );

        self.out.forward(out)
    }

    /// Forward pass with cached encoder keys/values.
    ///
    /// On first call, computes K/V from encoder output and caches them.
    /// On subsequent calls, reuses cached K/V.
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        encoder_out: &Tensor<B, 3>,
        k_cache: Option<Tensor<B, 3>>,
        v_cache: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, q_len, _] = x.dims();

        let q = self.query.forward(x);

        let (k, v) = match (k_cache, v_cache) {
            (Some(kc), Some(vc)) => (kc, vc),
            _ => {
                let k = self.key.forward(encoder_out.clone());
                let v = self.value.forward(encoder_out.clone());
                (k, v)
            }
        };

        let kv_len = k.dims()[1];

        let out = scaled_dot_product_attention(
            q, k.clone(), v.clone(), self.n_heads, self.head_dim, batch, q_len, kv_len, false,
        );

        let out = self.out.forward(out);
        (out, k, v)
    }
}

/// Scaled dot-product attention.
///
/// Q, K, V shapes: [B, T, D] where D = n_heads * head_dim
/// Returns: [B, T_q, D]
fn scaled_dot_product_attention(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    n_heads: usize,
    head_dim: usize,
    batch: usize,
    q_len: usize,
    kv_len: usize,
    causal: bool,
) -> Tensor<B, 3> {
    // Reshape [B, T, D] -> [B, T, H, Dh] -> [B, H, T, Dh]
    let q: Tensor<B, 4> = q
        .reshape([batch, q_len, n_heads, head_dim])
        .swap_dims(1, 2);
    let k: Tensor<B, 4> = k
        .reshape([batch, kv_len, n_heads, head_dim])
        .swap_dims(1, 2);
    let v: Tensor<B, 4> = v
        .reshape([batch, kv_len, n_heads, head_dim])
        .swap_dims(1, 2);

    // Attention scores: [B, H, T_q, T_kv]
    let scale = (head_dim as f32).sqrt();
    let scores = q.matmul(k.transpose()) / scale;

    // Apply causal mask if needed
    let scores = if causal && q_len > 1 {
        let device = scores.device();
        // Create upper-triangular mask filled with -inf
        let mut mask_data = vec![0.0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in (i + 1)..kv_len {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask: Tensor<B, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(mask_data, [q_len, kv_len]),
            &device,
        );
        let mask: Tensor<B, 4> = mask.unsqueeze::<4>();
        scores + mask
    } else {
        scores
    };

    // Softmax over key dimension
    let attn_weights = burn::tensor::activation::softmax(scores, 3);

    // Weighted sum: [B, H, T_q, Dh]
    let out = attn_weights.matmul(v);

    // Reshape back: [B, H, T_q, Dh] -> [B, T_q, H, Dh] -> [B, T_q, D]
    let out = out.swap_dims(1, 2);
    out.reshape([batch, q_len, n_heads * head_dim])
}
