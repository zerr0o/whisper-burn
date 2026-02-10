//! Whisper text decoder.
//!
//! Token + learned positional embeddings followed by a stack of decoder blocks
//! (self-attention + cross-attention + FFN), with KV caching for autoregressive
//! generation.

use burn::backend::Wgpu;
use burn::tensor::Tensor;

use super::attention::{Q4CrossAttention, Q4MultiHeadAttention};
use super::layers::{LayerNorm, Q4FFN};

type B = Wgpu;

/// KV cache for a single decoder block.
pub struct BlockKVCache {
    pub self_k: Option<Tensor<B, 3>>,
    pub self_v: Option<Tensor<B, 3>>,
    pub cross_k: Option<Tensor<B, 3>>,
    pub cross_v: Option<Tensor<B, 3>>,
}

impl BlockKVCache {
    pub fn empty() -> Self {
        Self {
            self_k: None,
            self_v: None,
            cross_k: None,
            cross_v: None,
        }
    }
}

/// Full KV cache for all decoder blocks.
pub struct KVCache {
    pub blocks: Vec<BlockKVCache>,
}

impl KVCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            blocks: (0..n_layers).map(|_| BlockKVCache::empty()).collect(),
        }
    }
}

/// A single decoder transformer block.
pub struct DecoderBlock {
    attn_ln: LayerNorm,
    attn: Q4MultiHeadAttention,
    cross_attn_ln: LayerNorm,
    cross_attn: Q4CrossAttention,
    mlp_ln: LayerNorm,
    mlp: Q4FFN,
}

impl DecoderBlock {
    pub fn new(
        attn_ln: LayerNorm,
        attn: Q4MultiHeadAttention,
        cross_attn_ln: LayerNorm,
        cross_attn: Q4CrossAttention,
        mlp_ln: LayerNorm,
        mlp: Q4FFN,
    ) -> Self {
        Self {
            attn_ln,
            attn,
            cross_attn_ln,
            cross_attn,
            mlp_ln,
            mlp,
        }
    }

    /// Forward with KV cache for autoregressive decoding.
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        encoder_out: &Tensor<B, 3>,
        cache: &mut BlockKVCache,
    ) -> Tensor<B, 3> {
        // Self-attention with cache
        let residual = x.clone();
        let x = self.attn_ln.forward(x);
        let (x, new_self_k, new_self_v) =
            self.attn
                .forward_with_cache(x, cache.self_k.take(), cache.self_v.take());
        cache.self_k = Some(new_self_k);
        cache.self_v = Some(new_self_v);
        let x = residual + x;

        // Cross-attention with cache
        let residual = x.clone();
        let x = self.cross_attn_ln.forward(x);
        let (x, new_cross_k, new_cross_v) = self.cross_attn.forward_with_cache(
            x,
            encoder_out,
            cache.cross_k.take(),
            cache.cross_v.take(),
        );
        cache.cross_k = Some(new_cross_k);
        cache.cross_v = Some(new_cross_v);
        let x = residual + x;

        // FFN
        let residual = x.clone();
        let x = self.mlp_ln.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }

    /// Forward without cache (for initial prompt processing).
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        encoder_out: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // Self-attention (causal)
        let residual = x.clone();
        let x = self.attn_ln.forward(x);
        let x = self.attn.forward(x, true);
        let x = residual + x;

        // Cross-attention
        let residual = x.clone();
        let x = self.cross_attn_ln.forward(x);
        let x = self.cross_attn.forward(x, encoder_out.clone());
        let x = residual + x;

        // FFN
        let residual = x.clone();
        let x = self.mlp_ln.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

/// Whisper text decoder.
pub struct WhisperDecoder {
    /// Token embedding [vocab_size, d_model]
    token_embedding: Tensor<B, 2>,
    /// Learned positional embedding [max_ctx, d_model]
    positional_embedding: Tensor<B, 2>,
    /// Decoder transformer blocks
    blocks: Vec<DecoderBlock>,
    /// Final layer norm
    ln: LayerNorm,
}

impl WhisperDecoder {
    pub fn new(
        token_embedding: Tensor<B, 2>,
        positional_embedding: Tensor<B, 2>,
        blocks: Vec<DecoderBlock>,
        ln: LayerNorm,
    ) -> Self {
        Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
        }
    }

    /// Get logits for a sequence of tokens (no cache, used for initial prompt).
    ///
    /// `tokens` shape: [B, T]
    /// `encoder_out` shape: [B, T_enc, D]
    /// Returns logits: [B, T, vocab_size]
    pub fn forward(
        &self,
        token_ids: &[i32],
        encoder_out: &Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let _device = encoder_out.device();
        let seq_len = token_ids.len();

        // Token embeddings: look up each token
        let mut embed_rows = Vec::with_capacity(seq_len);
        for &tid in token_ids {
            let row = self
                .token_embedding
                .clone()
                .slice([tid as usize..tid as usize + 1, 0..self.token_embedding.dims()[1]]);
            embed_rows.push(row);
        }
        let token_embed: Tensor<B, 2> = Tensor::cat(embed_rows, 0);

        // Positional embeddings
        let pos_embed = self
            .positional_embedding
            .clone()
            .slice([0..seq_len, 0..self.positional_embedding.dims()[1]]);

        let x: Tensor<B, 3> = (token_embed + pos_embed).unsqueeze_dim(0);

        // Decoder blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, encoder_out);
        }

        // Final layer norm
        let x = self.ln.forward(x);

        // Project to vocab: x @ token_embedding^T
        // x shape: [1, T, D] -> reshape to [T, D]
        let d_model = self.token_embedding.dims()[1];
        let x_2d: Tensor<B, 2> = x.reshape([seq_len, d_model]);
        let logits = x_2d.matmul(self.token_embedding.clone().transpose());

        logits
    }

    /// Decode a single token step with KV cache.
    ///
    /// `token_id`: the current token
    /// `position`: position in the sequence (for positional embedding)
    /// `encoder_out`: encoder hidden states
    /// `cache`: mutable KV cache
    ///
    /// Returns logits [1, vocab_size]
    pub fn decode_step(
        &self,
        token_id: i32,
        position: usize,
        encoder_out: &Tensor<B, 3>,
        cache: &mut KVCache,
    ) -> Vec<f32> {
        let _device = encoder_out.device();
        let d_model = self.token_embedding.dims()[1];

        // Token embedding for single token
        let token_embed = self
            .token_embedding
            .clone()
            .slice([token_id as usize..token_id as usize + 1, 0..d_model]);

        // Positional embedding for this position
        let pos_embed = self
            .positional_embedding
            .clone()
            .slice([position..position + 1, 0..d_model]);

        // [1, 1, D]
        let x: Tensor<B, 3> = (token_embed + pos_embed).unsqueeze_dim(0);

        // Through each decoder block with cache
        let mut x = x;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_cache(x, encoder_out, &mut cache.blocks[i]);
        }

        // Final layer norm
        let x = self.ln.forward(x);

        // Project to vocab: x is [1, 1, D] -> reshape to [1, D]
        let d_model = self.token_embedding.dims()[1];
        let x_2d: Tensor<B, 2> = x.reshape([1, d_model]);
        let logits = x_2d.matmul(self.token_embedding.clone().transpose());

        // Return logits as Vec<f32>
        let logits_data = logits.into_data();
        logits_data.to_vec::<f32>().unwrap()
    }
}
