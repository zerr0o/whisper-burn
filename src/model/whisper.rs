//! Composite Whisper model (encoder + decoder).

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use tracing::info;

use super::config::WhisperConfig;
use super::decoder::{KVCache, WhisperDecoder};
use super::encoder::WhisperEncoder;

type B = Wgpu;

// Special token IDs (shared across all Whisper models)
const SOT: i32 = 50258; // Start of transcript
const EOT: i32 = 50257; // End of transcript
// TRANSCRIBE and NO_TIMESTAMPS vary by model (99 vs 100 languages).
// Use self.config.transcribe_token() and self.config.no_timestamps_token().

/// Maximum number of generated tokens before stopping.
const MAX_TOKENS: usize = 224;

pub struct WhisperModel {
    encoder: WhisperEncoder,
    decoder: WhisperDecoder,
    config: WhisperConfig,
}

impl WhisperModel {
    pub fn new(encoder: WhisperEncoder, decoder: WhisperDecoder, config: WhisperConfig) -> Self {
        Self {
            encoder,
            decoder,
            config,
        }
    }

    /// Encode mel spectrogram to hidden states.
    ///
    /// Input: [B, 128, 3000]
    /// Output: [B, 1500, 1280]
    pub fn encode(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    /// Transcribe mel spectrogram to token IDs using greedy decoding.
    ///
    /// `lang_code`: language code like "en", "fr", "es", or `None` for auto-detect.
    ///
    /// Input: mel [1, 128, 3000]
    /// Returns: Vec of token IDs (without special tokens)
    pub fn transcribe(&self, mel: Tensor<B, 3>, lang_code: Option<&str>) -> Vec<i32> {
        info!("Encoding audio...");
        let encoder_out = self.encode(mel);
        info!("Encoder done, starting decoder...");

        // Initialize KV cache
        let mut cache = KVCache::new(self.config.n_text_layer);
        let mut position = 0;
        let mut logits;

        // Token IDs that vary by model (99 vs 100 language tokens)
        let transcribe_token = self.config.transcribe_token();
        let no_timestamps_token = self.config.no_timestamps_token();
        let lang_range = self.config.lang_token_range();

        // Build prompt based on language selection
        let prompt = if let Some(lang) = lang_code {
            // Explicit language: [SOT, lang, TRANSCRIBE, NO_TIMESTAMPS]
            let lang_token = crate::tokenizer::WhisperTokenizer::lang_token(lang);
            vec![SOT, lang_token as i32, transcribe_token, no_timestamps_token]
        } else {
            // Auto-detect: feed [SOT], pick language from logits, then continue
            logits = self.decoder.decode_step(SOT, 0, &encoder_out, &mut cache);
            position = 1;

            // Restrict to language tokens and pick the best
            let lang_token = lang_range
                .max_by(|&a, &b| {
                    logits[a]
                        .partial_cmp(&logits[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(50259) as i32;

            info!("Auto-detected language token: {}", lang_token);

            vec![lang_token, transcribe_token, no_timestamps_token]
        };

        // Process all prompt tokens in a single batched forward pass
        // (much faster than sequential decode_step calls)
        logits = self.decoder.forward_prompt(&prompt, &encoder_out, &mut cache);
        position += prompt.len();

        // Suppress EOT at the first position (matches Whisper's SuppressBlank filter)
        // to prevent premature termination from Q4 quantization noise
        const MIN_TOKENS: usize = 3;
        logits[EOT as usize] = f32::NEG_INFINITY;
        let mut next_token = argmax(&logits) as i32;

        // Autoregressive generation
        let mut generated = Vec::new();

        for step in 0..MAX_TOKENS {
            if next_token == EOT {
                info!("EOT after {} tokens", step);
                break;
            }

            generated.push(next_token);

            if step > 0 && step % 50 == 0 {
                info!("Generated {} tokens...", step);
            }

            logits = self.decoder.decode_step(next_token, position, &encoder_out, &mut cache);
            position += 1;

            // Suppress EOT for the first few tokens
            if step + 1 < MIN_TOKENS {
                logits[EOT as usize] = f32::NEG_INFINITY;
            }

            next_token = argmax(&logits) as i32;
        }

        generated
    }
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
