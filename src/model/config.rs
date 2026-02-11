//! Whisper model configuration.

/// Configuration for Whisper Large V3 and V3 Turbo models.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Number of mel frequency bins (128)
    pub n_mels: usize,
    /// Maximum audio context length in frames (1500)
    pub n_audio_ctx: usize,
    /// Audio encoder hidden dimension (1280)
    pub n_audio_state: usize,
    /// Number of attention heads in encoder (20)
    pub n_audio_head: usize,
    /// Number of encoder layers (32)
    pub n_audio_layer: usize,
    /// Maximum text context length in tokens (448)
    pub n_text_ctx: usize,
    /// Text decoder hidden dimension (1280)
    pub n_text_state: usize,
    /// Number of attention heads in decoder (20)
    pub n_text_head: usize,
    /// Number of decoder layers (32 for V3, 4 for V3 Turbo)
    pub n_text_layer: usize,
    /// Vocabulary size (51865)
    pub n_vocab: usize,
    /// Number of language tokens (99 for Medium, 100 for Large V3+)
    pub n_lang: usize,
}

impl WhisperConfig {
    /// Whisper Large V3 configuration (1.55B parameters).
    pub fn large_v3() -> Self {
        Self {
            n_mels: 128,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
            n_vocab: 51865,
            n_lang: 100,
        }
    }

    /// Whisper Medium configuration (769M parameters).
    pub fn medium() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 1024,
            n_audio_head: 16,
            n_audio_layer: 24,
            n_text_ctx: 448,
            n_text_state: 1024,
            n_text_head: 16,
            n_text_layer: 24,
            n_vocab: 51865,
            n_lang: 99,
        }
    }

    /// Transcribe task token ID (varies by model: 50359 for Medium, 50360 for Large V3).
    pub fn transcribe_token(&self) -> i32 {
        // Layout: SOT(50258), langs(50259..+n_lang), TRANSLATE, TRANSCRIBE
        // TRANSCRIBE = 50259 + n_lang + 1
        50260i32 + self.n_lang as i32
    }

    /// No-timestamps token ID (varies by model: 50363 for Medium, 50364 for Large V3).
    pub fn no_timestamps_token(&self) -> i32 {
        self.transcribe_token() + 4
    }

    /// Range of language token IDs for auto-detection.
    pub fn lang_token_range(&self) -> std::ops::Range<usize> {
        50259..(50259 + self.n_lang)
    }

    /// Head dimension (state / heads).
    pub fn head_dim(&self) -> usize {
        self.n_audio_state / self.n_audio_head
    }

    /// FFN intermediate dimension (4 * state).
    pub fn ffn_dim(&self) -> usize {
        self.n_audio_state * 4
    }
}
