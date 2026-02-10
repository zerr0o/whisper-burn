//! Whisper BPE tokenizer wrapper.
//!
//! Wraps the HuggingFace `tokenizers` crate to decode Whisper token IDs.

use anyhow::Result;
use std::path::Path;

pub struct WhisperTokenizer {
    inner: tokenizers::Tokenizer,
}

impl WhisperTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))
    }

    /// Start of transcript token.
    pub fn sot_token() -> u32 {
        50258
    }

    /// End of transcript token.
    pub fn eot_token() -> u32 {
        50257
    }

    /// Language token for a given language code.
    pub fn lang_token(lang: &str) -> u32 {
        crate::ALL_LANGUAGES
            .iter()
            .find(|l| l.code == Some(lang))
            .map(|l| l.token_id)
            .unwrap_or(50259) // default to English
    }

    /// Transcribe task token.
    pub fn transcribe_token() -> u32 {
        50359
    }

    /// No timestamps token.
    pub fn no_timestamps_token() -> u32 {
        50363
    }
}
