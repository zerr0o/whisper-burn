//! Transcription pipeline: audio -> mel -> encode -> decode -> text.

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use std::time::Instant;
use tracing::info;

use crate::audio::{
    io::AudioBuffer,
    mel::{MelConfig, MelSpectrogram},
    resample::resample_to_16k,
};
use crate::model::whisper::WhisperModel;
use crate::tokenizer::WhisperTokenizer;
use crate::Language;

type Backend = Wgpu;

/// Whisper samples per 30-second chunk at 16kHz.
const WHISPER_CHUNK_SAMPLES: usize = 480_000; // 30s * 16000

/// Expected number of mel frames for a 30s chunk.
const WHISPER_MEL_FRAMES: usize = 3000; // 480000 / 160

pub struct InferenceState {
    pub model: WhisperModel,
    pub tokenizer: WhisperTokenizer,
    pub device: <Backend as burn::tensor::backend::Backend>::Device,
}

/// Transcribe an AudioBuffer, returning (text, inference_ms).
pub fn transcribe(state: &InferenceState, audio: AudioBuffer, language: Language) -> Result<(String, u128)> {
    let device = &state.device;

    // Resample to 16kHz if needed
    let audio = if audio.sample_rate != 16000 {
        info!("Resampling {} Hz -> 16 kHz", audio.sample_rate);
        resample_to_16k(&audio).context("Failed to resample audio")?
    } else {
        audio
    };

    // Pad or truncate to 30 seconds
    let mut samples = audio.samples;
    if samples.len() < WHISPER_CHUNK_SAMPLES {
        samples.resize(WHISPER_CHUNK_SAMPLES, 0.0);
    } else if samples.len() > WHISPER_CHUNK_SAMPLES {
        // For now, truncate to 30s. Multi-chunk support can be added later.
        samples.truncate(WHISPER_CHUNK_SAMPLES);
    }

    // Compute mel spectrogram
    let mel_extractor = MelSpectrogram::new(MelConfig::whisper());
    let mel = mel_extractor.compute_log(&samples);
    let n_frames = mel.len();
    let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

    if n_frames == 0 {
        bail!("Audio too short to produce mel frames");
    }
    info!(frames = n_frames, bins = n_mels, "Mel spectrogram computed");

    // Transpose from [frames, mels] to [mels, frames] for the model
    let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
    for (frame_idx, frame) in mel.iter().enumerate() {
        for (mel_idx, &val) in frame.iter().enumerate() {
            mel_transposed[mel_idx][frame_idx] = val;
        }
    }

    // Pad or truncate mel frames to exactly 3000
    for channel in &mut mel_transposed {
        channel.resize(WHISPER_MEL_FRAMES, 0.0);
    }

    let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
    let mel_tensor: Tensor<Backend, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat, [1, n_mels, WHISPER_MEL_FRAMES]),
        device,
    );

    let start = Instant::now();
    info!("Starting Whisper inference (lang={:?})", language);
    let generated = state.model.transcribe(mel_tensor, language.code());
    let inference_ms = start.elapsed().as_millis();
    info!(
        elapsed_ms = inference_ms as u64,
        tokens = generated.len(),
        "Whisper inference complete"
    );

    // Decode tokens to text (filter out special tokens >= 50257)
    let text_tokens: Vec<u32> = generated
        .iter()
        .filter(|&&t| t >= 0 && t < 50257)
        .map(|&t| t as u32)
        .collect();

    let text = state
        .tokenizer
        .decode(&text_tokens)
        .context("Failed to decode tokens")?;

    Ok((text.trim().to_string(), inference_ms))
}
