//! Audio resampling utilities.
//!
//! Uses rubato for high-quality resampling to match Whisper's expected 16kHz input.

use crate::audio::AudioBuffer;
use anyhow::{Context, Result};
use audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Resampler};

/// Resample audio to 16kHz (Whisper's expected sample rate).
pub fn resample_to_16k(audio: &AudioBuffer) -> Result<AudioBuffer> {
    resample(audio, 16000)
}

/// Resample audio to a target sample rate.
pub fn resample(audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    if audio.sample_rate == target_rate {
        return Ok(audio.clone());
    }

    let mut resampler = Fft::<f32>::new(
        audio.sample_rate as usize,
        target_rate as usize,
        1024,
        2,
        1,
        FixedSync::Input,
    )
    .context("Failed to create resampler")?;

    let output_len = resampler.process_all_needed_output_len(audio.samples.len());

    let input_buf = InterleavedOwned::new_from(audio.samples.clone(), 1, audio.samples.len())
        .context("Failed to create input buffer")?;

    let mut output_buf = InterleavedOwned::new(0.0f32, 1, output_len);

    let (_, actual_output_len) = resampler
        .process_all_into_buffer(&input_buf, &mut output_buf, audio.samples.len(), None)
        .context("Failed to resample audio")?;

    let output = output_buf.take_data();
    let output = output[..actual_output_len].to_vec();

    Ok(AudioBuffer::new(output, target_rate))
}
