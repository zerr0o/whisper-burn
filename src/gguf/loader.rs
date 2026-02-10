//! GGUF model loader for Whisper.
//!
//! Loads Q4_0 quantized Whisper weights from a GGUF file and constructs
//! the full WhisperModel.

use anyhow::{bail, Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};
use std::io::{BufReader, Read, Seek};
use tracing::info;

use super::linear::Q4Linear;
use super::reader::{GgmlDtype, GgufReader};
use super::tensor::Q4Tensor;
use crate::model::attention::{Q4CrossAttention, Q4MultiHeadAttention};
use crate::model::config::WhisperConfig;
use crate::model::decoder::{DecoderBlock, WhisperDecoder};
use crate::model::encoder::{EncoderBlock, WhisperEncoder};
use crate::model::layers::{Conv1D, LayerNorm, Q4FFN};
use crate::model::whisper::WhisperModel;

type B = Wgpu;

/// Load a Whisper model from a GGUF file on disk.
pub fn load_whisper_from_gguf(
    path: &std::path::Path,
    config: &WhisperConfig,
    device: &WgpuDevice,
) -> Result<WhisperModel> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut gguf = GgufReader::open(reader).context("Failed to parse GGUF")?;

    info!(
        "GGUF v{}, {} tensors",
        gguf.version(),
        gguf.tensor_count()
    );

    let encoder = load_encoder(&mut gguf, config, device)?;
    let decoder = load_decoder(&mut gguf, config, device)?;

    Ok(WhisperModel::new(encoder, decoder, config.clone()))
}

fn load_f32_tensor_1d<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<B, 1>> {
    let info = gguf
        .tensor_info(name)
        .with_context(|| format!("Tensor '{name}' not found"))?
        .clone();
    let data = gguf.tensor_data(name)?;
    match info.dtype() {
        GgmlDtype::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let shape = [floats.len()];
            Ok(Tensor::from_data(TensorData::new(floats, shape), device))
        }
        GgmlDtype::F16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            let shape = [floats.len()];
            Ok(Tensor::from_data(TensorData::new(floats, shape), device))
        }
        other => bail!("Expected F32/F16 for '{name}', got {other:?}"),
    }
}

fn load_f32_tensor_2d<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<B, 2>> {
    let info = gguf
        .tensor_info(name)
        .with_context(|| format!("Tensor '{name}' not found"))?
        .clone();
    let shape = info.shape().to_vec();
    let data = gguf.tensor_data(name)?;
    match info.dtype() {
        GgmlDtype::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            // GGUF stores dimensions in reverse order (row-major: [cols, rows])
            let s = [shape[1] as usize, shape[0] as usize];
            Ok(Tensor::from_data(TensorData::new(floats, s), device))
        }
        GgmlDtype::F16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            let s = [shape[1] as usize, shape[0] as usize];
            Ok(Tensor::from_data(TensorData::new(floats, s), device))
        }
        other => bail!("Expected F32/F16 for '{name}', got {other:?}"),
    }
}

fn load_q4_linear<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    weight_name: &str,
    bias_name: Option<&str>,
    device: &WgpuDevice,
) -> Result<Q4Linear> {
    let info = gguf
        .tensor_info(weight_name)
        .with_context(|| format!("Tensor '{weight_name}' not found"))?
        .clone();
    let data = gguf.tensor_data(weight_name)?;

    let weights = match info.dtype() {
        GgmlDtype::Q4_0 => {
            // GGUF stores shape as [in_features, out_features] (reversed)
            let shape = info.shape();
            let k = shape[0] as usize;
            let n = shape[1] as usize;
            Q4Tensor::from_q4_bytes(&data, [n, k], device)?
        }
        GgmlDtype::F32 | GgmlDtype::F16 => {
            bail!("Expected Q4_0 for weight '{weight_name}', got {:?}. Use the conversion script.", info.dtype());
        }
    };

    let bias = if let Some(bn) = bias_name {
        if gguf.tensor_info(bn).is_some() {
            Some(load_f32_tensor_1d(gguf, bn, device)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(Q4Linear::new(weights, bias))
}

fn load_layer_norm<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    prefix: &str,
    device: &WgpuDevice,
) -> Result<LayerNorm> {
    let weight = load_f32_tensor_1d(gguf, &format!("{prefix}.weight"), device)?;
    let bias = load_f32_tensor_1d(gguf, &format!("{prefix}.bias"), device)?;
    Ok(LayerNorm::new(weight, bias))
}

fn load_mha<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    prefix: &str,
    n_heads: usize,
    device: &WgpuDevice,
) -> Result<Q4MultiHeadAttention> {
    let query = load_q4_linear(
        gguf,
        &format!("{prefix}.query.weight"),
        Some(&format!("{prefix}.query.bias")),
        device,
    )?;
    let key = load_q4_linear(
        gguf,
        &format!("{prefix}.key.weight"),
        Some(&format!("{prefix}.key.bias")),
        device,
    )?;
    let value = load_q4_linear(
        gguf,
        &format!("{prefix}.value.weight"),
        Some(&format!("{prefix}.value.bias")),
        device,
    )?;
    let out = load_q4_linear(
        gguf,
        &format!("{prefix}.out.weight"),
        Some(&format!("{prefix}.out.bias")),
        device,
    )?;
    Ok(Q4MultiHeadAttention::new(query, key, value, out, n_heads))
}

fn load_cross_attn<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    prefix: &str,
    n_heads: usize,
    device: &WgpuDevice,
) -> Result<Q4CrossAttention> {
    let query = load_q4_linear(
        gguf,
        &format!("{prefix}.query.weight"),
        Some(&format!("{prefix}.query.bias")),
        device,
    )?;
    let key = load_q4_linear(
        gguf,
        &format!("{prefix}.key.weight"),
        None, // cross-attention key has no bias in Whisper
        device,
    )?;
    let value = load_q4_linear(
        gguf,
        &format!("{prefix}.value.weight"),
        Some(&format!("{prefix}.value.bias")),
        device,
    )?;
    let out = load_q4_linear(
        gguf,
        &format!("{prefix}.out.weight"),
        Some(&format!("{prefix}.out.bias")),
        device,
    )?;
    Ok(Q4CrossAttention::new(query, key, value, out, n_heads))
}

fn load_ffn<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    prefix: &str,
    device: &WgpuDevice,
) -> Result<Q4FFN> {
    let fc1 = load_q4_linear(
        gguf,
        &format!("{prefix}.mlp.0.weight"),
        Some(&format!("{prefix}.mlp.0.bias")),
        device,
    )?;
    let fc2 = load_q4_linear(
        gguf,
        &format!("{prefix}.mlp.2.weight"),
        Some(&format!("{prefix}.mlp.2.bias")),
        device,
    )?;
    Ok(Q4FFN::new(fc1, fc2))
}

fn load_conv1d<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    weight_name: &str,
    bias_name: &str,
    device: &WgpuDevice,
) -> Result<Conv1D> {
    let w_info = gguf
        .tensor_info(weight_name)
        .with_context(|| format!("Tensor '{weight_name}' not found"))?
        .clone();
    let w_data = gguf.tensor_data(weight_name)?;

    // Conv weight shape in GGUF: [kernel_size, in_channels, out_channels]
    let dims = w_info.shape().to_vec();
    let (kernel_size, in_ch, out_ch) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);

    let floats: Vec<f32> = match w_info.dtype() {
        GgmlDtype::F32 => w_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        GgmlDtype::F16 => w_data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => bail!("Unexpected dtype {other:?} for conv weight '{weight_name}'"),
    };

    // Reshape to [out_ch, in_ch, kernel_size] for our Conv1D
    let weight: Tensor<B, 3> = Tensor::from_data(
        TensorData::new(floats, [out_ch, in_ch, kernel_size]),
        device,
    );
    let bias = load_f32_tensor_1d(gguf, bias_name, device)?;

    Ok(Conv1D::new(weight, bias))
}

fn load_encoder<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    config: &WhisperConfig,
    device: &WgpuDevice,
) -> Result<WhisperEncoder> {
    info!("Loading encoder...");

    let conv1 = load_conv1d(
        gguf,
        "encoder.conv1.weight",
        "encoder.conv1.bias",
        device,
    )?;
    let conv2 = load_conv1d(
        gguf,
        "encoder.conv2.weight",
        "encoder.conv2.bias",
        device,
    )?;

    let positional_embedding =
        load_f32_tensor_2d(gguf, "encoder.positional_embedding", device)?;

    let mut blocks = Vec::with_capacity(config.n_audio_layer);
    for i in 0..config.n_audio_layer {
        let prefix = format!("encoder.blocks.{i}");
        let attn_ln = load_layer_norm(gguf, &format!("{prefix}.attn_ln"), device)?;
        let attn = load_mha(gguf, &format!("{prefix}.attn"), config.n_audio_head, device)?;
        let mlp_ln = load_layer_norm(gguf, &format!("{prefix}.mlp_ln"), device)?;
        let mlp = load_ffn(gguf, &prefix, device)?;
        blocks.push(EncoderBlock::new(attn_ln, attn, mlp_ln, mlp));
        if (i + 1) % 8 == 0 {
            info!("  Encoder block {}/{}", i + 1, config.n_audio_layer);
        }
    }

    let ln_post = load_layer_norm(gguf, "encoder.ln_post", device)?;

    Ok(WhisperEncoder::new(
        conv1,
        conv2,
        positional_embedding,
        blocks,
        ln_post,
    ))
}

fn load_decoder<R: Read + Seek>(
    gguf: &mut GgufReader<R>,
    config: &WhisperConfig,
    device: &WgpuDevice,
) -> Result<WhisperDecoder> {
    info!("Loading decoder...");

    let token_embedding =
        load_f32_tensor_2d(gguf, "decoder.token_embedding.weight", device)?;
    let positional_embedding =
        load_f32_tensor_2d(gguf, "decoder.positional_embedding", device)?;

    let mut blocks = Vec::with_capacity(config.n_text_layer);
    for i in 0..config.n_text_layer {
        let prefix = format!("decoder.blocks.{i}");
        let attn_ln = load_layer_norm(gguf, &format!("{prefix}.attn_ln"), device)?;
        let attn = load_mha(gguf, &format!("{prefix}.attn"), config.n_text_head, device)?;
        let cross_attn_ln =
            load_layer_norm(gguf, &format!("{prefix}.cross_attn_ln"), device)?;
        let cross_attn = load_cross_attn(
            gguf,
            &format!("{prefix}.cross_attn"),
            config.n_text_head,
            device,
        )?;
        let mlp_ln = load_layer_norm(gguf, &format!("{prefix}.mlp_ln"), device)?;
        let mlp = load_ffn(gguf, &prefix, device)?;
        blocks.push(DecoderBlock::new(
            attn_ln,
            attn,
            cross_attn_ln,
            cross_attn,
            mlp_ln,
            mlp,
        ));
        info!("  Decoder block {}/{}", i + 1, config.n_text_layer);
    }

    let ln = load_layer_norm(gguf, "decoder.ln", device)?;

    Ok(WhisperDecoder::new(
        token_embedding,
        positional_embedding,
        blocks,
        ln,
    ))
}
