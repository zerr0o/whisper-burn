# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

whisper-burn is a native Rust implementation of OpenAI's Whisper Large V3 / V3 Turbo speech recognition using the Burn ML framework with GPU acceleration via wgpu (Vulkan/Metal/DirectX). Models are loaded from Q4_0 quantized GGUF files and inference runs entirely on GPU through custom WGSL compute shaders.

## Build Commands

```bash
# Build (release, optimized with LTO)
cargo build --release

# Run the native desktop app
cargo run --release --bin whisper-native

# Build library only (no native GUI)
cargo build --release --no-default-features --features wgpu
```

Release profile uses LTO + single codegen unit + opt-level 3. Debug builds are significantly slower for inference.

## Model Conversion

```bash
python scripts/convert_whisper.py --model openai/whisper-large-v3-turbo --output models/whisper-large-v3-turbo-q4.gguf
```

Converts HuggingFace Whisper models to Q4_0 GGUF format. 2D weight matrices > 256 dims are quantized to 4-bit; embeddings, biases, and layer norms stay F32.

## Architecture

### Inference Pipeline (`src/transcribe.rs`)
Audio → Resample 16kHz → Pad/truncate 30s → Mel spectrogram [1,128,3000] → Encoder [1,1500,1280] → Greedy decoder with KV cache → Token IDs → Text

### Module Structure

- **`src/model/`** — Whisper model architecture (encoder, decoder, attention, layers, config). `config.rs` defines Large V3 (32 layers) vs V3 Turbo (4 layers).
- **`src/gguf/`** — GGUF file parsing, Q4_0 tensor storage, fused dequant+matmul GPU kernel. `shader.wgsl` is the custom WGSL compute shader for Q4 matrix multiplication (dequantizes on-the-fly, no intermediate F32 buffer).
- **`src/audio/`** — Mel spectrogram computation (STFT + mel filterbank), WAV I/O, resampling to 16kHz.
- **`src/native/`** — eframe/egui desktop app with state machine (CheckModel → Download → Load → Ready ↔ Recording ↔ Transcribing). Inference runs on a dedicated background thread.
- **`src/tokenizer.rs`** — BPE tokenizer wrapper. Special tokens: SOT=50258, EOT=50257, TRANSCRIBE=50359, NO_TIMESTAMPS=50363.

### Key Design Decisions

- **Patched cubecl-wgpu** (`patches/cubecl-wgpu-0.9.0/`): Custom patch of CubeCL's wgpu runtime applied via `[patch.crates-io]` in Cargo.toml. Any wgpu backend changes go here.
- **Fused Q4 matmul shader**: The WGSL shader in `src/gguf/shader.wgsl` reads Q4_0 blocks (18 bytes = f16 scale + 16 bytes of packed nibbles) and performs dequantization + matmul in a single kernel pass.
- **Batched prompt processing**: All prompt tokens are processed in one forward pass before autoregressive generation begins (see `src/model/whisper.rs`).
- **Im2col Conv1D**: 1D convolution in the encoder uses im2col transformation for efficient GPU matmul (see `src/model/layers.rs`).
- **KV cache**: Decoder caches attention keys/values across autoregressive steps to avoid recomputation (`src/model/decoder.rs`).

## Testing

No automated test suite. Validation is done by running the native app with real audio input.

## Feature Flags

- `wgpu` (default) — GPU backend via Burn
- `native` (default) — Desktop app with eframe GUI, cpal audio capture, ureq HTTP for model download
