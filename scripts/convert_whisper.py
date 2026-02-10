#!/usr/bin/env python3
"""Convert Whisper safetensors weights to Q4_0 GGUF format.

Usage:
    python scripts/convert_whisper.py --model openai/whisper-large-v3 --output models/whisper-large-v3-q4.gguf
    python scripts/convert_whisper.py --model openai/whisper-large-v3-turbo --output models/whisper-large-v3-turbo-q4.gguf

Requirements:
    pip install torch safetensors transformers numpy struct
"""

import argparse
import struct
import numpy as np
from pathlib import Path


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
ALIGNMENT = 32

# GGML dtype codes
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2

# Q4_0 block size
Q4_BLOCK_SIZE = 32
Q4_BLOCK_BYTES = 18  # 2 bytes f16 scale + 16 bytes nibbles


def quantize_q4_0(tensor: np.ndarray) -> bytes:
    """Quantize a float32 tensor to Q4_0 format.

    Each block of 32 values is quantized to 18 bytes:
    - 2 bytes: f16 scale factor (max absolute value / 7)
    - 16 bytes: packed 4-bit nibbles (lower/upper)
    """
    flat = tensor.flatten().astype(np.float32)
    n_elements = len(flat)
    assert n_elements % Q4_BLOCK_SIZE == 0, f"Element count {n_elements} not divisible by {Q4_BLOCK_SIZE}"

    n_blocks = n_elements // Q4_BLOCK_SIZE
    output = bytearray()

    for block_idx in range(n_blocks):
        block = flat[block_idx * Q4_BLOCK_SIZE:(block_idx + 1) * Q4_BLOCK_SIZE]

        # Compute scale: max absolute value / 7
        amax = np.max(np.abs(block))
        d = amax / 7.0 if amax > 0 else 0.0

        # Convert scale to f16 bytes
        d_f16 = np.float16(d)
        scale_bytes = d_f16.tobytes()

        # Quantize: round(value / scale) + 8, clamp to [0, 15]
        if d > 0:
            quantized = np.round(block / d).astype(np.int8)
        else:
            quantized = np.zeros(Q4_BLOCK_SIZE, dtype=np.int8)

        # Pack nibbles: elements 0-15 in lower nibble, 16-31 in upper nibble
        nibble_bytes = bytearray(16)
        for i in range(16):
            lo = int(quantized[i] + 8) & 0x0F
            hi = int(quantized[i + 16] + 8) & 0x0F
            nibble_bytes[i] = lo | (hi << 4)

        output.extend(scale_bytes)
        output.extend(nibble_bytes)

    return bytes(output)


def should_quantize(name: str, shape: tuple) -> bool:
    """Decide whether a tensor should be quantized to Q4_0."""
    # Only quantize 2D weight matrices that are large enough
    if len(shape) != 2:
        return False
    if min(shape) < 256:
        return False
    # Don't quantize embeddings, biases, or normalization weights
    if "bias" in name:
        return False
    if "ln" in name or "layer_norm" in name:
        return False
    if "positional_embedding" in name:
        return False
    if "token_embedding" in name:
        return False
    # Don't quantize conv weights (they're small and 3D originally)
    if "conv" in name:
        return False
    return True


def write_gguf_string(f, s: str):
    """Write a GGUF string (u64 length + bytes)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def write_gguf_metadata_kv(f, key: str, value_type: int, value):
    """Write a metadata key-value pair."""
    write_gguf_string(f, key)
    f.write(struct.pack("<I", value_type))
    if value_type == 4:  # u32
        f.write(struct.pack("<I", value))
    elif value_type == 8:  # string
        write_gguf_string(f, value)


def align_offset(offset: int, alignment: int = ALIGNMENT) -> int:
    """Align offset to boundary."""
    return ((offset + alignment - 1) // alignment) * alignment


def convert_model(model_name: str, output_path: str):
    """Convert a Whisper model to Q4_0 GGUF."""
    try:
        from transformers import WhisperForConditionalGeneration
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers torch safetensors")
        return

    print(f"Loading model: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")

    state_dict = model.state_dict()
    print(f"Loaded {len(state_dict)} tensors")

    # Map HuggingFace names to GGUF names
    tensor_map = {}
    for hf_name, tensor in state_dict.items():
        gguf_name = hf_name_to_gguf(hf_name)
        if gguf_name is not None:
            tensor_map[gguf_name] = tensor.float().numpy()

    print(f"Mapped {len(tensor_map)} tensors to GGUF names")

    # Prepare tensor data
    tensor_entries = []
    tensor_data_list = []
    current_offset = 0

    for name, array in sorted(tensor_map.items()):
        shape = array.shape
        if should_quantize(name, shape):
            data = quantize_q4_0(array)
            dtype = GGML_TYPE_Q4_0
            print(f"  Q4_0: {name} {shape} -> {len(data)} bytes")
        else:
            data = array.astype(np.float32).tobytes()
            dtype = GGML_TYPE_F32
            print(f"  F32:  {name} {shape} -> {len(data)} bytes")

        # GGUF stores dimensions in reverse order
        dims = list(reversed(shape))

        # Align data offset
        aligned_offset = align_offset(current_offset)
        padding_needed = aligned_offset - current_offset

        tensor_entries.append({
            "name": name,
            "dims": dims,
            "dtype": dtype,
            "offset": aligned_offset,
        })
        tensor_data_list.append((padding_needed, data))
        current_offset = aligned_offset + len(data)

    # Write GGUF file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata = {
        "general.architecture": ("whisper", 8),  # string
        "general.name": (model_name, 8),
        "whisper.encoder.layer_count": (len([n for n in tensor_map if n.startswith("encoder.blocks.")]) // 8, 4),  # u32 (approx)
        "whisper.decoder.layer_count": (len([n for n in tensor_map if n.startswith("decoder.blocks.")]) // 16, 4),
    }

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_entries)))  # tensor count
        f.write(struct.pack("<Q", len(metadata)))  # metadata kv count

        # Metadata
        for key, (value, vtype) in metadata.items():
            write_gguf_metadata_kv(f, key, vtype, value)

        # Tensor index
        for entry in tensor_entries:
            write_gguf_string(f, entry["name"])
            f.write(struct.pack("<I", len(entry["dims"])))
            for d in entry["dims"]:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", entry["dtype"]))
            f.write(struct.pack("<Q", entry["offset"]))

        # Align to data section
        current_pos = f.tell()
        data_start = align_offset(current_pos)
        f.write(b"\x00" * (data_start - current_pos))

        # Tensor data
        for padding, data in tensor_data_list:
            if padding > 0:
                f.write(b"\x00" * padding)
            f.write(data)

    file_size = output_path.stat().st_size
    print(f"\nWritten: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"Tensors: {len(tensor_entries)}")


def hf_name_to_gguf(hf_name: str) -> str | None:
    """Map a HuggingFace Whisper parameter name to GGUF convention."""
    # Remove 'model.' prefix
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]

    # Encoder mappings
    if name.startswith("encoder."):
        name = name.replace("encoder.layers.", "encoder.blocks.")
        name = name.replace("encoder.layer_norm.", "encoder.ln_post.")
        name = name.replace("encoder.embed_positions.weight", "encoder.positional_embedding")
        name = name.replace(".self_attn.", ".attn.")
        name = name.replace(".q_proj.", ".query.")
        name = name.replace(".k_proj.", ".key.")
        name = name.replace(".v_proj.", ".value.")
        name = name.replace(".out_proj.", ".out.")
        name = name.replace(".self_attn_layer_norm.", ".attn_ln.")
        name = name.replace(".final_layer_norm.", ".mlp_ln.")
        name = name.replace(".fc1.", ".mlp.0.")
        name = name.replace(".fc2.", ".mlp.2.")
        return name

    # Decoder mappings
    if name.startswith("decoder."):
        name = name.replace("decoder.layers.", "decoder.blocks.")
        name = name.replace("decoder.layer_norm.", "decoder.ln.")
        name = name.replace("decoder.embed_tokens.weight", "decoder.token_embedding.weight")
        name = name.replace("decoder.embed_positions.weight", "decoder.positional_embedding")
        name = name.replace(".self_attn.", ".attn.")
        name = name.replace(".encoder_attn.", ".cross_attn.")
        name = name.replace(".q_proj.", ".query.")
        name = name.replace(".k_proj.", ".key.")
        name = name.replace(".v_proj.", ".value.")
        name = name.replace(".out_proj.", ".out.")
        name = name.replace(".self_attn_layer_norm.", ".attn_ln.")
        name = name.replace(".encoder_attn_layer_norm.", ".cross_attn_ln.")
        name = name.replace(".final_layer_norm.", ".mlp_ln.")
        name = name.replace(".fc1.", ".mlp.0.")
        name = name.replace(".fc2.", ".mlp.2.")
        return name

    # Conv layers
    if name.startswith("encoder.conv1.") or name.startswith("encoder.conv2."):
        return name

    # Projection head (not needed for inference)
    if name.startswith("proj_out."):
        return None

    print(f"  SKIP: {hf_name}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper to Q4_0 GGUF")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace model name (default: openai/whisper-large-v3-turbo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GGUF path (default: models/<model>-q4.gguf)",
    )
    args = parser.parse_args()

    if args.output is None:
        model_short = args.model.split("/")[-1]
        args.output = f"models/{model_short}-q4.gguf"

    convert_model(args.model, args.output)
