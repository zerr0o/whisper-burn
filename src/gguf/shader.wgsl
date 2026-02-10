// Q4_0 Dequantization + Matrix Multiplication Compute Shader
//
// Performs a fused dequant-matmul for Q4_0 quantized weight tensors on GPU.
// Computes: output[B, M, N] = input[B, M, K] x weights[N, K]^T
// where weights are stored in GGML Q4_0 block format.
//
// == Q4_0 Block Format (GGML standard, interleaved) ==
//
// Each block encodes 32 weights into 18 bytes:
//
//   Bytes 0-1:  f16 scale `d` -- rescales the quantized ints
//   Bytes 2-17: 16 bytes of packed 4-bit quantized values
//
// Nibble packing within the 16 data bytes:
//   - Each byte holds two 4-bit values (nibbles)
//   - Lower nibble (bits 0-3) -> elements 0-15
//   - Upper nibble (bits 4-7) -> elements 16-31
//   - Each nibble stores (value + 8) & 0xF, so dequantized value = (nibble - 8) * d
//
// The weight tensor [N, K] (out_features, in_features) is flattened row-major
// and divided into consecutive blocks of 32 elements.

// -- Bindings --
@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

// Read a single byte from the weights buffer at the given byte offset.
fn read_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_pos = byte_offset % 4u;
    return (weights[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Read the f16 scale factor at the start of a Q4_0 block. Returns f32.
fn read_f16_scale(block_byte_offset: u32) -> f32 {
    let lo = read_byte(block_byte_offset);
    let hi = read_byte(block_byte_offset + 1u);
    let bits = lo | (hi << 8u);
    return unpack2x16float(bits).x;
}

// ---------------------------------------------------------------------------
// Main kernel: one thread per output element.
//
// Optimized inner loop processes Q4_0 blocks directly:
// - Scale factor read once per 32-element block (not 32 times)
// - Nibble pairs (elements i and i+16) processed together from one byte read
// ---------------------------------------------------------------------------
@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = info[0];
    let M = info[1];
    let K = info[2];
    let N = info[3];

    let n = gid.x;
    let bm = gid.y;
    let m = bm % M;
    let b = bm / M;

    if (n >= N || b >= B) {
        return;
    }

    var acc: f32 = 0.0;
    let input_base = b * M * K + m * K;
    let blocks_per_row = K / 32u;

    // Iterate over Q4_0 blocks along the K dimension
    for (var blk: u32 = 0u; blk < blocks_per_row; blk = blk + 1u) {
        let block_idx = n * blocks_per_row + blk;
        let block_byte = block_idx * 18u;

        // Read scale factor ONCE per block (saves 31 redundant reads)
        let d = read_f16_scale(block_byte);
        let block_k = blk * 32u;

        // Process 16 data bytes: each byte holds two nibbles
        // Lower nibble -> element i (0..15), upper nibble -> element i+16
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let data_byte = read_byte(block_byte + 2u + i);
            let lo = (f32(data_byte & 0xFu) - 8.0) * d;
            let hi = (f32((data_byte >> 4u) & 0xFu) - 8.0) * d;
            acc += lo * input[input_base + block_k + i];
            acc += hi * input[input_base + block_k + i + 16u];
        }
    }

    output[b * M * N + m * N + n] = acc;
}
