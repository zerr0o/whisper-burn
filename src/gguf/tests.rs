//! Unit tests for Q4_0 quantization pipeline.
//!
//! Tests cover: GGUF parsing, Q4_0 block dequantization, and GPU q4_matmul
//! at various shapes including real Whisper model dimensions.

#[cfg(test)]
mod tests {
    use crate::gguf::*;
    use burn::backend::Wgpu;
    use burn::tensor::{Tensor, TensorData};

    type TestBackend = Wgpu;

    // =========================================================================
    // CPU-side Q4_0 helpers (test-only)
    // =========================================================================

    const Q4_BLOCK_SIZE: usize = 32;
    const Q4_BLOCK_BYTES: usize = 18;

    /// Quantize f32 data to Q4_0 format (GGML standard).
    /// Input length must be a multiple of 32.
    /// Returns raw bytes: 18 bytes per block (2 f16 scale + 16 packed nibbles).
    fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
        assert_eq!(
            data.len() % Q4_BLOCK_SIZE,
            0,
            "Data length {} is not a multiple of {}",
            data.len(),
            Q4_BLOCK_SIZE
        );
        let n_blocks = data.len() / Q4_BLOCK_SIZE;
        let mut output = Vec::with_capacity(n_blocks * Q4_BLOCK_BYTES);

        for block_idx in 0..n_blocks {
            let block = &data[block_idx * Q4_BLOCK_SIZE..(block_idx + 1) * Q4_BLOCK_SIZE];

            // Find absmax to compute scale
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 7.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };

            // Write scale as f16 (2 bytes LE)
            let d_f16 = half::f16::from_f32(d);
            output.extend_from_slice(&d_f16.to_le_bytes());

            // Pack nibbles: lower nibble = elements 0..16, upper nibble = elements 16..32
            for i in 0..16 {
                let v0 = block[i];
                let v1 = block[i + 16];
                let q0 = ((v0 * id + 8.5) as u8).min(15);
                let q1 = ((v1 * id + 8.5) as u8).min(15);
                output.push(q0 | (q1 << 4));
            }
        }
        output
    }

    /// Dequantize Q4_0 bytes back to f32.
    fn dequantize_q4_0_to_f32(q4_bytes: &[u8], n_elements: usize) -> Vec<f32> {
        assert_eq!(
            n_elements % Q4_BLOCK_SIZE,
            0,
            "Element count {} is not a multiple of {}",
            n_elements,
            Q4_BLOCK_SIZE
        );
        let n_blocks = n_elements / Q4_BLOCK_SIZE;
        assert_eq!(q4_bytes.len(), n_blocks * Q4_BLOCK_BYTES);
        let mut output = vec![0.0f32; n_elements];

        for block_idx in 0..n_blocks {
            let offset = block_idx * Q4_BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([q4_bytes[offset], q4_bytes[offset + 1]]);
            let d = half::f16::from_bits(d_bits).to_f32();

            let base = block_idx * Q4_BLOCK_SIZE;
            for i in 0..16 {
                let byte = q4_bytes[offset + 2 + i];
                let lo = (byte & 0x0F) as f32 - 8.0;
                let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                output[base + i] = lo * d;
                output[base + i + 16] = hi * d;
            }
        }
        output
    }

    /// Build a GGUF v3 file with multiple Q4_0 tensors.
    fn build_multi_tensor_gguf(tensors: &[(&str, &[u64], &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        // Metadata KV
        write_gguf_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_gguf_string(&mut buf, "whisper");

        // Tensor infos (offsets are cumulative)
        let mut data_offset: u64 = 0;
        for (name, shape, q4_data) in tensors {
            write_gguf_string(&mut buf, name);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in *shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0
            buf.extend_from_slice(&data_offset.to_le_bytes());
            data_offset += q4_data.len() as u64;
        }

        // Alignment padding
        let alignment = 32;
        let padding = (alignment - (buf.len() % alignment)) % alignment;
        buf.extend(std::iter::repeat_n(0u8, padding));

        // All tensor data concatenated
        for (_, _, q4_data) in tensors {
            buf.extend_from_slice(q4_data);
        }

        buf
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// f32 matmul reference: [M, K] x [N, K]^T -> [M, N] (row-major)
    /// `b_t` is in transposed layout [N, K], matching PyTorch/GGUF convention.
    fn reference_matmul(a: &[f32], b_t: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[i * k + l] * b_t[j * k + l];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    // =========================================================================
    // Block-level dequantization tests (CPU-only, no GPU needed)
    // =========================================================================

    #[test]
    fn test_q4_block_dequant() {
        // 32 values spanning [-1, 1]
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 15.5).collect();

        let q4_bytes = quantize_f32_to_q4_0(&original);
        assert_eq!(q4_bytes.len(), Q4_BLOCK_BYTES, "One Q4_0 block = 18 bytes");

        // Verify scale
        let d_bits = u16::from_le_bytes([q4_bytes[0], q4_bytes[1]]);
        let d = half::f16::from_bits(d_bits).to_f32();
        let expected_d = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 7.0;
        assert!(
            (d - expected_d).abs() < 0.01,
            "Scale mismatch: got {}, expected {}",
            d,
            expected_d
        );

        // Dequantize and compare
        let dequantized = dequantize_q4_0_to_f32(&q4_bytes, 32);
        assert_eq!(dequantized.len(), 32);

        let mut max_diff: f32 = 0.0;
        for (a, b) in dequantized.iter().zip(original.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!(
            "Q4 block dequant max diff: {:.4e} (scale d={:.6})",
            max_diff, d
        );
        assert!(
            max_diff < 0.08,
            "Max diff {:.4e} exceeds tolerance 0.08",
            max_diff
        );
    }

    #[test]
    fn test_q4_block_edge_cases() {
        // All zeros
        let zeros = vec![0.0f32; 32];
        let q4_zeros = quantize_f32_to_q4_0(&zeros);
        let deq_zeros = dequantize_q4_0_to_f32(&q4_zeros, 32);
        for (i, v) in deq_zeros.iter().enumerate() {
            assert_eq!(*v, 0.0, "Zero block element {} should be 0.0, got {}", i, v);
        }

        // All same value
        let uniform = vec![0.5f32; 32];
        let q4_uniform = quantize_f32_to_q4_0(&uniform);
        let deq_uniform = dequantize_q4_0_to_f32(&q4_uniform, 32);
        let mut max_diff: f32 = 0.0;
        for (a, b) in deq_uniform.iter().zip(uniform.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 0.08,
            "Uniform block max diff {:.4e} exceeds tolerance",
            max_diff
        );

        // Large values
        let large: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 100.0).collect();
        let q4_large = quantize_f32_to_q4_0(&large);
        let deq_large = dequantize_q4_0_to_f32(&q4_large, 32);
        let d_large = large.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 7.0;
        let mut max_diff: f32 = 0.0;
        for (a, b) in deq_large.iter().zip(large.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!(
            "Large block max diff: {:.4e}, scale d={:.4}, bound d/2={:.4}",
            max_diff,
            d_large,
            d_large / 2.0
        );
        assert!(
            max_diff < d_large / 2.0 + 1.0,
            "Large block max diff {:.4e} exceeds bound {:.4e}",
            max_diff,
            d_large / 2.0 + 1.0
        );
    }

    // =========================================================================
    // GGUF reader tests
    // =========================================================================

    #[test]
    fn test_gguf_reader_parse_header() {
        // Create a 32x64 Q4_0 tensor
        let n_elements = 32 * 64;
        let original: Vec<f32> = (0..n_elements)
            .map(|i| ((i as f32) * 0.001 - 1.0).sin())
            .collect();
        let q4_data = quantize_f32_to_q4_0(&original);

        let gguf_bytes =
            build_multi_tensor_gguf(&[("test.weight", &[32u64, 64][..], q4_data.as_slice())]);

        let mut reader = GgufReader::from_bytes(&gguf_bytes).expect("Failed to parse GGUF");

        assert_eq!(reader.version(), 3);
        assert_eq!(reader.tensor_count(), 1);

        let tensor_info = reader.tensor_info("test.weight").expect("Tensor not found");
        assert_eq!(tensor_info.shape(), &[32, 64]);
        assert_eq!(tensor_info.dtype(), GgmlDtype::Q4_0);

        let raw = reader.tensor_data("test.weight").expect("Data not found");
        assert_eq!(raw.len(), q4_data.len());
        assert_eq!(raw, q4_data.as_slice());
    }

    #[test]
    fn test_gguf_metadata_skipping() {
        fn kv(buf: &mut Vec<u8>, key: &str, ty: u32, value: &[u8]) {
            write_gguf_string(buf, key);
            buf.extend_from_slice(&ty.to_le_bytes());
            buf.extend_from_slice(value);
        }
        fn str_val(s: &str) -> Vec<u8> {
            let mut v = (s.len() as u64).to_le_bytes().to_vec();
            v.extend_from_slice(s.as_bytes());
            v
        }

        let q4_data = quantize_f32_to_q4_0(&[0.5f32; 32]);
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&14u64.to_le_bytes()); // metadata_kv_count

        // Scalar types 0..=7 and 10..=12: one entry per GGUF type code.
        for (ty, width) in [
            (0u32, 1),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 1),
        ] {
            kv(&mut buf, "scalar", ty, &[0xAB; 4][..width]);
        }
        for ty in 10u32..=12 {
            kv(&mut buf, "scalar", ty, &[0xAB; 8]);
        }
        kv(&mut buf, "string", 8, &str_val("hello"));
        // Array of 2 u32 values
        let mut arr = 4u32.to_le_bytes().to_vec(); // elem type u32
        arr.extend_from_slice(&2u64.to_le_bytes());
        arr.extend_from_slice(&[0xCD; 8]);
        kv(&mut buf, "array", 9, &arr);
        // Nested array: 2 arrays of 2 strings each
        let mut inner = 8u32.to_le_bytes().to_vec(); // elem type string
        inner.extend_from_slice(&2u64.to_le_bytes());
        inner.extend_from_slice(&str_val("a"));
        inner.extend_from_slice(&str_val("bcd"));
        let mut outer = 9u32.to_le_bytes().to_vec(); // elem type array
        outer.extend_from_slice(&2u64.to_le_bytes());
        outer.extend_from_slice(&inner);
        outer.extend_from_slice(&inner);
        kv(&mut buf, "nested", 9, &outer);
        // Tensor info
        write_gguf_string(&mut buf, "t.weight");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0
        buf.extend_from_slice(&0u64.to_le_bytes());
        let padding = (32 - buf.len() % 32) % 32;
        buf.extend(std::iter::repeat_n(0u8, padding));
        buf.extend_from_slice(&q4_data);

        let mut reader = GgufReader::from_bytes(&buf).expect("Failed to skip metadata");
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.tensor_info("t.weight").unwrap().shape(), &[32]);
        assert_eq!(reader.tensor_data("t.weight").unwrap(), q4_data);

        // Unknown metadata value types are rejected.
        let mut bad = Vec::new();
        bad.extend_from_slice(&0x46554747u32.to_le_bytes());
        bad.extend_from_slice(&3u32.to_le_bytes());
        bad.extend_from_slice(&0u64.to_le_bytes());
        bad.extend_from_slice(&1u64.to_le_bytes());
        kv(&mut bad, "bad", 99, &[]);
        assert!(GgufReader::from_bytes(&bad).is_err());
    }

    #[test]
    fn test_gguf_multiple_tensors() {
        let data_a = quantize_f32_to_q4_0(&vec![0.1f32; 1024]);
        let data_b = quantize_f32_to_q4_0(&vec![0.2f32; 2048]);
        let data_c = quantize_f32_to_q4_0(&vec![-0.1f32; 2048]);

        let tensors: Vec<(&str, &[u64], &[u8])> = vec![
            ("weight_a", &[32, 32], &data_a),
            ("weight_b", &[64, 32], &data_b),
            ("weight_c", &[32, 64], &data_c),
        ];

        let gguf_bytes = build_multi_tensor_gguf(&tensors);
        let reader = GgufReader::from_bytes(&gguf_bytes).expect("Failed to parse GGUF");

        assert_eq!(reader.tensor_count(), 3);
        assert!(reader.tensor_info("weight_a").is_some());
        assert!(reader.tensor_info("weight_b").is_some());
        assert!(reader.tensor_info("weight_c").is_some());
        assert!(reader.tensor_info("nonexistent").is_none());
    }

    // =========================================================================
    // GPU dequantization test
    // =========================================================================

    #[test]
    fn test_q4_dequantize_gpu() {
        let device = Default::default();

        let rows = 16;
        let cols = 16;
        let n_elements = rows * cols;
        let original: Vec<f32> = (0..n_elements)
            .map(|i| ((i as f32) * 0.05 - 6.4).sin() * 0.3)
            .collect();

        let q4_bytes = quantize_f32_to_q4_0(&original);
        let expected = dequantize_q4_0_to_f32(&q4_bytes, n_elements);

        let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [rows, cols], &device)
            .expect("Failed to create Q4Tensor");
        let dequantized = q4_tensor.dequantize();

        assert_eq!(dequantized.dims(), [rows, cols]);

        let deq_data = dequantized.to_data();
        let deq_slice = deq_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in deq_slice.iter().zip(expected.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("Q4 GPU dequantize max diff: {:.4e}", max_diff);
        assert!(
            max_diff < 1e-5,
            "Max diff {:.4e} exceeds tolerance 1e-5",
            max_diff
        );
    }

    // =========================================================================
    // q4_matmul kernel tests
    // =========================================================================

    #[test]
    fn test_q4_matmul_small() {
        let device = Default::default();

        let k = 32;
        let n = 32;

        // Known weights [N, K] = [32, 32] (out_features, in_features)
        let weights_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();

        let q4_bytes = quantize_f32_to_q4_0(&weights_f32);
        let weights_deq = dequantize_q4_0_to_f32(&q4_bytes, n * k);

        // Activations [1, 1, K]
        let act_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        // CPU reference: [1, K] x [N, K]^T -> [1, N]
        let expected = reference_matmul(&act_data, &weights_deq, 1, k, n);

        // GPU path
        let activations =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(act_data, [1, 1, k]), &device);
        let q4_tensor =
            Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).expect("Failed to create Q4Tensor");
        let output = q4_matmul(activations, &q4_tensor);

        assert_eq!(output.dims(), [1, 1, n]);

        let output_data = output.to_data();
        let output_slice = output_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in output_slice.iter().zip(expected.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("Q4 matmul small max diff: {:.4e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "Max diff {:.4e} exceeds tolerance 1e-3",
            max_diff
        );
    }

    #[test]
    fn test_q4_matmul_shapes() {
        let device = Default::default();

        // Whisper-specific shapes: d_model=1280, n_heads=20, head_dim=64, ffn=5120
        // (batch, seq, K, N, tolerance, description)
        let shapes: &[(usize, usize, usize, usize, f32, &str)] = &[
            (1, 1, 128, 64, 1e-2, "small projection"),
            (1, 1, 1280, 1280, 1e-2, "decoder attn qkv (M=1)"),
            (1, 10, 1280, 1280, 1e-2, "encoder attn qkv"),
            (4, 10, 128, 64, 1e-3, "batched projection"),
            (1, 32, 1280, 1280, 1e-2, "encoder projection"),
            (1, 1, 1280, 5120, 1e-2, "decoder FFN fc1"),
            (1, 1, 5120, 1280, 1e-2, "decoder FFN fc2"),
        ];

        for &(batch, seq, k, n, tol, desc) in shapes {
            println!(
                "Testing shape: {} [{}x{}x{}] x [{}x{}]^T",
                desc, batch, seq, k, n, k
            );

            let act_data: Vec<f32> = (0..batch * seq * k)
                .map(|i| ((i as f32) * 0.001).sin() * 0.1)
                .collect();
            // Weights in [N, K] layout (out_features, in_features)
            let weight_data: Vec<f32> = (0..n * k)
                .map(|i| ((i as f32) * 0.0007).cos() * 0.05)
                .collect();

            let q4_bytes = quantize_f32_to_q4_0(&weight_data);
            let weight_deq = dequantize_q4_0_to_f32(&q4_bytes, n * k);

            // Reference: Burn f32 matmul -- dequantized weights are [N, K],
            // transpose to [K, N] for standard matmul
            let act_tensor = Tensor::<TestBackend, 3>::from_data(
                TensorData::new(act_data, [batch, seq, k]),
                &device,
            );
            let weight_deq_tensor =
                Tensor::<TestBackend, 2>::from_data(TensorData::new(weight_deq, [n, k]), &device);
            let expected = act_tensor
                .clone()
                .matmul(weight_deq_tensor.transpose().unsqueeze::<3>());

            // Q4 GPU path
            let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device)
                .expect("Failed to create Q4Tensor");
            let output = q4_matmul(act_tensor, &q4_tensor);

            assert_eq!(output.dims(), [batch, seq, n]);

            let output_data = output.to_data();
            let expected_data = expected.to_data();
            let out_slice = output_data.as_slice::<f32>().unwrap();
            let exp_slice = expected_data.as_slice::<f32>().unwrap();

            let mut max_diff: f32 = 0.0;
            for (a, b) in out_slice.iter().zip(exp_slice.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            println!("  Q4 matmul {} max diff: {:.4e}", desc, max_diff);
            assert!(
                max_diff < tol,
                "Q4 matmul {} max diff {:.4e} exceeds tolerance {:.4e}",
                desc,
                max_diff,
                tol
            );
        }
    }

    // =========================================================================
    // Q4Linear tests
    // =========================================================================

    #[test]
    fn test_q4_linear_forward_shape() {
        let device = Default::default();

        let in_features = 128;
        let out_features = 64;

        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.001).sin() * 0.1)
            .collect();
        let q4_bytes = quantize_f32_to_q4_0(&weight_data);

        let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [out_features, in_features], &device)
            .expect("Failed to create Q4Tensor");
        let linear = Q4Linear::new(q4_tensor, None);

        let input = Tensor::<TestBackend, 3>::zeros([2, 5, in_features], &device);
        let output = linear.forward(input);

        assert_eq!(output.dims(), [2, 5, out_features]);
    }

    #[test]
    fn test_q4_linear_forward_with_bias() {
        let device = Default::default();

        let in_features = 64;
        let out_features = 32;

        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.001).sin() * 0.1)
            .collect();
        let q4_bytes = quantize_f32_to_q4_0(&weight_data);
        let weight_deq = dequantize_q4_0_to_f32(&q4_bytes, out_features * in_features);

        // Create bias
        let bias_data: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.01).collect();
        let bias = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(bias_data.clone(), [out_features]),
            &device,
        );

        let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [out_features, in_features], &device)
            .expect("Failed to create Q4Tensor");
        let linear = Q4Linear::new(q4_tensor, Some(bias));

        // Input
        let act_data: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.1).collect();
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(act_data.clone(), [1, 1, in_features]),
            &device,
        );
        let output = linear.forward(input);

        assert_eq!(output.dims(), [1, 1, out_features]);

        // Reference: matmul + bias
        let expected_matmul =
            reference_matmul(&act_data, &weight_deq, 1, in_features, out_features);
        let expected: Vec<f32> = expected_matmul
            .iter()
            .zip(bias_data.iter())
            .map(|(m, b)| m + b)
            .collect();

        let output_data = output.to_data();
        let output_slice = output_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in output_slice.iter().zip(expected.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("Q4Linear with bias max diff: {:.4e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "Max diff {:.4e} exceeds tolerance 1e-3",
            max_diff
        );
    }

    // =========================================================================
    // Q4FFN tests
    // =========================================================================

    #[test]
    fn test_q4_ffn_forward_shape() {
        use crate::model::layers::Q4FFN;

        let device = Default::default();
        let d_model = 64;
        let ffn_dim = 256; // Whisper uses 4x multiplier

        // fc1: [ffn_dim, d_model], fc2: [d_model, ffn_dim]
        let make_q4_linear = |rows: usize, cols: usize| -> Q4Linear {
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32) * 0.001).sin() * 0.05)
                .collect();
            let bytes = quantize_f32_to_q4_0(&data);
            let tensor = Q4Tensor::from_q4_bytes(&bytes, [rows, cols], &device)
                .expect("Failed to create Q4Tensor");
            Q4Linear::new(tensor, None)
        };

        let fc1 = make_q4_linear(ffn_dim, d_model);
        let fc2 = make_q4_linear(d_model, ffn_dim);
        let ffn = Q4FFN::new(fc1, fc2);

        let input = Tensor::<TestBackend, 3>::zeros([1, 4, d_model], &device);
        let output = ffn.forward(input);

        assert_eq!(output.dims(), [1, 4, d_model]);
    }

    // =========================================================================
    // Q4 roundtrip test (quantize -> upload -> dequantize -> compare)
    // =========================================================================

    #[test]
    fn test_q4_roundtrip_small() {
        let device = Default::default();

        let rows = 32;
        let cols = 64;
        let n_elements = rows * cols;

        let original: Vec<f32> = (0..n_elements)
            .map(|i| ((i as f32) * 0.003 - 3.0).sin() * 0.5)
            .collect();

        let q4_bytes = quantize_f32_to_q4_0(&original);
        let cpu_deq = dequantize_q4_0_to_f32(&q4_bytes, n_elements);

        let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [rows, cols], &device)
            .expect("Failed to create Q4Tensor");
        let gpu_deq = q4_tensor.dequantize();

        let gpu_data = gpu_deq.to_data();
        let gpu_slice = gpu_data.as_slice::<f32>().unwrap();

        // CPU and GPU dequantization should match exactly
        let mut max_diff: f32 = 0.0;
        for (a, b) in gpu_slice.iter().zip(cpu_deq.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("Q4 roundtrip CPU vs GPU max diff: {:.4e}", max_diff);
        assert!(
            max_diff < 1e-5,
            "CPU/GPU dequant mismatch: max diff {:.4e}",
            max_diff
        );

        // Check that dequantized values are reasonably close to originals
        let mut max_quant_err: f32 = 0.0;
        for (a, b) in gpu_slice.iter().zip(original.iter()) {
            max_quant_err = max_quant_err.max((a - b).abs());
        }
        println!("Q4 quantization error max: {:.4e}", max_quant_err);
        assert!(
            max_quant_err < 0.1,
            "Quantization error too large: {:.4e}",
            max_quant_err
        );
    }

    // =========================================================================
    // Q4CrossAttention cache initialization
    // =========================================================================

    #[test]
    fn test_q4_cross_attention_init_cache_matches_forward() {
        use crate::model::attention::Q4CrossAttention;

        let device = Default::default();
        let (d, q_len, enc_len) = (64usize, 2usize, 3usize);

        // Tiny layer (1 head, head_dim 64); each projection gets different weights.
        let linear = |seed: f32| {
            let w: Vec<f32> = (0..d * d)
                .map(|i| ((i as f32) * 0.013 + seed).sin() * 0.1)
                .collect();
            let q4 = Q4Tensor::from_q4_bytes(&quantize_f32_to_q4_0(&w), [d, d], &device)
                .expect("Failed to create Q4Tensor");
            Q4Linear::new(q4, None)
        };
        let attn = Q4CrossAttention::new(linear(0.1), linear(0.7), linear(1.3), linear(1.9), 1);

        let input = |len: usize, scale: f32| {
            let data: Vec<f32> = (0..len * d).map(|i| ((i as f32) * scale).cos()).collect();
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [1, len, d]), &device)
        };
        let x = input(q_len, 0.05);
        let encoder = input(enc_len, 0.03);

        let expected = attn.forward(x.clone(), encoder.clone());
        let (out, k, v) = attn.forward_init_cache(x.clone(), &encoder);
        assert_eq!(out.dims(), [1, q_len, d]);
        assert_eq!(k.dims(), [1, enc_len, d]);
        assert_eq!(v.dims(), [1, enc_len, d]);

        let max_diff = |a: Tensor<TestBackend, 3>, b: Tensor<TestBackend, 3>| -> f32 {
            let (a, b) = (a.to_data(), b.to_data());
            let (a, b) = (a.as_slice::<f32>().unwrap(), b.as_slice::<f32>().unwrap());
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max)
        };
        let init_diff = max_diff(out, expected.clone());
        assert!(
            init_diff < 1e-4,
            "init_cache vs forward diff {:.4e}",
            init_diff
        );

        // Reusing the returned K/V must reproduce the same output.
        let (reused, _, _) = attn.forward_with_cache(x, &encoder, Some(k), Some(v));
        let reuse_diff = max_diff(reused, expected);
        assert!(
            reuse_diff < 1e-4,
            "cached vs forward diff {:.4e}",
            reuse_diff
        );
    }
}
