use super::*;
use burn::tensor::{Device, TensorData};

fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn tensor<const D: usize>(data: &[f32], shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(data.to_vec(), shape), device)
}

fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().iter::<f32>().collect()
}

/// Host `a · b + bias` for one batch, with the stored layouts of [`Layout`].
fn host_product(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    [m, n, k]: [usize; 3],
    layout: Layout,
) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = bias[col];
            for i in 0..k {
                let av = if layout.trans_a {
                    a[i * m + row]
                } else {
                    a[row * k + i]
                };
                let bv = if layout.trans_b {
                    b[col * k + i]
                } else {
                    b[i * n + col]
                };
                sum += av * bv;
            }
            out[row * n + col] = if layout.relu { sum.max(0.0) } else { sum };
        }
    }
    out
}

fn host_convolution(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    [cin, h, w]: [usize; 3],
    [cout, kernel, stride]: [usize; 3],
) -> Vec<f32> {
    let pad = kernel / 2;
    let (ho, wo) = (
        (h + 2 * pad - kernel) / stride + 1,
        (w + 2 * pad - kernel) / stride + 1,
    );
    let mut out = vec![0.0; cout * ho * wo];
    for co in 0..cout {
        for y in 0..ho {
            for x in 0..wo {
                let mut sum = bias[co];
                for ci in 0..cin {
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let iy = (y * stride + ky) as isize - pad as isize;
                            let ix = (x * stride + kx) as isize - pad as isize;
                            if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                sum += weight[((co * cin + ci) * kernel + ky) * kernel + kx]
                                    * input[(ci * h + iy as usize) * w + ix as usize];
                            }
                        }
                    }
                }
                out[(co * ho + y) * wo + x] = sum;
            }
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4 * (1.0 + e.abs()),
            "{what}[{i}]: {a} against {e}"
        );
    }
}

const LAYOUTS: [Layout; 4] = [
    Layout {
        trans_a: false,
        trans_b: false,
        relu: false,
    },
    Layout {
        trans_a: true,
        trans_b: false,
        relu: false,
    },
    Layout {
        trans_a: false,
        trans_b: true,
        relu: true,
    },
    Layout {
        trans_a: true,
        trans_b: true,
        relu: true,
    },
];

/// Check `product` on `device` against the host for each layout, with a broadcast `b`.
fn check_product(device: &Device, [m, n, k]: [usize; 3], batch: usize) {
    for layout in LAYOUTS {
        let a = values(batch * m * k, 1);
        let b = values(k * n, 2);
        let bias = values(n, 3);
        let a_shape = if layout.trans_a {
            [batch, k, m]
        } else {
            [batch, m, k]
        };
        let b_shape = if layout.trans_b { [1, n, k] } else { [1, k, n] };
        let out = read(product(
            tensor(&a, a_shape, device),
            tensor(&b, b_shape, device),
            tensor(&bias, [n], device),
            layout,
        ));
        let mut expected = Vec::new();
        for z in 0..batch {
            let batch_a = &a[z * m * k..(z + 1) * m * k];
            expected.extend(host_product(batch_a, &b, &bias, [m, n, k], layout));
        }
        assert_close(&out, &expected, &format!("product {layout:?} {m}x{n}x{k}"));
    }
}

fn check_convolution(device: &Device, [cin, h, w]: [usize; 3], [cout, kernel, stride]: [usize; 3]) {
    let input = values(2 * cin * h * w, 4);
    let weight = values(cout * cin * kernel * kernel, 5);
    let bias = values(cout, 6);
    let out = read(convolution(
        tensor(&input, [2, cin, h, w], device),
        tensor(&weight, [cout, cin, kernel, kernel], device),
        tensor(&bias, [cout], device),
        stride,
        false,
    ));
    let plane = cin * h * w;
    let sizes = ([cin, h, w], [cout, kernel, stride]);
    let mut expected = host_convolution(&input[..plane], &weight, &bias, sizes.0, sizes.1);
    expected.extend(host_convolution(
        &input[plane..],
        &weight,
        &bias,
        sizes.0,
        sizes.1,
    ));
    assert_close(
        &out,
        &expected,
        &format!("convolution {kernel}x{kernel}/{stride}"),
    );
}

#[test]
fn the_cpu_product_follows_the_layout_contract() {
    check_product(&Device::flex(), [5, 7, 3], 2);
}

#[test]
fn the_cpu_convolution_pads_by_half_the_kernel() {
    for (kernel, stride) in [(1, 1), (3, 1), (3, 2), (7, 2)] {
        check_convolution(&Device::flex(), [3, 9, 11], [4, kernel, stride]);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a GPU; the Models workflow runs it on a software Vulkan device"]
fn the_gpu_product_matches_the_contract_across_tile_edges() {
    let device = Device::wgpu(Default::default());
    check_product(&device, [130, 67, 33], 3);
    check_product(&device, [1, 257, 129], 1);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a GPU; the Models workflow runs it on a software Vulkan device"]
fn the_gpu_convolution_matches_the_contract_across_tile_edges() {
    let device = Device::wgpu(Default::default());
    for (kernel, stride) in [(1, 1), (3, 1), (3, 2), (7, 2)] {
        check_convolution(&device, [5, 23, 19], [131, kernel, stride]);
    }
}
