use super::*;
use ort::{
    session::{Session, builder::AutoDevicePolicy},
    value::TensorRef,
};

#[test]
#[ignore = "requires NAVIGATE_TEST_ORT, NAVIGATE_XFEAT_MODEL and NAVIGATE_XFEAT_INPUT; run alone"]
fn dense_recipe_matches_original_model_intermediates() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = std::env::var_os("NAVIGATE_TEST_ORT").ok_or("set NAVIGATE_TEST_ORT")?;
    let model = std::env::var_os("NAVIGATE_XFEAT_MODEL").ok_or("set NAVIGATE_XFEAT_MODEL")?;
    let input = std::env::var_os("NAVIGATE_XFEAT_INPUT").ok_or("set NAVIGATE_XFEAT_INPUT")?;
    ort::init_from(runtime)?.commit();
    let bytes = std::fs::read(model)?;
    let prepared = prepare(&bytes)?;
    let mut original = ModelProto::decode(bytes.as_slice())?;
    let graph = original.graph.as_mut().ok_or("model graph")?;
    // Expose the source activations directly; leave every original operation and weight intact.
    graph.output = OUTPUTS
        .iter()
        .map(|(name, _, channels)| graph::value(name, 1, &[1, *channels, 72, 100]))
        .collect();
    let baseline = original.encode_to_vec();
    let load = |bytes: &[u8]| -> ort::Result<Session> {
        Session::builder()?
            .with_auto_device(AutoDevicePolicy::Default)?
            .with_intra_threads(4)?
            .commit_from_memory(bytes)
    };
    let mut old = load(&baseline)?;
    let mut new = load(&prepared)?;
    let raw = std::fs::read(input)?;
    assert_eq!(
        raw.len(),
        1_440_000 * 4,
        "input must be float32 LE NCHW 1×3×600×800"
    );
    let values: Vec<_> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()));
    let tensor = || TensorRef::from_array_view(([1, 3, 600, 800], values.as_slice()));
    let expected = old.run(ort::inputs!["images" => tensor()?])?;
    let actual = new.run(ort::inputs!["images" => tensor()?])?;
    for (source, name, channels) in OUTPUTS {
        let (left_shape, left) = expected
            .get(source)
            .ok_or("missing source output")?
            .try_extract_tensor::<f32>()?;
        let (right_shape, right) = actual
            .get(name)
            .ok_or("missing prepared output")?
            .try_extract_tensor::<f32>()?;
        assert_eq!(left_shape.as_ref(), [1, channels, 72, 100]);
        assert_eq!(left_shape, right_shape);
        for (index, (a, b)) in left.iter().zip(right).enumerate() {
            assert!(a.is_finite() && b.is_finite());
            assert!(
                (a - b).abs() <= 3e-5 + a.abs() * 1e-4,
                "{name}[{index}]: {a} != {b}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires NAVIGATE_TEST_ORT and NAVIGATE_XFEAT_MODEL; run alone"]
fn grayscale_recipe_matches_repeated_grayscale_activations()
-> Result<(), Box<dyn std::error::Error>> {
    let runtime = std::env::var_os("NAVIGATE_TEST_ORT").ok_or("set NAVIGATE_TEST_ORT")?;
    let model = std::env::var_os("NAVIGATE_XFEAT_MODEL").ok_or("set NAVIGATE_XFEAT_MODEL")?;
    ort::init_from(runtime)?.commit();
    let bytes = std::fs::read(model)?;
    let mut rgb = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_memory(&prepare_channels(&bytes, 3)?)?;
    let mut gray = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_memory(&prepare_channels(&bytes, 1)?)?;
    let values: Vec<f32> = (0..480_000).map(|i| (i % 251) as f32 / 250.0).collect();
    let repeated = values.repeat(3);
    let expected = rgb.run(ort::inputs!["images" => TensorRef::from_array_view(([1, 3, 600, 800], repeated.as_slice()))?])?;
    let actual = gray.run(ort::inputs!["images" => TensorRef::from_array_view(([1, 1, 600, 800], values.as_slice()))?])?;
    for (_, name, _) in OUTPUTS {
        let (left_shape, left) = expected
            .get(name)
            .ok_or("missing RGB output")?
            .try_extract_tensor::<f32>()?;
        let (right_shape, right) = actual
            .get(name)
            .ok_or("missing gray output")?
            .try_extract_tensor::<f32>()?;
        assert_eq!(left_shape, right_shape);
        assert_eq!(left, right, "{name}");
    }
    Ok(())
}
