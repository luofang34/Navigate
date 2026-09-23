//! Decode dense detector logits into model coordinates and normalized features.
use crate::{InferenceError, features::Features, native_runtime::runtime, preprocessing};
use image::GrayImage;
use ort::{session::Session, value::Tensor};

pub(crate) fn extract(
    session: &mut Session,
    image: &GrayImage,
    size: [usize; 2],
    limit: usize,
) -> Result<Features, InferenceError> {
    let [width, height] = detector_size(session, image)?;
    let input = preprocessing::prepare(image, width, height, 1)?;
    let tensor = Tensor::from_array(([1, 1, height, width], input.data.clone()))
        .map_err(|e| runtime("build SuperPoint input", e))?;
    let output = session
        .run(ort::inputs!["image"=>tensor])
        .map_err(|e| runtime("run SuperPoint", e))?;
    let get = |name| {
        output
            .get(name)
            .ok_or_else(|| InferenceError::Invalid(format!("missing SuperPoint {name}")))
    };
    let (shape, logits) = get("scores")?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("read SuperPoint logits", e))?;
    let (dshape, descriptors) = get("descriptors")?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("read SuperPoint descriptors", e))?;
    let gw = width / 8;
    let gh = height / 8;
    let cells = gw * gh;
    if shape.as_ref() != [1, 65, gh as i64, gw as i64]
        || dshape.as_ref() != [1, 256, gh as i64, gw as i64]
        || !logits.iter().all(|v| v.is_finite())
    {
        return Err(InferenceError::Invalid(
            "incompatible dense SuperPoint outputs".into(),
        ));
    }
    let heat = heatmap(logits, width, height);
    let heat = suppress(&heat, width, height);
    let mut selected: Vec<_> = heat
        .iter()
        .enumerate()
        .filter_map(|(i, score)| {
            let x = i % width;
            let y = i / width;
            let p = input.pixel([x as f64, y as f64]);
            (*score > 0.005 && preprocessing::inside(p, image)).then_some((x, y, p, *score))
        })
        .collect();
    selected.sort_by(|a, b| b.3.total_cmp(&a.3));
    let mut features = Features::empty(256);
    for (x, y, p, score) in selected.into_iter().take(limit) {
        let sx = (x as f64 - 3.5) / (width as f64 - 4.5) * (gw - 1) as f64;
        let sy = (y as f64 - 3.5) / (height as f64 - 4.5) * (gh - 1) as f64;
        let mut d = vec![0.0; 256];
        for (c, value) in d.iter_mut().enumerate() {
            *value = sample(&descriptors[c * cells..(c + 1) * cells], gw, gh, sx, sy);
        }
        features.push(
            p,
            normalized_pixel([x as f32, y as f32], [width, height], size),
            score,
            &d,
        )?;
    }
    Ok(features)
}
fn heatmap(logits: &[f32], width: usize, height: usize) -> Vec<f32> {
    let cells = width * height / 64;
    let mut heat = vec![0.0; width * height];
    for cell in 0..cells {
        let maximum = (0..65)
            .map(|c| logits[c * cells + cell])
            .fold(f32::NEG_INFINITY, f32::max);
        let sum = (0..65)
            .map(|c| (logits[c * cells + cell] - maximum).exp())
            .sum::<f32>();
        for c in 0..64 {
            let x = cell % (width / 8) * 8 + c % 8;
            let y = cell / (width / 8) * 8 + c / 8;
            heat[y * width + x] = (logits[c * cells + cell] - maximum).exp() / sum;
        }
    }
    heat
}
fn maximum(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0_f32;
            for yy in y.saturating_sub(4)..=(y + 4).min(height - 1) {
                for xx in x.saturating_sub(4)..=(x + 4).min(width - 1) {
                    value = value.max(values[yy * width + xx]);
                }
            }
            output[y * width + x] = value;
        }
    }
    output
}
fn suppress(heat: &[f32], width: usize, height: usize) -> Vec<f32> {
    let max = maximum(heat, width, height);
    let mut keep: Vec<_> = heat.iter().zip(max).map(|(v, m)| *v == m).collect();
    for _ in 0..2 {
        let mask: Vec<_> = keep.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect();
        let suppressed = maximum(&mask, width, height);
        let remaining: Vec<_> = heat
            .iter()
            .zip(&suppressed)
            .map(|(v, s)| if *s > 0.0 { 0.0 } else { *v })
            .collect();
        let maximum = maximum(&remaining, width, height);
        for i in 0..keep.len() {
            keep[i] |= suppressed[i] == 0.0 && remaining[i] == maximum[i];
        }
    }
    heat.iter()
        .zip(keep)
        .map(|(v, k)| if k { *v } else { 0.0 })
        .collect()
}
fn sample(data: &[f32], width: usize, height: usize, x: f64, y: f64) -> f32 {
    let left = x.floor();
    let top = y.floor();
    let fx = x - left;
    let fy = y - top;
    let mut value = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let xx = left + f64::from(dx);
            let yy = top + f64::from(dy);
            if xx >= 0.0 && yy >= 0.0 && xx < width as f64 && yy < height as f64 {
                value += f64::from(data[yy as usize * width + xx as usize])
                    * if dx == 0 { 1.0 - fx } else { fx }
                    * if dy == 0 { 1.0 - fy } else { fy };
            }
        }
    }
    value as f32
}

fn detector_size(session: &Session, image: &GrayImage) -> Result<[usize; 2], InferenceError> {
    let input = session
        .inputs()
        .first()
        .ok_or_else(|| InferenceError::Invalid("missing detector input".into()))?;
    let ort::value::ValueType::Tensor { shape, .. } = input.dtype() else {
        return Err(InferenceError::Invalid(
            "detector input is not a tensor".into(),
        ));
    };
    if shape.len() != 4 {
        return Err(InferenceError::Invalid(
            "detector input must be NCHW".into(),
        ));
    }
    let side = |declared: i64, actual: u32| {
        if declared > 0 {
            declared as usize
        } else {
            (actual as usize).div_ceil(8) * 8
        }
    };
    let size = [
        side(shape[3], image.width()),
        side(shape[2], image.height()),
    ];
    if size.iter().any(|v| *v < 16 || *v > 1920 || v % 8 != 0) {
        return Err(InferenceError::Invalid(
            "unsupported detector dimensions".into(),
        ));
    }
    Ok(size)
}
fn normalized_pixel(point: [f32; 2], source: [usize; 2], target: [usize; 2]) -> [f32; 2] {
    let ratio = target[0].max(target[1]) as f32 / source[0].max(source[1]) as f32;
    [
        (point[0] - source[0] as f32 / 2.0) * ratio + target[0] as f32 / 2.0,
        (point[1] - source[1] as f32 / 2.0) * ratio + target[1] as f32 / 2.0,
    ]
}
#[cfg(test)]
mod tests;
