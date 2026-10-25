//! Run one fixed-feature masked correlation comparison.
mod beam;
mod score;
use score::{Correlator, Grid, Score};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
type Failure = Box<dyn std::error::Error>;
fn load_blocking(path: &Path, border: usize) -> Result<Grid, Failure> {
    let meta: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let bytes = std::fs::read(meta["tensor"].as_str().ok_or("tensor path")?)?;
    let shape = meta["shape_chw"].as_array().ok_or("shape")?;
    let dims: Vec<_> = shape
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or("dimension"))
        .collect::<Result<_, _>>()?;
    let [c, h, w] = dims.as_slice() else {
        return Err("shape rank".into());
    };
    if bytes.len() != c * h * w * 4 {
        return Err("tensor length".into());
    }
    let raw: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut values = vec![0.0; raw.len()];
    let mut mask = vec![false; w * h];
    for y in 0..*h {
        for x in 0..*w {
            let i = y * w + x;
            mask[i] = x >= border && y >= border && x + border < *w && y + border < *h;
            if !mask[i] {
                continue;
            }
            let norm = (0..*c)
                .map(|ch| f64::from(raw[ch * w * h + i]).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            for ch in 0..*c {
                values[i * c + ch] = (f64::from(raw[ch * w * h + i]) / norm) as f32;
            }
        }
    }
    Ok(Grid::new(*w, *h, *c, values, mask)?)
}
fn crop(g: &Grid, value: &Value) -> Result<Grid, Failure> {
    let v: Vec<usize> = serde_json::from_value(value.clone())?;
    let [x, y, w, h] = v.as_slice() else {
        return Err("crop must be x,y,w,h".into());
    };
    Ok(g.crop(*x, *y, *w, *h)?)
}
fn order(scores: &[Score]) -> Vec<usize> {
    let mut indices: Vec<_> = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| s.value.is_some())
        .map(|(i, _)| i)
        .collect();
    indices.sort_by(|a, b| {
        scores[*b]
            .value
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&scores[*a].value.unwrap_or(f64::NEG_INFINITY))
            .then(a.cmp(b))
    });
    indices
}
fn errors(a: &[Score], b: &[Score]) -> Value {
    let differences: Vec<_> = a
        .iter()
        .zip(b)
        .filter_map(|(x, y)| {
            x.value
                .zip(y.value)
                .map(|(x, y)| ((x - y).abs(), (x - y).abs() / x.abs().max(1e-6)))
        })
        .collect();
    json!({"max_abs":differences.iter().map(|d|d.0).fold(0.0,f64::max),"max_relative_with_1e_6_floor":differences.iter().map(|d|d.1).fold(0.0,f64::max),"support_exact":a.iter().zip(b).all(|(x,y)|x.count==y.count&&x.value.is_some()==y.value.is_some()),"top10_equal":order(a).into_iter().take(10).eq(order(b).into_iter().take(10))})
}
fn measure(
    q: &Grid,
    m: &Grid,
    center: [i32; 2],
    radius: i32,
    repeats: usize,
    min: usize,
) -> Result<Value, Failure> {
    let lags: Vec<_> = (-radius..=radius)
        .flat_map(|dy| (-radius..=radius).map(move |dx| [center[0] + dx, center[1] + dy]))
        .collect();
    let started = Instant::now();
    let mut fft = Correlator::new(q, m)?;
    let setup_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let first = fft.score(q, &lags, min)?;
    let first_ms = started.elapsed().as_secs_f64() * 1000.0;
    let oracle = direct_score(q, m, &lags, min, true);
    let d32 = direct_score(q, m, &lags, min, false);
    let mut direct_ms = Vec::new();
    let mut fft_ms = Vec::new();
    for i in 0..repeats {
        for kind in if i % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let start = Instant::now();
            if kind {
                let scores = fft.score(q, &lags, min)?;
                std::hint::black_box(order(&scores));
                fft_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            } else {
                let scores = direct_score(q, m, &lags, min, false);
                std::hint::black_box(order(&scores));
                direct_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    let rank = order(&oracle);
    let boundary = rank.first().is_some_and(|i| {
        (lags[*i][0] - center[0]).abs() == radius || (lags[*i][1] - center[1]).abs() == radius
    });
    let top: Vec<_> = rank
        .iter()
        .take(10)
        .map(|i| json!({"lag":lags[*i],"score":oracle[*i].value,"overlap":oracle[*i].count}))
        .collect();
    Ok(
        json!({"best_on_search_boundary":boundary,"radius_cells":radius,"candidate_count":lags.len(),"minimum_overlap_cells":min,"fft_setup_map_ms":setup_ms,"fft_first_query_ms":first_ms,"fft_cached_map_bytes":fft.cached_bytes(),"direct_cached_map_bytes":m.values.len()*4+m.mask.len(),"direct_ms":direct_ms,"fft_ms":fft_ms,"fft_error":errors(&oracle,&first),"direct_error":errors(&oracle,&d32),"oracle_top10":top}),
    )
}
fn direct_score(q: &Grid, m: &Grid, lags: &[[i32; 2]], min: usize, double: bool) -> Vec<Score> {
    score::direct(q, m, lags, min, double)
}
fn main() -> Result<(), Failure> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [input, output] = args.as_slice() else {
        return Err("usage: dense-correlation-probe CONFIG.json OUTPUT.json".into());
    };
    let config: Value = serde_json::from_slice(&std::fs::read(input)?)?;
    let start = Instant::now();
    let border = config["border_cells"].as_u64().ok_or("border")? as usize;
    let q = load_blocking(
        Path::new(config["query_export"].as_str().ok_or("query")?),
        border,
    )?;
    let m = load_blocking(
        Path::new(config["map_export"].as_str().ok_or("map")?),
        border,
    )?;
    let q = crop(&q, &config["query_crop"])?;
    let m = crop(&m, &config["map_crop"])?;
    if q.c != m.c {
        return Err("channel mismatch".into());
    }
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
    let center: [i32; 2] = serde_json::from_value(config["center"].clone())?;
    let radii: Vec<i32> = serde_json::from_value(config["radii"].clone())?;
    let repeats = config["repeats"].as_u64().ok_or("repeats")? as usize;
    if repeats == 0 || repeats > 100 || radii.iter().any(|r| *r < 0 || *r > 128) {
        return Err("budget".into());
    }
    let fraction = config["minimum_overlap_fraction"]
        .as_f64()
        .ok_or("overlap")?;
    if !(0.0..=1.0).contains(&fraction) {
        return Err("overlap range".into());
    }
    let min = (q.mask.iter().filter(|v| **v).count() as f64 * fraction).ceil() as usize;
    let beam_report = beam::compare(
        &q,
        &m,
        center,
        *radii.iter().max().ok_or("empty radii")?,
        min,
    )?;
    let rows = radii
        .into_iter()
        .map(|r| measure(&q, &m, center, r, repeats, min))
        .collect::<Result<Vec<_>, _>>()?;
    let result = json!({"config":config,"timing_scope":"hot query scores and full candidate ranking; input preparation and map FFT setup separate", "direct_kernel":"four independent f32 accumulation lanes", "score":"masked mean dot of per-cell L2-normalized f32 features; joint binary mask; no confidence weighting","scope":"fixed tensor translation diagnostic; no geographic truth or pose acceptance","prepare_io_normalize_crop_ms":prepare_ms,"query_chw":[q.c,q.h,q.w],"map_chw":[m.c,m.h,m.w],"rows":rows,"beam":beam_report});
    serde_json::to_writer_pretty(std::fs::File::create(output)?, &result)?;
    tracing::info!(?output, "Saved correlation comparison");
    Ok(())
}
