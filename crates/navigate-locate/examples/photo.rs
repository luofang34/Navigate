//! Locate one photo natively with LoFTR-DS on the GPU, Esri World Imagery and
//! Terrarium elevation tiles. Tiles are kept in a cache directory, so a run with the
//! same cache uses the same map data.
//!
//! ```sh
//! cargo run --release -p navigate-locate --example photo -- \
//!     WEIGHTS PHOTO FOCAL_PX TARGET_LAT TARGET_LON HEADING_DEG CACHE_DIR REPORT.json
//! ```
use burn::tensor::Device;
use navigate_locate::{CachedHttpSource, LocateRequest, LookAt, Outcome, SceneRequest, locate};
use navigate_match::{Loftr, LoftrMatcher, Weights};
use navigate_visual::CameraModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [weights, photo, focal, lat, lon, heading, cache, report] = <[String; 8]>::try_from(args)
        .map_err(|_| "usage: photo WEIGHTS PHOTO FOCAL_PX TARGET_LAT TARGET_LON HEADING_DEG CACHE_DIR REPORT")?;
    let image = image::open(&photo)?.to_luma8();
    let (width, height) = image.dimensions();
    let focal: f64 = focal.parse()?;
    let camera = CameraModel {
        width,
        height,
        fx: focal,
        fy: focal,
        cx: (f64::from(width) - 1.0) / 2.0,
        cy: (f64::from(height) - 1.0) / 2.0,
    };
    let device = Device::wgpu(Default::default());
    let model = Loftr::load(
        Weights::from_safetensors(&std::fs::read(&weights)?)?,
        &device,
    )?;
    let request = LocateRequest {
        image,
        camera,
        target: [lat.parse()?, lon.parse()?],
        look_at: LookAt::standard(heading.parse()?),
        scene: SceneRequest {
            anchor: [0.0, 0.0],
            radius_m: 800.0,
            imagery_zoom: 18,
            terrain_zoom: 14,
            imagery_tile_pixels: 256,
            max_bytes: 512 << 20,
            concurrency: 16,
        },
        position_radius_m: 400.0,
        attitude_radius_rad: 0.9,
    };
    let started = std::time::Instant::now();
    let source = CachedHttpSource::new(cache);
    let outcome = pollster::block_on(locate(
        &source,
        LoftrMatcher::new(model, "local"),
        request,
        |p| {
            tracing::info!(?p, "progress");
        },
    ))?;
    let elapsed_s = started.elapsed().as_secs_f64();
    let json = match outcome {
        Outcome::Found(located) => serde_json::json!({
            "found": true,
            "elapsed_s": elapsed_s,
            "geodetic": located.geodetic,
            "angles_deg": located.angles_deg,
            "refined": located.refined,
            "inliers": located.estimate.quality.inliers,
            "rms_px": located.estimate.quality.reprojection_rms_px,
        }),
        Outcome::NotFound {
            candidates,
            best_matches,
        } => serde_json::json!({
            "found": false, "elapsed_s": elapsed_s, "candidates": candidates, "best_matches": best_matches,
        }),
    };
    tracing::info!(%json, "result");
    std::fs::write(report, serde_json::to_vec_pretty(&json)?)?;
    Ok(())
}
