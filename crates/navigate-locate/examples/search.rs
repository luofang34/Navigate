//! Search a region for the camera of one photo with a place index, natively.
//!
//! ```sh
//! cargo run --release -p navigate-locate --features http --example search -- \
//!     LOFTR.safetensors DINOV2.safetensors INDEX_DIR PHOTO FOCAL_PX LAT LON RADIUS_M \
//!     HEIGHT_M HEIGHT_SIGMA_M CACHE_DIR [TRUE_LAT TRUE_LON]
//! ```
use burn::tensor::Device;
use navigate_locate::{
    CachedHttpSource, IndexSource, PlaceEncoder, SearchOutcome, SearchSources, search,
};
use navigate_match::{Dinov2, Loftr, LoftrMatcher, Weights};
use navigate_place::{Canvas, GRID, HeightPrior, IndexManifest, Region, SearchRequest, Tokens};
use std::path::PathBuf;

type Error = Box<dyn std::error::Error>;

struct Burn(Dinov2);

impl PlaceEncoder for Burn {
    async fn tokens(&mut self, canvases: &[Canvas]) -> Result<Vec<Tokens>, String> {
        let n = self.0.input_px();
        let input: Vec<f32> = canvases.iter().flat_map(|c| c.model_input(n)).collect();
        let values = self
            .0
            .tokens(&input, canvases.len())
            .await
            .map_err(|e| e.to_string())?;
        let each = (1 + GRID * GRID) * navigate_match::DINOV2_WIDTH;
        Ok(values
            .chunks(each)
            .map(|v| Tokens {
                width: navigate_match::DINOV2_WIDTH,
                values: v.to_vec(),
            })
            .collect())
    }
}

struct Files(PathBuf);

impl IndexSource for Files {
    async fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        std::fs::read(self.0.join(path)).map_err(|e| format!("{path}: {e}"))
    }
}

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 11 {
        return Err("usage: search LOFTR DINOV2 INDEX PHOTO FOCAL LAT LON RADIUS HEIGHT SIGMA CACHE [TRUE_LAT TRUE_LON]".into());
    }
    let device = Device::wgpu_options().init()?;
    let loftr = Loftr::load(
        Weights::from_safetensors(&std::fs::read(&args[0])?)?,
        &device,
    )?;
    let dinov2 = Dinov2::load(
        Weights::from_safetensors(&std::fs::read(&args[1])?)?,
        224,
        &device,
    )?;
    let index = Files(PathBuf::from(&args[2]));
    let manifest =
        IndexManifest::from_json(&std::fs::read_to_string(index.0.join("manifest.json"))?)?;
    let photo = image::open(&args[3])?.to_rgb8();
    let request =
        SearchRequest {
            region: Region::Circle {
                centre: [args[5].parse()?, args[6].parse()?],
                radius_m: args[7].parse()?,
            },
            image_px: [photo.width(), photo.height()],
            focal_px: args[4].parse()?,
            heading: args.get(13).map(|h| h.parse()).transpose()?.map(|deg| {
                navigate_place::HeadingPrior {
                    deg,
                    sigma_deg: 5.0,
                }
            }),
            height: Some(HeightPrior {
                m: args[8].parse()?,
                sigma_m: args[9].parse()?,
            }),
            pitch: args.get(14).map(|p| p.parse()).transpose()?.map(|deg| {
                navigate_place::PitchPrior {
                    deg,
                    sigma_deg: 2.0,
                }
            }),
        };
    let tiles = CachedHttpSource::new(&args[10]);
    let sources = SearchSources {
        tiles: &tiles,
        index: &index,
        manifest: &manifest,
    };
    let started = std::time::Instant::now();
    let mut encoder = Burn(dinov2);
    let outcome = pollster::block_on(search(
        &sources,
        &mut encoder,
        LoftrMatcher::new(loftr, "search"),
        &photo,
        &request,
        |p| {
            tracing::info!(?p, seconds = started.elapsed().as_secs_f64(), "search");
        },
    ))?;
    let seconds = started.elapsed().as_secs_f64();
    match outcome {
        SearchOutcome::Found {
            located,
            inliers,
            checked,
        } => {
            let [lat, lon, h] = located.geodetic;
            let error_m = match (args.get(11), args.get(12)) {
                (Some(a), Some(b)) => {
                    let (tl, tn): (f64, f64) = (a.parse()?, b.parse()?);
                    Some(
                        ((lat - tl) * 111_320.0)
                            .hypot((lon - tn) * 111_320.0 * tl.to_radians().cos()),
                    )
                }
                _ => None,
            };
            tracing::info!(lat, lon, h, angles = ?located.angles_deg, inliers, checked, ?error_m, seconds, "found");
        }
        SearchOutcome::NotFound {
            ranked,
            checked,
            best_inliers,
        } => {
            tracing::info!(ranked, checked, best_inliers, seconds, "not found");
        }
        SearchOutcome::Fallback(f) => tracing::info!(?f, "fallback"),
    }
    Ok(())
}
