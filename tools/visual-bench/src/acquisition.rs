//! Cold geographic acquisition with resident, replaceable Rust adapters.
mod files;
mod planning;
mod search;
mod terrain;
use crate::cli::args::{path, positional, take, value};
use crate::{
    BenchError,
    flight::inference::InferenceArgs,
    package::MapPackage,
    stream::write_record_blocking,
    trial::{config_blocking, writer_blocking},
};
use clap::{ArgAction, ArgMatches, Command, value_parser};
use navigate_visual::{Frame, FrameStamp, ImageMatcher, place_retrieval::ImageRetriever};
use navigate_visual_onnx::{CampRetriever, ExecutionConfig};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

pub(crate) struct AcquisitionArgs {
    pub package: PathBuf,
    pub images: Vec<PathBuf>,
    pub prior: PathBuf,
    pub output: PathBuf,
    pub retrieval_model: PathBuf,
    pub index: PathBuf,
    pub catalog: PathBuf,
    pub scales: Vec<f64>,
    pub orientations: Option<PathBuf>,
    pub width: u32,
    pub inference: InferenceArgs,
}

impl AcquisitionArgs {
    pub fn args(command: Command) -> Command {
        let command = command
            .arg(positional("package"))
            .arg(
                positional("images")
                    .num_args(1..)
                    .action(ArgAction::Append)
                    .help("Source images. Each uses the same prior and has no previous fitted pose."),
            )
            .arg(path("prior").required(true).help(
                "Source-resolution camera calibration and navigation admission bounds.",
            ))
            .arg(path("output").required(true))
            .arg(
                path("retrieval-model")
                    .required(true)
                    .help("CAMP model export. No model download occurs during acquisition."),
            )
            .arg(
                path("index")
                    .required(true)
                    .help("CAMP index file with model, catalog, map, row and descriptor identities."),
            )
            .arg(path("catalog").required(true))
            .arg(
                value("scales", "100,200,50,400", value_parser!(f64))
                    .value_delimiter(',')
                    .action(ArgAction::Append)
                    .help("Crop widths to search in priority order. They initialize camera height."),
            )
            .arg(path("orientations").help(
                "JSON array of camera-to-local unit quaternions `[x,y,z,w]`.\nWithout this file, this demo proposes eight downward-looking yaw seeds.",
            ))
            .arg(value("width", "640", value_parser!(u32)));
        InferenceArgs::args(command)
    }

    pub fn from_matches(matches: &mut ArgMatches) -> Result<Self, BenchError> {
        Ok(Self {
            package: take(matches, "package")?,
            images: matches
                .remove_many("images")
                .map(Iterator::collect)
                .unwrap_or_default(),
            prior: take(matches, "prior")?,
            output: take(matches, "output")?,
            retrieval_model: take(matches, "retrieval-model")?,
            index: take(matches, "index")?,
            catalog: take(matches, "catalog")?,
            scales: matches
                .remove_many("scales")
                .map(Iterator::collect)
                .unwrap_or_default(),
            orientations: matches.remove_one("orientations"),
            width: take(matches, "width")?,
            inference: InferenceArgs::from_matches(matches)?,
        })
    }
}

pub(crate) async fn run_blocking(args: &AcquisitionArgs) -> Result<(), BenchError> {
    let started = Instant::now();
    let input = config_blocking(&args.prior)?;
    let source_camera = input.camera.model();
    source_camera.validate()?;
    let camera = crate::flight::scaled_camera(source_camera, args.width)?;
    let package = MapPackage::open_blocking(&args.package)?;
    let prior = input.prior.prior(package.frame)?;
    prior.validate()?;
    let (index, catalog) = files::load_blocking(&args.index, &args.catalog, &package.revision)?;
    let plan = planning::Plan::new(&catalog, &args.scales, &prior)?;
    let orientations = files::orientations_blocking(args.orientations.as_deref())?;
    let terrain = terrain::Terrain::new(&package);
    let mut matcher = args.inference.load_blocking()?;
    let mut retriever =
        CampRetriever::load_blocking(&args.retrieval_model, index, &ExecutionConfig::default())?;
    let mut search = search::Search::new(package, camera, prior).await?;
    let initialization_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut writer = writer_blocking(&args.output)?;
    for (sequence, path) in args.images.iter().enumerate() {
        let report = acquire_blocking(
            SourceInput {
                path,
                camera: source_camera,
                sequence: sequence as u64,
            },
            &plan,
            &orientations,
            &terrain,
            &mut retriever,
            matcher.as_mut(),
            &mut search,
        )?;
        let mut report = report;
        report["initialization_ms"] = initialization_ms.into();
        report["retrieval_seed_orientation"] = if args.orientations.is_some() {
            "supplied camera orientations"
        } else {
            "eight near-nadir yaw seeds; final pose attitude is unrestricted"
        }
        .into();
        write_record_blocking(&mut writer, &args.output, &report)?;
        tracing::info!(image=%path.display(),decision=?report["decision"],processing_ms=?report["processing_ms"],"geographic acquisition");
    }
    Ok(())
}

struct SourceInput<'a> {
    path: &'a Path,
    camera: navigate_visual::CameraModel,
    sequence: u64,
}
fn acquire_blocking(
    input: SourceInput<'_>,
    plan: &planning::Plan<'_>,
    orientations: &[nalgebra::UnitQuaternion<f64>],
    terrain: &terrain::Terrain,
    retriever: &mut dyn ImageRetriever,
    matcher: &mut dyn ImageMatcher,
    search: &mut search::Search,
) -> Result<serde_json::Value, BenchError> {
    let SourceInput {
        path,
        camera: source_camera,
        sequence,
    } = input;
    let started = Instant::now();
    let bytes = crate::read_blocking(path)?;
    let rgb = image::load_from_memory(&bytes)
        .map_err(|source| BenchError::Image {
            path: path.to_owned(),
            source,
        })?
        .to_rgb8();
    if rgb.dimensions() != (source_camera.width, source_camera.height) {
        return Err(invalid(
            "source image dimensions differ from source calibration",
        ));
    }
    let gray = image::GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let p = rgb.get_pixel(x, y).0;
        image::Luma([
            ((u32::from(p[0]) * 77 + u32::from(p[1]) * 150 + u32::from(p[2]) * 29) >> 8) as u8,
        ])
    });
    let image = image::imageops::resize(
        &gray,
        search.camera.width,
        search.camera.height,
        image::imageops::FilterType::Triangle,
    );
    let frame = Frame {
        image,
        camera: search.camera,
        stamp: FrameStamp {
            sequence,
            capture_time_ns: 0,
        },
    };
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    let retrieved = Instant::now();
    let ids = plan.rank_blocking(retriever, &rgb)?;
    let (candidates, unsupported) = plan.candidates(&ids, orientations, search.camera, terrain)?;
    let retrieval_ms = retrieved.elapsed().as_secs_f64() * 1000.0;
    let mut report = search.evaluate_blocking(frame, &candidates, matcher)?;
    report["input"] = path.display().to_string().into();
    report["source_sha256"] = crate::package::digest(&bytes).into();
    report["retrieval"] = serde_json::json!({"stage":"retrieval_only","reference_ids":ids.iter().map(|id|id.0).collect::<Vec<_>>(),"backend":retriever.identity(),"unsupported_references":unsupported,"pose_candidates":candidates.len(),"input_width":rgb.width(),"input_height":rgb.height(),"scope":"bounded reference retrieval; unexamined geographic alternatives remain"});
    report["pixel_processing"] =
        "encoded RGB; integer luma; triangle resize to scaled calibration".into();
    report["capture_timing"] =
        "independent still image; capture time unknown; zero is a serialization marker".into();
    report["decode_ms"] = decode_ms.into();
    report["retrieval_ms"] = retrieval_ms.into();
    report["processing_ms"] = (started.elapsed().as_secs_f64() * 1000.0).into();
    Ok(report)
}

fn invalid(reason: &str) -> BenchError {
    BenchError::Record {
        reason: reason.into(),
    }
}
