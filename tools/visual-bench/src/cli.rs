//! CLI contracts for every visual-bench subcommand.

pub(crate) mod args;
#[cfg(test)]
mod tests;

use crate::{BenchError, backend::BackendKind};
use args::{path, positional, take};
use clap::{Arg, ArgMatches, Command, value_parser};
use std::path::{Path, PathBuf};

const ABOUT: &str = "Estimate camera coordinates from calibrated images and a nearby pose prior.";
const AFTER_HELP: &str = "Start with: visual-bench demo target/visual-demo\nThe demo writes a shared map, sample images, prior.json, and frames.jsonl.\nUse image --help or video --help for input contracts.\nAPI docs: cargo doc -p navigate-visual --open";
const REFINE_HELP: &str = "Use render to export the candidate image and optical depth. Supply correspondence JSON from an external matcher with the exported image and depth digests. Supply a grayscale query PNG. --prior controls admission bounds; --reference-prior initializes the image search. Geometry covariance excludes map and camera calibration errors.";
const RENDER_HELP: &str = "The output PNG has matching .depth.bin and .depth.json sidecars. Depth is row-major little-endian float32 in metres along the optical axis. Zero marks missing imagery or terrain. The digests bind external correspondences to the exact reference.";
const VIDEO_HELP: &str = "Requires ffmpeg and ffprobe in PATH. Frames retain presentation timing relative to the first frame. Video dimensions must match the calibration. No rotation or undistortion is applied. Without --priors, every frame uses the fixed --prior pose. Rejected observations remain in the output. The result is a sequence of visual fixes, not a fused trajectory.";

fn positionals(name: &'static str, about: &'static str, ids: &[&'static str]) -> Command {
    Command::new(name)
        .about(about)
        .args(ids.iter().map(|id| positional(id)))
}

pub(crate) fn command() -> Command {
    Command::new("visual-bench")
        .about(ABOUT)
        .after_help(AFTER_HELP)
        .subcommand_required(true)
        .arg(
            Arg::new("backend")
                .long("backend")
                .global(true)
                .default_value("cpu")
                .value_parser(value_parser!(BackendKind))
                .help("CPU patch matching or GPU compute shaders. GPU failures do not fall back to CPU."),
        )
        .subcommand(positionals(
            "worker",
            "Keep one offline renderer resident and handle JSON Lines requests.",
            &["package", "prior"],
        ))
        .subcommand(positionals(
            "verify-pack",
            "Verify immutable package chunks through the native byte-range store.",
            &["manifest", "root"],
        ))
        .subcommand(
            TrialArgs::args(Command::new("refine"))
                .about("Fit image-bound external correspondences using rendered terrain depth.")
                .after_help(REFINE_HELP)
                .arg(
                    path("reference-prior")
                        .required(true)
                        .help("Candidate camera configuration used to render the matched reference."),
                )
                .arg(
                    path("matches")
                        .required(true)
                        .help("JSON correspondences bound to query pixels, reference pixels and depth."),
                ),
        )
        .subcommand(
            positionals(
                "render",
                "Export a candidate PNG, float32 optical depth, and content digests.",
                &["package", "prior", "output"],
            )
            .after_help(RENDER_HELP),
        )
        .subcommand(positionals(
            "prepare",
            "Generate a procedural offline imagery and elevation package.",
            &["package"],
        ))
        .subcommand(positionals(
            "evaluate",
            "Compare estimates with withheld synthetic camera poses. Fails if limits are exceeded.",
            &["package", "output"],
        ))
        .subcommand(positionals(
            "demo",
            "Create a map, sample images, priors, and a validated synthetic example.",
            &["directory"],
        ))
        .subcommand(positionals(
            "localize",
            "Read calibrated FrameRecord JSON Lines. Use '-' for standard input.",
            &["package", "frames", "output"],
        ))
        .subcommand(
            TrialArgs::args(Command::new("image"))
                .about("Estimate one undistorted PNG/JPEG frame from a camera and pose prior JSON file.")
                .after_help(PRIOR_HELP),
        )
        .subcommand(
            TrialArgs::args(Command::new("video"))
                .about("Decode a video with FFmpeg and write timestamped coordinates for each frame.")
                .after_help(VIDEO_HELP)
                .arg(path("priors").help(
                    "Optional FrameRecord JSONL with one prior per decoded frame. Sequence and relative timestamps must agree.",
                )),
        )
        .subcommand(positionals(
            "track",
            "Export coordinate observations as GeoJSON points and track segments. Rejections split segments.",
            &["input", "output"],
        ))
}

pub(crate) struct TrialArgs {
    pub package: PathBuf,
    pub input: PathBuf,
    pub prior: PathBuf,
    pub output: PathBuf,
    pub track: Option<PathBuf>,
}

impl TrialArgs {
    fn args(command: Command) -> Command {
        command
            .arg(
                positional("package")
                    .help("Published package manifest, or a source folder with `map.json`."),
            )
            .arg(positional("input").help("Image or video path."))
            .arg(
                path("prior").required(true).help(
                    "JSON object with 'camera' intrinsics and 'prior' pose. See image --help.",
                ),
            )
            .arg(
                path("output")
                    .required(true)
                    .help("New JSONL file for accepted coordinates and explicit rejections."),
            )
            .arg(path("track").help("Optional new GeoJSON file with points and track segments."))
    }

    fn from_matches(matches: &mut ArgMatches) -> Result<Self, BenchError> {
        Ok(Self {
            package: take(matches, "package")?,
            input: take(matches, "input")?,
            prior: take(matches, "prior")?,
            output: take(matches, "output")?,
            track: matches.remove_one("track"),
        })
    }
}

const PRIOR_HELP: &str = r#"Prior JSON:
{"camera":{"width":640,"height":480,"fx":550,"fy":550,"cx":319.5,"cy":239.5},
 "prior":{"geodetic_lat_lon_alt_m":[47.324,11.492,1680],
          "heading_tilt_roll_deg":[0,45,0],"position_radius_m":300,"attitude_radius_rad":0.15}}

Replace the example calibration and pose with measured values.
Heading is clockwise from north. Tilt is 0 degrees down and 90 degrees horizontal.
Roll is clockwise about the viewing axis. The pose describes the camera, not the aircraft body.
Alternatively use position_enu_m:[east,north,up] and eye_to_enu_xyzw:[x,y,z,w].
Supply exactly one position and one orientation representation. Altitude uses the package datum.
Coordinates use the renderer's local Mercator frame. The output labels this coordinate model.
The image must be undistorted. Appearance must resemble the map, and the prior must provide overlap.
Real-camera and night accuracy require separate validation."#;

pub(crate) async fn run_blocking() -> Result<(), BenchError> {
    let mut matches = match command().try_get_matches() {
        Ok(matches) => matches,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error.print().map_err(|source| BenchError::Io {
                path: PathBuf::from("<terminal>"),
                source,
            })?;
            return Ok(());
        }
        Err(error) => return Err(BenchError::Arguments(error)),
    };
    let backend: BackendKind = take(&mut matches, "backend")?;
    let Some((name, mut m)) = matches.remove_subcommand() else {
        return Err(BenchError::Arguments(command().error(
            clap::error::ErrorKind::MissingSubcommand,
            "a subcommand is required",
        )));
    };
    let m = &mut m;
    match name.as_str() {
        "worker" => {
            crate::worker::run_blocking(
                &take::<PathBuf>(m, "package")?,
                &take::<PathBuf>(m, "prior")?,
            )
            .await
        }
        "verify-pack" => {
            crate::offline_pack::verify_blocking(
                &take::<PathBuf>(m, "manifest")?,
                &take::<PathBuf>(m, "root")?,
            )
            .await
        }
        "refine" => {
            let input = TrialArgs::from_matches(m)?;
            crate::trial::refine_blocking(
                &input,
                &take::<PathBuf>(m, "reference-prior")?,
                &take::<PathBuf>(m, "matches")?,
            )
            .await
        }
        "render" => {
            crate::trial::render_blocking(
                &take::<PathBuf>(m, "package")?,
                &take::<PathBuf>(m, "prior")?,
                &take::<PathBuf>(m, "output")?,
            )
            .await
        }
        "prepare" => crate::fixture::prepare_blocking(&take::<PathBuf>(m, "package")?),
        "evaluate" => {
            crate::scenario::evaluate_blocking(
                &take::<PathBuf>(m, "package")?,
                &take::<PathBuf>(m, "output")?,
                backend,
            )
            .await
        }
        "demo" => demo_blocking(&take::<PathBuf>(m, "directory")?, backend).await,
        "localize" => {
            crate::stream::localize_blocking(
                &take::<PathBuf>(m, "package")?,
                &take::<PathBuf>(m, "frames")?,
                &take::<PathBuf>(m, "output")?,
                backend,
            )
            .await
        }
        "image" => crate::trial::image_blocking(&TrialArgs::from_matches(m)?, backend).await,
        "video" => {
            let input = TrialArgs::from_matches(m)?;
            let priors: Option<PathBuf> = m.remove_one("priors");
            crate::video::run_blocking(&input, priors.as_deref(), backend).await
        }
        "track" => crate::track::export_blocking(
            &take::<PathBuf>(m, "input")?,
            &take::<PathBuf>(m, "output")?,
        ),
        other => Err(BenchError::Arguments(command().error(
            clap::error::ErrorKind::InvalidSubcommand,
            format!("unknown subcommand {other}"),
        ))),
    }
}

async fn demo_blocking(directory: &Path, backend: BackendKind) -> Result<(), BenchError> {
    crate::fixture::prepare_blocking(&directory.join("map"))?;
    crate::scenario::evaluate_blocking(&directory.join("map"), &directory.join("frames"), backend)
        .await?;
    tracing::info!(directory=%directory.display(),"demo ready: use frames/prior.json with frames/pitch-0-offset-0-clean-query.png");
    Ok(())
}
