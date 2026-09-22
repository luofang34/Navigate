//! CLI contracts are generated from these command and argument types.

use crate::{BenchError, backend::BackendKind};
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    about = "Estimate camera coordinates from calibrated images and a nearby pose prior.",
    after_help = "Start with: visual-bench demo target/visual-demo\nThe demo writes a shared map, sample images, prior.json, and frames.jsonl.\nUse image --help or video --help for input contracts.\nAPI docs: cargo doc -p navigate-visual --open"
)]
struct Cli {
    /// CPU patch matching or GPU compute shaders. GPU failures do not fall back to CPU.
    #[arg(long,global=true,value_enum,default_value_t=BackendKind::Cpu)]
    backend: BackendKind,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a procedural offline imagery and elevation package.
    Prepare { package: PathBuf },
    /// Compare estimates with withheld synthetic camera poses. Fails if limits are exceeded.
    Evaluate { package: PathBuf, output: PathBuf },
    /// Create a map, sample images, priors, and a validated synthetic example.
    Demo { directory: PathBuf },
    /// Read calibrated FrameRecord JSON Lines. Use '-' for standard input.
    Localize {
        package: PathBuf,
        frames: PathBuf,
        output: PathBuf,
    },
    /// Estimate one undistorted PNG/JPEG frame from a camera and pose prior JSON file.
    #[command(after_help=PRIOR_HELP)]
    Image(TrialArgs),
    /// Decode a video with FFmpeg and write timestamped coordinates for each frame.
    #[command(
        after_help = "Requires ffmpeg and ffprobe in PATH. Frames retain presentation timing relative to the first frame. Video dimensions must match the calibration. No rotation or undistortion is applied. Without --priors, every frame uses the fixed --prior pose. Rejected observations remain in the output. The result is a sequence of visual fixes, not a fused trajectory."
    )]
    Video {
        #[command(flatten)]
        input: TrialArgs,
        /// Optional FrameRecord JSONL with one prior per decoded frame. Sequence and relative timestamps must agree.
        #[arg(long)]
        priors: Option<PathBuf>,
    },
    /// Export coordinate observations as GeoJSON points and track segments. Rejections split segments.
    Track { input: PathBuf, output: PathBuf },
}

#[derive(Args)]
pub(crate) struct TrialArgs {
    /// Shared verified imagery and elevation package directory.
    pub package: PathBuf,
    /// Image or video path.
    pub input: PathBuf,
    /// JSON object with 'camera' intrinsics and 'prior' pose. See image --help.
    #[arg(long)]
    pub prior: PathBuf,
    /// New JSONL file for accepted coordinates and explicit rejections.
    #[arg(long)]
    pub output: PathBuf,
    /// Optional new GeoJSON file with points and track segments.
    #[arg(long)]
    pub track: Option<PathBuf>,
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
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
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
    match cli.command {
        Command::Prepare { package } => crate::fixture::prepare_blocking(&package),
        Command::Evaluate { package, output } => {
            crate::scenario::evaluate_blocking(&package, &output, cli.backend).await
        }
        Command::Demo { directory } => demo_blocking(&directory, cli.backend).await,
        Command::Localize {
            package,
            frames,
            output,
        } => crate::stream::localize_blocking(&package, &frames, &output, cli.backend).await,
        Command::Image(input) => crate::trial::image_blocking(&input, cli.backend).await,
        Command::Video { input, priors } => {
            crate::video::run_blocking(&input, priors.as_deref(), cli.backend).await
        }
        Command::Track { input, output } => crate::track::export_blocking(&input, &output),
    }
}

async fn demo_blocking(directory: &Path, backend: BackendKind) -> Result<(), BenchError> {
    crate::fixture::prepare_blocking(&directory.join("map"))?;
    crate::scenario::evaluate_blocking(&directory.join("map"), &directory.join("frames"), backend)
        .await?;
    tracing::info!(directory=%directory.display(),"demo ready: use frames/prior.json with frames/pitch-0-offset-0-clean-query.png");
    Ok(())
}
