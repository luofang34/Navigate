//! Rank local reference IDs through the resident Rust retrieval library.
use clap::{Arg, ArgAction, ArgMatches, Command, ValueEnum, builder::PossibleValue, value_parser};
use navigate_visual::{
    MapRevision,
    place_retrieval::{ImageRetriever, ReferenceCatalog, ReferenceId},
};
use navigate_visual_onnx::{
    CampIndex, CampRetriever, ExecutionConfig, Provider, initialize_blocking,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::json;
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy)]
enum Device {
    Cpu,
    CoremlAne,
    CoremlGpu,
}
impl ValueEnum for Device {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Cpu, Self::CoremlAne, Self::CoremlGpu]
    }
    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(match self {
            Self::Cpu => "cpu",
            Self::CoremlAne => "coreml-ane",
            Self::CoremlGpu => "coreml-gpu",
        }))
    }
}
fn take<T: Clone + Send + Sync + 'static>(
    matches: &mut ArgMatches,
    id: &'static str,
) -> Result<T, Box<dyn std::error::Error>> {
    matches
        .remove_one(id)
        .ok_or_else(|| format!("missing --{id}").into())
}
fn path(id: &'static str) -> Arg {
    Arg::new(id).long(id).value_parser(value_parser!(PathBuf))
}
fn flag(id: &'static str, long: &'static str) -> Arg {
    Arg::new(id).long(long).action(ArgAction::SetTrue)
}
fn device() -> Arg {
    Arg::new("device")
        .long("device")
        .default_value("cpu")
        .value_parser(value_parser!(Device))
}
struct Args {
    library: PathBuf,
    model: PathBuf,
    index: PathBuf,
    queries: PathBuf,
    output: PathBuf,
    device: Device,
    threads: usize,
    cpu_spinning: bool,
}
fn command() -> Command {
    Command::new("retrieve_places")
        .args(["library", "model", "index", "queries", "output"].map(|id| path(id).required(true)))
        .arg(device())
        .arg(
            Arg::new("threads")
                .long("threads")
                .default_value("4")
                .value_parser(value_parser!(usize)),
        )
        .arg(flag("cpu_spinning", "cpu-spinning"))
}
impl Args {
    fn from_matches(mut m: ArgMatches) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            library: take(&mut m, "library")?,
            model: take(&mut m, "model")?,
            index: take(&mut m, "index")?,
            queries: take(&mut m, "queries")?,
            output: take(&mut m, "output")?,
            device: take(&mut m, "device")?,
            threads: take(&mut m, "threads")?,
            cpu_spinning: m.get_flag("cpu_spinning"),
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexFile {
    map_release: String,
    map_manifest_sha256: String,
    catalog_manifest_sha256: String,
    model_sha256: String,
    descriptors: PathBuf,
    ids: Vec<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    id: String,
    image: PathBuf,
    eligible_groups: Vec<Vec<u64>>,
    limit_per_group: usize,
}
#[derive(Debug, thiserror::Error)]
enum FileError {
    #[error("read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("decode image {path}: {source}")]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("descriptor file {0} has a partial float32 value")]
    Shape(PathBuf),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let matches = match command().try_get_matches() {
        Ok(matches) => matches,
        Err(error) if error.kind() == clap::error::ErrorKind::DisplayHelp => {
            error.print()?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let args = Args::from_matches(matches)?;
    initialize_blocking(&args.library)?;
    let start = Instant::now();
    let index = read_index_blocking(&args.index)?;
    let execution = ExecutionConfig {
        provider: match args.device {
            Device::Cpu => Provider::Cpu,
            Device::CoremlAne => Provider::CoreMlAne,
            Device::CoremlGpu => Provider::CoreMlGpu,
        },
        threads: args.threads,
        cpu_spinning: args.cpu_spinning,
        ..Default::default()
    };
    let mut retriever = CampRetriever::load_blocking(&args.model, index, &execution)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut results = Vec::new();
    for query in read_json_blocking::<Vec<Query>>(&args.queries)? {
        let decode_started = Instant::now();
        let image = image::open(&query.image)
            .map_err(|source| FileError::Image {
                path: query.image.clone(),
                source,
            })?
            .to_rgb8();
        let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let mut groups = Vec::new();
        for eligible in query.eligible_groups {
            let ids: Vec<_> = eligible.into_iter().map(ReferenceId).collect();
            groups.push(
                retriever
                    .rank_blocking(&image, &ids, query.limit_per_group)?
                    .into_iter()
                    .map(|id| id.0)
                    .collect::<Vec<_>>(),
            );
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(case = query.id, elapsed_ms, "reference retrieval complete");
        results.push(json!({"id": query.id, "image": query.image, "groups": groups, "elapsed_ms": elapsed_ms, "decode_ms": decode_ms, "total_ms": decode_started.elapsed().as_secs_f64() * 1000.0}));
    }
    let catalog = retriever.catalog();
    let report = json!({"scope": "ranked reference IDs only; no camera pose or geographic acceptance", "backend": retriever.identity(), "load_ms": load_ms, "execution_config": {"cpu_threads": execution.threads, "cpu_spinning": execution.cpu_spinning},
        "catalog": {"map_release": catalog.map.release_id, "map_manifest_sha256": catalog.map.manifest_sha256, "catalog_manifest_sha256": catalog.manifest_sha256}, "cases": results});
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    output.write_all(&serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn read_json_blocking<T: DeserializeOwned>(path: &Path) -> Result<T, FileError> {
    let bytes = std::fs::read(path).map_err(|source| FileError::Read {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| FileError::Json {
        path: path.to_owned(),
        source,
    })
}
fn read_index_blocking(path: &Path) -> Result<CampIndex, Box<dyn std::error::Error>> {
    let index: IndexFile = read_json_blocking(path)?;
    let bytes = std::fs::read(&index.descriptors).map_err(|source| FileError::Read {
        path: index.descriptors.clone(),
        source,
    })?;
    let (values, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(FileError::Shape(index.descriptors).into());
    }
    let values = values
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect();
    let catalog = ReferenceCatalog {
        map: MapRevision {
            release_id: index.map_release,
            manifest_sha256: index.map_manifest_sha256,
        },
        manifest_sha256: index.catalog_manifest_sha256,
    };
    Ok(CampIndex::new(
        catalog,
        index.model_sha256,
        index.ids.into_iter().map(ReferenceId).collect(),
        values,
    )?)
}
