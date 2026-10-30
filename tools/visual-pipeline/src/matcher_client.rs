//! Unified matcher client supporting in-process `burn` engines and ONNX worker processes.

use crate::{PipelineError, util::file_sha256};
use burn::tensor::Device;
use image::{DynamicImage, GrayImage, RgbImage};
use navigate_match::{EngineKind, NativeMlEngine, Weights};
use navigate_visual::{ImageMatcher, PixelMatch, PyramidalMatcher};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

/// Image correspondence matcher used by offline pipeline stages.
pub enum PipelineMatcher {
    /// In-process pure-Rust `burn` matcher (`LoFTR-DS` or `XFeat + LighterGlue`).
    Native(Box<NativeMlEngine>),
    /// Classical pyramidal optical-flow/patch matcher (`navigate_visual::PyramidalMatcher`).
    Pyramidal(PyramidalMatcher),
    /// External `navigate-recording-worker match-worker` JSON-lines subprocess for `.onnx`.
    Worker {
        /// Matcher backend identity string reported by the worker.
        identity: String,
        /// Child process handle.
        child: Child,
        /// Standard input pipe.
        stdin: ChildStdin,
        /// Buffered standard output reader.
        stdout: BufReader<ChildStdout>,
    },
}

impl PipelineMatcher {
    /// Open an in-process LoFTR-DS `.safetensors` model on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when reading or loading weights fails.
    pub fn load_loftr(
        safetensors_path: &Path,
        device: &Device,
        engine: EngineKind,
    ) -> Result<Self, PipelineError> {
        let sha = file_sha256(safetensors_path)?;
        let bytes = std::fs::read(safetensors_path).map_err(|source| PipelineError::Io {
            path: safetensors_path.to_path_buf(),
            source,
        })?;
        let weights = Weights::from_safetensors(&bytes)?;
        let ml = NativeMlEngine::load_loftr(weights, &sha, device, engine)?;
        Ok(Self::Native(Box::new(ml)))
    }

    /// Open an in-process XFeat (and optional LighterGlue) `.safetensors` matcher on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when reading or loading weights fails.
    pub fn load_xfeat(
        xfeat_path: &Path,
        lighterglue_path: Option<&Path>,
        limit: usize,
        device: &Device,
        engine: EngineKind,
    ) -> Result<Self, PipelineError> {
        let det_sha = file_sha256(xfeat_path)?;
        let det_bytes = std::fs::read(xfeat_path).map_err(|source| PipelineError::Io {
            path: xfeat_path.to_path_buf(),
            source,
        })?;
        let det_w = Weights::from_safetensors(&det_bytes)?;
        let lg_pair = if let Some(lg_p) = lighterglue_path {
            let lg_sha = file_sha256(lg_p)?;
            let lg_bytes = std::fs::read(lg_p).map_err(|source| PipelineError::Io {
                path: lg_p.to_path_buf(),
                source,
            })?;
            Some((Weights::from_safetensors(&lg_bytes)?, lg_sha))
        } else {
            None
        };
        let ml = NativeMlEngine::load_xfeat(
            det_w,
            &det_sha,
            lg_pair.as_ref().map(|(w, s)| (w.clone(), s.as_str())),
            limit,
            device,
            engine,
        )?;
        Ok(Self::Native(Box::new(ml)))
    }

    /// Construct a deterministic classical pyramidal matcher.
    pub fn pyramidal() -> Self {
        Self::Pyramidal(PyramidalMatcher)
    }

    /// Spawn `worker_bin match-worker` for an `.onnx` model with ONNX Runtime `runtime_lib`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when spawning or handshaking with the worker fails.
    pub fn spawn_onnx_worker(
        worker_bin: &Path,
        model_path: &Path,
        runtime_lib: &Path,
    ) -> Result<Self, PipelineError> {
        let mut child = Command::new(worker_bin)
            .arg("match-worker")
            .arg("--model")
            .arg(model_path)
            .arg("--runtime")
            .arg(runtime_lib)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| PipelineError::Tool {
                tool: worker_bin.display().to_string(),
                message: e.to_string(),
            })?;
        let stdin = child.stdin.take().ok_or_else(|| PipelineError::Tool {
            tool: "match-worker".into(),
            message: "missing stdin pipe".into(),
        })?;
        let out = child.stdout.take().ok_or_else(|| PipelineError::Tool {
            tool: "match-worker".into(),
            message: "missing stdout pipe".into(),
        })?;
        let mut stdout = BufReader::new(out);
        let mut banner = String::new();
        stdout
            .read_line(&mut banner)
            .map_err(|e| PipelineError::Tool {
                tool: "match-worker".into(),
                message: e.to_string(),
            })?;
        let doc: Value = serde_json::from_str(&banner).map_err(|source| PipelineError::Json {
            context: "match-worker banner".into(),
            source,
        })?;
        if doc.get("status").and_then(Value::as_str) != Some("ready") {
            return Err(PipelineError::Tool {
                tool: "match-worker".into(),
                message: format!("expected a ready banner, got {}", banner.trim_end()),
            });
        }
        let identity = doc
            .get("backend")
            .or_else(|| doc.get("matcher"))
            .and_then(Value::as_str)
            .unwrap_or("onnx-match-worker")
            .to_owned();
        Ok(Self::Worker {
            identity,
            child,
            stdin,
            stdout,
        })
    }

    /// Open a matcher from CLI options (`backend` + `model_path` + optional `lighterglue` / `runtime`).
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when configuration or model loading fails.
    pub fn from_options(
        backend: &str,
        model_path: Option<&Path>,
        lighterglue_path: Option<&Path>,
        worker_bin: Option<&Path>,
        runtime_lib: Option<&Path>,
    ) -> Result<Self, PipelineError> {
        if backend == "pyramidal" {
            return Ok(Self::pyramidal());
        }
        let path = model_path.ok_or_else(|| {
            PipelineError::Invalid(format!("backend '{backend}' requires --model"))
        })?;
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if ext == "onnx" {
            let worker = worker_bin.unwrap_or_else(|| Path::new("navigate-recording-worker"));
            let rt = runtime_lib.ok_or_else(|| {
                PipelineError::Invalid("ONNX model requires --onnx-runtime path".into())
            })?;
            return Self::spawn_onnx_worker(worker, path, rt);
        }
        let (device, engine) = default_burn_device();
        match backend {
            "loftr" => Self::load_loftr(path, &device, engine),
            "xfeat" => Self::load_xfeat(path, lighterglue_path, 2048, &device, engine),
            other => Err(PipelineError::Invalid(format!(
                "unsupported matcher backend '{other}'; use loftr, xfeat, or pyramidal"
            ))),
        }
    }

    /// Return the stable matcher backend identity string.
    pub fn identity(&self) -> &str {
        match self {
            Self::Native(m) => m.identity(),
            Self::Pyramidal(m) => m.identity(),
            Self::Worker { identity, .. } => identity.as_str(),
        }
    }

    /// Match RGB `reference` and `query` images and return pixel-centre pairs.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when matching or IPC fails.
    pub fn match_rgb(
        &mut self,
        reference: &RgbImage,
        query: &RgbImage,
    ) -> Result<Vec<PixelMatch>, PipelineError> {
        let ref_gray = DynamicImage::ImageRgb8(reference.clone()).into_luma8();
        let qry_gray = DynamicImage::ImageRgb8(query.clone()).into_luma8();
        self.match_gray(&ref_gray, &qry_gray)
    }

    /// Match grayscale `reference` and `query` images and return pixel-centre pairs.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when matching or IPC fails.
    pub fn match_gray(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, PipelineError> {
        match self {
            Self::Native(m) => Ok(m.match_images_blocking(reference, query)?),
            Self::Pyramidal(m) => Ok(m.match_images_blocking(reference, query)?),
            Self::Worker { stdin, stdout, .. } => {
                let tmp_dir = std::env::temp_dir();
                let pid = std::process::id();
                let ref_p = tmp_dir.join(format!("nav-ref-{pid}.png"));
                let qry_p = tmp_dir.join(format!("nav-qry-{pid}.png"));
                reference
                    .save(&ref_p)
                    .map_err(|source| PipelineError::Image {
                        path: ref_p.clone(),
                        source,
                    })?;
                query.save(&qry_p).map_err(|source| PipelineError::Image {
                    path: qry_p.clone(),
                    source,
                })?;
                let req = json!({
                    "reference": ref_p.display().to_string(),
                    "query": qry_p.display().to_string(),
                });
                let line = serde_json::to_string(&req).map_err(|source| PipelineError::Json {
                    context: "match-worker request".into(),
                    source,
                })?;
                writeln!(stdin, "{line}").map_err(|e| PipelineError::Tool {
                    tool: "match-worker".into(),
                    message: e.to_string(),
                })?;
                stdin.flush().map_err(|e| PipelineError::Tool {
                    tool: "match-worker".into(),
                    message: e.to_string(),
                })?;
                let mut resp = String::new();
                stdout
                    .read_line(&mut resp)
                    .map_err(|e| PipelineError::Tool {
                        tool: "match-worker".into(),
                        message: e.to_string(),
                    })?;
                drop(std::fs::remove_file(&ref_p));
                drop(std::fs::remove_file(&qry_p));
                let parsed: Value =
                    serde_json::from_str(&resp).map_err(|source| PipelineError::Json {
                        context: "match-worker response".into(),
                        source,
                    })?;
                let pairs: Vec<[[f64; 2]; 2]> = parsed
                    .get("pairs")
                    .or_else(|| parsed.get("matches"))
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|source| PipelineError::Json {
                        context: "match-worker pairs".into(),
                        source,
                    })?
                    .unwrap_or_default();
                Ok(pairs
                    .into_iter()
                    .map(|p| PixelMatch {
                        reference: nalgebra::Vector2::new(p[0][0], p[0][1]),
                        query: nalgebra::Vector2::new(p[1][0], p[1][1]),
                    })
                    .collect())
            }
        }
    }
}

impl Drop for PipelineMatcher {
    fn drop(&mut self) {
        if let Self::Worker { child, .. } = self {
            drop(child.kill());
            drop(child.wait());
        }
    }
}

pub(crate) fn default_burn_device() -> (Device, EngineKind) {
    #[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
    {
        (Device::default(), EngineKind::BurnGpu)
    }
    #[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cuda")))]
    {
        (Device::flex(), EngineKind::BurnCpu)
    }
}
