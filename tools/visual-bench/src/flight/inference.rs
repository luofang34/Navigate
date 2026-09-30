//! Explicit model and runtime choices for resident video matching.
use crate::BenchError;
use crate::cli::args::{flag, path, take, value};
use clap::{ArgMatches, Command, ValueEnum, builder::PossibleValue, value_parser};
use navigate_visual::ImageMatcher;
use navigate_visual_onnx::{
    ExecutionConfig, LoFtrMatcher, MatcherFiles, OnnxMatcher, Provider, initialize_blocking,
};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Device {
    Cpu,
    CoremlGpu,
    CoremlAne,
    Cuda,
    TensorRt,
}

impl ValueEnum for Device {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Cpu,
            Self::CoremlGpu,
            Self::CoremlAne,
            Self::Cuda,
            Self::TensorRt,
        ]
    }
    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(match self {
            Self::Cpu => "cpu",
            Self::CoremlGpu => "coreml-gpu",
            Self::CoremlAne => "coreml-ane",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensor-rt",
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatcherKind {
    Loftr,
    Lightglue,
    Superglue,
}

impl ValueEnum for MatcherKind {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Loftr, Self::Lightglue, Self::Superglue]
    }
    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(match self {
            Self::Loftr => "loftr",
            Self::Lightglue => "lightglue",
            Self::Superglue => "superglue",
        }))
    }
}

pub(crate) struct InferenceArgs {
    pub model: PathBuf,
    pub runtime: PathBuf,
    pub coreml_cache: Option<PathBuf>,
    pub matcher: MatcherKind,
    pub detector: Option<PathBuf>,
    pub model_width: usize,
    pub model_height: usize,
    pub keypoints: usize,
    pub profile_keypoints: bool,
    pub refinement_patches: bool,
    pub quarter_turns: bool,
    pub device: Device,
}

impl InferenceArgs {
    pub fn args(command: Command) -> Command {
        command
            .arg(
                path("model")
                    .required(true)
                    .help("Matcher export with embedded weights. No model is downloaded."),
            )
            .arg(
                path("runtime")
                    .required(true)
                    .help("Host ONNX Runtime dynamic library."),
            )
            .arg(
                path("coreml-cache")
                    .help("Local compiled-model cache for Core ML and embedded weights."),
            )
            .arg(value("matcher", "loftr", value_parser!(MatcherKind)))
            .arg(
                path("detector")
                    .required_if_eq_any([("matcher", "lightglue"), ("matcher", "superglue")])
                    .help("Dense SuperPoint export required for LightGlue or SuperGlue."),
            )
            .arg(
                value("model-width", "640", value_parser!(usize))
                    .help("Image width embedded in the sparse matcher's coordinate normalization."),
            )
            .arg(
                value("model-height", "360", value_parser!(usize))
                    .help("Image height embedded in the sparse matcher's coordinate normalization."),
            )
            .arg(
                value("keypoints", "1024", value_parser!(usize))
                    .help("Maximum SuperPoint features. Lower limits can reduce geometric support."),
            )
            .arg(flag("profile-keypoints").help(
                "Enable a validated full-feature LightGlue profile; other counts stay dynamic.",
            ))
            .arg(flag("refinement-patches").help(
                "Use a full view and four overlapping crops with LoFTR.\nThe host must budget up to five inference calls for each match.",
            ))
            .arg(flag("quarter-turns").help(
                "Enable up to three LoFTR rotation attempts when relative geometry fails.",
            ))
            .arg(
                value(
                    "device",
                    if cfg!(target_os = "macos") {
                        "coreml-gpu"
                    } else {
                        "cpu"
                    },
                    value_parser!(Device),
                )
                .help("Default to Core ML GPU on macOS and CPU on other hosts.\nUse an explicit device to select a different validated execution path."),
            )
    }

    pub fn from_matches(matches: &mut ArgMatches) -> Result<Self, BenchError> {
        Ok(Self {
            model: take(matches, "model")?,
            runtime: take(matches, "runtime")?,
            coreml_cache: matches.remove_one("coreml-cache"),
            matcher: take(matches, "matcher")?,
            detector: matches.remove_one("detector"),
            model_width: take(matches, "model-width")?,
            model_height: take(matches, "model-height")?,
            keypoints: take(matches, "keypoints")?,
            profile_keypoints: matches.get_flag("profile-keypoints"),
            refinement_patches: matches.get_flag("refinement-patches"),
            quarter_turns: matches.get_flag("quarter-turns"),
            device: take(matches, "device")?,
        })
    }

    pub fn provider(&self) -> Provider {
        match self.device {
            Device::Cpu => Provider::Cpu,
            Device::CoremlGpu => Provider::CoreMlGpu,
            Device::CoremlAne => Provider::CoreMlAne,
            Device::Cuda => Provider::Cuda,
            Device::TensorRt => Provider::TensorRt,
        }
    }
    pub fn load_blocking(&self) -> Result<Box<dyn ImageMatcher>, BenchError> {
        let files = self.sparse_files()?;
        initialize_blocking(&self.runtime)?;
        let execution = ExecutionConfig {
            provider: self.provider(),
            coreml_cache_directory: self.coreml_cache.clone(),
            ..ExecutionConfig::default()
        };
        match files {
            Some(files) if self.profile_keypoints => Ok(Box::new(
                OnnxMatcher::load_profiled_blocking(files, execution, self.keypoints)?,
            )),
            Some(files) => Ok(Box::new(OnnxMatcher::load_blocking(
                files,
                execution,
                self.keypoints,
            )?)),
            None => {
                let matcher = LoFtrMatcher::load_blocking(&self.model, &execution)?;
                let matcher = if self.refinement_patches {
                    matcher.with_refinement_patches()
                } else {
                    matcher
                };
                Ok(Box::new(if self.quarter_turns {
                    matcher.with_quarter_turns()
                } else {
                    matcher
                }))
            }
        }
    }
    fn sparse_files(&self) -> Result<Option<MatcherFiles>, BenchError> {
        if self.quarter_turns && self.matcher != MatcherKind::Loftr {
            return Err(BenchError::Record {
                reason: "quarter turns require LoFTR".into(),
            });
        }
        if self.refinement_patches && self.matcher != MatcherKind::Loftr {
            return Err(BenchError::Record {
                reason: "refinement patches require LoFTR".into(),
            });
        }
        if self.profile_keypoints && self.matcher != MatcherKind::Lightglue {
            return Err(BenchError::Record {
                reason: "feature profiles require LightGlue".into(),
            });
        }
        if self.matcher == MatcherKind::Loftr {
            if self.detector.is_some() {
                return Err(BenchError::Record {
                    reason: "LoFTR does not use a SuperPoint detector".into(),
                });
            }
            return Ok(None);
        }
        let detector = self.detector.clone().ok_or_else(|| BenchError::Record {
            reason: "LightGlue and SuperGlue require a SuperPoint detector".into(),
        })?;
        let matcher = self.model.clone();
        let image_size = [self.model_width, self.model_height];
        Ok(Some(match self.matcher {
            MatcherKind::Lightglue => MatcherFiles::LightGlue {
                detector,
                matcher,
                image_size,
            },
            MatcherKind::Superglue => MatcherFiles::SuperGlue {
                detector,
                matcher,
                image_size,
            },
            MatcherKind::Loftr => return Ok(None),
        }))
    }
}

#[cfg(test)]
mod tests;
