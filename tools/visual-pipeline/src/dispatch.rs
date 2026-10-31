//! Subcommand dispatcher for `navigate-visual-pipeline`.

use clap::ArgMatches;
use navigate_segment::{ColorMode, TrainPolicy};
use navigate_visual_pipeline::{
    PipelineError, PipelineMatcher, build_reference_catalog_zoom,
    clock::seed_live_prior,
    compose_survey_debrief, export_preview, export_publication, freeze_anchors,
    interpolate_flight_priors, prepare_photo_testset,
    realtime::{BankBuildOptions, BankMatchOptions, build_reference_bank, run_realtime_bank_match},
    reproduce_bundle,
    segment_cli::{run_airframe_compose, run_photo_masks, run_segment_finetune, run_video_masks},
    stages::{
        MatchStageOptions, run_match_stage, run_pipeline_recipe, run_refine_stage,
        run_terrain_bundle_stage, run_track_stage,
    },
    util::write_json,
    video_ops::{
        apply_video_priors, import_video_media, prepare_navigate_debrief, prepare_video_testset,
    },
};
use std::path::PathBuf;

fn req<T: Clone + Send + Sync + 'static>(
    args: &ArgMatches,
    name: &str,
) -> Result<T, PipelineError> {
    args.get_one::<T>(name)
        .cloned()
        .ok_or_else(|| PipelineError::Invalid(format!("missing --{name}")))
}

fn paths(args: &ArgMatches, name: &str) -> Result<Vec<PathBuf>, PipelineError> {
    let list: Vec<PathBuf> = args
        .get_many::<PathBuf>(name)
        .map(|it| it.cloned().collect())
        .unwrap_or_default();
    if list.is_empty() {
        return Err(PipelineError::Invalid(format!("missing --{name}")));
    }
    Ok(list)
}

fn matcher_from(args: &ArgMatches) -> Result<PipelineMatcher, PipelineError> {
    let backend: String = req(args, "backend")?;
    let model = args.get_one::<PathBuf>("model").map(PathBuf::as_path);
    let lg = args.get_one::<PathBuf>("lighterglue").map(PathBuf::as_path);
    PipelineMatcher::from_options(&backend, model, lg)
}

pub(crate) fn dispatch(sub: &str, args: &ArgMatches) -> Result<serde_json::Value, PipelineError> {
    match sub {
        "prepare" => prepare_photo_testset(
            &req::<PathBuf>(args, "source")?,
            &req::<PathBuf>(args, "output")?,
            req::<u32>(args, "long-edge")?,
        ),
        "priors" => {
            let filled = interpolate_flight_priors(&req::<PathBuf>(args, "dataset")?)?;
            Ok(serde_json::json!({ "interpolated_priors": filled }))
        }
        "reference" => build_reference_catalog_zoom(
            &req::<PathBuf>(args, "profile")?,
            &req::<PathBuf>(args, "output")?,
            &req::<String>(args, "imagery-url")?,
            &req::<String>(args, "dem-url")?,
            args.get_one::<PathBuf>("cache").map(PathBuf::as_path),
            req::<f64>(args, "radius-m")?,
            req::<u32>(args, "imagery-zoom")?,
            req::<u32>(args, "dem-zoom")?,
        ),
        "mask" => run_photo_masks(
            &req::<PathBuf>(args, "dataset")?,
            &req::<PathBuf>(args, "model")?,
            &req::<String>(args, "policy")?,
            req::<usize>(args, "margin")?,
        ),
        "video-masks" => run_video_masks(
            &req::<PathBuf>(args, "frames")?,
            &req::<PathBuf>(args, "model")?,
            &req::<PathBuf>(args, "output")?,
            req::<f32>(args, "threshold")?,
            req::<usize>(args, "margin")?,
        ),
        "segment-finetune" => {
            let color_mode = ColorMode::parse(&req::<String>(args, "color")?)?;
            let policy = TrainPolicy {
                size: req::<usize>(args, "size")?,
                batch_size: req::<usize>(args, "batch-size")?,
                epochs: req::<usize>(args, "epochs")?,
                color_mode,
                seed: req::<u64>(args, "seed")?,
                ..TrainPolicy::default()
            };
            run_segment_finetune(
                &req::<PathBuf>(args, "dataset")?,
                args.get_one::<PathBuf>("base-model").map(PathBuf::as_path),
                &req::<PathBuf>(args, "output")?,
                &policy,
            )
        }
        "airframe-labels" => run_airframe_compose(
            &req::<PathBuf>(args, "image")?,
            &req::<PathBuf>(args, "sky-mask")?,
            args.get_one::<PathBuf>("airframe-mask")
                .map(PathBuf::as_path),
            args.get_one::<PathBuf>("propeller-mask")
                .map(PathBuf::as_path),
            &req::<PathBuf>(args, "output")?,
            args.get_one::<PathBuf>("sheet").map(PathBuf::as_path),
        ),
        "corridor-build" | "corridor-track" => crate::corridor_cli::dispatch(sub, args),
        "match" | "bank-build" | "bank-match" | "track" | "refine" | "terrain-bundle"
        | "pipeline" => dispatch_pose_stage(sub, args),
        _ => dispatch_export_or_video(sub, args),
    }
}

fn dispatch_pose_stage(sub: &str, args: &ArgMatches) -> Result<serde_json::Value, PipelineError> {
    let only_cases = args
        .try_get_many::<String>("only")
        .ok()
        .flatten()
        .map(|it| it.cloned().collect::<std::collections::BTreeSet<_>>());
    match sub {
        "match" => {
            let mut matcher = matcher_from(args)?;
            let opts = MatchStageOptions {
                dataset_dir: req(args, "dataset")?,
                profile_path: req(args, "profile")?,
                catalog_path: req(args, "catalog")?,
                context_catalog: args.get_one::<PathBuf>("context-catalog").cloned(),
                output_dir: req(args, "output")?,
                attempts: req(args, "attempts")?,
                steps: req(args, "steps")?,
                heading_step_deg: req(args, "heading-step")?,
                height_factors: vec![1.0],
                only_cases,
            };
            run_match_stage(&opts, &mut matcher)
        }
        "bank-build" => build_reference_bank(&BankBuildOptions {
            dataset_dir: req(args, "dataset")?,
            profile_path: req(args, "profile")?,
            catalog_path: req(args, "catalog")?,
            xfeat_weights_path: req(args, "model")?,
            output_bank_path: req(args, "output")?,
            attempts: req(args, "attempts")?,
            heading_step_deg: req(args, "heading-step")?,
            height_factors: vec![1.0],
            max_keypoints: req(args, "max-keypoints")?,
            adaptive_stride: req(args, "adaptive-stride")?,
            only_cases,
            xfeat_runtime: crate::corridor_cli::parse_xfeat_runtime(args)?,
        }),
        "bank-match" => run_realtime_bank_match(&BankMatchOptions {
            dataset_dir: req(args, "dataset")?,
            profile_path: req(args, "profile")?,
            catalog_path: req(args, "catalog")?,
            bank_path: req(args, "bank")?,
            xfeat_weights_path: req(args, "model")?,
            lighterglue_weights_path: args.get_one::<PathBuf>("lighterglue").cloned(),
            output_dir: req(args, "output")?,
            shortlist_k: req(args, "shortlist")?,
            query_keypoints: req(args, "query-keypoints")?,
            polish: args.get_flag("polish"),
            only_cases,
            xfeat_runtime: crate::corridor_cli::parse_xfeat_runtime(args)?,
        }),
        "track" => {
            let mut matcher = matcher_from(args)?;
            let opts = MatchStageOptions {
                dataset_dir: req(args, "dataset")?,
                profile_path: req(args, "profile")?,
                catalog_path: req(args, "catalog")?,
                context_catalog: None,
                output_dir: req(args, "output")?,
                attempts: 8,
                steps: 1,
                heading_step_deg: 0.0,
                height_factors: vec![1.0],
                only_cases: None,
            };
            run_track_stage(&opts, &paths(args, "parents")?, &mut matcher)
        }
        "refine" => {
            let mut matcher = matcher_from(args)?;
            let opts = MatchStageOptions {
                dataset_dir: req(args, "dataset")?,
                profile_path: req(args, "profile")?,
                catalog_path: req(args, "catalog")?,
                context_catalog: None,
                output_dir: req(args, "output")?,
                attempts: 1,
                steps: 4,
                heading_step_deg: 0.0,
                height_factors: vec![1.0],
                only_cases: None,
            };
            run_refine_stage(&opts, &paths(args, "parents")?, &mut matcher)
        }
        "terrain-bundle" => run_terrain_bundle_stage(
            &req::<PathBuf>(args, "catalog")?,
            &paths(args, "parents")?,
            &req::<PathBuf>(args, "output")?,
        ),
        "pipeline" => {
            let mut matcher = matcher_from(args)?;
            run_pipeline_recipe(&req::<PathBuf>(args, "recipe")?, &mut matcher)
        }
        other => Err(PipelineError::Invalid(format!(
            "unknown pose subcommand `{other}`"
        ))),
    }
}

fn dispatch_export_or_video(
    sub: &str,
    args: &ArgMatches,
) -> Result<serde_json::Value, PipelineError> {
    match sub {
        "preview" => export_preview(
            &req::<PathBuf>(args, "dataset")?,
            &req::<PathBuf>(args, "catalog")?,
            &paths(args, "runs")?,
            &req::<PathBuf>(args, "output")?,
        ),
        "freeze" => freeze_anchors(&paths(args, "runs")?, &req::<PathBuf>(args, "output")?),
        "publish" => export_publication(
            &req::<PathBuf>(args, "files-dir")?,
            &req::<String>(args, "base-url")?,
            &req::<PathBuf>(args, "output")?,
        ),
        "reproduce" => reproduce_bundle(
            &req::<PathBuf>(args, "publication")?,
            args.get_one::<PathBuf>("local-files").map(PathBuf::as_path),
            &req::<PathBuf>(args, "target")?,
            args.get_one::<String>("prefix").map(String::as_str),
            args.get_one::<usize>("limit").copied(),
        ),
        "live-prior" => {
            let (acq, summary) = seed_live_prior(
                &req::<PathBuf>(args, "replay-dir")?,
                &req::<PathBuf>(args, "video")?,
                &req::<PathBuf>(args, "map-manifest")?,
                req::<f64>(args, "offset-s")?,
                [1280, 720],
                72.0,
                req::<f64>(args, "terrain-m")?,
                30.0,
            )?;
            write_json(&req::<PathBuf>(args, "output")?, &acq)?;
            Ok(summary)
        }
        "import-video" => import_video_media(
            &req::<PathBuf>(args, "video")?,
            &req::<PathBuf>(args, "output")?,
        ),
        "video-prepare" => prepare_video_testset(
            &req::<PathBuf>(args, "video")?,
            &req::<PathBuf>(args, "output")?,
            req::<f64>(args, "fps")?,
            req::<u32>(args, "long-edge")?,
        ),
        "video-priors" => apply_video_priors(
            &req::<PathBuf>(args, "dataset")?,
            &req::<PathBuf>(args, "acquisition")?,
            30.0,
        ),
        "navigate-debrief" => prepare_navigate_debrief(
            &req::<PathBuf>(args, "evidence")?,
            &req::<PathBuf>(args, "output")?,
        ),
        "compose-survey" => {
            compose_survey_debrief(&paths(args, "flights")?, &req::<PathBuf>(args, "output")?)
        }
        other => Err(PipelineError::Invalid(format!(
            "unknown subcommand `{other}`"
        ))),
    }
}
