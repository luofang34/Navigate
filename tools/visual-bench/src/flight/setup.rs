//! Prepare immutable data and resident execution adapters once per video.
use super::*;
pub(super) struct Prepared {
    pub camera: CameraModel,
    pub timestamps: Vec<u64>,
    pub metadata: serde_json::Value,
    pub tracker: tracker::Tracker,
}
pub(super) async fn prepare_blocking(
    args: &FlightArgs,
    backend: crate::backend::BackendKind,
) -> Result<Prepared, BenchError> {
    let config = config_blocking(&args.prior)?;
    let source_camera = config.camera.model();
    source_camera.validate()?;
    let camera = scaled_camera(source_camera, args.width)?;
    let timestamps = match &args.timeline_cache {
        Some(cache) => probe::cache::timestamps_blocking(&args.video, source_camera, cache)?,
        None => probe::timestamps_blocking(&args.video, source_camera)?,
    };
    let package = MapPackage::open_blocking(&args.package)?;
    let prior = config.prior.prior(package.frame)?;
    let poses = match &args.candidates {
        Some(path) => serde_json::from_slice::<Vec<PoseRecord>>(&crate::read_blocking(path)?)
            .map_err(|source| BenchError::Json {
                path: path.clone(),
                source,
            })?
            .into_iter()
            .map(PoseRecord::pose)
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![prior.pose],
    };
    let mut metadata = json!({"source_video":args.video,"source_camera":config.camera,
        "camera":crate::stream::CameraRecord::from(camera),"map_manifest_sha256":package.revision.manifest_sha256,
        "map_release_id":package.revision.release_id,"anchor_lat_lon":package.manifest.anchor_lat_lon,
        "renderer_minimum_revision":crate::RENDERER_REVISION.trim(),"clock_domain":"video-relative-pts",
        "surface_geometry":"rendered terrain; not independently verified",
        "navigation_prior":config.prior,"realtime":args.realtime,"adaptive":args.adaptive});
    metadata["source_video_sha256"] = probe::cache::digest_blocking(&args.video)?.into();
    metadata["model_sha256"] = probe::cache::digest_blocking(&args.inference.model)?.into();
    metadata["runtime_sha256"] = probe::cache::digest_blocking(&args.inference.runtime)?.into();
    if let Some(detector) = &args.inference.detector {
        metadata["detector_sha256"] = probe::cache::digest_blocking(detector)?.into();
    }
    metadata["settings"] = json!({"fps":args.fps,"utilization":args.utilization,"budget_file":args.budget_file,
        "map_interval_s":args.map_interval,"reanchor":args.reanchor,"fixed_tilt":args.fixed_tilt,
        "surface_tracks":args.surface_tracks,"keyframe_interval_s":args.keyframe_interval,"start_s":args.start,"duration_s":args.duration,"dense_only":args.dense_only,"warm_start":args.warm_start});
    let matcher = args.inference.load_blocking()?;
    metadata["matcher_identity"] = matcher.identity().into();
    let renderer = ReferenceRenderer::new(package, camera).await?;
    let identity = navigate_visual::ReferenceRenderer::identity(&renderer);
    metadata["style_sha256"] = identity.style_sha256.into();
    metadata["provider_request"] = format!("{:?}", args.inference.provider()).into();
    metadata["provider_scope"] =
        "requested provider; this does not prove operator placement".into();
    let tracker = tracker::Tracker::new(
        Box::new(renderer),
        matcher,
        Box::new(crate::backend::Backend::new(backend).await?),
        prior,
        poses,
        tracker::Policy {
            dense_only: args.dense_only,
            map_interval_ns: (args.map_interval * 1e9) as u64,
            reanchor: args.reanchor,
            fixed_tilt: args.fixed_tilt,
            surface_tracks: args.surface_tracks,
            keyframe_interval_ns: (args.keyframe_interval * 1e9) as u64,
        },
    )?;
    Ok(Prepared {
        camera,
        timestamps,
        metadata,
        tracker,
    })
}
