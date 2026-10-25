//! Record causal session output without labelling declared bounds as truth.
use super::Output;
use navigate_recording::Pose;
use navigate_visual_session::{
    AnchorDecision, Confirmation, EventBatch, FramePose, MapPose, SessionEvent, SessionUsage,
    to_camera,
};
use serde_json::{Value, json};

pub(super) fn output(
    pose: &FramePose,
    decision: Option<&AnchorDecision>,
    events: &EventBatch,
    usage: SessionUsage,
) -> Output {
    let mut record = json!({
        "continuity":pose.frame.continuity.0,"revision":pose.revision,
        "anchor":decision.map(anchor),
        "odometry":pose.odometry.as_ref().map(|o| json!({
            "segment":o.segment.0,"pose":Pose::from(to_camera(&o.pose)),"path_m":o.path_m,
            "scale":o.scale.as_ref().map(|s|format!("{s:?}"))})),
        "revisions":events.events.iter().filter_map(|event| match event {
            SessionEvent::TrajectoryRevised(r) => Some(json!({
                "revision":r.revision,"cause":format!("{:?}",r.cause),
                "ranges":r.ranges.iter().map(|range|json!({"from":range.from.index,"to":range.to.index,
                    "max_shift_m":range.max_shift_m,"max_rotation_rad":range.max_rotation_rad})).collect::<Vec<_>>()
            })),
            _ => None,
        }).collect::<Vec<_>>(),
        "dropped_events":events.dropped,"full_revision_required":events.full_revision_required,
        "usage":{"frames":usage.frames,"keyframes":usage.keyframes,"anchors":usage.anchors,"pending_anchors":usage.pending_anchors},
        "scope":"Causal session output. Bounds use declared map errors and the standard drift policy; they are not measured accuracy. Lens, mount, vertical datum, and map age are unknown. No age growth is modelled. Seeded map motion shares map error. The session does not change frontend matching or tracking."
    });
    let position = match &pose.map {
        MapPose::Located {
            pose,
            position_bound_m,
            nearest_anchor_ns,
            confirmation,
            usability,
            local_frame,
            ..
        } => {
            let camera = to_camera(pose);
            let p = local_frame.geodetic(camera.position);
            record["map"] = json!({"pose":Pose::from(camera),"geodetic":p,
                "position_bound_m":position_bound_m,"nearest_anchor_ns":nearest_anchor_ns,
                "confirmed":matches!(confirmation,Confirmation::Confirmed { .. }),
                "navigation_usable":usability.navigation,"ground_projection_usable":usability.ground_projection,
                "recent_map_support":usability.recent_map_support});
            Some(p)
        }
        MapPose::Unlocated(reason) => {
            record["unlocated"] = json!(format!("{reason:?}"));
            None
        }
    };
    Output { record, position }
}

fn anchor(decision: &AnchorDecision) -> Value {
    match decision {
        AnchorDecision::Accepted { eligibility } => {
            json!({"status":"accepted","fusion_eligibility":format!("{eligibility:?}")})
        }
        AnchorDecision::Relocated {
            agreeing,
            retracted,
        } => json!({"status":"relocated","agreeing":agreeing,"retracted":retracted}),
        AnchorDecision::SameViewpoint => json!({"status":"same_viewpoint"}),
        AnchorDecision::Rejected(reason) => {
            json!({"status":"rejected","reason":format!("{reason:?}")})
        }
    }
}
