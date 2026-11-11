use super::*;

/// The names that hosts count. A change here is a change of the contract
/// with each host.
#[test]
fn the_record_names_do_not_change() {
    assert_eq!(MEASUREMENT, "terrain_position");
    assert_eq!([KEYFRAME, RENDER], ["keyframe", "render"]);
    assert_eq!(
        [
            FEATURELESS_FRAME,
            ACQUIRED_AFTER_DELIVERY,
            CAPTURE_LAG,
            NO_SOLUTION,
            BELIEF_OUT_OF_RANGE,
            OUTSIDE_COVERAGE,
            RENDER_REFUSED,
            OUTSIDE_GRID,
            SHARED_MAP_ERROR,
            UNTRACKED_MAP_CELL,
            FIX_REFUSED,
        ],
        [
            "featureless_frame",
            "acquired_after_delivery",
            "capture_lag",
            "no_solution",
            "belief_out_of_range",
            "outside_coverage",
            "render_refused",
            "outside_grid",
            "shared_map_error",
            "untracked_map_cell",
            "fix_refused",
        ]
    );
    let source = || Box::new(std::io::Error::other("backend")) as Box<_>;
    let rejections = [
        VisualError::InsufficientMatches {
            found: 1,
            required: 20,
        },
        VisualError::InsufficientSpatialSupport {
            found: 1,
            required: 20,
        },
        VisualError::DegenerateGeometry,
        VisualError::OutsidePrior {
            meters: 1.0,
            radians: 0.0,
        },
        VisualError::FrameOrder {
            previous_ns: 2,
            received_ns: 1,
        },
        VisualError::Backend {
            backend: "matcher".into(),
            source: source(),
        },
        VisualError::Invalid { field: "frame" },
        VisualError::Dimensions {
            query: (1, 1),
            reference: (2, 2),
        },
    ];
    assert_eq!(
        rejections.iter().map(rejection).collect::<Vec<_>>(),
        [
            "insufficient_matches",
            "insufficient_spatial_support",
            "degenerate_geometry",
            "outside_prior",
            "frame_order",
            "matcher_failed",
            "invalid_view",
            "invalid_view",
        ]
    );
}
