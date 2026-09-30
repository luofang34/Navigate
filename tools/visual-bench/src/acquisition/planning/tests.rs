use super::*;
use navigate_visual::{
    MapRevision,
    place_retrieval::{ReferenceCatalog, RetrievalError},
};

struct Ranked {
    identity: ReferenceCatalog,
    requests: Vec<Vec<ReferenceId>>,
    invalid: bool,
}
impl ImageRetriever for Ranked {
    fn identity(&self) -> &str {
        "test-retriever"
    }
    fn catalog(&self) -> &ReferenceCatalog {
        &self.identity
    }
    fn rank_blocking(
        &mut self,
        _: &image::RgbImage,
        eligible: &[ReferenceId],
        _: usize,
    ) -> Result<Vec<ReferenceId>, RetrievalError> {
        self.requests.push(eligible.to_vec());
        Ok(if self.invalid {
            vec![ReferenceId(999)]
        } else {
            eligible.iter().take(8).copied().collect()
        })
    }
}
fn fixture() -> (Catalog, PosePrior, Ranked) {
    let catalog = Catalog {
        release: "fixture".into(),
        gallery: vec![
            Entry {
                id: 4,
                center_enu_m: [0.0, 0.0],
                width_m: 100.0,
            },
            Entry {
                id: 8,
                center_enu_m: [10.0, 10.0],
                width_m: 200.0,
            },
            Entry {
                id: 12,
                center_enu_m: [1000.0, 0.0],
                width_m: 100.0,
            },
        ],
    };
    let prior = PosePrior {
        pose: CameraPose {
            position: Vector3::new(0.0, 0.0, 110.0),
            orientation: UnitQuaternion::identity(),
        },
        position_radius_m: 100.0,
        attitude_radius_rad: std::f64::consts::PI,
    };
    let retriever = Ranked {
        identity: ReferenceCatalog {
            map: MapRevision {
                release_id: "fixture".into(),
                manifest_sha256: "a".repeat(64),
            },
            manifest_sha256: "b".repeat(64),
        },
        requests: vec![],
        invalid: false,
    };
    (catalog, prior, retriever)
}
#[test]
fn navigation_bounds_filter_retrieval_and_repeated_channels_do_not_duplicate_ids() {
    let (catalog, prior, mut retriever) = fixture();
    let plan = Plan::new(&catalog, &[100.0, 200.0], &prior).expect("plan");
    let ids = plan
        .rank_blocking(&mut retriever, &image::RgbImage::new(8, 8))
        .expect("rank");
    assert_eq!(ids, vec![ReferenceId(4), ReferenceId(8)]);
    assert_eq!(
        retriever.requests,
        vec![
            vec![ReferenceId(4)],
            vec![ReferenceId(8)],
            vec![ReferenceId(4)],
            vec![ReferenceId(8)]
        ]
    );
    retriever.invalid = true;
    assert!(
        plan.rank_blocking(&mut retriever, &image::RgbImage::new(8, 8))
            .is_err()
    );
}
#[test]
fn malformed_catalogs_and_scale_budgets_fail_before_inference() {
    let (mut catalog, prior, _) = fixture();
    assert!(Plan::new(&catalog, &[], &prior).is_err());
    assert!(Plan::new(&catalog, &[100.0, 100.0], &prior).is_err());
    catalog.gallery[1].id = 4;
    assert!(Plan::new(&catalog, &[100.0], &prior).is_err());
}

#[test]
fn duplicate_channels_fill_the_unique_reference_budget_in_scale_order() {
    let (mut catalog, prior, mut retriever) = fixture();
    let scales = [100.0, 200.0, 50.0, 400.0];
    catalog.gallery = scales
        .iter()
        .enumerate()
        .flat_map(|(group, &width_m)| {
            (0..8).map(move |rank| Entry {
                id: group as u64 * 10 + rank,
                center_enu_m: [0.0, 0.0],
                width_m,
            })
        })
        .collect();
    let plan = Plan::new(&catalog, &scales, &prior).expect("plan");
    let ids = plan
        .rank_blocking(&mut retriever, &image::RgbImage::new(8, 8))
        .expect("rank");
    let expected: Vec<_> = (0..4)
        .flat_map(|rank| (0..4).map(move |group| ReferenceId(group * 10 + rank)))
        .collect();
    assert_eq!(ids, expected);
    assert_eq!(ids.len() * 8, 128, "eight orientations keep the pose cap");
}
