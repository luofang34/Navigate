use super::*;
#[test]
fn geometry_requires_both_sources_at_the_same_location() {
    let coverage = Coverage {
        imagery: BTreeMap::from([(2, HashSet::from([(1, 1), (3, 3)]))]),
        terrain: BTreeMap::from([(1, HashSet::from([(0, 0)]))]),
    };
    assert!(coverage.supports([0.3, 0.3]));
    for xy in [
        [0.6, 0.3],
        [0.1, 0.1],
        [0.9, 0.9],
        [1.0, 1.0],
        [-0.1, 0.3],
        [f64::NAN, 0.3],
    ] {
        assert!(!coverage.supports(xy));
    }
}
