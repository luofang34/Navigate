use super::*;

#[test]
fn pyramid_budget_contains_all_children_and_rejects_invalid_depth() {
    let options = Options {
        side: 1024,
        levels: 3,
        beam: 8,
    };
    assert_eq!(options.plan((4000, 4000), 772).unwrap().1, 772);
    assert!(options.plan((4000, 4000), 771).is_err());
    for options in [
        Options {
            levels: 0,
            ..options
        },
        Options {
            levels: 6,
            ..options
        },
        Options { beam: 0, ..options },
        Options {
            beam: 17,
            ..options
        },
        Options {
            side: 66,
            ..options
        },
        Options {
            side: 64,
            levels: 3,
            ..options
        },
    ] {
        assert!(options.plan((4000, 4000), 4096).is_err());
    }
}

#[test]
fn parent_scores_keep_ties_stable_and_suppress_mostly_duplicate_regions() {
    let a = [0, 0, 1024, 1024];
    let b = [128, 0, 1024, 1024];
    let c = [1024, 0, 1024, 1024];
    let d = [0, 1024, 1024, 1024];
    assert_eq!(
        parents(&[(a, 10), (b, 9), (c, 8), (d, 8)], 3),
        vec![a, c, d]
    );
    assert_eq!(parents(&[(a, 0), (c, 0)], 4), Vec::<[u32; 4]>::new());
}

#[test]
fn children_stay_inside_parents_cover_edges_and_share_exact_windows() {
    let parents = [[2048, 2048, 1024, 1024], [2560, 2048, 1024, 1024]];
    let windows = children(&parents);
    assert_eq!(windows.len(), 15);
    assert_eq!(windows[0], [2304, 2304, 512, 512]);
    assert_eq!(windows.iter().collect::<BTreeSet<_>>().len(), windows.len());
    for w in &windows {
        assert!(parents.iter().any(|p| w[0] >= p[0]
            && w[1] >= p[1]
            && w[0] + w[2] <= p[0] + p[2]
            && w[1] + w[3] <= p[1] + p[3]));
    }
    for p in parents {
        assert!(windows.contains(&[p[0], p[1], 512, 512]));
        assert!(windows.contains(&[p[0] + 512, p[1] + 512, 512, 512]));
    }
}
