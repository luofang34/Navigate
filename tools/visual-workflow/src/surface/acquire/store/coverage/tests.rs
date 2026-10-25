use super::*;

#[test]
fn pyramid_catalog_covers_all_possible_children_without_query_selection() {
    let top = Coverage::Pyramid {
        side: 1024,
        levels: 1,
    }
    .windows((4000, 4000))
    .unwrap();
    let two = Coverage::Pyramid {
        side: 1024,
        levels: 2,
    }
    .windows((4000, 4000))
    .unwrap();
    assert_eq!(top.len(), 49);
    assert_eq!(two.len(), 305);
    assert_eq!(two.iter().collect::<BTreeSet<_>>().len(), two.len());
    for parent in top.into_iter().flatten() {
        assert!(two.contains(&Some(parent)));
        for child in pyramid::children(&[parent]) {
            assert!(two.contains(&Some(child)));
        }
    }
    assert!(two.contains(&Some([3488, 3488, 512, 512])));
}

#[test]
fn coverage_rejects_excess_packets_and_invalid_scale_before_encoding() {
    assert!(
        Coverage::Pyramid {
            side: 1024,
            levels: 3
        }
        .windows((4000, 4000))
        .is_err()
    );
    for (side, levels) in [(31, 1), (1023, 2), (1024, 0), (1024, 6)] {
        assert!(
            Coverage::Pyramid { side, levels }
                .windows((4000, 4000))
                .is_err()
        );
    }
    let all = Coverage::Pyramid {
        side: 128,
        levels: 3,
    }
    .windows((256, 256))
    .unwrap();
    let mut level = pyramid::Options {
        side: 128,
        levels: 3,
        beam: 1,
    }
    .plan((256, 256), 4096)
    .unwrap()
    .0;
    for _ in 0..3 {
        assert!(level.iter().all(|w| all.contains(&Some(*w))));
        level = pyramid::children(&level);
    }
}
