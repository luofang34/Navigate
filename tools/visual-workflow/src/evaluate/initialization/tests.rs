use super::*;

#[test]
fn initial_permutations_preserve_records_and_leave_later_observations_unchanged() {
    let pairs: Vec<_> = (0..100)
        .map(|i| PixelMatch {
            reference: [f64::from(i), 40.0].into(),
            query: [f64::from(100 - i), 50.0].into(),
        })
        .collect();
    let key = |p: &PixelMatch| [p.reference.x, p.reference.y, p.query.x, p.query.y];
    let original: Vec<_> = pairs.iter().map(key).collect();
    for order in [
        MatchOrder::Original,
        MatchOrder::Reverse,
        MatchOrder::QueryX,
        MatchOrder::Shuffle0,
        MatchOrder::Shuffle1,
        MatchOrder::Shuffle2,
        MatchOrder::Shuffle3,
    ] {
        let mut permuted = pairs.clone();
        order.apply(0, &mut permuted);
        let mut ordered = permuted.clone();
        ordered.sort_by(|a, b| a.reference.x.total_cmp(&b.reference.x));
        assert_eq!(ordered.iter().map(key).collect::<Vec<_>>(), original);
        for sequence in [1, 5, 100] {
            let mut later = pairs.clone();
            order.apply(sequence, &mut later);
            assert_eq!(later.iter().map(key).collect::<Vec<_>>(), original);
        }
    }
}
