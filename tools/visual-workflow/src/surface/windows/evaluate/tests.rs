use super::*;

#[test]
fn window_selection_keeps_score_alignment_and_pair_order() {
    let pairs = [10.0, 200.0, 30.0]
        .into_iter()
        .map(|x| PixelMatch {
            reference: [x, 40.0].into(),
            query: [x + 1.0, 50.0].into(),
        })
        .collect::<Vec<_>>();
    let ordinary = Observations {
        pairs: pairs.clone(),
        scores: None,
    }
    .select([0, 0, 100, 100]);
    let scored = Observations {
        pairs,
        scores: Some(vec![0.3, 0.9, 0.7]),
    }
    .select([0, 0, 100, 100]);
    assert_eq!(scored.scores, Some(vec![0.3, 0.7]));
    assert_eq!(ordinary.pairs.len(), scored.pairs.len());
    for (a, b) in ordinary.pairs.iter().zip(&scored.pairs) {
        assert_eq!(a.reference, b.reference);
        assert_eq!(a.query, b.query);
    }
}
