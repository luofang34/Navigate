//! A pooled score can remove the strongest fine-scale mode.
use super::*;
#[test]
fn beam_must_report_loss_when_pooling_removes_fine_texture() {
    let values: Vec<_> = (0..16)
        .map(|i| if (i / 4 + i % 4) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let q = Grid::new(4, 4, 1, values, vec![true; 16]).unwrap();
    let mut m = Grid::new(12, 12, 1, vec![0.0; 144], vec![true; 144]).unwrap();
    for y in 0..4 {
        for x in 0..4 {
            m.values[(y + 8) * 12 + x + 8] = q.values[y * 4 + x];
        }
    }
    let result = compare(&q, &m, [4, 4], 4, 16).unwrap();
    let rows = result["rows"].as_array().unwrap();
    for row in rows {
        assert_eq!(row["oracle_lag"], json!([8, 8]));
        if row["beam"] == 1 {
            assert_eq!(row["full_fine_argmax_retained"], false);
        }
        if row["beam"] == 4 {
            assert_eq!(row["full_fine_argmax_retained"], true);
        }
    }
}
