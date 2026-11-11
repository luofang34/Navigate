use super::*;

#[test]
fn unknown_values_stay_unknown_and_known_values_grow_with_age() {
    let mut a = Accuracy::unknown();
    assert_eq!(a.horizontal_sigma_m(DayNumber(1_000)), None);
    a.horizontal_mm = Some(3_000);
    a.acquired = Some(DayNumber(0));
    assert_eq!(a.horizontal_sigma_m(DayNumber(1_000)), None);
    a.age_growth_mm_per_year = Some(1_000);
    let four_years = DayNumber(1_461);
    assert!((a.horizontal_sigma_m(four_years).unwrap() - 5.0).abs() < 1e-9);
    assert_eq!(a.horizontal_sigma_m(DayNumber(-5)), Some(3.0));
}

#[test]
fn the_encoding_has_integers_and_explicit_nulls() {
    let text = serde_json::to_string(&Accuracy::unknown()).unwrap();
    assert_eq!(
        text,
        r#"{"horizontal_mm":null,"vertical_mm":null,"acquired":null,"age_growth_mm_per_year":null,"surface":"unknown"}"#
    );
    assert!(serde_json::from_str::<Accuracy>(r#"{"horizontal_mm":1.5,"vertical_mm":null,"acquired":null,"age_growth_mm_per_year":null,"surface":"unknown"}"#).is_err());
}
