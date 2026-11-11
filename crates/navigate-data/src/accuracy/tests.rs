use super::*;

#[test]
fn unknown_values_stay_unknown_and_known_values_grow_with_age() {
    let mut a = Accuracy::unknown();
    assert_eq!(a.horizontal_after(3.0), Sigma::Unknown);
    a.horizontal = Sigma::Known(3.0);
    assert_eq!(a.horizontal_after(3.0), Sigma::Unknown);
    a.age_growth_per_year = Sigma::Known(4.0 / 3.0);
    assert_eq!(a.horizontal_after(3.0), Sigma::Known(5.0));
    assert_eq!(a.horizontal_after(-1.0), Sigma::Known(3.0));
}

#[test]
fn unknown_is_explicit_in_the_encoding() {
    let text = serde_json::to_string(&Accuracy::unknown()).unwrap();
    assert_eq!(
        text,
        r#"{"horizontal":"unknown","vertical":"unknown","acquired":"unknown","age_growth_per_year":"unknown","surface":"unknown"}"#
    );
    let known = serde_json::to_string(&Sigma::Known(2.5)).unwrap();
    assert_eq!(known, r#"{"known":2.5}"#);
}
