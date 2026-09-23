use super::*;

#[test]
fn stac_year_accepts_provider_strings_and_numeric_values() {
    assert_eq!(
        source_year(&json!({"properties":{"naip:year":"2023"}})),
        Some(2023)
    );
    assert_eq!(
        source_year(&json!({"properties":{"naip:year":2023}})),
        Some(2023)
    );
    assert_eq!(
        source_year(&json!({"properties":{"naip:year":"unknown"}})),
        None
    );
}
