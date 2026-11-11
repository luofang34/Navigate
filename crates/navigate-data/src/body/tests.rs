use super::*;

#[test]
fn day_numbers_count_from_2000() {
    assert_eq!(DayNumber::from_ymd(2000, 1, 1).unwrap(), DayNumber(0));
    assert_eq!(DayNumber::from_ymd(2000, 3, 1).unwrap(), DayNumber(60));
    assert_eq!(DayNumber::from_ymd(1999, 12, 31).unwrap(), DayNumber(-1));
    assert_eq!(DayNumber::from_ymd(2026, 6, 15).unwrap(), DayNumber(9_662));
    assert!(DayNumber::from_ymd(2025, 2, 29).is_err());
    assert!(DayNumber::from_ymd(2024, 13, 1).is_err());
}

#[test]
fn unknown_names_decode_as_unsupported() {
    let body: Body = serde_json::from_str("\"mars\"").unwrap();
    assert_eq!(body, Body::Unsupported);
    assert_eq!(serde_json::to_string(&Body::Earth).unwrap(), "\"earth\"");
}
