#![allow(clippy::expect_used)]
use crate::{DataUri, check_range};
#[test]
fn invalid_ranges_are_rejected_before_io() {
    let uri = DataUri::parse("pilotage://chunks/test").expect("valid URI");
    assert!(check_range(&uri, 32, 31, 1).is_ok());
    assert!(check_range(&uri, 32, 32, 0).is_ok());
    assert!(check_range(&uri, 32, 32, 1).is_err());
    assert!(check_range(&uri, u64::MAX, u64::MAX, 1).is_err());
}
