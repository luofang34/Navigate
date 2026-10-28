use super::{RECORDS, SolarEclipse, SolarKind, solar_eclipses};
use crate::CalendarDate;

#[test]
fn the_table_holds_every_row_in_catalogue_order() {
    assert_eq!(RECORDS.len(), 11_898);
    let mut previous = f64::MIN;
    for (index, eclipse) in solar_eclipses().enumerate() {
        assert_eq!(usize::from(eclipse.number), index + 1);
        assert!(eclipse.greatest_eclipse_jd_tt() > previous);
        previous = eclipse.greatest_eclipse_jd_tt();
    }
}

#[test]
fn the_kinds_have_the_canon_totals() {
    let count = |kind| solar_eclipses().filter(|e| e.kind == kind).count();
    assert_eq!(count(SolarKind::Partial), 4_200);
    assert_eq!(count(SolarKind::Annular), 3_956);
    assert_eq!(count(SolarKind::Total), 3_173);
    assert_eq!(count(SolarKind::Hybrid), 569);
    // A type code with another first character would decode as partial.
    assert!(
        solar_eclipses()
            .all(|e| e.kind != SolarKind::Partial || e.type_code.as_str().starts_with('P'))
    );
}

#[test]
fn rows_decode_as_the_catalogue_pages_give_them() {
    // 09561  2024 Apr 08  18:18:29  74  300  139  T  n-  0.3431  1.0566  25N 104W  70  198  04m28s
    let eclipse = SolarEclipse::by_number(9561).unwrap();
    assert_eq!(
        eclipse.date,
        CalendarDate {
            year: 2024,
            month: 4,
            day: 8
        }
    );
    assert_eq!(eclipse.greatest_eclipse_td_s, 18 * 3600 + 18 * 60 + 29);
    assert_eq!(
        (eclipse.delta_t_s, eclipse.lunation, eclipse.saros),
        (74, 300, 139)
    );
    assert_eq!(eclipse.type_code.as_str(), "T");
    assert_eq!(eclipse.fortnight_code.as_str(), "n-");
    assert!((eclipse.gamma - 0.3431).abs() < 1e-12);
    assert!((eclipse.magnitude - 1.0566).abs() < 1e-12);
    assert_eq!(
        (
            eclipse.latitude_deg,
            eclipse.longitude_deg,
            eclipse.sun_altitude_deg
        ),
        (25, -104, 70)
    );
    assert_eq!(eclipse.path_width_km, Some(198));
    assert_eq!(eclipse.central_duration_s, Some(268));

    // 00001 -1999 Jun 12  03:14:51  46438 -49456  5  T  -n  -0.2701  1.0733  6N  33W  74  247  06m37s
    let first = SolarEclipse::by_number(1).unwrap();
    assert_eq!(first.date.year, -1999);
    assert_eq!(
        (first.delta_t_s, first.lunation, first.saros),
        (46_438, -49_456, 5)
    );
    assert!((first.gamma + 0.2701).abs() < 1e-12);
    assert_eq!((first.latitude_deg, first.longitude_deg), (6, -33));
    assert_eq!(first.central_duration_s, Some(397));

    // 11898  3000 Oct 19  16:10:16  4428  12378  169  H  nn  -0.2303  1.0049  23S  52W  77  17  00m29s
    let last = SolarEclipse::by_number(11_898).unwrap();
    assert_eq!(last.kind, SolarKind::Hybrid);
    assert_eq!(last.fortnight_code.as_str(), "nn");
    assert_eq!(
        (last.path_width_km, last.central_duration_s),
        (Some(17), Some(29))
    );
    assert_eq!(SolarEclipse::by_number(0), None);
    assert_eq!(SolarEclipse::by_number(11_899), None);
}

#[test]
fn a_partial_eclipse_has_no_path() {
    // 09484  1989 Mar 07  18:08:41  56  -134  149  P  t-  1.0981  0.8268  61N 170W  0
    let eclipse = SolarEclipse::by_number(9484).unwrap();
    assert_eq!(eclipse.kind, SolarKind::Partial);
    assert_eq!(eclipse.sun_altitude_deg, 0);
    assert_eq!(
        (eclipse.path_width_km, eclipse.central_duration_s),
        (None, None)
    );
}

#[test]
fn greatest_eclipse_is_in_tt() {
    let eclipse = SolarEclipse::by_number(9561).unwrap();
    let expected = 2_460_408.5 + (18.0 * 3600.0 + 18.0 * 60.0 + 29.0) / 86_400.0;
    assert!((eclipse.greatest_eclipse_jd_tt() - expected).abs() < 1e-9);
}
