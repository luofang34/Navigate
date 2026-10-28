use super::{LunarEclipse, LunarKind, RECORDS, lunar_eclipses};

#[test]
fn the_table_holds_every_row_in_catalogue_order() {
    assert_eq!(RECORDS.len(), 12_064);
    let mut previous = f64::MIN;
    for (index, eclipse) in lunar_eclipses().enumerate() {
        assert_eq!(usize::from(eclipse.number), index + 1);
        assert!(eclipse.greatest_eclipse_jd_tt() > previous);
        previous = eclipse.greatest_eclipse_jd_tt();
    }
}

#[test]
fn the_kinds_have_the_canon_totals() {
    let count = |kind| lunar_eclipses().filter(|e| e.kind == kind).count();
    assert_eq!(count(LunarKind::Penumbral), 4_378);
    assert_eq!(count(LunarKind::Partial), 4_207);
    assert_eq!(count(LunarKind::Total), 3_479);
    // A type code with another first character would decode as penumbral.
    assert!(
        lunar_eclipses()
            .all(|e| e.kind != LunarKind::Penumbral || e.type_code.as_str().starts_with('N'))
    );
}

#[test]
fn rows_decode_as_the_catalogue_pages_give_them() {
    // 09701  2022 Nov 08  11:00:22  73  282  136  T+  p-  0.2570  2.4143  1.3589  353.9  219.8  85.0  17N  169W
    let eclipse = LunarEclipse::by_number(9701).unwrap();
    assert_eq!(
        (eclipse.date.year, eclipse.date.month, eclipse.date.day),
        (2022, 11, 8)
    );
    assert_eq!(eclipse.greatest_eclipse_td_s, 11 * 3600 + 22);
    assert_eq!(
        (eclipse.delta_t_s, eclipse.lunation, eclipse.saros),
        (73, 282, 136)
    );
    assert_eq!(eclipse.kind, LunarKind::Total);
    assert_eq!(eclipse.type_code.as_str(), "T+");
    assert_eq!(eclipse.fortnight_code.as_str(), "p-");
    assert!((eclipse.gamma - 0.2570).abs() < 1e-12);
    assert!((eclipse.penumbral_magnitude - 2.4143).abs() < 1e-12);
    assert!((eclipse.umbral_magnitude - 1.3589).abs() < 1e-12);
    assert_eq!(eclipse.penumbral_duration_min, Some(353.9));
    assert_eq!(eclipse.partial_duration_min, Some(219.8));
    assert_eq!(eclipse.total_duration_min, Some(85.0));
    assert_eq!(
        (eclipse.zenith_latitude_deg, eclipse.zenith_longitude_deg),
        (17, -169)
    );

    // 00001 -1999 Jun 26  14:13:28  46437 -49456  17  N  t-  -1.0981  0.8791 -0.1922  268.8  -  -  24S  22W
    let first = LunarEclipse::by_number(1).unwrap();
    assert_eq!(first.kind, LunarKind::Penumbral);
    assert!((first.umbral_magnitude + 0.1922).abs() < 1e-12);
    assert_eq!(first.penumbral_duration_min, Some(268.8));
    assert_eq!(
        (first.partial_duration_min, first.total_duration_min),
        (None, None)
    );
    assert_eq!(
        (first.zenith_latitude_deg, first.zenith_longitude_deg),
        (-24, -22)
    );

    // 12064  3000 Nov 04  05:45:18  4428  12378  181  N  h-  -1.2170  0.6326 -0.3828  214.8  -  -  14N  72W
    let last = LunarEclipse::by_number(12_064).unwrap();
    assert_eq!(last.fortnight_code.as_str(), "h-");
    assert!((last.gamma + 1.2170).abs() < 1e-12);
    assert_eq!(LunarEclipse::by_number(12_065), None);
}
