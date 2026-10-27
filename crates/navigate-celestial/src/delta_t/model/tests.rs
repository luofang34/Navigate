use super::*;

/// Morrison and Stephenson (2004) Delta T in seconds, from the NASA Five
/// Millennium Canon. The polynomials fit these values.
#[rustfmt::skip]
const PUBLISHED: [[f64; 2]; 24] = [
    [-500.0, 17_190.0], [-400.0, 15_530.0], [-300.0, 14_080.0], [-200.0, 12_790.0],
    [-100.0, 11_640.0], [0.0, 10_580.0], [100.0, 9_600.0], [200.0, 8_640.0],
    [300.0, 7_680.0], [400.0, 6_700.0], [500.0, 5_710.0], [600.0, 4_740.0],
    [700.0, 3_810.0], [800.0, 2_960.0], [900.0, 2_200.0], [1000.0, 1_570.0],
    [1100.0, 1_090.0], [1200.0, 740.0], [1300.0, 490.0], [1400.0, 320.0],
    [1500.0, 200.0], [1600.0, 120.0], [1700.0, 9.0], [1750.0, 13.0],
];

#[test]
fn polynomials_fit_the_published_values() {
    for [year, published] in PUBLISHED {
        let error = (polynomial(year) - published).abs();
        // The published values are rounded to 10 s before 1600, and the
        // polynomials fit them to about 0.1 percent.
        let tolerance = 10.0 + 0.001 * published;
        assert!(
            error < tolerance,
            "{year}: {} vs {published}",
            polynomial(year)
        );
    }
}

#[test]
fn canon_tidal_correction_is_about_minus_202_s_at_minus_2000() {
    let correction = at_year(-2000.0).0 - polynomial(-2000.0);
    assert!((correction + 202.0).abs() < 1.0, "{correction}");
    assert!((at_year(1955.0).0 - polynomial(1955.0)).abs() < 1e-12);
}

#[test]
fn uncertainty_follows_morrison_and_stephenson() {
    assert_eq!(uncertainty(-500.0), 430.0);
    assert_eq!(uncertainty(-450.0), 410.0);
    let u: f64 = (-2000.0 - 1820.0) / 100.0;
    assert!((uncertainty(-2000.0) - 0.8 * u * u).abs() < 1e-9);
    // The fit agrees with the first table row within a few percent.
    assert!((uncertainty(-500.0 - 1e-9) - 430.0).abs() < 15.0);
}
