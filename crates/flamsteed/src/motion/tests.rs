use crate::Catalog;

const REFERENCE: &str = include_str!("../../fixtures/star-space-motion.txt");

fn separation_arcsec(a: [f64; 3], b: [f64; 3]) -> f64 {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let sin = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
    let cos: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    sin.atan2(cos).to_degrees() * 3600.0
}

#[test]
fn space_motion_matches_the_reference_from_minus_2000_to_3000() {
    let stars: Vec<_> = Catalog::bright_star_catalogue().stars().collect();
    let mut rows = 0;
    for line in REFERENCE.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        let star = stars
            .iter()
            .find(|star| f64::from(star.hr) == v[0])
            .unwrap();
        let error = separation_arcsec(star.direction_at(v[1]), [v[2], v[3], v[4]]);
        // The packed positions round to 0.3 milliarcsecond.
        assert!(error < 0.01, "HR {} at {} years: {error}\"", v[0], v[1]);
        rows += 1;
    }
    assert_eq!(rows, 150);
}

#[test]
fn at_j2000_the_direction_is_the_catalogue_position() {
    for star in Catalog::bright_star_catalogue().stars().take(50) {
        let (sin_ra, cos_ra) = star.right_ascension_rad.sin_cos();
        let (sin_dec, cos_dec) = star.declination_rad.sin_cos();
        let expected = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
        assert!(separation_arcsec(star.direction_at(0.0), expected) < 1e-9);
    }
}
