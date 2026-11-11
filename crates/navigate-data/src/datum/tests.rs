use super::*;
use navigate_geodesy::ZeroUndulation;

struct Constant(f64);

impl GeoidUndulation for Constant {
    fn ellipsoid(&self) -> Ellipsoid {
        Ellipsoid::WGS84
    }
    fn undulation_m(&self, _: f64, _: f64) -> Option<f64> {
        Some(self.0)
    }
}

#[test]
fn orthometric_heights_convert_only_with_the_named_model() {
    let datum = VerticalDatum::Orthometric {
        geoid_model: "egm96".into(),
    };
    let model = Constant(49.5);
    assert_eq!(
        datum.to_ellipsoidal(300.0, 0.7, -0.02, Some(("egm96", &model))),
        EllipsoidalHeight::Known(349.5)
    );
    let unknown = EllipsoidalHeight::UnknownUndulation {
        height_m: 300.0,
        bound_m: UNKNOWN_UNDULATION_BOUND_M,
    };
    assert_eq!(
        datum.to_ellipsoidal(300.0, 0.7, -0.02, Some(("egm2008", &model))),
        unknown
    );
    assert_eq!(
        datum.to_ellipsoidal::<Constant>(300.0, 0.7, -0.02, None),
        unknown
    );
}

#[test]
fn ellipsoidal_and_unknown_datums() {
    let zero = ZeroUndulation(Ellipsoid::WGS84);
    assert_eq!(
        VerticalDatum::Ellipsoidal.to_ellipsoidal(10.0, 0.0, 0.0, Some(("egm96", &zero))),
        EllipsoidalHeight::Known(10.0)
    );
    assert!(matches!(
        VerticalDatum::Unknown.to_ellipsoidal(10.0, 0.0, 0.0, Some(("egm96", &zero))),
        EllipsoidalHeight::UnknownUndulation { .. }
    ));
}
