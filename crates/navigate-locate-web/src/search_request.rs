//! Search messages to and from the pipeline types, independent of the browser.
use image::{RgbImage, imageops::FilterType};
use navigate_locate::SearchOutcome;
use navigate_locate::messages::search::{SearchRegion, SearchRequest, SearchResponse};
use navigate_place::{
    Fallback, HeadingPrior, HeightPrior, PitchPrior, Region, SearchRequest as Plan,
};
use navigate_visual::CameraModel;

/// Long edge of the photo in the locate step, as `navigate_locate::search` uses it.
const LOCATE_EDGE: u32 = 640;

/// Decode `photo`, reduce it to the request's maximum width, and give the plan
/// request at that size.
///
/// # Errors
///
/// Returns a reason for a person for an unsupported version, a value out of range,
/// or a photo that cannot be decoded.
pub fn prepare_search(request: &SearchRequest, photo: &[u8]) -> Result<(RgbImage, Plan), String> {
    if request.version != navigate_locate::messages::search::SEARCH_MESSAGES_VERSION {
        return Err(format!("unsupported request version {}", request.version));
    }
    if !(request.focal_px.is_finite() && request.focal_px > 0.0)
        || !(64..=4096).contains(&request.max_width_px)
    {
        return Err("the focal length or the maximum width is out of range".into());
    }
    let decoded = image::load_from_memory(photo)
        .map_err(|e| format!("the photo cannot be read: {e}"))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    let scale = (f64::from(request.max_width_px) / f64::from(width)).min(1.0);
    let image = if scale < 1.0 {
        let w = (f64::from(width) * scale).round().max(1.0) as u32;
        let h = (f64::from(height) * scale).round().max(1.0) as u32;
        image::imageops::resize(&decoded, w, h, FilterType::Triangle)
    } else {
        decoded
    };
    let focal_px = request.focal_px * f64::from(image.width()) / f64::from(width);
    let region = match &request.region {
        SearchRegion::Circle {
            latitude_rad,
            longitude_rad,
            radius_m,
        } => Region::Circle {
            centre: [latitude_rad.to_degrees(), longitude_rad.to_degrees()],
            radius_m: *radius_m,
        },
        SearchRegion::Polygon { points_rad } => Region::Polygon {
            points: points_rad
                .iter()
                .map(|[a, b]| [a.to_degrees(), b.to_degrees()])
                .collect(),
        },
    };
    let priors = &request.priors;
    let plan = Plan {
        region,
        image_px: [image.width(), image.height()],
        focal_px,
        heading: priors.heading_rad.map(|p| HeadingPrior {
            deg: p.value.to_degrees(),
            sigma_deg: p.sigma.to_degrees(),
        }),
        height: priors.height_m.map(|p| HeightPrior {
            m: p.value,
            sigma_m: p.sigma,
        }),
        pitch: priors.pitch_rad.map(|p| PitchPrior {
            deg: p.value.to_degrees(),
            sigma_deg: p.sigma.to_degrees(),
        }),
    };
    Ok((image, plan))
}

/// The response message for a search outcome on a photo `width` pixels wide with
/// focal length `focal_px`.
pub fn search_response_of(
    outcome: SearchOutcome,
    width: u32,
    height: u32,
    focal_px: f64,
) -> SearchResponse {
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    match outcome {
        SearchOutcome::Found {
            located, checked, ..
        } => {
            let scale = (f64::from(LOCATE_EDGE) / f64::from(width.max(height))).min(1.0);
            let w = (f64::from(width) * scale).round() as u32;
            let h = (f64::from(height) * scale).round() as u32;
            let camera = CameraModel {
                width: w,
                height: h,
                fx: focal_px * scale,
                fy: focal_px * scale,
                cx: (f64::from(w) - 1.0) / 2.0,
                cy: (f64::from(h) - 1.0) / 2.0,
            };
            match crate::response_of(navigate_locate::Outcome::Found(located), &camera) {
                navigate_locate::messages::LocateResponse::Found { camera } => {
                    SearchResponse::Found {
                        camera,
                        checked: count(checked),
                    }
                }
                _ => SearchResponse::Failed {
                    reason: "the located camera has no response".into(),
                },
            }
        }
        SearchOutcome::NotFound {
            ranked,
            checked,
            best_inliers,
        } => SearchResponse::NotFound {
            ranked: count(ranked),
            checked: count(checked),
            best_inliers: count(best_inliers),
        },
        SearchOutcome::Fallback(Fallback::DenseGrid) => SearchResponse::Uncovered {
            dense_search: true,
            reason: "no place index covers this region; it is small enough for the dense search"
                .into(),
        },
        SearchOutcome::Fallback(Fallback::Uncovered { reason }) => SearchResponse::Uncovered {
            dense_search: false,
            reason,
        },
        SearchOutcome::Fallback(Fallback::None) => SearchResponse::Failed {
            reason: "the plan has no queries".into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use navigate_locate::messages::search::{Prior, SEARCH_MESSAGES_VERSION, SearchPriors};

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        RgbImage::from_pixel(width, height, image::Rgb([90, 120, 60]))
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        bytes
    }

    #[test]
    fn a_request_becomes_a_plan_at_the_working_size() {
        let request = SearchRequest {
            version: SEARCH_MESSAGES_VERSION,
            region: SearchRegion::Circle {
                latitude_rad: 41_f64.to_radians(),
                longitude_rad: (-75_f64).to_radians(),
                radius_m: 5_000.0,
            },
            focal_px: 2400.0,
            max_width_px: 1600,
            priors: SearchPriors {
                heading_rad: Some(Prior {
                    value: 1.0,
                    sigma: 0.1,
                }),
                ..SearchPriors::default()
            },
            index_url: "x/".into(),
        };
        let (image, plan) = prepare_search(&request, &png(3200, 2400)).unwrap();
        assert_eq!(image.dimensions(), (1600, 1200));
        assert!((plan.focal_px - 1200.0).abs() < 1e-9);
        assert!((plan.heading.unwrap().deg - 57.2958).abs() < 1e-3);
        assert!(plan.region.contains([41.0, -75.0]));
        let old = SearchRequest {
            version: 0,
            ..request
        };
        assert!(prepare_search(&old, &png(10, 10)).is_err());
    }

    #[test]
    fn an_uncovered_region_says_whether_the_dense_search_can_serve_it() {
        let dense = search_response_of(
            SearchOutcome::Fallback(Fallback::DenseGrid),
            640,
            480,
            500.0,
        );
        assert!(matches!(
            dense,
            SearchResponse::Uncovered {
                dense_search: true,
                ..
            }
        ));
        let far = search_response_of(
            SearchOutcome::Fallback(Fallback::Uncovered {
                reason: "far".into(),
            }),
            640,
            480,
            500.0,
        );
        assert!(matches!(
            far,
            SearchResponse::Uncovered {
                dense_search: false,
                ..
            }
        ));
    }
}
