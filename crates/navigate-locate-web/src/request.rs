//! Photo preparation and the response message, independent of the browser.
use image::{GrayImage, imageops::FilterType};
use navigate_contract::locate::{LocateRequest, LocateResponse, LocatedCamera};
use navigate_locate::Outcome;
use navigate_visual::CameraModel;

/// A photo at its working size, with the calibration at that size.
#[derive(Debug)]
pub struct PreparedPhoto {
    /// Grayscale photo.
    pub image: GrayImage,
    /// Pinhole calibration of `image`.
    pub camera: CameraModel,
}

/// Decode `photo`, reduce it to the request's maximum width, and scale the focal length.
///
/// # Errors
///
/// Returns a reason for a person when the photo cannot be decoded or a value is
/// out of range.
pub fn prepare_photo(request: &LocateRequest, photo: &[u8]) -> Result<PreparedPhoto, String> {
    if request.version != navigate_contract::locate::LOCATE_MESSAGES_VERSION {
        return Err(format!("unsupported request version {}", request.version));
    }
    if !(request.focal_px.is_finite() && request.focal_px > 0.0)
        || !(64..=4096).contains(&request.max_width_px)
    {
        return Err("the focal length or the maximum width is out of range".into());
    }
    let decoded =
        image::load_from_memory(photo).map_err(|e| format!("the photo cannot be read: {e}"))?;
    let gray = decoded.to_luma8();
    let (width, height) = gray.dimensions();
    let scale = (f64::from(request.max_width_px) / f64::from(width)).min(1.0);
    let image = if scale < 1.0 {
        let w = (f64::from(width) * scale).round().max(1.0) as u32;
        let h = (f64::from(height) * scale).round().max(1.0) as u32;
        image::imageops::resize(&gray, w, h, FilterType::Triangle)
    } else {
        gray
    };
    let (w, h) = image.dimensions();
    let focal = request.focal_px * f64::from(w) / f64::from(width);
    Ok(PreparedPhoto {
        camera: CameraModel {
            width: w,
            height: h,
            fx: focal,
            fy: focal,
            cx: (f64::from(w) - 1.0) / 2.0,
            cy: (f64::from(h) - 1.0) / 2.0,
        },
        image,
    })
}

/// The response message for a pipeline outcome on a photo of `camera`.
pub fn response_of(outcome: Outcome, camera: &CameraModel) -> LocateResponse {
    match outcome {
        Outcome::Found(located) => {
            let [lat, lon, height] = located.geodetic;
            let [heading, pitch, roll] = located.angles_deg;
            LocateResponse::Found {
                camera: LocatedCamera {
                    latitude_rad: lat.to_radians(),
                    longitude_rad: lon.to_radians(),
                    height_m: height,
                    heading_rad: heading.to_radians(),
                    pitch_rad: pitch.to_radians(),
                    roll_rad: roll.to_radians(),
                    width_px: camera.width,
                    height_px: camera.height,
                    focal_px: camera.fx,
                    inliers: u32::try_from(located.estimate.quality.inliers).unwrap_or(u32::MAX),
                    reprojection_rms_px: located.estimate.quality.reprojection_rms_px,
                    confirmed: located.refined,
                    map_sha256: located.estimate.map.manifest_sha256.clone(),
                },
            }
        }
        Outcome::NotFound {
            candidates,
            best_matches,
        } => LocateResponse::NotFound {
            candidates: u32::try_from(candidates).unwrap_or(u32::MAX),
            best_matches: u32::try_from(best_matches).unwrap_or(u32::MAX),
        },
    }
}

#[cfg(test)]
mod tests;
