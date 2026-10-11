//! Transport an anchor without changing the physical camera it describes.
use super::VisualSession;
use crate::{AnchorObservation, SessionError, pose::Pose};
use nalgebra::{Matrix6, UnitQuaternion};
use navigate_visual::LocalFrame;

impl VisualSession {
    pub(super) fn to_session_frame(
        &self,
        anchor: &mut AnchorObservation,
    ) -> Result<(), SessionError> {
        let frame = self.local_frame.unwrap_or(anchor.local_frame);
        if frame == anchor.local_frame {
            return Ok(());
        }
        let (rotation, covariance) = match (anchor.local_frame, frame) {
            (LocalFrame::SphericalEnu { .. }, LocalFrame::SphericalEnu { .. })
            | (LocalFrame::Wgs84Enu { .. }, LocalFrame::Wgs84Enu { .. }) => {
                let rotation =
                    frame
                        .local_rotation(anchor.local_frame.anchor())
                        .map_err(|source| SessionError::Visual {
                            frame: anchor.frame,
                            source,
                        })?;
                // Rotation errors are right perturbations in camera axes. Only the
                // position axes change, including their covariance with rotation.
                let mut transport = Matrix6::identity();
                transport
                    .fixed_view_mut::<3, 3>(0, 0)
                    .copy_from(rotation.to_rotation_matrix().matrix());
                (
                    rotation,
                    transport * anchor.geometry_covariance * transport.transpose(),
                )
            }
            (LocalFrame::AnchorMercator { .. }, LocalFrame::AnchorMercator { .. }) => (
                UnitQuaternion::identity(),
                crate::anchor::rescaled(
                    &anchor.geometry_covariance,
                    frame.mercator_scale_m() / anchor.local_frame.mercator_scale_m(),
                ),
            ),
            _ => {
                return Err(SessionError::Invalid {
                    field: "anchor and session frame projections differ",
                });
            }
        };
        let geodetic = anchor.local_frame.geodetic(anchor.pose.translation.vector);
        let position = frame
            .local(geodetic)
            .map_err(|source| SessionError::Visual {
                frame: anchor.frame,
                source,
            })?;
        anchor.pose = Pose::from_parts(position.into(), rotation * anchor.pose.rotation);
        anchor.geometry_covariance = covariance;
        anchor.local_frame = frame;
        Ok(())
    }
}
