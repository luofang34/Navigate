//! The port through which a host renderer makes reference views (ADR-0010).
//!
//! The navigation core does not depend on a map renderer. A host implements
//! [`ReferenceRenderer`] on its renderer, for example the MapLibre fork, a
//! precomputed orthophoto, or a test double.

use crate::{CameraPose, ReferenceView};

/// Identity of the code and style that made a reference view.
///
/// With the map revision in [`ReferenceView`], this identity lets a later
/// reader reproduce the render.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RendererIdentity {
    /// Revision of the renderer code, for example a fork commit.
    pub revision: String,
    /// SHA-256 of the reference style, in lowercase hexadecimal.
    pub style_sha256: String,
}

/// Makes reference views of one installed map package for one camera.
///
/// An implementation must meet these requirements:
///
/// - It draws only the reference content: imagery and terrain, with no
///   labels, fades, or atmosphere.
/// - It selects tile detail from the source data, not from the display size.
/// - It returns a view only when the view is complete. It does not depend on
///   a fixed count of frames.
/// - It reports depth along the optical axis in metres, and zero where no
///   surface or imagery is present.
pub trait ReferenceRenderer {
    /// A render failure.
    type Error: std::error::Error + 'static;

    /// The code and style identity of every view that this renderer makes.
    fn identity(&self) -> RendererIdentity;

    /// Render one candidate pose. This call can wait for the GPU.
    ///
    /// # Errors
    /// Returns the renderer failure.
    fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, Self::Error>;

    /// Render several candidate poses in order.
    ///
    /// The default renders them one at a time. A renderer that can draw
    /// several views in one pass overrides this method.
    ///
    /// # Errors
    /// Returns the first renderer failure.
    fn render_batch_blocking(
        &mut self,
        poses: &[CameraPose],
    ) -> Result<Vec<ReferenceView>, Self::Error> {
        poses
            .iter()
            .map(|pose| self.render_blocking(*pose))
            .collect()
    }
}

#[cfg(test)]
mod tests;
