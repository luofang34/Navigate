//! Native provider adapters and verified filesystem package publication.
mod files;
mod provider;
mod raster;

pub use files::{Region, import_region_blocking, load_catalog_blocking};
pub use provider::NaipProvider;
pub use raster::{GdalRaster, RasterSource};

use crate::ImageryError;

fn raster_error(operation: &'static str) -> impl FnOnce(gdal::errors::GdalError) -> ImageryError {
    move |source| ImageryError::Raster { operation, source }
}
