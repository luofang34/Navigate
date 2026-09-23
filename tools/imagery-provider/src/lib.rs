//! NAIP imagery provider, GDAL raster reader and verified file package publication.
//!
//! The navigation core consumes imagery packages and never fetches them. This
//! tool crate fetches and publishes them for the visual tools. It links GDAL.
mod error;
mod files;
mod provider;
mod raster;

pub use error::ProviderError;
pub use files::{Region, import_region_blocking, load_catalog_blocking};
pub use provider::NaipProvider;
pub use raster::{GdalRaster, RasterSource};

fn raster_error(operation: &'static str) -> impl FnOnce(gdal::errors::GdalError) -> ProviderError {
    move |source| ProviderError::Raster { operation, source }
}
