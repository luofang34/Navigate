//! Scalar field maps: the trait that the particle filter samples and a
//! bilinear grid on latitude and longitude.

use navigate_contract::GeodeticPosition;

use super::MpfError;

/// Distance in cells by which a coordinate can pass the first or the last
/// node and still read that node. It absorbs the rounding of
/// `min + k·step`.
const NODE_TOLERANCE: f64 = 1e-9;

/// A scalar field at a geodetic position, such as a crustal magnetic
/// anomaly map in tesla or a terrain map in meters.
pub trait ScalarFieldMap {
    /// The field value at `position`, or `None` outside the map.
    fn sample(&self, position: &GeodeticPosition) -> Option<f64>;
}

/// A regular grid of values on latitude and longitude with bilinear
/// interpolation. Row `i` is at `lat_min_rad + i·lat_step_rad`, and
/// column `j` is at `lon_min_rad + j·lon_step_rad`. The values are in
/// row order.
#[derive(Debug, Clone, PartialEq)]
pub struct BilinearGridMap {
    lat_min_rad: f64,
    lat_step_rad: f64,
    lon_min_rad: f64,
    lon_step_rad: f64,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl BilinearGridMap {
    /// Builds a grid.
    ///
    /// # Errors
    ///
    /// [`MpfError::InvalidMap`] when the grid has fewer than two rows or
    /// two columns, a step is not finite and positive, a corner is not
    /// finite, or `values` does not hold `rows·cols` finite values.
    pub fn new(
        lat_min_rad: f64,
        lat_step_rad: f64,
        lon_min_rad: f64,
        lon_step_rad: f64,
        rows: usize,
        cols: usize,
        values: Vec<f64>,
    ) -> Result<Self, MpfError> {
        let steps_valid = [lat_step_rad, lon_step_rad]
            .iter()
            .all(|s| s.is_finite() && *s > 0.0);
        let valid = rows >= 2
            && cols >= 2
            && steps_valid
            && lat_min_rad.is_finite()
            && lon_min_rad.is_finite()
            && rows.checked_mul(cols) == Some(values.len())
            && values.iter().all(|v| v.is_finite());
        if !valid {
            return Err(MpfError::InvalidMap { rows, cols });
        }
        Ok(Self {
            lat_min_rad,
            lat_step_rad,
            lon_min_rad,
            lon_step_rad,
            rows,
            cols,
            values,
        })
    }

    fn value(&self, row: usize, col: usize) -> Option<f64> {
        self.values
            .get(row.checked_mul(self.cols)?.checked_add(col)?)
            .copied()
    }

    /// The cell index and the fraction in the cell along one axis, or
    /// `None` outside the grid.
    fn locate(coordinate: f64, min: f64, step: f64, count: usize) -> Option<(usize, f64)> {
        let last = count.checked_sub(1)? as f64;
        let raw = (coordinate - min) / step;
        if !(raw >= -NODE_TOLERANCE && raw <= last + NODE_TOLERANCE) {
            return None;
        }
        let index = raw.clamp(0.0, last);
        // The last node belongs to the last cell.
        let cell = index.floor().min(last - 1.0);
        Some((cell as usize, index - cell))
    }
}

impl ScalarFieldMap for BilinearGridMap {
    fn sample(&self, position: &GeodeticPosition) -> Option<f64> {
        let (row, fy) = Self::locate(
            position.latitude_rad,
            self.lat_min_rad,
            self.lat_step_rad,
            self.rows,
        )?;
        let (col, fx) = Self::locate(
            position.longitude_rad,
            self.lon_min_rad,
            self.lon_step_rad,
            self.cols,
        )?;
        let v00 = self.value(row, col)?;
        let v01 = self.value(row, col.checked_add(1)?)?;
        let v10 = self.value(row.checked_add(1)?, col)?;
        let v11 = self.value(row.checked_add(1)?, col.checked_add(1)?)?;
        let south = v00 + (v01 - v00) * fx;
        let north = v10 + (v11 - v10) * fx;
        Some(south + (north - south) * fy)
    }
}
