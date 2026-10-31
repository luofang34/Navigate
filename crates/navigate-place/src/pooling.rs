//! One unit-length descriptor from the tokens of a canvas.
use crate::{GRID, PlaceError, Tokens};
use half::f16;

/// A unit-length descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct Descriptor {
    /// The values.
    pub values: Vec<f32>,
}

/// A linear map, row by row.
pub(crate) struct Matrix<'a> {
    pub rows: usize,
    pub columns: usize,
    pub values: &'a [u8],
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// The mean of the patch tokens of the cells that hold imagery, made unit length,
/// then mapped by the query head when the index has one. The head file is
/// little-endian `f16`, `rows` × `columns`, row by row.
///
/// # Errors
///
/// Returns an error when the token or head widths do not agree.
pub fn describe(
    tokens: &Tokens,
    cells: &[bool],
    head: Option<(&[u8], usize, usize)>,
) -> Result<Descriptor, PlaceError> {
    let width = tokens.width;
    if tokens.values.len() != (1 + GRID * GRID) * width || cells.len() != GRID * GRID {
        return Err(PlaceError::Width {
            expected: (1 + GRID * GRID) * width,
            found: tokens.values.len(),
        });
    }
    let mut sum = vec![0.0_f32; width];
    for (c, keep) in cells.iter().enumerate() {
        if *keep {
            let token = &tokens.values[(1 + c) * width..(2 + c) * width];
            sum.iter_mut().zip(token).for_each(|(s, v)| *s += v);
        }
    }
    let mean = unit(sum);
    let Some((bytes, rows, columns)) = head else {
        return Ok(Descriptor { values: mean });
    };
    let matrix = Matrix {
        rows,
        columns,
        values: bytes,
    };
    if matrix.columns != width || matrix.values.len() != 2 * rows * columns {
        return Err(PlaceError::Width {
            expected: matrix.columns,
            found: width,
        });
    }
    let mapped = (0..matrix.rows)
        .map(|r| {
            matrix.values[2 * r * matrix.columns..2 * (r + 1) * matrix.columns]
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&mean)
                .map(|(b, x)| f16::from_le_bytes(*b).to_f32() * x)
                .sum()
        })
        .collect();
    Ok(Descriptor {
        values: unit(mapped),
    })
}
