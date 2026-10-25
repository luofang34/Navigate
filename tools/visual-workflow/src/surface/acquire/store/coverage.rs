//! Offline coverage is independent of query scores and runtime beam selection.
use super::*;
use crate::surface::acquire::pyramid;

#[derive(Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Coverage {
    Acquisition { include_fine: bool },
    Pyramid { side: u32, levels: usize },
}
impl Coverage {
    pub(super) fn windows(self, size: (u32, u32)) -> Result<Vec<Option<[u32; 4]>>, Error> {
        let (side, levels) = match self {
            Self::Acquisition { include_fine } => return super::windows(size, include_fine),
            Self::Pyramid { side, levels } => (side, levels),
        };
        let (mut level, _) = pyramid::Options {
            side,
            levels,
            beam: 1,
        }
        .plan(size, 4096)?;
        let mut all = BTreeSet::new();
        for depth in 0..levels {
            all.extend(level.iter().copied());
            if all.len().saturating_mul(4) > 4096 {
                return Err(Error::Input(
                    "pyramid feature catalog exceeds 4096 packet bound; prepare fewer levels"
                        .into(),
                ));
            }
            if depth + 1 < levels {
                level = pyramid::children(&level);
            }
        }
        Ok(all.into_iter().map(Some).collect())
    }
}

#[cfg(test)]
mod tests;
