//! A timeline cache is bound to all source bytes and source image dimensions.
use crate::BenchError;
use navigate_visual::CameraModel;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Timeline {
    video_sha256: String,
    source_size: [u32; 2],
    timestamps: Vec<u64>,
}
pub(crate) fn timestamps_blocking(
    path: &Path,
    camera: CameraModel,
    cache: &Path,
) -> Result<Vec<u64>, BenchError> {
    let hash = digest_blocking(path)?;
    if cache.exists() {
        let record: Timeline =
            serde_json::from_slice(&crate::read_blocking(cache)?).map_err(|source| {
                BenchError::Json {
                    path: cache.to_owned(),
                    source,
                }
            })?;
        if record.video_sha256 != hash
            || record.source_size != [camera.width, camera.height]
            || record.timestamps.first() != Some(&0)
            || record.timestamps.windows(2).any(|v| v[0] >= v[1])
        {
            return Err(BenchError::Record {
                reason: format!(
                    "timeline identity or ordering mismatch: {}",
                    cache.display()
                ),
            });
        }
        return Ok(record.timestamps);
    }
    let timestamps = super::timestamps_blocking(path, camera)?;
    let record = Timeline {
        video_sha256: hash,
        source_size: [camera.width, camera.height],
        timestamps,
    };
    let mut writer = crate::trial::writer_blocking(cache)?;
    crate::stream::write_record_blocking(&mut writer, cache, &record)?;
    Ok(record.timestamps)
}
pub(crate) fn digest_blocking(path: &Path) -> Result<String, BenchError> {
    let error = |source| BenchError::Io {
        path: path.to_owned(),
        source,
    };
    let mut file = std::fs::File::open(path).map_err(error)?;
    let mut digest = Sha256::new();
    let mut bytes = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut bytes).map_err(error)?;
        if n == 0 {
            break;
        }
        digest.update(&bytes[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests;
