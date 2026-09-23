//! FFmpeg child lifetime and exact grayscale frame reads.

use crate::{BenchError, read_blocking};
use image::GrayImage;
use navigate_visual::CameraModel;
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
};

pub(super) struct Decoder {
    child: Child,
    stdout: ChildStdout,
    stderr: tempfile::NamedTempFile,
    camera: CameraModel,
    path: PathBuf,
}

impl Decoder {
    pub fn new_blocking(path: &Path, camera: CameraModel) -> Result<Self, BenchError> {
        let stderr = tempfile::NamedTempFile::new().map_err(|source| BenchError::Io {
            path: "<decoder-log>".into(),
            source,
        })?;
        let log = stderr.reopen().map_err(|source| BenchError::Io {
            path: stderr.path().to_owned(),
            source,
        })?;
        let mut child = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-xerror", "-noautorotate", "-i"])
            .arg(path)
            .args([
                "-map", "0:v:0", "-vsync", "0", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log))
            .spawn()
            .map_err(|source| BenchError::Spawn {
                program: "ffmpeg",
                source,
            })?;
        let Some(stdout) = child.stdout.take() else {
            child.kill().ok();
            child.wait().ok();
            return Err(BenchError::Record {
                reason: "FFmpeg stdout is unavailable".into(),
            });
        };
        Ok(Self {
            child,
            stdout,
            stderr,
            camera,
            path: path.to_owned(),
        })
    }

    pub fn next_blocking(&mut self) -> Result<GrayImage, BenchError> {
        let mut bytes =
            vec![0; (u64::from(self.camera.width) * u64::from(self.camera.height)) as usize];
        if let Err(source) = self.stdout.read_exact(&mut bytes) {
            let status = self.child.wait().map_err(|source| BenchError::Io {
                path: self.path.clone(),
                source,
            })?;
            if !status.success() {
                return Err(self.process_error(status)?);
            }
            return Err(BenchError::Io {
                path: self.path.clone(),
                source,
            });
        }
        GrayImage::from_raw(self.camera.width, self.camera.height, bytes).ok_or_else(|| {
            BenchError::Record {
                reason: "decoded frame size does not match calibration".into(),
            }
        })
    }

    pub fn finish_blocking(mut self) -> Result<(), BenchError> {
        let mut extra = [0];
        if self
            .stdout
            .read(&mut extra)
            .map_err(|source| BenchError::Io {
                path: self.path.clone(),
                source,
            })?
            != 0
        {
            return Err(BenchError::Record {
                reason: "decoder produced more frames than the video timeline".into(),
            });
        }
        let status = self.child.wait().map_err(|source| BenchError::Io {
            path: self.path.clone(),
            source,
        })?;
        if !status.success() {
            return Err(self.process_error(status)?);
        }
        Ok(())
    }

    fn process_error(&self, status: std::process::ExitStatus) -> Result<BenchError, BenchError> {
        Ok(BenchError::Process {
            program: "ffmpeg",
            status,
            stderr: String::from_utf8_lossy(&read_blocking(self.stderr.path())?).into_owned(),
        })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}
