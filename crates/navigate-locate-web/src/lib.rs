//! Navigate locate pipeline compiled as a WebAssembly module for a Web Worker.
//!
//! A browser page starts a dedicated Web Worker, provides the model weight bytes,
//! and exchanges [`navigate_locate::messages`] JSON payloads.
//! The worker fetches map tiles, executes [`navigate_locate::locate`], and returns
//! the located camera pose.
//! The input photo stays inside the worker memory.
#[cfg(target_arch = "wasm32")]
mod locator;
mod request;
#[cfg(target_arch = "wasm32")]
mod start;
#[cfg(target_arch = "wasm32")]
mod tiles;

pub use request::{PreparedPhoto, photo_info, prepare_photo, response_of};
