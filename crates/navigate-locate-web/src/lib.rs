//! The Navigate locate pipeline as a WebAssembly module for a Web Worker.
//!
//! A page starts a dedicated Worker, gives it the LoFTR-DS weight file, and sends
//! [`navigate_contract::locate`] messages as JSON. The module fetches map tiles
//! with the Worker's `fetch`, runs [`navigate_locate::locate`] on the browser's
//! WebGPU device, and returns the located camera. The photo never leaves the
//! Worker. A page that only shows a pose link never loads this module.
#[cfg(target_arch = "wasm32")]
mod locator;
mod request;
#[cfg(target_arch = "wasm32")]
mod start;
#[cfg(target_arch = "wasm32")]
mod tiles;

pub use request::{PreparedPhoto, prepare_photo, response_of};
