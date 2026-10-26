//! Navigate locate pipeline compiled as a WebAssembly module for a Web Worker.
//!
//! A browser page starts a dedicated Web Worker, provides the model weight bytes,
//! and exchanges [`navigate_locate::messages`] JSON payloads.
//! The worker fetches map tiles, executes [`navigate_locate::locate`], and returns
//! the located camera pose.
//! A `Searcher` also loads DINOv2 and answers [`navigate_locate::messages::search`]
//! messages. It reads a place index with `fetch`, searches a region, and then
//! locates the camera at the best places.
//! The input photo stays inside the worker memory.
#[cfg(target_arch = "wasm32")]
mod locator;
mod request;
mod search_request;
#[cfg(target_arch = "wasm32")]
mod searcher;
#[cfg(target_arch = "wasm32")]
mod start;
#[cfg(target_arch = "wasm32")]
mod tiles;

pub use request::{PreparedPhoto, photo_info, prepare_photo, response_of};
pub use search_request::{prepare_search, search_response_of};
