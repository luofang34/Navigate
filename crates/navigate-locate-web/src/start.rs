//! One call of the linker's static constructors when the module starts.
//!
//! cubecl registers its IR types with the `inventory` crate, which uses static
//! constructors. When no export calls `__wasm_call_ctors`, wasm-ld treats the
//! module as a command and calls the constructors at the start of every export.
//! A tensor operation calls into JavaScript and back several times, so each
//! operation then paid about 3 ms; LoFTR-DS took 3.7 s instead of 0.35 s per pair
//! in Chrome. This export calls the constructors once, and wasm-ld then links the
//! module as a reactor without the wrappers.
#![allow(unsafe_code)]

use wasm_bindgen::prelude::wasm_bindgen;

unsafe extern "C" {
    fn __wasm_call_ctors();
}

/// Called once by the generated `init` before any other export.
#[wasm_bindgen(start)]
pub fn start() {
    // SAFETY: `__wasm_call_ctors` is the linker-generated constructor list. The
    // generated `init` calls this start function exactly once per instance,
    // before JavaScript can call any other export.
    unsafe { __wasm_call_ctors() }
}
