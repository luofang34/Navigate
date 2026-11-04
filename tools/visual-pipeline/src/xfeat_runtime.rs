//! Hardware-portable sparse `XFeat` and `LighterGlue` runtime re-exports.

pub use navigate_visual_flight::runtime::{
    LoadedMatcher, LoadedXFeat, MatcherRuntimeConfig, XFeatRuntime, XFeatRuntimeConfig,
    default_burn_device,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_parser_accepts_supported_names_and_rejects_unknown() {
        assert_eq!(
            XFeatRuntime::parse("burn-f32").ok(),
            Some(XFeatRuntime::BurnF32)
        );
        assert_eq!(
            XFeatRuntime::parse("burn-f16").ok(),
            Some(XFeatRuntime::BurnF16)
        );
        assert_eq!(
            XFeatRuntime::parse("coreml").ok(),
            Some(XFeatRuntime::CoreMl)
        );
        assert_eq!(
            XFeatRuntime::parse("tensorrt").ok(),
            Some(XFeatRuntime::TensorRt)
        );
        assert_eq!(
            XFeatRuntime::parse("onnx-cpu").ok(),
            Some(XFeatRuntime::OnnxCpu)
        );
        assert!(XFeatRuntime::parse("unknown-npu").is_err());
    }
}
