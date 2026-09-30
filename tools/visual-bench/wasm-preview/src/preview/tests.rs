use super::style_json;

fn raster_sources(style: &serde_json::Value) -> Vec<String> {
    let mut names: Vec<String> = style["sources"]
        .as_object()
        .map(|sources| {
            sources
                .iter()
                .filter(|(_, source)| source["type"] == "raster")
                .map(|(name, _)| name.clone())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

#[test]
fn reference_renderers_declare_only_uploaded_raster_sources() {
    let style = style_json([40.5, -74.4], 18, 14, false);
    assert_eq!(raster_sources(&style), ["imagery"]);
    let layers: Vec<&str> = style["layers"]
        .as_array()
        .map(|layers| layers.iter().filter_map(|l| l["id"].as_str()).collect())
        .unwrap_or_default();
    assert_eq!(layers, ["background", "imagery"]);
}

#[test]
fn globe_renderers_add_the_uploaded_context_source() {
    let style = style_json([40.5, -74.4], 18, 14, true);
    assert_eq!(raster_sources(&style), ["context", "imagery"]);
    assert_eq!(style["projection"]["type"], "vertical-perspective");
}

#[test]
fn both_styles_parse_as_maplibre_styles() {
    for globe in [false, true] {
        let parsed: Result<maplibre::style::Style, _> =
            serde_json::from_value(style_json([40.5, -74.4], 18, 14, globe));
        assert!(parsed.is_ok(), "globe={globe}");
    }
}
