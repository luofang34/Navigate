use super::*;
use navigate_visual::PixelMatch;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default)]
struct TestMatcher {
    calls: u32,
    fail: bool,
}

impl ImageMatcher for TestMatcher {
    fn identity(&self) -> &str {
        "test-resident-matcher"
    }
    fn match_images_blocking(
        &mut self,
        a: &GrayImage,
        b: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.calls = self.calls.wrapping_add(1);
        if self.fail {
            return Err(VisualError::Invalid {
                field: "test backend failure",
            });
        }
        assert_eq!(a.get_pixel(0, 0)[0], 7);
        assert_eq!(b.get_pixel(0, 0)[0], 19);
        Ok(vec![PixelMatch {
            reference: [1.0, 2.0].into(),
            query: [3.0, 4.0].into(),
        }])
    }
}

fn fixture_blocking(label: &str) -> Result<(PathBuf, String), Box<dyn std::error::Error>> {
    let directory = std::env::temp_dir().join(format!(
        "navigate-matcher-stream-{label}-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir(&directory)?;
    let reference = directory.join("reference.png");
    let query = directory.join("query.png");
    GrayImage::from_pixel(8, 8, image::Luma([7])).save(&reference)?;
    GrayImage::from_pixel(8, 8, image::Luma([19])).save(&query)?;
    let input = [41, 73]
        .map(|request_id| {
            json!({"request_id": request_id,
        "reference": reference, "query": query})
            .to_string()
        })
        .join("\n");
    Ok((directory, input))
}

#[test]
fn two_requests_share_the_matcher_and_keep_pixel_identity() -> Result<(), Box<dyn std::error::Error>>
{
    let (directory, input) = fixture_blocking("success")?;
    let mut matcher = TestMatcher::default();
    let mut output = Vec::new();
    let result = serve_blocking(&mut matcher, input.as_bytes(), &mut output, 12.5);
    std::fs::remove_dir_all(directory)?;
    result?;
    let text = String::from_utf8(output)?;
    let rows = text
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(matcher.calls, 2);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["ready"], true);
    assert_eq!(rows[0]["load_ms"], 12.5);
    for (row, id) in rows[1..].iter().zip([41, 73]) {
        assert_eq!(row["request_id"], id);
        assert_eq!(row["backend"], "test-resident-matcher");
        assert_eq!(
            row["reference_image_sha256"],
            format!("{:x}", Sha256::digest([7; 64]))
        );
        assert_eq!(
            row["query_image_sha256"],
            format!("{:x}", Sha256::digest([19; 64]))
        );
        assert_eq!(
            row["pairs"],
            json!([{"reference": [1.0, 2.0], "query": [3.0, 4.0]}])
        );
    }
    Ok(())
}

#[test]
fn backend_failure_stops_the_stream_without_a_success_response()
-> Result<(), Box<dyn std::error::Error>> {
    let (directory, input) = fixture_blocking("failure")?;
    let mut matcher = TestMatcher {
        fail: true,
        ..Default::default()
    };
    let mut output = Vec::new();
    let result = serve_blocking(&mut matcher, input.as_bytes(), &mut output, 0.0);
    std::fs::remove_dir_all(directory)?;
    assert!(matches!(
        result,
        Err(StreamError::Match { request_id: 41, .. })
    ));
    assert_eq!(matcher.calls, 1);
    assert_eq!(String::from_utf8(output)?.lines().count(), 1);
    Ok(())
}

#[test]
fn missing_images_keep_request_and_path_context() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from("/nonexistent/navigate-image.png");
    let input = json!({"request_id": 23, "reference": path, "query": path}).to_string();
    let mut matcher = TestMatcher::default();
    let result = serve_blocking(&mut matcher, input.as_bytes(), Vec::new(), 0.0);
    assert!(matches!(result, Err(StreamError::Image { request_id: 23, path: p, .. }) if p == path));
    assert_eq!(matcher.calls, 0);
    Ok(())
}
