//! `navigate-visual-pipeline package` plans a flight, fetches the planned
//! tiles through `file://` templates, and seals a package whose identity
//! does not depend on the cache.

use navigate_data::{DirectoryStore, Package, Sha256Digest, TileAddress, TileRead};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

#[cfg(test)]
fn terrarium_png(height_m: f64) -> Vec<u8> {
    let v = height_m + 32_768.0;
    let pixel = image::Rgb([(v / 256.0).floor() as u8, (v % 256.0).floor() as u8, 0]);
    let mut out = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(8, 8, pixel)
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

const ROUTE: [[f64; 2]; 3] = [[41.70, -1.00], [41.70, -0.98], [41.715, -0.98]];

#[cfg(test)]
fn write_provider(root: &Path) {
    let put =
        |layer: &str, zoom: u32, radius: f64, bytes: &dyn Fn(TileAddress) -> Option<Vec<u8>>| {
            let mut set = BTreeSet::new();
            navigate_data::corridor::corridor_tiles(&ROUTE, radius, zoom, &mut set).unwrap();
            for a in set {
                if let Some(b) = bytes(a) {
                    let path = root.join(format!("{layer}/{}/{}/{}", a.zoom, a.x, a.y));
                    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                    std::fs::write(path, b).unwrap();
                }
            }
        };
    for zoom in [10, 11, 12, 13] {
        put("dem", zoom, 6_000.0, &|_| Some(terrarium_png(300.0)));
    }
    for zoom in [14, 15, 16] {
        put("img", zoom, 3_000.0, &|a| {
            (!(a.x + a.y).is_multiple_of(7)).then(|| format!("img {a:?}").into_bytes())
        });
    }
}

#[cfg(test)]
fn run(
    provider: &Path,
    plan: &Path,
    store: &Path,
    cache: &Path,
    extra: &[&str],
) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_navigate-visual-pipeline"))
        .arg("package")
        .arg("--plan")
        .arg(plan)
        .arg("--store")
        .arg(store)
        .arg("--cache")
        .arg(cache)
        .arg("--imagery-url")
        .arg(format!(
            "file://{}/img/{{z}}/{{x}}/{{y}}",
            provider.display()
        ))
        .args([
            "--imagery-source",
            "test-imagery",
            "--imagery-revision",
            "r1",
        ])
        .arg("--dem-url")
        .arg(format!(
            "file://{}/dem/{{z}}/{{x}}/{{y}}",
            provider.display()
        ))
        .args(["--dem-source", "test-terrarium"])
        .args(extra)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn package_subcommand_plans_fetches_and_seals_a_package() {
    let provider = tempfile::tempdir().unwrap();
    write_provider(provider.path());
    let work = tempfile::tempdir().unwrap();
    let plan = work.path().join("plan.json");
    let waypoints: Vec<_> = ROUTE
        .iter()
        .map(|[lat, lon]| serde_json::json!({"lat_deg": lat, "lon_deg": lon, "altitude_m": 1300.0}))
        .collect();
    let plan_doc = serde_json::json!({
        "release_id": "cli-test", "waypoints": waypoints,
        "cameras": [{"width_px": 320, "height_px": 240, "horizontal_fov_deg": 45.0, "depression_deg": 90.0}],
        "search_radius_m": 50.0, "deviation_m": 100.0, "max_range_m": 6000.0,
    });
    std::fs::write(&plan, serde_json::to_vec(&plan_doc).unwrap()).unwrap();

    let only = run(
        provider.path(),
        &plan,
        &work.path().join("s0"),
        &work.path().join("c0"),
        &["--plan-only"],
    );
    assert!(only["planned_tiles"].as_u64().unwrap() > 0, "{only}");
    assert_eq!(
        only["resource_plan"]["legs"][0]["agl_m"].as_f64(),
        Some(1000.0)
    );
    assert!(!work.path().join("s0").exists());

    let a = run(
        provider.path(),
        &plan,
        &work.path().join("s1"),
        &work.path().join("c1"),
        &[],
    );
    let b = run(
        provider.path(),
        &plan,
        &work.path().join("s2"),
        &work.path().join("c2"),
        &[],
    );
    assert_eq!(a["package"]["identity"], b["package"]["identity"]);
    assert!(a["package"]["gaps"].as_u64().unwrap() > 0, "{a}");

    let identity: Sha256Digest = a["package"]["identity"].as_str().unwrap().parse().unwrap();
    let store = DirectoryStore::new(work.path().join("s1"));
    let package = pollster::block_on(Package::open(&store, identity)).unwrap();
    // Identical tiles (the flat elevation) share one blob.
    let distinct: BTreeSet<_> = package.manifest().blobs().collect();
    let declared = package.manifest().blobs().count() as u64;
    assert_eq!(declared, a["package"]["tiles"].as_u64().unwrap());
    assert_eq!(
        pollster::block_on(package.verify_all(&store)).unwrap(),
        distinct.len()
    );
    let layer = package.manifest().layer("imagery").unwrap();
    for zoom in &layer.zooms {
        for t in &zoom.tiles {
            let address = TileAddress::new(zoom.zoom, t.x, t.y).unwrap();
            let read = pollster::block_on(package.tile(&store, "imagery", address)).unwrap();
            let want = std::fs::read(
                provider
                    .path()
                    .join(format!("img/{}/{}/{}", zoom.zoom, t.x, t.y)),
            )
            .unwrap();
            assert_eq!(read, TileRead::Present(want));
        }
    }
}
