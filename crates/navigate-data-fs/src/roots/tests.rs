use super::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn roots(base: &Path) -> StorageRoots {
    StorageRoots::new(
        base.join("data"),
        base.join("offline"),
        base.join("cache"),
        base.join("config"),
        base.join("temp"),
    )
}

fn runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
}

#[test]
fn each_class_reads_from_its_own_root() -> TestResult {
    runtime()?.block_on(async {
        let base = tempfile::tempdir()?;
        let roots = roots(base.path());
        let store = ClassStore::new(&roots).await?;
        for class in StorageClass::ALL {
            std::fs::write(roots.root(class).join("item.bin"), class.scheme())?;
        }
        for class in StorageClass::ALL {
            let uri = DataUri::in_class(class, "item.bin")?;
            let file = store.open(&uri).await?;
            let bytes = file.read_at(0, file.len() as usize).await?;
            assert_eq!(bytes, class.scheme().as_bytes());
        }
        Ok(())
    })
}

#[test]
fn a_host_scheme_is_not_a_storage_class() -> TestResult {
    runtime()?.block_on(async {
        let base = tempfile::tempdir()?;
        let store = ClassStore::new(&roots(base.path())).await?;
        let uri = DataUri::parse("pilotage://item.bin")?;
        assert!(matches!(
            store.open(&uri).await,
            Err(DataError::InvalidUri { .. })
        ));
        Ok(())
    })
}

#[test]
fn platform_roots_keep_offline_apart_from_data_and_cache() -> TestResult {
    let roots = StorageRoots::for_platform("Luofang", "Pilotage")?;
    let offline = roots.root(StorageClass::Offline);
    assert!(offline.ends_with("Offline"));
    assert_ne!(offline, roots.root(StorageClass::Cache));
    assert_ne!(offline, roots.root(StorageClass::Data));
    Ok(())
}
