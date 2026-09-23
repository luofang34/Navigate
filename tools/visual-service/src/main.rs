//! Python-free package service for browser-local visual estimation.
mod api;
mod config;
mod error;
mod jobs;

use clap::Parser;
use config::Config;
use error::ServiceError;
use std::time::Duration;

fn main() -> Result<(), ServiceError> {
    let config = Config::parse();
    config.validate()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(run(config));
    runtime.shutdown_timeout(Duration::from_secs(10));
    result
}

async fn run(config: Config) -> Result<(), ServiceError> {
    let init = config.clone();
    let regions = tokio::task::spawn_blocking(move || {
        navigate_imagery_provider::load_catalog_blocking(&init.state, init.catalog.as_deref())
    })
    .await??;
    let (jobs, handle) = jobs::start(config.state.clone(), regions);
    let router = api::router(config.clone(), jobs.clone());
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    tracing::info!(address=%listener.local_addr()?,processing="browser-worker",provider="Rust/GDAL","visual service ready");
    let result = axum::serve(listener, router)
        .with_graceful_shutdown(async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error,"shutdown signal failed");
            }
        })
        .await;
    jobs.stop().await;
    let mut handle = handle;
    match tokio::time::timeout(Duration::from_secs(10), &mut handle).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::error!(task="coverage-jobs",%error,"task failed"),
        Err(_) => {
            tracing::error!(task = "coverage-jobs", "shutdown deadline exceeded");
            handle.abort();
        }
    }
    result.map_err(ServiceError::from)
}
