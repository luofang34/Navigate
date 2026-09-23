use navigate_imagery::{
    CoveragePlan,
    native::{NaipProvider, Region},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::{RwLock, mpsc, oneshot};

#[derive(Default)]
pub(crate) struct Snapshot {
    pub regions: BTreeMap<String, Region>,
    pub jobs: BTreeMap<String, Value>,
}
#[derive(Clone)]
pub(crate) struct Jobs {
    pub snapshot: Arc<RwLock<Snapshot>>,
    sender: mpsc::Sender<Command>,
}
enum Command {
    Start(CoveragePlan, oneshot::Sender<Result<Value, &'static str>>),
    Progress(String, String),
    Complete(String, Result<Region, navigate_imagery::ImageryError>),
    Stop,
}

impl Jobs {
    pub async fn submit(&self, plan: CoveragePlan) -> Result<Value, &'static str> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(Command::Start(plan, tx))
            .await
            .map_err(|_| "coverage worker is stopped")?;
        rx.await
            .map_err(|_| "coverage worker stopped before acceptance")?
    }
    pub async fn stop(&self) {
        self.sender.send(Command::Stop).await.ok();
    }
}

pub(crate) fn start(
    state: PathBuf,
    regions: BTreeMap<String, Region>,
) -> (Jobs, tokio::task::JoinHandle<()>) {
    let (sender, receiver) = mpsc::channel(64);
    let snapshot = Arc::new(RwLock::new(Snapshot {
        regions,
        jobs: BTreeMap::new(),
    }));
    let jobs = Jobs { snapshot, sender };
    let actor = jobs.clone();
    let handle = tokio::spawn(async move {
        run(state, actor, receiver).await;
    });
    (jobs, handle)
}

async fn run(state: PathBuf, jobs: Jobs, mut receiver: mpsc::Receiver<Command>) {
    let mut queued = VecDeque::new();
    let mut active = false;
    let mut stopping = false;
    while let Some(command) = receiver.recv().await {
        match command {
            Command::Start(plan, reply) => {
                if stopping || queued.len() + usize::from(active) >= 3 {
                    reply.send(Err("coverage queue is full or stopping")).ok();
                    continue;
                }
                let id = uuid::Uuid::new_v4().simple().to_string();
                let record = json!({"id":id,"status":"queued","plan":plan,"progress":"Queued"});
                jobs.snapshot
                    .write()
                    .await
                    .jobs
                    .insert(id.clone(), record.clone());
                queued.push_back((id, plan));
                reply.send(Ok(record)).ok();
            }
            Command::Progress(id, text) => {
                if let Some(job) = jobs.snapshot.write().await.jobs.get_mut(&id) {
                    job["progress"] = text.into();
                }
            }
            Command::Complete(id, result) => {
                complete(&jobs, id, result).await;
                active = false;
            }
            Command::Stop => {
                stopping = true;
                queued.clear();
            }
        }
        if stopping && !active {
            break;
        }
        if !active && let Some((id, plan)) = queued.pop_front() {
            if let Some(job) = jobs.snapshot.write().await.jobs.get_mut(&id) {
                job["status"] = "running".into();
            }
            launch(state.clone(), jobs.sender.clone(), id, plan);
            active = true;
        }
    }
}

fn launch(state: PathBuf, sender: mpsc::Sender<Command>, id: String, plan: CoveragePlan) {
    tokio::task::spawn_blocking(move || {
        let result = NaipProvider::new_blocking().and_then(|p| {
            p.build_blocking(&plan, &state, |text| {
                sender
                    .blocking_send(Command::Progress(id.clone(), text))
                    .ok();
            })
        });
        sender.blocking_send(Command::Complete(id, result)).ok();
    });
}

async fn complete(jobs: &Jobs, id: String, result: Result<Region, navigate_imagery::ImageryError>) {
    let mut state = jobs.snapshot.write().await;
    match result {
        Ok(region) => {
            if let Some(job) = state.jobs.get_mut(&id) {
                job["status"] = "complete".into();
                job["region"] = json!(region);
            }
            state.regions.insert(region.id.clone(), region);
        }
        Err(error) => {
            tracing::warn!(job=%id,error=%error,"coverage package failed");
            if let Some(job) = state.jobs.get_mut(&id) {
                job["status"] = "failed".into();
                job["error"] = "Provider package failed. Check coverage and retry.".into();
            }
        }
    }
    while state.jobs.len() > 100 {
        let expired = state
            .jobs
            .iter()
            .find(|(_, v)| matches!(v["status"].as_str(), Some("complete" | "failed")))
            .map(|(k, _)| k.clone());
        if let Some(key) = expired {
            state.jobs.remove(&key);
        } else {
            break;
        }
    }
}
