"""Run provider package builds outside HTTP request threads."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import uuid
from .coverage import plan
from .naip import build


class Downloads:
    def __init__(self,state,regions):
        self.state=state; self.regions=regions; self.jobs={}; self.lock=Lock()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='coverage-download')

    def start(self,value):
        selection=plan(value)
        key=uuid.uuid4().hex
        with self.lock:
            if sum(job['status'] in ('queued','running') for job in self.jobs.values())>=3:
                raise ValueError('Download queue is full')
            self.jobs[key]=dict(id=key,status='queued',plan=selection,progress='Queued')
        self.pool.submit(self.run,key,selection)
        return self.read(key)

    def read(self,key):
        with self.lock:
            return dict(self.jobs[key])

    def update(self,key,**values):
        with self.lock: self.jobs[key].update(values)

    def run(self,key,selection):
        try:
            self.update(key,status='running')
            region=build(selection,self.state,lambda text:self.update(key,progress=text))
            self.regions[region['id']]=region
            self.update(key,status='complete',region=region)
        except Exception as error:
            # Provider exception text can include signed URLs.
            self.update(key,status='failed',error=f'Provider package failed ({type(error).__name__}). Check coverage and retry.')

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
