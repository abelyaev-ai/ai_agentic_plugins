# Ray Core Patterns Reference

This reference provides detailed patterns for Ray Core distributed primitives. Use alongside the main skill documentation for comprehensive implementation guidance.

## Actor Lifecycle Management

### Creation Patterns

**Standard Actor Creation:**

```python
import ray

@ray.remote
class StatefulWorker:
    def __init__(self, config):
        self.config = config
        self.state = {}

    def process(self, key, value):
        self.state[key] = value
        return len(self.state)

# Create actor with constructor arguments
worker = StatefulWorker.remote(config={"batch_size": 32})
```

**Deferred Actor Creation with Options:**

```python
# Define actor class without instantiation
@ray.remote
class ConfigurableActor:
    def __init__(self):
        self.initialized = True

# Create with runtime options
actor = ConfigurableActor.options(
    name="worker-1",
    namespace="production",
    num_cpus=2,
    num_gpus=0.5,
    memory=2 * 1024 * 1024 * 1024,  # 2GB
    max_restarts=3,
    max_task_retries=5,
    lifetime="detached",
).remote()
```

### Named Actor Patterns

**Global Singleton Pattern:**

```python
def get_or_create_coordinator():
    """Return existing coordinator or create new one."""
    try:
        return ray.get_actor("coordinator", namespace="system")
    except ValueError:
        return Coordinator.options(
            name="coordinator",
            namespace="system",
            lifetime="detached",
        ).remote()

# Usage from any process
coordinator = get_or_create_coordinator()
```

**Service Registry Pattern:**

```python
@ray.remote
class ServiceRegistry:
    def __init__(self):
        self.services = {}

    def register(self, name, actor_handle):
        self.services[name] = actor_handle

    def get(self, name):
        return self.services.get(name)

    def list_services(self):
        return list(self.services.keys())

# Create detached registry
registry = ServiceRegistry.options(
    name="service_registry",
    lifetime="detached"
).remote()

# Register services
ray.get(registry.register.remote("data_loader", DataLoader.remote()))
ray.get(registry.register.remote("model_server", ModelServer.remote()))

# Retrieve from any process
registry = ray.get_actor("service_registry")
data_loader = ray.get(registry.get.remote("data_loader"))
```

### Detached Actor Management

Detached actors persist beyond driver lifetime. Proper cleanup is essential.

```python
# Create detached actor
processor = DataProcessor.options(
    name="background_processor",
    lifetime="detached"
).remote()

# Later: graceful shutdown
def shutdown_processor():
    try:
        processor = ray.get_actor("background_processor")
        # Allow graceful cleanup
        ray.get(processor.shutdown.remote())
        # Force kill after cleanup
        ray.kill(processor)
    except ValueError:
        pass  # Actor already terminated
```

### Actor Termination

**Graceful Termination:**

```python
@ray.remote
class ManagedActor:
    def __init__(self):
        self.running = True

    def shutdown(self):
        self.running = False
        self._cleanup_resources()
        return "shutdown_complete"

    def _cleanup_resources(self):
        # Close connections, flush buffers, etc.
        pass

# Graceful shutdown
ray.get(actor.shutdown.remote())
ray.kill(actor)
```

**Force Termination:**

```python
# Immediate termination (no cleanup)
ray.kill(actor, no_restart=True)
```

## Task DAG Composition

### Linear Pipeline Pattern

```python
@ray.remote
def extract(source):
    return load_data(source)

@ray.remote
def transform(data, config):
    return apply_transformations(data, config)

@ray.remote
def load(data, destination):
    return write_data(data, destination)

# Build pipeline
def run_etl_pipeline(source, config, destination):
    data_ref = extract.remote(source)
    transformed_ref = transform.remote(data_ref, config)
    result_ref = load.remote(transformed_ref, destination)
    return ray.get(result_ref)
```

### Fan-Out / Fan-In Pattern

```python
@ray.remote
def split_data(data, num_chunks):
    chunk_size = len(data) // num_chunks
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

@ray.remote
def process_chunk(chunk):
    return [item * 2 for item in chunk]

@ray.remote
def merge_results(results):
    merged = []
    for result in results:
        merged.extend(result)
    return merged

def parallel_process(data, num_workers=4):
    # Fan-out
    chunks_ref = split_data.remote(data, num_workers)
    chunks = ray.get(chunks_ref)

    # Process in parallel
    processed_refs = [process_chunk.remote(chunk) for chunk in chunks]

    # Fan-in
    result_ref = merge_results.remote(processed_refs)
    return ray.get(result_ref)
```

### Dynamic DAG Pattern

```python
@ray.remote
def conditional_process(data, threshold):
    if data["value"] > threshold:
        return heavy_computation.remote(data)
    else:
        return light_computation.remote(data)

@ray.remote
def heavy_computation(data):
    # Expensive processing
    return {"result": data["value"] ** 2, "path": "heavy"}

@ray.remote
def light_computation(data):
    # Simple processing
    return {"result": data["value"], "path": "light"}

# Execute dynamic DAG
def process_with_branching(items, threshold):
    refs = [conditional_process.remote(item, threshold) for item in items]
    # Each ref is itself an ObjectRef to the chosen computation
    nested_refs = ray.get(refs)
    return ray.get(nested_refs)
```

### Recursive DAG Pattern

```python
@ray.remote
def parallel_merge_sort(arr, depth=0, max_depth=3):
    if len(arr) <= 1 or depth >= max_depth:
        return sorted(arr)

    mid = len(arr) // 2
    left_ref = parallel_merge_sort.remote(arr[:mid], depth + 1, max_depth)
    right_ref = parallel_merge_sort.remote(arr[mid:], depth + 1, max_depth)

    left, right = ray.get([left_ref, right_ref])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

## Object Store Best Practices

### Efficient Data Sharing

**Multi-Consumer Pattern:**

```python
import numpy as np

# BAD: Object serialized N times
def inefficient_broadcast(data, workers):
    return [worker.process.remote(data) for worker in workers]

# GOOD: Object serialized once, shared via object store
def efficient_broadcast(data, workers):
    data_ref = ray.put(data)
    return [worker.process.remote(data_ref) for worker in workers]

# Usage
large_array = np.random.rand(10000, 10000)
workers = [Worker.remote() for _ in range(10)]
refs = efficient_broadcast(large_array, workers)
```

### Zero-Copy Data Access

```python
import numpy as np
import pyarrow as pa

# NumPy arrays support zero-copy on same node
array = np.zeros((10000, 10000), dtype=np.float32)
ref = ray.put(array)

@ray.remote
def read_only_task(arr_ref):
    # Zero-copy read on same node
    arr = ray.get(arr_ref)
    return arr.sum()

# Arrow tables also support zero-copy
table = pa.table({"col1": range(1000000), "col2": range(1000000)})
table_ref = ray.put(table)
```

### Memory Management

**Explicit Reference Management:**

```python
# Delete object from object store when no longer needed
del ref

# Force garbage collection
import gc
gc.collect()

# Check object store memory usage
ray.available_resources()
```

**Spilling Configuration:**

```python
# Configure object spilling to disk
ray.init(
    _system_config={
        "object_spilling_config": {
            "type": "filesystem",
            "params": {"directory_path": "/tmp/ray_spill"}
        }
    }
)
```

### Serialization Patterns

**Custom Serializer Registration:**

```python
import ray
from ray import cloudpickle

class CustomObject:
    def __init__(self, data):
        self.data = data

def serialize_custom(obj):
    return {"data": obj.data}

def deserialize_custom(serialized):
    return CustomObject(serialized["data"])

# Register custom serializer
ray.util.register_serializer(
    CustomObject,
    serializer=serialize_custom,
    deserializer=deserialize_custom
)
```

**Handling Unpicklable Objects:**

```python
@ray.remote
class DatabaseActor:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None  # Not picklable

    def _ensure_connection(self):
        if self.connection is None:
            self.connection = create_connection(self.connection_string)

    def query(self, sql):
        self._ensure_connection()
        return self.connection.execute(sql)

    def __getstate__(self):
        # Exclude unpicklable connection from state
        state = self.__dict__.copy()
        state["connection"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Connection will be recreated on first use
```

## Resource Allocation Strategies

### Fractional Resource Allocation

```python
# Share GPU across multiple actors
@ray.remote(num_gpus=0.25)
class LightweightModel:
    def predict(self, data):
        pass

# 4 models can share 1 GPU
models = [LightweightModel.remote() for _ in range(4)]
```

### Custom Resource Scheduling

```python
# Initialize cluster with custom resources
ray.init(resources={"TPU": 8, "high_memory": 4})

@ray.remote(resources={"TPU": 1})
class TPUWorker:
    pass

@ray.remote(resources={"high_memory": 1, "CPU": 4})
def memory_intensive_task():
    pass
```

### Placement Group Strategies

**PACK Strategy (Minimize Nodes):**

```python
from ray.util.placement_group import placement_group

# All bundles on minimum number of nodes
pg = placement_group(
    bundles=[
        {"CPU": 4, "GPU": 1},
        {"CPU": 4, "GPU": 1},
        {"CPU": 4, "GPU": 1},
        {"CPU": 4, "GPU": 1},
    ],
    strategy="PACK"
)
ray.get(pg.ready())
```

**SPREAD Strategy (Maximize Distribution):**

```python
# Distribute bundles across maximum number of nodes
pg = placement_group(
    bundles=[
        {"CPU": 2},
        {"CPU": 2},
        {"CPU": 2},
        {"CPU": 2},
    ],
    strategy="SPREAD"
)
```

**STRICT Variants:**

```python
# STRICT_PACK: All bundles must fit on single node
pg = placement_group(bundles=[...], strategy="STRICT_PACK")

# STRICT_SPREAD: Each bundle on different node
pg = placement_group(bundles=[...], strategy="STRICT_SPREAD")
```

**Actor Scheduling on Placement Groups:**

```python
@ray.remote(num_cpus=4, num_gpus=1)
class TrainingWorker:
    pass

# Create workers on specific bundles
workers = []
for i in range(len(pg.bundle_specs)):
    worker = TrainingWorker.options(
        placement_group=pg,
        placement_group_bundle_index=i
    ).remote()
    workers.append(worker)
```

## Fault Tolerance Patterns

### Actor Checkpoint Pattern

```python
import json
import os

@ray.remote(max_restarts=5, max_task_retries=3)
class CheckpointedActor:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, "state.json")
        self.state = self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r") as f:
                return json.load(f)
        return {"counter": 0, "processed_items": []}

    def _save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(self.state, f)

    def process(self, item):
        self.state["counter"] += 1
        self.state["processed_items"].append(item)
        self._save_checkpoint()
        return self.state["counter"]
```

### Idempotent Task Pattern

```python
import hashlib

@ray.remote(max_retries=5)
def idempotent_write(data, output_path):
    """Write data only if not already written."""
    # Generate deterministic filename from content
    content_hash = hashlib.sha256(str(data).encode()).hexdigest()[:16]
    final_path = f"{output_path}_{content_hash}"

    if os.path.exists(final_path):
        return final_path  # Already processed

    # Write to temp file then rename (atomic)
    temp_path = f"{final_path}.tmp"
    with open(temp_path, "w") as f:
        f.write(str(data))
    os.rename(temp_path, final_path)
    return final_path
```

### Retry with Exponential Backoff

```python
import time
import random

@ray.remote(max_retries=0)  # Handle retries manually
def robust_external_call(url, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            response = make_request(url)
            return response
        except TransientError as e:
            if attempt == max_attempts - 1:
                raise
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

### Graceful Degradation Pattern

```python
@ray.remote
class ResilientService:
    def __init__(self):
        self.cache = {}
        self.fallback_enabled = True

    def get_data(self, key):
        try:
            # Primary data source
            return self._fetch_from_primary(key)
        except Exception as e:
            if self.fallback_enabled and key in self.cache:
                return self.cache[key]
            raise

    def _fetch_from_primary(self, key):
        data = external_service.fetch(key)
        self.cache[key] = data  # Update cache on success
        return data
```

## Performance Patterns

### Task Batching

```python
# BAD: Too many fine-grained tasks
def process_items_slow(items):
    refs = [process_single.remote(item) for item in items]
    return ray.get(refs)

# GOOD: Batch items into larger tasks
@ray.remote
def process_batch(items):
    return [process_single_local(item) for item in items]

def process_items_fast(items, batch_size=100):
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    refs = [process_batch.remote(batch) for batch in batches]
    results = ray.get(refs)
    return [item for batch in results for item in batch]
```

### Pipelining Pattern

```python
@ray.remote
class PipelineStage:
    def __init__(self, stage_fn):
        self.stage_fn = stage_fn
        self.pending = []

    def process(self, data_ref):
        result = self.stage_fn(ray.get(data_ref))
        return ray.put(result)

def run_pipeline(data_chunks, stages):
    """Process chunks through pipeline stages with overlap."""
    stage_actors = [PipelineStage.remote(fn) for fn in stages]

    # Start first chunk through all stages
    refs_in_flight = []
    for chunk in data_chunks:
        ref = ray.put(chunk)
        for stage in stage_actors:
            ref = stage.process.remote(ref)
        refs_in_flight.append(ref)

    return ray.get(refs_in_flight)
```

### Avoiding Common Anti-Patterns

**Anti-Pattern: Blocking ray.get in Loop:**

```python
# BAD
results = []
for i in range(1000):
    ref = task.remote(i)
    results.append(ray.get(ref))  # Blocks on each task

# GOOD
refs = [task.remote(i) for i in range(1000)]
results = ray.get(refs)  # Single blocking call
```

**Anti-Pattern: Large Return Values:**

```python
# BAD: Returns large data through driver
@ray.remote
def load_large_dataset():
    return load_from_disk()  # 10GB dataset

# GOOD: Store in object store, return reference
@ray.remote
def load_large_dataset():
    data = load_from_disk()
    return ray.put(data)

# Or use Ray Data for large datasets
```

**Anti-Pattern: Actor Method Storms:**

```python
# BAD: Overwhelming actor with requests
refs = [actor.process.remote(i) for i in range(100000)]

# GOOD: Batch requests or use backpressure
batch_size = 1000
for i in range(0, 100000, batch_size):
    batch_refs = [actor.process.remote(j) for j in range(i, min(i + batch_size, 100000))]
    ray.get(batch_refs)  # Wait for batch to complete
```

## Ray Dashboard Usage

### Accessing the Dashboard

```python
# Dashboard runs on port 8265 by default
ray.init(dashboard_host="0.0.0.0", dashboard_port=8265)

# Access at http://<head-node-ip>:8265
```

### Key Dashboard Sections

**Jobs View:**
- Monitor submitted Ray jobs
- View job logs and status
- Track job resource utilization

**Cluster View:**
- Node status and resource utilization
- Autoscaler activity
- Node logs and errors

**Actors View:**
- List all actors by state (alive, dead, pending)
- View actor logs and stack traces
- Monitor actor resource usage

**Metrics View:**
- Built-in Prometheus metrics
- Task throughput and latency
- Object store utilization
- Memory usage trends

### Debugging with Dashboard

**Finding Dead Actors:**

Navigate to Actors tab, filter by state="DEAD", examine death cause and stack trace.

**Identifying Resource Bottlenecks:**

Check Cluster view for resource utilization. Look for nodes at 100% CPU/GPU/memory.

**Tracing Task Failures:**

Jobs view shows failed tasks. Click through to see exception details and retry history.

### Programmatic Metrics Access

```python
from ray.util.metrics import Counter, Gauge, Histogram

# Custom application metrics
requests_counter = Counter(
    "requests_total",
    description="Total requests processed",
    tag_keys=("endpoint",)
)

latency_histogram = Histogram(
    "request_latency_seconds",
    description="Request latency",
    boundaries=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

@ray.remote
class MetricsActor:
    def process_request(self, endpoint, data):
        start = time.time()
        result = self._process(data)
        latency_histogram.observe(time.time() - start)
        requests_counter.inc(tags={"endpoint": endpoint})
        return result
```

Metrics are exported to the Ray metrics endpoint and can be scraped by Prometheus for visualization in Grafana dashboards.
