---
name: ray-core-patterns
description: >
  This skill should be used when the user asks to "create a Ray actor",
  "set up remote tasks", "use the object store", "handle Ray fault tolerance",
  or works with Ray Core distributed primitives.
---

## Purpose

This skill provides comprehensive patterns for Ray Core distributed computing primitives. It covers actors, tasks, the object store, resource management, and fault tolerance mechanisms that form the foundation of all Ray applications. Use this skill when building distributed Python applications that require stateful workers, parallel task execution, or shared memory between processes.

## Prerequisites

Before implementing Ray Core patterns, resolve the current API documentation:

1. Call `resolve-library-id` with library name "ray" to obtain the Context7-compatible library ID
2. Call `query-docs` with queries for:
   - "Ray Core actors remote decorator stateful actors actor lifecycle named actors"
   - "Ray tasks remote functions ray.get ray.wait task dependencies retries"
   - "Ray object store ray.put serialization fault tolerance placement groups"

Ground all implementations in the latest API surface rather than cached patterns.

## Core Workflow

### Actors

Actors are stateful workers that execute methods serially on a single process. Create an actor by applying the `@ray.remote` decorator to a Python class.

**Basic Actor Creation:**

```python
import ray

@ray.remote
class Counter:
    def __init__(self, initial_value=0):
        self.value = initial_value

    def increment(self):
        self.value += 1
        return self.value

    def get_value(self):
        return self.value

# Instantiate the actor (returns an ActorHandle)
counter = Counter.remote(initial_value=10)

# Call methods asynchronously (returns ObjectRef)
ref = counter.increment.remote()
result = ray.get(ref)  # Blocks until result is ready
```

**Resource Requests:**

Specify CPU, GPU, memory, and custom resource requirements at definition or instantiation time:

```python
@ray.remote(num_cpus=2, num_gpus=0.5, memory=1024 * 1024 * 1024)
class GPUActor:
    pass

# Override at instantiation
actor = GPUActor.options(num_cpus=4, num_gpus=1).remote()
```

Fractional GPU allocation allows multiple actors to share a single GPU. Ray enforces resource limits at scheduling time but does not prevent over-allocation within a process.

**Named Actors:**

Named actors are discoverable across the cluster and persist beyond the creating process:

```python
# Create a named actor
counter = Counter.options(name="global_counter", lifetime="detached").remote()

# Retrieve from another process
counter = ray.get_actor("global_counter")

# Namespace isolation
counter = Counter.options(name="counter", namespace="prod").remote()
retrieved = ray.get_actor("counter", namespace="prod")
```

Detached actors (`lifetime="detached"`) survive driver script termination and require explicit termination via `ray.kill()`.

**Actor Pools:**

Use `ray.util.ActorPool` to distribute work across a pool of actors:

```python
from ray.util import ActorPool

@ray.remote
class Worker:
    def process(self, item):
        return item * 2

workers = [Worker.remote() for _ in range(4)]
pool = ActorPool(workers)

# Map function across pool (returns generator)
results = list(pool.map(lambda a, v: a.process.remote(v), [1, 2, 3, 4, 5, 6, 7, 8]))
```

ActorPool handles actor assignment automatically and supports `map`, `submit`, `get_next`, and `has_next` operations.

**Actor Lifecycle:**

- Actors start when `.remote()` is called on the class
- Actor methods execute serially by default (single-threaded)
- Enable concurrency with `max_concurrency` or async methods
- Terminate actors with `ray.kill(actor)` or let them garbage collect when no references remain

### Tasks

Tasks are stateless functions that execute remotely. Apply `@ray.remote` to a function to create a task.

**Basic Task Definition and Invocation:**

```python
@ray.remote
def compute_square(x):
    return x * x

# Submit task (returns ObjectRef immediately)
ref = compute_square.remote(5)

# Retrieve result (blocks)
result = ray.get(ref)

# Submit multiple tasks in parallel
refs = [compute_square.remote(i) for i in range(100)]
results = ray.get(refs)
```

**Task Options:**

Configure resources, retries, and scheduling at definition or call time:

```python
@ray.remote(num_cpus=2, num_gpus=1, max_retries=3, retry_exceptions=True)
def gpu_task(data):
    pass

# Override at call time
ref = gpu_task.options(num_cpus=4, max_retries=5).remote(data)
```

**Task Dependencies:**

Pass ObjectRefs as arguments to create task DAGs. Ray automatically resolves dependencies before execution:

```python
@ray.remote
def load_data():
    return [1, 2, 3, 4, 5]

@ray.remote
def process(data):
    return [x * 2 for x in data]

@ray.remote
def aggregate(processed):
    return sum(processed)

# Create DAG - Ray resolves dependencies automatically
data_ref = load_data.remote()
processed_ref = process.remote(data_ref)  # Waits for load_data
result_ref = aggregate.remote(processed_ref)  # Waits for process

final = ray.get(result_ref)
```

**Waiting for Tasks:**

Use `ray.wait()` to process results as they complete:

```python
refs = [slow_task.remote(i) for i in range(10)]

# Wait for at least 3 results with 10-second timeout
ready, remaining = ray.wait(refs, num_returns=3, timeout=10.0)

# Process completed results
for ref in ready:
    print(ray.get(ref))
```

### Object Store

The Ray object store is a shared-memory system that enables zero-copy data sharing between tasks and actors on the same node.

**Storing and Retrieving Objects:**

```python
# Explicitly store data in object store
data = [1, 2, 3, 4, 5] * 1000000
ref = ray.put(data)

# Pass reference to multiple tasks (zero-copy on same node)
results = ray.get([process.remote(ref) for _ in range(10)])

# Retrieve data
data_copy = ray.get(ref)
```

**Object Ownership:**

Objects are owned by the process that created them. When the owner exits, objects become unavailable unless:
- The object was created by a detached actor
- Object reconstruction is enabled via lineage

**Serialization:**

Ray uses Apache Arrow for serialization. NumPy arrays and Arrow-compatible data structures benefit from zero-copy reads. Custom objects use pickle serialization.

```python
import numpy as np

# Zero-copy for numpy arrays on same node
large_array = np.zeros((10000, 10000))
ref = ray.put(large_array)

# Custom serialization for complex objects
@ray.remote
def process_custom(obj):
    return obj.compute()
```

**Large Object Handling:**

For objects exceeding 100MB, consider:
- Storing data externally (S3, GCS) and passing URIs
- Using Ray Data for large datasets
- Chunking data across multiple objects

### Resource Management

**Custom Resources:**

Define custom resources for specialized scheduling:

```python
# Start Ray with custom resources
ray.init(resources={"special_hardware": 2})

@ray.remote(resources={"special_hardware": 1})
def specialized_task():
    pass
```

**Placement Groups:**

Group resources for gang scheduling:

```python
from ray.util.placement_group import placement_group, remove_placement_group

# Create placement group with specific bundles
pg = placement_group([
    {"CPU": 2, "GPU": 1},
    {"CPU": 2, "GPU": 1},
], strategy="PACK")

# Wait for placement group to be ready
ray.get(pg.ready())

# Schedule tasks/actors on placement group
@ray.remote(num_cpus=2, num_gpus=1)
def train_worker():
    pass

refs = [
    train_worker.options(
        placement_group=pg,
        placement_group_bundle_index=i
    ).remote()
    for i in range(2)
]

# Clean up
remove_placement_group(pg)
```

Strategies: `PACK` (minimize nodes), `SPREAD` (maximize distribution), `STRICT_PACK`, `STRICT_SPREAD`.

### Fault Tolerance

**Actor Restarts:**

Configure automatic actor restart on failure:

```python
@ray.remote(max_restarts=3, max_task_retries=-1)
class ResilientActor:
    def __init__(self):
        self.state = self._load_checkpoint()

    def _load_checkpoint(self):
        # Restore state from persistent storage
        pass

    def process(self, data):
        result = self._compute(data)
        self._save_checkpoint()
        return result
```

- `max_restarts`: Number of times to restart actor (-1 for unlimited)
- `max_task_retries`: Retry count for methods on restarted actor (-1 for unlimited)

**Task Retries:**

```python
@ray.remote(max_retries=5, retry_exceptions=[ConnectionError, TimeoutError])
def unreliable_task():
    pass
```

Set `retry_exceptions=True` to retry on any exception, or provide a list of specific exception types.

**Lineage Reconstruction:**

Enable object reconstruction from lineage when objects are lost:

```python
ray.init(_system_config={"object_reconstruction_enabled": True})
```

This allows Ray to re-execute tasks to reconstruct lost objects, at the cost of storing task lineage metadata.

## Common Pitfalls

- **Passing large objects directly to tasks**: Always use `ray.put()` for objects passed to multiple tasks; otherwise Ray serializes the object separately for each task invocation, wasting memory and bandwidth.

- **Blocking the driver with too many `ray.get()` calls**: Avoid calling `ray.get()` inside loops. Instead, submit all tasks first, collect ObjectRefs, then call `ray.get()` once on the list.

- **Creating too many fine-grained tasks**: Task overhead is approximately 1ms. Tasks that complete faster than this threshold should be batched together.

- **Ignoring actor serialization constraints**: Actor state must be picklable for checkpointing. Unpicklable objects (file handles, network connections) require manual reconstruction after restart.

- **Assuming resource enforcement at runtime**: Ray enforces resources at scheduling time only. An actor requesting 1 GPU can still access all GPUs on the node unless application code enforces isolation (e.g., `CUDA_VISIBLE_DEVICES`).

## Additional Resources

Consult `references/patterns.md` for detailed patterns covering:
- Actor lifecycle management (creation, naming, detaching, termination)
- Task DAG composition and dependency patterns
- Object store best practices (serialization, zero-copy, memory management)
- Resource allocation strategies (fractional resources, placement groups)
- Fault tolerance patterns (retries, restarts, lineage reconstruction)
- Performance optimization (batching, pipelining, avoiding anti-patterns)
- Ray Dashboard usage for debugging and monitoring
