---
name: ray-cluster-config
description: >
  This skill should be used when the user asks to "set up a Ray cluster",
  "configure autoscaling", "allocate cluster resources",
  "deploy Ray to production", or works with Ray cluster management.
---

## Purpose

This skill provides comprehensive patterns for Ray cluster setup, resource allocation, and production deployment. Use this skill to configure Ray clusters for local development, cloud deployment, or Kubernetes-based production environments. The guidance covers cluster initialization options, autoscaling configuration, resource specification, and runtime environment packaging.

## Prerequisites

Before implementing Ray cluster configuration, resolve the Ray library documentation using context7:

1. Use `resolve-library-id` with query "ray" to obtain the Context7-compatible library ID
2. Use `query-docs` with the resolved library ID and queries like:
   - "Ray cluster configuration setup autoscaling"
   - "ray.init options local cluster"
   - "runtime_env pip conda dependencies"
   - "KubeRay operator RayCluster deployment"

## Core Workflow

### Cluster Initialization with ray.init()

Ray provides flexible initialization options depending on the deployment target. The `ray.init()` function serves as the entry point for all Ray applications.

**Local Development Mode**

Start a single-node Ray cluster for local development and testing:

```python
import ray

# Start a local Ray cluster with default settings
ray.init()

# Start with specific resource allocation
ray.init(
    num_cpus=8,
    num_gpus=2,
    object_store_memory=10 * 1024 * 1024 * 1024,  # 10 GB
)

# Include runtime environment for dependency management
ray.init(
    runtime_env={
        "pip": ["numpy", "pandas", "scikit-learn"],
        "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}
    }
)
```

**Connecting to Existing Clusters**

Connect to a running Ray cluster using the cluster address:

```python
# Connect to a cluster via Ray address
ray.init(address="auto")  # Auto-detect from RAY_ADDRESS env var

# Connect to a specific cluster address
ray.init(address="ray://192.168.1.100:10001")

# Connect with namespace isolation
ray.init(
    address="ray://192.168.1.100:10001",
    namespace="production",
    runtime_env={
        "working_dir": "./my_project",
        "pip": ["transformers", "torch"]
    }
)
```

### Cluster YAML Configuration

For cloud deployments, define cluster configuration in YAML files. The cluster launcher uses these configurations to provision and manage infrastructure.

**Basic Cluster Configuration Structure**

```yaml
cluster_name: my-ray-cluster

# Cloud provider configuration
provider:
  type: aws  # aws, gcp, azure, local
  region: us-west-2
  availability_zone: us-west-2a

# Authentication and access
auth:
  ssh_user: ubuntu
  ssh_private_key: ~/.ssh/ray-cluster-key.pem

# Head node configuration
head_node:
  InstanceType: m5.2xlarge
  ImageId: ami-0abcdef1234567890
  BlockDeviceMappings:
    - DeviceName: /dev/sda1
      Ebs:
        VolumeSize: 200
        VolumeType: gp3

# Worker node configuration
worker_nodes:
  InstanceType: m5.4xlarge
  ImageId: ami-0abcdef1234567890

# Resource limits
min_workers: 2
max_workers: 10
initial_workers: 2
```

### Head Node vs Worker Node Setup

**Head Node Responsibilities**

The head node manages cluster orchestration and runs critical services:

- Global Control Store (GCS) for cluster state management
- Autoscaler for dynamic worker scaling
- Dashboard for monitoring and debugging
- Driver programs and job submission endpoint

Configure head node resources appropriately:

```yaml
head_node_type:
  node_config:
    InstanceType: m5.2xlarge
  resources:
    CPU: 4
    memory: 16000000000  # 16 GB
  # Reserve resources for cluster management
  # Avoid scheduling compute-intensive tasks on head
```

**Worker Node Configuration**

Worker nodes execute tasks and actors. Configure worker node types based on workload requirements:

```yaml
worker_node_types:
  cpu_workers:
    min_workers: 2
    max_workers: 20
    node_config:
      InstanceType: c5.4xlarge
    resources:
      CPU: 16
      memory: 32000000000

  gpu_workers:
    min_workers: 0
    max_workers: 8
    node_config:
      InstanceType: p3.2xlarge
    resources:
      CPU: 8
      GPU: 1
      memory: 61000000000
```

### Autoscaler Configuration

The Ray autoscaler dynamically adjusts cluster size based on resource demand. Configure autoscaling behavior through these parameters:

```yaml
# Autoscaling configuration
autoscaling_config:
  # Upscaling behavior
  upscaling_speed: 1.0  # Aggressive scaling multiplier

  # Downscaling behavior
  idle_timeout_minutes: 5  # Time before removing idle workers

  # Resource utilization targets
  target_utilization_fraction: 0.8

# Worker scaling bounds
min_workers: 1
max_workers: 100

# Initial cluster size
initial_workers: 5
```

**Programmatic Autoscaling Control**

```python
from ray.util.spark import setup_ray_cluster

# Configure autoscaling cluster on Spark
setup_ray_cluster(
    max_worker_nodes=50,
    min_worker_nodes=5,
    num_cpus_worker_node=4,
    num_gpus_worker_node=1,
    autoscale_upscaling_speed=2.0,
    autoscale_idle_timeout_minutes=10,
)
```

### Resource Specification

**CPU and GPU Allocation**

Specify resource requirements for tasks and actors:

```python
@ray.remote(num_cpus=4, num_gpus=1)
def train_model(data):
    # Uses 4 CPUs and 1 GPU
    pass

@ray.remote(num_cpus=2, num_gpus=0.5)
class ModelServer:
    # Uses 2 CPUs and half a GPU (GPU sharing)
    pass
```

**Custom Resources**

Define custom resources for specialized hardware or logical resource management:

```yaml
# In cluster YAML
worker_nodes:
  resources:
    CPU: 16
    GPU: 4
    TPU: 1
    custom_accelerator: 2
    memory_tier_fast: 100
```

```python
# Request custom resources in code
@ray.remote(resources={"TPU": 1, "custom_accelerator": 1})
def tpu_computation():
    pass
```

**Accelerator Type Specification**

For heterogeneous GPU clusters, specify accelerator types:

```python
@ray.remote(
    num_gpus=1,
    accelerator_type="nvidia-tesla-v100"
)
def v100_training():
    pass

@ray.remote(
    num_gpus=1,
    accelerator_type="nvidia-a100"
)
def a100_training():
    pass
```

### Runtime Environment Packaging

The `runtime_env` parameter enables dependency isolation and reproducibility across cluster nodes.

**Pip Dependencies**

```python
runtime_env = {
    "pip": [
        "torch==2.0.0",
        "transformers>=4.30.0",
        "numpy",
        "-e git+https://github.com/org/repo.git#egg=package"
    ]
}
ray.init(runtime_env=runtime_env)
```

**Conda Environments**

```python
# Using conda environment dictionary
runtime_env = {
    "conda": {
        "dependencies": [
            "pytorch",
            "torchvision",
            "pip",
            {"pip": ["pendulum", "ray[default]"]}
        ],
        "channels": ["pytorch", "conda-forge"]
    }
}

# Using existing conda environment
runtime_env = {
    "conda": "pytorch_env"  # Name of existing environment
}

# Using environment.yml file
runtime_env = {
    "conda": "./environment.yml"
}
```

**Working Directory and Modules**

```python
runtime_env = {
    "working_dir": "./my_project",  # Upload local directory
    "py_modules": [
        "./utils",
        "./models",
        "s3://bucket/shared_module.zip"
    ],
    "excludes": ["*.pyc", "__pycache__", "*.log"]
}
```

**Environment Variables**

```python
runtime_env = {
    "env_vars": {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": "/opt/custom/lib",
        "API_KEY": "secret_value"  # Caution: visible in logs
    }
}
```

**Complete Runtime Environment Example**

```python
from ray.runtime_env import RuntimeEnv

runtime_env = RuntimeEnv(
    working_dir="./src",
    pip=["torch>=2.0", "transformers", "datasets"],
    env_vars={
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_LAUNCH_BLOCKING": "1"
    },
    config={
        "setup_timeout_seconds": 600,
        "eager_install": True
    }
)

ray.init(runtime_env=runtime_env)
```

## Common Pitfalls

- **Object Store Memory Exhaustion**: Avoid storing large objects in the Ray object store without proper lifecycle management. Use `ray.put()` judiciously and delete references when no longer needed. Configure `object_store_memory` based on workload requirements.

- **Head Node Overload**: Do not schedule compute-intensive tasks on the head node. Reserve head node resources for cluster management, GCS, and autoscaler operations. Use resource specifications to prevent accidental scheduling.

- **Runtime Environment Conflicts**: Mixing pip and conda in the same `runtime_env` causes validation errors. Use conda's pip integration by specifying pip dependencies within the conda environment definition.

- **Autoscaler Thrashing**: Setting `idle_timeout_minutes` too low causes frequent scale-up/scale-down cycles. Balance responsiveness with stability by configuring appropriate timeout values based on workload patterns.

- **Resource Over-Specification**: Requesting more resources than physically available causes tasks to hang indefinitely. Verify cluster capacity before specifying resource requirements. Use `ray.cluster_resources()` to check available resources.

## Additional Resources

Refer to `references/production-config.md` for detailed guidance on:

- KubeRay operator setup and CRD configuration
- Helm chart deployment patterns
- Cloud-specific deployment (AWS, GCP, Azure)
- Prometheus and Grafana monitoring integration
- Security configuration (TLS, authentication)
- GPU scheduling and node affinity patterns
