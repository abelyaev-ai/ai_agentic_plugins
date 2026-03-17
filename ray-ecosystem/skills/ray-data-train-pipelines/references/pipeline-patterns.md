# Ray Data and Ray Train Pipeline Patterns

This reference provides detailed patterns for building production-grade distributed data preprocessing and training pipelines with Ray Data and Ray Train.

## Data Loading Patterns

### Basic File Loading

Load data from common file formats with automatic parallelization:

```python
import ray

# Parquet with column pruning (read only needed columns)
dataset = ray.data.read_parquet(
    "s3://bucket/data/",
    columns=["id", "features", "label"],
    filter=pyarrow.compute.field("year") == 2024
)

# CSV with explicit schema
dataset = ray.data.read_csv(
    "/data/files/*.csv",
    parse_options=pyarrow.csv.ParseOptions(delimiter="\t"),
    convert_options=pyarrow.csv.ConvertOptions(
        column_types={"id": pyarrow.int64(), "score": pyarrow.float32()}
    )
)

# JSON with per-line parsing
dataset = ray.data.read_json(
    "gs://bucket/logs/*.jsonl",
    lines=True
)
```

### Partitioned Data Loading

Handle partitioned datasets efficiently:

```python
# Hive-style partitioned data
dataset = ray.data.read_parquet(
    "s3://bucket/data/",
    partitioning=ray.data.Partitioning("hive", base_dir="s3://bucket/data/")
)

# Filter partitions at read time
dataset = ray.data.read_parquet(
    "s3://bucket/data/year=2024/month=01/",
)

# Custom partitioning scheme
from ray.data.datasource.partitioning import PathPartitionFilter

filter_fn = PathPartitionFilter.of(
    style="hive",
    filter_fn=lambda d: d["region"] == "us-west"
)
dataset = ray.data.read_parquet("s3://bucket/", partition_filter=filter_fn)
```

### Image and Binary Data Loading

Load unstructured data for computer vision and multimedia pipelines:

```python
# Load images with metadata
dataset = ray.data.read_images(
    "s3://bucket/images/",
    include_paths=True,
    mode="RGB",
    size=(224, 224)  # Resize on load
)

# Load binary files
dataset = ray.data.read_binary_files(
    "/data/documents/",
    include_paths=True
)

# Custom image preprocessing on load
def decode_and_preprocess(batch):
    from PIL import Image
    import io

    images = []
    for item in batch["bytes"]:
        img = Image.open(io.BytesIO(item))
        img = img.resize((224, 224))
        images.append(np.array(img))

    batch["image"] = images
    return batch

dataset = ray.data.read_binary_files("s3://bucket/images/").map_batches(
    decode_and_preprocess,
    batch_size=64
)
```

### Streaming Data Ingestion

Process data in streaming fashion for memory-efficient pipelines:

```python
# Enable streaming execution
ray.data.DataContext.get_current().execution_options.locality_with_output = True

# Stream through transformations without materializing
dataset = ray.data.read_parquet("s3://bucket/large-dataset/")
processed = (
    dataset
    .map_batches(preprocess_fn, batch_size=1000)
    .filter(quality_filter)
    .map_batches(feature_extraction, batch_size=500)
)

# Iterate in streaming fashion
for batch in processed.iter_batches(batch_size=256):
    # Process batch
    pass
```

## Windowed vs Batch Transforms

### Batch Transformations

Standard batch processing for independent row transformations:

```python
def normalize_batch(batch):
    """Normalize features within each batch independently."""
    features = batch["features"]
    batch["normalized"] = (features - features.mean(axis=0)) / features.std(axis=0)
    return batch

dataset = dataset.map_batches(
    normalize_batch,
    batch_format="numpy",
    batch_size=1024
)
```

### Windowed Transformations

Process data with temporal or sequential dependencies:

```python
def compute_rolling_stats(batch, window_size=10):
    """Compute rolling statistics requiring window context."""
    import pandas as pd

    df = pd.DataFrame(batch)
    df["rolling_mean"] = df["value"].rolling(window=window_size, min_periods=1).mean()
    df["rolling_std"] = df["value"].rolling(window=window_size, min_periods=1).std()
    return df.to_dict("list")

# Sort before windowed operations
sorted_dataset = dataset.sort("timestamp")
windowed = sorted_dataset.map_batches(
    compute_rolling_stats,
    batch_format="pandas",
    batch_size=1000,
    fn_kwargs={"window_size": 10}
)
```

### Stateful Transformations with Actors

Maintain state across batches using class-based transforms:

```python
class StatefulProcessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.batch_count = 0

    def __call__(self, batch):
        # Update running statistics
        batch_mean = batch["features"].mean(axis=0)
        batch_std = batch["features"].std(axis=0)

        if self.global_mean is None:
            self.global_mean = batch_mean
            self.global_std = batch_std
        else:
            # Incremental update
            self.global_mean = (self.global_mean * self.batch_count + batch_mean) / (self.batch_count + 1)

        self.batch_count += 1

        # Apply normalization with global stats
        batch["normalized"] = (batch["features"] - self.global_mean) / (self.global_std + 1e-8)
        return batch

dataset = dataset.map_batches(
    StatefulProcessor,
    batch_format="numpy",
    batch_size=512,
    concurrency=1  # Single instance for consistent state
)
```

## GPU Preprocessing

### GPU-Accelerated Transformations

Leverage GPUs for compute-intensive preprocessing:

```python
class GPUPreprocessor:
    def __init__(self):
        import torch
        self.device = torch.device("cuda")
        self.model = load_embedding_model().to(self.device)
        self.model.eval()

    def __call__(self, batch):
        import torch

        with torch.no_grad():
            # Move data to GPU
            inputs = torch.tensor(batch["features"]).to(self.device)

            # GPU computation
            embeddings = self.model(inputs)

            # Move back to CPU for output
            batch["embeddings"] = embeddings.cpu().numpy()

        return batch

dataset = dataset.map_batches(
    GPUPreprocessor,
    batch_size=128,
    num_gpus=1,
    concurrency=4  # 4 GPU workers
)
```

### Multi-GPU Image Processing

Distribute image preprocessing across multiple GPUs:

```python
class ImageAugmentor:
    def __init__(self):
        import torch
        import torchvision.transforms as T

        self.device = torch.device("cuda")
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, batch):
        import torch

        images = torch.tensor(batch["image"]).permute(0, 3, 1, 2).float() / 255.0
        images = images.to(self.device)

        augmented = torch.stack([self.transform(img) for img in images])
        batch["augmented"] = augmented.cpu().numpy()
        return batch

# Distribute across 8 GPUs
dataset = dataset.map_batches(
    ImageAugmentor,
    batch_size=64,
    num_gpus=1,
    concurrency=8
)
```

### Hybrid CPU-GPU Pipelines

Combine CPU preprocessing with GPU-accelerated feature extraction:

```python
# CPU: Data loading and basic preprocessing
dataset = ray.data.read_images("s3://bucket/images/")

# CPU: Resize and basic augmentation
def cpu_preprocess(batch):
    from PIL import Image
    processed = []
    for img in batch["image"]:
        img = Image.fromarray(img).resize((256, 256))
        processed.append(np.array(img))
    batch["resized"] = processed
    return batch

dataset = dataset.map_batches(cpu_preprocess, batch_size=256)

# GPU: Feature extraction
class FeatureExtractor:
    def __init__(self):
        import torch
        import torchvision.models as models

        self.device = torch.device("cuda")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()

    def __call__(self, batch):
        import torch

        with torch.no_grad():
            inputs = torch.tensor(batch["resized"]).permute(0, 3, 1, 2).float()
            inputs = inputs.to(self.device) / 255.0
            features = self.model(inputs)
            batch["features"] = features.cpu().numpy()
        return batch

dataset = dataset.map_batches(
    FeatureExtractor,
    batch_size=32,
    num_gpus=1,
    concurrency=4
)
```

## Distributed Training Strategies

### Data Distributed Parallel (DDP)

Standard data parallelism where each worker has a full model copy:

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.torch

def train_func_ddp(config):
    import torch
    import torch.nn as nn

    # Get data shard for this worker
    train_data = ray.train.get_dataset_shard("train")

    # Initialize model
    model = MyModel()

    # Wrap model for DDP - this handles gradient synchronization
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        for batch in train_data.iter_torch_batches(batch_size=config["batch_size"]):
            optimizer.zero_grad()
            loss = model(batch["features"], batch["labels"])
            loss.backward()  # Gradients automatically synchronized
            optimizer.step()

        ray.train.report(metrics={"epoch": epoch, "loss": loss.item()})

trainer = TorchTrainer(
    train_loop_per_worker=train_func_ddp,
    train_loop_config={"lr": 1e-4, "epochs": 10, "batch_size": 32},
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 1}
    ),
    datasets={"train": train_dataset}
)
```

### Fully Sharded Data Parallel (FSDP)

Shard model parameters across workers for large models:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def train_func_fsdp(config):
    import torch

    # Get data shard
    train_data = ray.train.get_dataset_shard("train")

    # Initialize model
    model = LargeTransformerModel()

    # Configure FSDP wrapping policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerEncoderLayer}
    )

    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device()
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        for batch in train_data.iter_torch_batches(batch_size=config["batch_size"]):
            optimizer.zero_grad()
            loss = model(batch["input_ids"], batch["labels"])
            loss.backward()
            optimizer.step()

        # Save FSDP checkpoint
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()

        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save(state_dict, f"{temp_dir}/model.pt")
            ray.train.report(
                metrics={"loss": loss.item()},
                checkpoint=ray.train.Checkpoint.from_directory(temp_dir)
            )

trainer = TorchTrainer(
    train_loop_per_worker=train_func_fsdp,
    train_loop_config={"lr": 1e-5, "epochs": 3, "batch_size": 4},
    scaling_config=ScalingConfig(
        num_workers=16,
        use_gpu=True,
        resources_per_worker={"CPU": 8, "GPU": 1}
    ),
    datasets={"train": train_dataset}
)
```

### DeepSpeed Integration

Use DeepSpeed for ZeRO optimization and mixed precision:

```python
from ray.train.torch import TorchTrainer
import deepspeed

def train_func_deepspeed(config):
    import torch

    train_data = ray.train.get_dataset_shard("train")

    model = LargeLanguageModel()

    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": config["global_batch_size"],
        "train_micro_batch_size_per_gpu": config["micro_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation"],
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": config["lr"], "weight_decay": 0.01}
        },
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
            "overlap_comm": True,
            "contiguous_gradients": True
        }
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    for epoch in range(config["epochs"]):
        for batch in train_data.iter_torch_batches(batch_size=config["micro_batch_size"]):
            loss = model_engine(batch["input_ids"], batch["labels"])
            model_engine.backward(loss)
            model_engine.step()

        ray.train.report(metrics={"loss": loss.item()})

trainer = TorchTrainer(
    train_loop_per_worker=train_func_deepspeed,
    train_loop_config={
        "lr": 1e-5,
        "epochs": 3,
        "micro_batch_size": 2,
        "global_batch_size": 64,
        "gradient_accumulation": 8
    },
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"CPU": 8, "GPU": 1}
    ),
    datasets={"train": train_dataset}
)
```

## Checkpointing in Training

### Basic Checkpointing

Save model state periodically during training:

```python
def train_func_with_checkpoints(config):
    import torch
    import tempfile

    model = MyModel()
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Resume from checkpoint if available
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            state = torch.load(f"{checkpoint_dir}/checkpoint.pt")
            model.module.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
    else:
        start_epoch = 0

    train_data = ray.train.get_dataset_shard("train")

    for epoch in range(start_epoch, config["epochs"]):
        total_loss = 0
        for batch in train_data.iter_torch_batches(batch_size=config["batch_size"]):
            loss = train_step(model, optimizer, batch)
            total_loss += loss.item()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, f"{temp_dir}/checkpoint.pt")

            ray.train.report(
                metrics={"loss": total_loss, "epoch": epoch},
                checkpoint=ray.train.Checkpoint.from_directory(temp_dir)
            )
```

### Distributed Checkpointing

Save model shards in parallel across workers:

```python
def train_func_distributed_checkpoint(config):
    import torch
    import tempfile
    import os

    model = LargeModel()
    model = ray.train.torch.prepare_model(model)

    train_data = ray.train.get_dataset_shard("train")
    rank = ray.train.get_context().get_world_rank()

    for epoch in range(config["epochs"]):
        for batch in train_data.iter_torch_batches(batch_size=config["batch_size"]):
            # Training step
            pass

        # Each worker saves its shard
        with tempfile.TemporaryDirectory() as temp_dir:
            shard_path = os.path.join(temp_dir, f"model-rank={rank}.pt")
            torch.save(model.module.state_dict(), shard_path)

            ray.train.report(
                metrics={"epoch": epoch},
                checkpoint=ray.train.Checkpoint.from_directory(temp_dir)
            )

# Cloud storage will contain: model-rank=0.pt, model-rank=1.pt, ...
trainer = TorchTrainer(
    train_loop_per_worker=train_func_distributed_checkpoint,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    run_config=RunConfig(storage_path="s3://bucket/checkpoints/")
)
```

### Checkpoint Selection and Retention

Configure which checkpoints to keep:

```python
from ray.train import RunConfig, CheckpointConfig

run_config = RunConfig(
    name="training-experiment",
    storage_path="s3://bucket/results/",
    checkpoint_config=CheckpointConfig(
        num_to_keep=5,                          # Keep top 5 checkpoints
        checkpoint_score_attribute="val_accuracy",
        checkpoint_score_order="max",           # Higher accuracy is better
        checkpoint_frequency=1,                 # Checkpoint every epoch
        checkpoint_at_end=True                  # Always checkpoint at end
    )
)
```

## Fault Tolerance

### Automatic Failure Recovery

Configure Ray Train to recover from worker failures:

```python
from ray.train import RunConfig, FailureConfig

run_config = RunConfig(
    failure_config=FailureConfig(
        max_failures=3,  # Retry up to 3 times on failure
    ),
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_frequency=1  # Checkpoint every epoch for recovery
    ),
    storage_path="s3://bucket/results/"  # Persistent storage for recovery
)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    run_config=run_config
)
```

### Handling Spot Instance Preemption

Design training for resilience to spot instance termination:

```python
def robust_train_func(config):
    import torch
    import signal

    # Handle termination signals gracefully
    def handle_sigterm(signum, frame):
        ray.train.report(
            metrics={"interrupted": True},
            checkpoint=create_checkpoint()
        )

    signal.signal(signal.SIGTERM, handle_sigterm)

    # Always resume from checkpoint if available
    checkpoint = ray.train.get_checkpoint()
    start_epoch, model, optimizer = restore_from_checkpoint(checkpoint)

    train_data = ray.train.get_dataset_shard("train")

    for epoch in range(start_epoch, config["epochs"]):
        for batch in train_data.iter_torch_batches(batch_size=config["batch_size"]):
            train_step(model, optimizer, batch)

        # Frequent checkpointing for spot instances
        save_checkpoint(model, optimizer, epoch)

# Use with spot instances
scaling_config = ScalingConfig(
    num_workers=8,
    use_gpu=True,
    resources_per_worker={"CPU": 4, "GPU": 1}
)

run_config = RunConfig(
    failure_config=FailureConfig(max_failures=10),  # Higher for spot
    checkpoint_config=CheckpointConfig(
        checkpoint_frequency=1,
        num_to_keep=3
    ),
    storage_path="s3://bucket/spot-training/"
)
```

### Elastic Training

Scale workers dynamically based on cluster availability:

```python
# Configure elastic training bounds
scaling_config = ScalingConfig(
    num_workers=8,
    use_gpu=True,
    # Elastic training configuration (if supported)
    min_workers=2,  # Minimum workers to continue training
    max_workers=16  # Maximum workers when resources available
)
```

## Data-to-Training Pipeline Orchestration

### End-to-End Pipeline

Orchestrate the complete data preprocessing and training workflow:

```python
import ray
from ray.data import Dataset
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

def create_pipeline():
    # Stage 1: Load raw data
    raw_dataset = ray.data.read_parquet("s3://bucket/raw-data/")

    # Stage 2: Data cleaning
    cleaned = raw_dataset.filter(
        lambda row: row["label"] is not None and row["features"] is not None
    )

    # Stage 3: Feature engineering
    def engineer_features(batch):
        batch["processed_features"] = preprocess(batch["features"])
        batch["normalized"] = normalize(batch["processed_features"])
        return batch

    features = cleaned.map_batches(engineer_features, batch_format="pandas")

    # Stage 4: Train/validation split
    train_ds, valid_ds = features.train_test_split(test_size=0.2)

    return train_ds, valid_ds

def train_func(config):
    train_data = ray.train.get_dataset_shard("train")
    valid_data = ray.train.get_dataset_shard("valid")

    model = create_model(config)
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        for batch in train_data.iter_torch_batches(batch_size=config["batch_size"]):
            train_step(model, optimizer, batch)

        # Validation
        model.eval()
        val_metrics = evaluate(model, valid_data)

        ray.train.report(metrics=val_metrics)

# Execute pipeline
train_ds, valid_ds = create_pipeline()

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"lr": 1e-4, "epochs": 20, "batch_size": 64},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    run_config=RunConfig(storage_path="s3://bucket/results/"),
    datasets={"train": train_ds, "valid": valid_ds}
)

result = trainer.fit()
```

### Decoupled Preprocessing and Training

Scale preprocessing and training independently:

```python
# Preprocessing job (CPU-heavy)
@ray.remote(num_cpus=4)
def preprocess_partition(partition_path):
    dataset = ray.data.read_parquet(partition_path)
    processed = dataset.map_batches(cpu_preprocess_fn, batch_size=1000)
    output_path = partition_path.replace("raw", "processed")
    processed.write_parquet(output_path)
    return output_path

# Run preprocessing in parallel
raw_partitions = list_partitions("s3://bucket/raw-data/")
processed_paths = ray.get([
    preprocess_partition.remote(p) for p in raw_partitions
])

# Training job (GPU-heavy)
processed_dataset = ray.data.read_parquet(processed_paths)
train_ds, valid_ds = processed_dataset.train_test_split(test_size=0.1)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"CPU": 2, "GPU": 1}  # Fewer CPUs for training
    ),
    datasets={"train": train_ds, "valid": valid_ds}
)
```

### Streaming Data into Training

Stream data from preprocessing directly into training:

```python
def create_streaming_dataset():
    """Create a dataset that streams through preprocessing."""
    dataset = ray.data.read_parquet("s3://bucket/data/")

    # Transformations execute lazily in streaming fashion
    return (
        dataset
        .map_batches(normalize_fn, batch_size=512)
        .map_batches(augment_fn, batch_size=256)
        .random_shuffle()
    )

def train_func_streaming(config):
    # Data streams through as training consumes it
    train_data = ray.train.get_dataset_shard("train")

    model = ray.train.torch.prepare_model(MyModel())
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(config["epochs"]):
        # iter_torch_batches triggers streaming execution
        for batch in train_data.iter_torch_batches(
            batch_size=config["batch_size"],
            prefetch_batches=4  # Prefetch for pipeline efficiency
        ):
            train_step(model, optimizer, batch)

        ray.train.report(metrics={"epoch": epoch})

streaming_dataset = create_streaming_dataset()

trainer = TorchTrainer(
    train_loop_per_worker=train_func_streaming,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    datasets={"train": streaming_dataset}
)
```

### Multi-Stage Pipeline with Intermediate Storage

For complex pipelines, materialize intermediate results:

```python
def multi_stage_pipeline():
    # Stage 1: Extract
    raw = ray.data.read_json("s3://bucket/raw/")

    # Stage 2: Transform (materialize for reuse)
    transformed = raw.map_batches(transform_fn).materialize()
    transformed.write_parquet("s3://bucket/transformed/")

    # Stage 3: Feature engineering
    features = transformed.map_batches(feature_fn)

    # Stage 4: Split for training
    train_ds, valid_ds, test_ds = features.split_proportionately([0.8, 0.1, 0.1])

    # Materialize test set for later evaluation
    test_ds.write_parquet("s3://bucket/test-set/")

    return train_ds, valid_ds

train_ds, valid_ds = multi_stage_pipeline()

# Training uses streaming train/valid sets
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    datasets={"train": train_ds, "valid": valid_ds},
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True)
)

result = trainer.fit()

# Evaluate on materialized test set
test_ds = ray.data.read_parquet("s3://bucket/test-set/")
evaluate_model(result.best_checkpoints[0], test_ds)
```

## Performance Optimization Tips

1. **Match batch sizes to GPU memory**: Use the largest batch size that fits in GPU memory to maximize throughput.

2. **Prefetch data batches**: Set `prefetch_batches` in `iter_torch_batches()` to overlap data loading with computation.

3. **Use appropriate concurrency**: Set `concurrency` in `map_batches()` to control parallel preprocessing workers.

4. **Minimize data serialization**: Keep data in Ray object store format; avoid unnecessary conversions between formats.

5. **Leverage locality**: Enable `locality_with_output` in execution options to co-locate data with compute.

6. **Profile pipeline stages**: Use Ray Dashboard to identify bottlenecks in data loading vs. training.

7. **Consider memory pressure**: Monitor object store memory; use streaming execution for large datasets.

8. **Optimize checkpoint frequency**: Balance between recovery granularity and checkpoint overhead.
