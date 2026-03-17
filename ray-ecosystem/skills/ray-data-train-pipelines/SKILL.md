---
name: ray-data-train-pipelines
description: >
  This skill should be used when the user asks to "build a data pipeline with Ray",
  "set up distributed training", "preprocess data with Ray Data",
  or works with Ray Data or Ray Train.
---

## Purpose

Provide patterns and guidance for building distributed data preprocessing pipelines with Ray Data and distributed training workflows with Ray Train. This skill covers the complete data-to-model lifecycle: loading data from various sources, applying transformations at scale, ingesting data into training loops, and orchestrating distributed training across multiple workers and GPUs.

## Prerequisites

Before implementing data pipelines or training workflows, resolve context7 documentation for the latest Ray Data and Ray Train APIs:

1. Resolve the Ray library ID using `resolve-library-id` with query "Ray Data distributed data preprocessing and Ray Train distributed training workflows"
2. Query Ray Data documentation for dataset operations, transformations, and I/O patterns
3. Query Ray Train documentation for trainer configurations, scaling, and checkpointing

## Core Workflow

### Dataset Creation and Loading

Ray Data provides multiple entry points for creating distributed datasets from various sources.

**Creating Datasets from In-Memory Data:**

```python
import ray

# From Python objects
items = [{"id": i, "value": i * 2} for i in range(1000)]
dataset = ray.data.from_items(items)

# From pandas DataFrame
import pandas as pd
df = pd.DataFrame({"col1": range(100), "col2": ["a", "b"] * 50})
dataset = ray.data.from_pandas(df)

# From NumPy arrays
import numpy as np
arr = np.random.randn(1000, 10)
dataset = ray.data.from_numpy(arr)
```

**Reading from External Storage:**

```python
# Read Parquet files (supports S3, GCS, Azure, local)
dataset = ray.data.read_parquet("s3://bucket/data/*.parquet")

# Read CSV files with schema inference
dataset = ray.data.read_csv("/path/to/data/*.csv")

# Read JSON/JSONL files
dataset = ray.data.read_json("gs://bucket/data.jsonl")

# Read images for computer vision pipelines
dataset = ray.data.read_images("s3://bucket/images/", include_paths=True)

# Read binary files
dataset = ray.data.read_binary_files("/path/to/files/")
```

### Data Transformations

Ray Data executes transformations lazily in a streaming fashion, enabling efficient processing of datasets larger than cluster memory.

**Row-Level Transformations with `map`:**

```python
def transform_row(row):
    row["normalized"] = row["value"] / row["value"].max()
    row["label_encoded"] = 1 if row["category"] == "positive" else 0
    return row

transformed = dataset.map(transform_row)
```

**Filtering Rows:**

```python
# Keep rows matching a condition
filtered = dataset.filter(lambda row: row["score"] > 0.5)

# Chain multiple filters
filtered = dataset.filter(lambda r: r["age"] >= 18).filter(lambda r: r["country"] == "US")
```

**Expanding Rows with `flat_map`:**

```python
def expand_row(row):
    # Generate multiple output rows from one input row
    for token in row["text"].split():
        yield {"token": token, "source_id": row["id"]}

expanded = dataset.flat_map(expand_row)
```

**Batch Transformations with `map_batches`:**

Use `map_batches` for vectorized operations and GPU preprocessing:

```python
def preprocess_batch(batch):
    # batch is a pandas DataFrame or dict of numpy arrays
    batch["features"] = batch["raw_features"].apply(normalize)
    batch["embeddings"] = model.encode(batch["text"].tolist())
    return batch

# Process in pandas format for vectorized operations
processed = dataset.map_batches(
    preprocess_batch,
    batch_format="pandas",
    batch_size=256
)

# GPU-accelerated preprocessing
processed = dataset.map_batches(
    gpu_preprocess_fn,
    batch_size=64,
    num_gpus=1,  # Request GPU per batch worker
    concurrency=4  # Number of parallel workers
)
```

**Selecting and Dropping Columns:**

```python
# Keep only specific columns
selected = dataset.select_columns(["id", "features", "label"])

# Drop unnecessary columns
dropped = dataset.drop_columns(["metadata", "raw_text"])
```

### Writing Data to Storage

```python
# Write to Parquet (recommended for large datasets)
dataset.write_parquet("s3://bucket/output/")

# Write to CSV
dataset.write_csv("/local/output/")

# Write to JSON
dataset.write_json("gs://bucket/output.jsonl")

# Control output partitioning
dataset.repartition(100).write_parquet(
    "s3://bucket/output/",
    num_rows_per_file=100000
)
```

### Ray Train Integration

Ray Train provides high-level trainers for distributed training with automatic setup for data parallelism, gradient synchronization, and fault tolerance.

**TorchTrainer for PyTorch:**

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import ray.train

def train_func_per_worker(config):
    # Access hyperparameters
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    # Get data shards for this worker
    train_shard = ray.train.get_dataset_shard("train")
    valid_shard = ray.train.get_dataset_shard("valid")

    # Create iterators for training
    train_loader = train_shard.iter_torch_batches(batch_size=batch_size)
    valid_loader = valid_shard.iter_torch_batches(batch_size=batch_size)

    # Define model and wrap for distributed training
    model = MyModel()
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch["features"], batch["labels"]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch["features"], batch["labels"]
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        # Report metrics and save checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save(model.module.state_dict(), f"{temp_dir}/model.pt")
            ray.train.report(
                metrics={"loss": val_loss, "epoch": epoch},
                checkpoint=ray.train.Checkpoint.from_directory(temp_dir)
            )
```

**Configuring ScalingConfig:**

```python
scaling_config = ScalingConfig(
    num_workers=4,              # Number of training workers
    use_gpu=True,               # Allocate GPU per worker
    resources_per_worker={
        "CPU": 4,               # CPUs per worker
        "GPU": 1                # GPUs per worker
    },
    trainer_resources={
        "CPU": 1                # Resources for trainer coordinator
    }
)
```

**Configuring Checkpoints and Storage:**

```python
run_config = RunConfig(
    name="my-training-run",
    storage_path="s3://bucket/ray-results/",
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,                          # Keep top 3 checkpoints
        checkpoint_score_attribute="val_accuracy",
        checkpoint_score_order="max"            # Higher is better
    )
)
```

**Launching Training:**

```python
trainer = TorchTrainer(
    train_loop_per_worker=train_func_per_worker,
    train_loop_config={
        "lr": 1e-4,
        "epochs": 10,
        "batch_size_per_worker": 32
    },
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={
        "train": train_dataset,
        "valid": valid_dataset
    }
)

result = trainer.fit()
print(f"Best checkpoint: {result.best_checkpoints}")
```

**LightningTrainer for PyTorch Lightning:**

```python
from ray.train.lightning import LightningTrainer, LightningConfigBuilder

lightning_config = (
    LightningConfigBuilder()
    .module(MyLightningModule, hidden_size=256, learning_rate=1e-3)
    .trainer(max_epochs=10, accelerator="gpu")
    .fit_params(datamodule=MyDataModule())
    .checkpointing(monitor="val_loss", mode="min")
    .build()
)

trainer = LightningTrainer(
    lightning_config=lightning_config,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    run_config=RunConfig(storage_path="/tmp/ray_results")
)

result = trainer.fit()
```

**HuggingFaceTrainer for Transformers:**

```python
from ray.train.huggingface import TransformersTrainer
from transformers import TrainingArguments, AutoModelForSequenceClassification

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    save_strategy="epoch"
)

trainer = TransformersTrainer(
    trainer_init_per_worker=lambda: Trainer(
        model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased"),
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval
    ),
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    datasets={"train": ray_train_ds, "eval": ray_eval_ds}
)

result = trainer.fit()
```

### Reporting Metrics with train.report()

Use `ray.train.report()` to send metrics and checkpoints to Ray Train:

```python
import ray.train

# Report metrics only
ray.train.report(metrics={"loss": 0.5, "accuracy": 0.85})

# Report metrics with checkpoint
with tempfile.TemporaryDirectory() as temp_dir:
    torch.save(model.state_dict(), f"{temp_dir}/model.pt")
    ray.train.report(
        metrics={"loss": 0.5, "accuracy": 0.85},
        checkpoint=ray.train.Checkpoint.from_directory(temp_dir)
    )
```

## Common Pitfalls

- **Memory exhaustion during data loading**: Use `streaming=True` for datasets larger than cluster memory, and configure appropriate `batch_size` in `map_batches` to control memory usage per worker.

- **Incorrect batch format in transformations**: Specify `batch_format="pandas"` or `batch_format="numpy"` explicitly in `map_batches` to ensure the expected data structure; the default may vary based on the source data.

- **Forgetting to wrap models for distributed training**: Always call `ray.train.torch.prepare_model(model)` inside the training function to enable gradient synchronization across workers; skipping this results in each worker training an independent model.

- **Blocking on data iteration outside training loop**: Call `iter_torch_batches()` inside the training loop, not outside; creating the iterator prematurely can cause data loading to block or fail in distributed settings.

- **Not using dataset shards in training workers**: Always retrieve data via `ray.train.get_dataset_shard("train")` inside the training function; passing raw datasets directly bypasses Ray Train's distributed data loading and causes each worker to process the full dataset.

## Additional Resources

Refer to `references/pipeline-patterns.md` for detailed patterns including:

- Advanced data loading patterns (windowed, streaming, partitioned)
- GPU preprocessing pipelines
- Distributed training strategies (DDP, FSDP, DeepSpeed)
- Checkpointing and fault tolerance
- End-to-end data-to-training orchestration
