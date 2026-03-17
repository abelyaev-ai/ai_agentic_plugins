# Search Algorithms and Schedulers Reference

This reference provides detailed guidance for selecting and configuring Ray Tune search algorithms and schedulers. Use this material to make informed decisions about optimization strategies based on problem characteristics, computational constraints, and performance requirements.

## Search Algorithm Comparison

### Algorithm Selection Matrix

| Algorithm | Best For | Search Space | Sample Efficiency | Parallelism | Dependencies |
|-----------|----------|--------------|-------------------|-------------|--------------|
| Random Search | Baseline, high-dimensional spaces | Any | Low | Excellent | None |
| Grid Search | Small discrete spaces, exhaustive coverage | Discrete only | N/A (exhaustive) | Excellent | None |
| OptunaSearch (TPE) | General purpose, mixed spaces | Any | High | Moderate | `optuna` |
| BayesOptSearch (GP) | Continuous spaces, expensive evaluations | Continuous preferred | Very High | Low | `bayesian-optimization` |
| HyperOptSearch | Mixed spaces, conditional parameters | Any | High | Moderate | `hyperopt` |
| TuneBOHB | Large budgets with early stopping | Any | High | Moderate | `hpbandster`, `ConfigSpace` |
| AxSearch | Multi-objective, constraints | Any | Very High | Moderate | `ax-platform` |
| NevergradSearch | Gradient-free, noisy objectives | Continuous preferred | Moderate | Good | `nevergrad` |

### Random and Grid Search

Random search provides a strong baseline that often outperforms sophisticated methods in high-dimensional spaces where only a few parameters matter (the "blessing of dimensionality"). Use random search when:

- Starting exploration of a new problem
- The search space exceeds 10 dimensions
- Computational budget allows many parallel trials
- No prior knowledge about good hyperparameter regions exists

```python
from ray import tune

# Random search is the default when no search_alg is specified
tuner = Tuner(
    training_function,
    tune_config=TuneConfig(
        num_samples=100,
        metric="loss",
        mode="min",
    ),
    param_space={
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "hidden_dim": tune.randint(64, 512),
    },
)
```

Grid search exhaustively evaluates all combinations of discrete values. Reserve for small spaces requiring complete coverage:

```python
param_space = {
    "optimizer": tune.grid_search(["adam", "sgd", "rmsprop"]),
    "lr": tune.grid_search([0.001, 0.01, 0.1]),
    "weight_decay": tune.grid_search([0.0, 0.0001, 0.001]),
}
# Total trials: 3 * 3 * 3 = 27
```

### Optuna (Tree-structured Parzen Estimator)

OptunaSearch implements TPE, which models the conditional probability of hyperparameters given good vs. bad performance. TPE handles mixed continuous/discrete spaces and conditional parameters effectively.

```python
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
import optuna

# Basic setup
searcher = OptunaSearch(
    metric="val_loss",
    mode="min",
)
search_alg = ConcurrencyLimiter(searcher, max_concurrent=4)

# With sampler configuration
sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,      # Random trials before TPE kicks in
    n_ei_candidates=24,       # Candidates for EI evaluation
    multivariate=True,        # Model parameter dependencies
)
searcher = OptunaSearch(
    metric="val_loss",
    mode="min",
    sampler=sampler,
)

# Provide initial points to seed the search
initial_params = [
    {"lr": 0.001, "batch_size": 32, "hidden_dim": 256},
    {"lr": 0.0001, "batch_size": 64, "hidden_dim": 128},
]
searcher = OptunaSearch(
    points_to_evaluate=initial_params,
    metric="val_loss",
    mode="min",
)
```

**Multi-objective optimization** with Optuna enables Pareto-optimal hyperparameter discovery:

```python
def multi_objective_train(config):
    model = build_model(config)
    for epoch in range(100):
        train(model)
        loss = evaluate_loss(model)
        latency = measure_inference_latency(model)
        tune.report({"loss": loss, "latency": latency})

searcher = OptunaSearch(
    metric=["loss", "latency"],
    mode=["min", "min"],
)

tuner = Tuner(
    multi_objective_train,
    tune_config=TuneConfig(
        search_alg=ConcurrencyLimiter(searcher, max_concurrent=4),
        num_samples=100,
    ),
    param_space=param_space,
)
results = tuner.fit()

# Access Pareto front
pareto_results = [r for r in results if r.metrics.get("pareto_optimal", False)]
```

### Bayesian Optimization (Gaussian Process)

BayesOptSearch uses Gaussian Process regression to model the objective function, providing uncertainty estimates that guide exploration vs. exploitation. Excels with expensive-to-evaluate objectives and continuous search spaces.

```python
from ray.tune.search.bayesopt import BayesOptSearch

# UCB acquisition function (exploration-exploitation balance)
searcher = BayesOptSearch(
    metric="val_loss",
    mode="min",
    utility_kwargs={
        "kind": "ucb",
        "kappa": 2.5,    # Higher = more exploration
        "xi": 0.0,
    },
)

# EI acquisition function (exploitation-focused)
searcher = BayesOptSearch(
    metric="val_loss",
    mode="min",
    utility_kwargs={
        "kind": "ei",
        "xi": 0.01,      # Improvement threshold
    },
)

# POI acquisition function (probability of improvement)
searcher = BayesOptSearch(
    metric="val_loss",
    mode="min",
    utility_kwargs={
        "kind": "poi",
        "xi": 0.01,
    },
)
```

BayesOptSearch requires continuous search spaces. Convert discrete parameters to continuous approximations or use Optuna/HyperOpt for mixed spaces.

**Acquisition function selection guide:**

- **UCB (Upper Confidence Bound)**: Balanced exploration. Increase `kappa` for more exploration early in optimization.
- **EI (Expected Improvement)**: Exploitation-focused. Good when confident about the objective landscape.
- **POI (Probability of Improvement)**: Conservative. Use when small improvements matter.

### HyperOpt (TPE Variant)

HyperOptSearch provides an alternative TPE implementation with native support for conditional search spaces:

```python
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

# HyperOpt native search space syntax
hyperopt_space = {
    "lr": hp.loguniform("lr", -7, -1),  # exp(uniform(-7, -1))
    "optimizer": hp.choice("optimizer", [
        {
            "type": "adam",
            "beta1": hp.uniform("adam_beta1", 0.8, 0.99),
            "beta2": hp.uniform("adam_beta2", 0.9, 0.999),
        },
        {
            "type": "sgd",
            "momentum": hp.uniform("sgd_momentum", 0.0, 0.99),
        },
    ]),
}

searcher = HyperOptSearch(
    space=hyperopt_space,
    metric="val_loss",
    mode="min",
    n_initial_points=20,  # Random exploration before TPE
)
```

Use HyperOpt when conditional parameters require explicit tree structure or when migrating from existing HyperOpt experiments.

## Scheduler Selection Guide

### Scheduler Comparison Matrix

| Scheduler | Early Stopping | Adaptive LR | Checkpointing Required | Best For |
|-----------|---------------|-------------|------------------------|----------|
| FIFO | No | No | No | Debugging, baselines |
| ASHA | Yes | No | No | Fast elimination of bad configs |
| HyperBand | Yes | No | No | Balanced early stopping |
| PBT | Yes | Yes | **Yes** | Long training, adaptive schedules |
| PB2 | Yes | Yes | **Yes** | Improved PBT with Bayesian updates |
| MedianStopping | Yes | No | No | Simple early stopping |
| HyperBandForBOHB | Yes | No | No | Paired with TuneBOHB only |

### ASHA Configuration Deep Dive

ASHA (Asynchronous Successive Halving Algorithm) provides efficient early stopping by progressively filtering trials:

```python
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    time_attr="training_iteration",  # Metric for progress (epoch, step, etc.)
    metric="val_loss",
    mode="min",
    max_t=100,              # Maximum iterations per trial
    grace_period=10,        # Minimum iterations before stopping eligible
    reduction_factor=3,     # Keep top 1/3 at each rung
    brackets=1,             # Single bracket (standard SHA)
)
```

**Parameter tuning guidelines:**

- `grace_period`: Set to the minimum iterations needed for meaningful performance signal. Too low causes premature termination; too high wastes resources.
- `reduction_factor`: Lower values (2) are more aggressive; higher values (4) are conservative. Default of 3 balances speed and thoroughness.
- `max_t`: Total training budget per trial. ASHA is most effective when `max_t / grace_period` is large (10x or more).
- `brackets`: Multiple brackets provide robustness against suboptimal `grace_period` choice at the cost of more total trials.

```python
# Conservative ASHA for expensive evaluations
scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=200,
    grace_period=50,        # Allow substantial training before decisions
    reduction_factor=2,     # Gentle pruning
)

# Aggressive ASHA for cheap evaluations
scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=100,
    grace_period=5,         # Quick initial assessment
    reduction_factor=4,     # Aggressive pruning
)
```

### Population Based Training Configuration

PBT evolves hyperparameters during training by exploiting top performers and exploring around them. Requires checkpoint support in the trainable:

```python
from ray.tune.schedulers import PopulationBasedTraining

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="val_loss",
    mode="min",
    perturbation_interval=10,    # Iterations between exploit/explore cycles
    hyperparam_mutations={
        # Continuous: resample from distribution
        "lr": tune.loguniform(1e-5, 1e-2),
        # Discrete: choose from list
        "batch_size": [16, 32, 64, 128, 256],
        # Perturbation factors (multiply current value)
        "weight_decay": lambda: random.choice([0.5, 1.0, 2.0]),
    },
    quantile_fraction=0.25,      # Bottom 25% replaced by top 25%
    resample_probability=0.25,   # Probability of resampling vs. perturbing
    log_config=True,             # Log hyperparameter changes
)
```

**Trainable checkpoint requirements for PBT:**

```python
from ray import train

def pbt_trainable(config):
    model = create_model(config)
    optimizer = create_optimizer(model, config)

    # Restore from checkpoint if available
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            state = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"]
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config["epochs"]):
        train_epoch(model, optimizer, config)
        val_loss = evaluate(model)

        # Save checkpoint for PBT weight transfer
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, os.path.join(temp_dir, "checkpoint.pt"))
            train.report(
                {"val_loss": val_loss},
                checkpoint=train.Checkpoint.from_directory(temp_dir),
            )
```

**PBT hyperparameter schedule visualization:**

PBT automatically discovers schedules like learning rate warmup and decay. Extract the schedule from results:

```python
results = tuner.fit()

# Analyze hyperparameter evolution for best trial
best_result = results.get_best_result()
for checkpoint_data in best_result.checkpoint_history:
    print(f"Iteration {checkpoint_data['training_iteration']}: lr={checkpoint_data['config']['lr']}")
```

### BOHB Configuration

BOHB combines HyperBand scheduling with Bayesian optimization. Always pair `HyperBandForBOHB` scheduler with `TuneBOHB` search algorithm:

```python
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search import ConcurrencyLimiter

search_alg = TuneBOHB(
    metric="val_loss",
    mode="min",
)
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)

scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=100,
    reduction_factor=4,
    stop_last_trials=False,  # Continue final bracket to completion
)

tuner = Tuner(
    training_function,
    tune_config=TuneConfig(
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=50,
        metric="val_loss",
        mode="min",
    ),
    param_space=param_space,
)
```

## Distributed Tune with Resource Budgets

### Cluster Resource Planning

Calculate resource requirements for Tune experiments:

```python
# Single trial resources
resources_per_trial = {"cpu": 4, "gpu": 1}

# Concurrent trials = cluster_resources / resources_per_trial
# Example: 8 GPUs, 32 CPUs -> 8 concurrent trials (GPU-bound)

tuner = Tuner(
    tune.with_resources(training_function, resources=resources_per_trial),
    tune_config=TuneConfig(
        num_samples=100,
        max_concurrent_trials=8,  # Explicit concurrency limit
    ),
    param_space=param_space,
)
```

### Fractional GPU Allocation

Run multiple trials per GPU for memory-efficient models:

```python
# 4 trials per GPU
tuner = Tuner(
    tune.with_resources(training_function, resources={"cpu": 1, "gpu": 0.25}),
    tune_config=TuneConfig(num_samples=100),
    param_space=param_space,
)
```

Set `CUDA_VISIBLE_DEVICES` appropriately within the trainable to prevent memory conflicts:

```python
def training_function(config):
    # Ray handles GPU assignment; access via CUDA_VISIBLE_DEVICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config).to(device)
    # Training logic...
```

### Placement Groups for Distributed Training

Use placement groups when trials require multiple co-located resources (e.g., data parallel training within each trial):

```python
from ray.tune import PlacementGroupFactory

tuner = Tuner(
    tune.with_resources(
        distributed_training_function,
        resources=PlacementGroupFactory(
            bundles=[
                {"CPU": 1, "GPU": 1},  # Trainer head
                {"CPU": 4, "GPU": 1},  # Worker 1
                {"CPU": 4, "GPU": 1},  # Worker 2
                {"CPU": 4, "GPU": 1},  # Worker 3
            ],
            strategy="PACK",  # Co-locate on same node if possible
        ),
    ),
    tune_config=TuneConfig(num_samples=20),
    param_space=param_space,
)
```

## Integration with Ray Train

### TorchTrainer Integration

Combine Ray Train's distributed training with Ray Tune's hyperparameter optimization:

```python
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

def train_func(config):
    import ray.train as train

    model = create_model(config["hidden_dim"], config["num_layers"])
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_loader = train.torch.prepare_data_loader(create_dataloader(config["batch_size"]))

    for epoch in range(config["epochs"]):
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)

        val_loss = evaluate(model)
        train.report({"val_loss": val_loss, "epoch": epoch})

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"CPU": 2, "GPU": 1},
    ),
)

tuner = Tuner(
    trainer,
    tune_config=TuneConfig(
        metric="val_loss",
        mode="min",
        num_samples=20,
        scheduler=ASHAScheduler(max_t=100, grace_period=10),
    ),
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([32, 64, 128]),
            "hidden_dim": tune.randint(128, 512),
            "num_layers": tune.randint(2, 6),
            "epochs": 100,
        }
    },
)

results = tuner.fit()
```

### Checkpointing with Ray Train

Ray Train handles checkpoint management automatically. Configure checkpoint strategy in `RunConfig`:

```python
from ray.train import RunConfig, CheckpointConfig

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    run_config=RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,                    # Keep best 2 checkpoints
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    ),
)
```

## Custom Stopper and Reporter Patterns

### Custom Stoppers

Implement custom stopping logic by subclassing `tune.stopper.Stopper`:

```python
from ray.tune.stopper import Stopper

class PlateauStopper(Stopper):
    def __init__(self, metric, patience=10, min_delta=0.001):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self._trial_history = {}

    def __call__(self, trial_id, result):
        history = self._trial_history.setdefault(trial_id, [])
        history.append(result[self.metric])

        if len(history) < self.patience:
            return False

        recent = history[-self.patience:]
        improvement = max(recent) - min(recent)
        return improvement < self.min_delta

    def stop_all(self):
        return False  # Don't stop entire experiment

# Use in TuneConfig
tuner = Tuner(
    training_function,
    tune_config=TuneConfig(
        num_samples=50,
        metric="val_loss",
        mode="min",
    ),
    run_config=RunConfig(
        stop=PlateauStopper("val_loss", patience=15, min_delta=0.0001),
    ),
    param_space=param_space,
)
```

### Combined Stopping Conditions

Combine multiple stoppers with `CombinedStopper`:

```python
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TimeoutStopper

stopper = CombinedStopper(
    MaximumIterationStopper(max_iter=100),
    TimeoutStopper(timeout=3600),  # 1 hour timeout
    PlateauStopper("val_loss", patience=20),
)

tuner = Tuner(
    training_function,
    run_config=RunConfig(stop=stopper),
    tune_config=TuneConfig(num_samples=100),
    param_space=param_space,
)
```

### Custom Callbacks

Implement custom experiment-level callbacks:

```python
from ray.tune import Callback

class MetricsLoggerCallback(Callback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_trial_result(self, iteration, trials, trial, result, **info):
        with open(self.log_file, "a") as f:
            f.write(f"{trial.trial_id},{result['val_loss']},{result['training_iteration']}\n")

    def on_trial_complete(self, iteration, trials, trial, **info):
        print(f"Trial {trial.trial_id} completed with status {trial.status}")

tuner = Tuner(
    training_function,
    run_config=RunConfig(
        callbacks=[MetricsLoggerCallback("/path/to/metrics.csv")],
    ),
    tune_config=TuneConfig(num_samples=50),
    param_space=param_space,
)
```

## Experiment Analysis and Visualization

### DataFrame Analysis

Convert results to pandas DataFrame for analysis:

```python
results = tuner.fit()
df = results.get_dataframe()

# Filter successful trials
successful = df[df["status"] == "TERMINATED"]

# Find correlation between hyperparameters and performance
correlations = successful[["config/lr", "config/batch_size", "val_loss"]].corr()

# Group analysis by categorical hyperparameters
grouped = successful.groupby("config/optimizer")["val_loss"].agg(["mean", "std", "min"])
```

### Visualization Patterns

Visualize hyperparameter importance and performance:

```python
import matplotlib.pyplot as plt

df = results.get_dataframe()

# Learning rate vs. performance scatter
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df["config/lr"], df["val_loss"], alpha=0.5)
axes[0].set_xscale("log")
axes[0].set_xlabel("Learning Rate")
axes[0].set_ylabel("Validation Loss")

# Batch size distribution of top trials
top_trials = df.nsmallest(20, "val_loss")
axes[1].hist(top_trials["config/batch_size"], bins=10)
axes[1].set_xlabel("Batch Size")
axes[1].set_ylabel("Count in Top 20")

plt.tight_layout()
plt.savefig("hyperparameter_analysis.png")
```

### TensorBoard Integration

Log Tune experiments to TensorBoard:

```python
from ray.tune.logger import TBXLoggerCallback

tuner = Tuner(
    training_function,
    run_config=RunConfig(
        callbacks=[TBXLoggerCallback()],
        storage_path="/path/to/tune_results",
    ),
    tune_config=TuneConfig(num_samples=50),
    param_space=param_space,
)

# View with: tensorboard --logdir /path/to/tune_results
```

## Checkpoint Management in Tune

### Checkpoint Configuration

Control checkpoint frequency and retention:

```python
from ray.train import RunConfig, CheckpointConfig

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,                        # Keep top 3 checkpoints per trial
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",         # Keep checkpoints with lowest val_loss
        checkpoint_frequency=5,               # Checkpoint every 5 iterations
        checkpoint_at_end=True,               # Always checkpoint on trial completion
    ),
    storage_path="s3://bucket/tune_results",  # Cloud storage for durability
)

tuner = Tuner(
    training_function,
    run_config=run_config,
    tune_config=TuneConfig(num_samples=100),
    param_space=param_space,
)
```

### Resuming Experiments

Resume interrupted experiments from checkpoints:

```python
# Resume from existing experiment
tuner = Tuner.restore(
    path="/path/to/tune_results/experiment_name",
    trainable=training_function,
    resume_unfinished=True,      # Resume incomplete trials
    resume_errored=False,        # Don't retry failed trials
    restart_errored=True,        # Restart failed trials from scratch
)
results = tuner.fit()
```

### Accessing Trial Checkpoints

Load models from best trial checkpoints:

```python
results = tuner.fit()
best_result = results.get_best_result()

# Get the best checkpoint
best_checkpoint = best_result.checkpoint

# Load model from checkpoint
with best_checkpoint.as_directory() as checkpoint_dir:
    model_state = torch.load(os.path.join(checkpoint_dir, "model.pt"))
    model = create_model(best_result.config)
    model.load_state_dict(model_state)

# Deploy or evaluate the model
final_metrics = evaluate_model(model, test_data)
```

### Checkpoint Storage Backends

Configure remote storage for distributed experiments:

```python
# S3 storage
run_config = RunConfig(
    storage_path="s3://my-bucket/tune-experiments",
    name="distributed_experiment",
)

# GCS storage
run_config = RunConfig(
    storage_path="gs://my-bucket/tune-experiments",
    name="distributed_experiment",
)

# NFS or shared filesystem
run_config = RunConfig(
    storage_path="/shared/nfs/tune-experiments",
    name="distributed_experiment",
)
```

Ensure all nodes in the cluster have access to the storage backend. For cloud storage, configure credentials via environment variables or IAM roles.
