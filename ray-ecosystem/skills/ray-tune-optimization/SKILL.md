---
name: ray-tune-optimization
description: >
  This skill should be used when the user asks to "tune hyperparameters with Ray",
  "configure a search algorithm", "set up a Tune scheduler",
  or works with Ray Tune experiments.
---

## Purpose

Provide patterns and guidance for hyperparameter optimization using Ray Tune. This skill covers the complete workflow from defining search spaces and configuring search algorithms to running distributed experiments and analyzing results. Apply these patterns when building scalable hyperparameter tuning pipelines that leverage Ray's distributed execution capabilities.

## Prerequisites

Before implementing any Ray Tune solution, resolve the relevant documentation via Context7:

1. Call `resolve-library-id` with `libraryName: "ray"` to obtain the Context7-compatible library ID.
2. Call `query-docs` with the resolved library ID and a query targeting Tune-specific APIs (e.g., "Tuner TuneConfig search algorithms schedulers").
3. Ground all recommendations in the latest API surface rather than memorized patterns.

## Core Workflow

### Tuner and TuneConfig Setup

The `Tuner` class serves as the primary entry point for hyperparameter optimization experiments. Configure experiments through the `TuneConfig` object, which specifies the search algorithm, scheduler, metric to optimize, and number of trials.

```python
from ray import tune
from ray.tune import Tuner, TuneConfig

def training_function(config):
    # Training logic using config hyperparameters
    for epoch in range(config["epochs"]):
        loss = train_epoch(config["lr"], config["batch_size"])
        tune.report({"loss": loss, "epoch": epoch})

tuner = Tuner(
    training_function,
    tune_config=TuneConfig(
        metric="loss",
        mode="min",
        num_samples=100,
        search_alg=search_algorithm,
        scheduler=scheduler,
    ),
    param_space=search_space,
    run_config=tune.RunConfig(
        name="experiment_name",
        storage_path="/path/to/results",
    ),
)

results = tuner.fit()
```

The `run_config` parameter accepts a `RunConfig` object for specifying experiment name, storage location, checkpoint configuration, and failure handling policies. Always set a meaningful experiment name and persistent storage path for production experiments.

### Search Space Definition

Define hyperparameter search spaces using Ray Tune's sampling primitives. Each primitive generates values according to a specific distribution or selection strategy.

**Continuous Distributions:**

```python
param_space = {
    # Uniform distribution between 0.0001 and 0.1
    "lr": tune.uniform(0.0001, 0.1),

    # Log-uniform distribution (better for learning rates)
    "lr_log": tune.loguniform(1e-5, 1e-1),

    # Normal distribution with mean=0.5, std=0.1
    "dropout": tune.normal(0.5, 0.1),

    # Quantized uniform (steps of 0.1)
    "momentum": tune.quniform(0.1, 0.9, 0.1),
}
```

**Discrete Distributions:**

```python
param_space = {
    # Random integer between 16 and 256
    "batch_size": tune.randint(16, 256),

    # Categorical choice from list
    "activation": tune.choice(["relu", "tanh", "leaky_relu"]),

    # Grid search over explicit values
    "hidden_units": tune.grid_search([64, 128, 256, 512]),

    # Quantized log-uniform for exponential scales
    "num_layers": tune.qlograndint(1, 8, 1),
}
```

**Nested and Conditional Spaces:**

```python
param_space = {
    "optimizer": tune.choice(["adam", "sgd"]),
    "adam_config": {
        "beta1": tune.uniform(0.8, 0.99),
        "beta2": tune.uniform(0.9, 0.999),
    },
    "sgd_config": {
        "momentum": tune.uniform(0.0, 0.99),
        "nesterov": tune.choice([True, False]),
    },
}
```

Use `tune.sample_from()` for complex dependent sampling where one hyperparameter depends on another:

```python
param_space = {
    "num_layers": tune.randint(1, 5),
    "hidden_per_layer": tune.sample_from(
        lambda spec: [tune.randint(32, 256).sample() for _ in range(spec.config.num_layers)]
    ),
}
```

### Search Algorithms

Search algorithms determine how Ray Tune explores the hyperparameter space. Select algorithms based on the search space characteristics and computational budget.

**Optuna Integration:**

```python
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

searcher = OptunaSearch(
    metric="loss",
    mode="min",
    points_to_evaluate=[{"lr": 0.001, "batch_size": 32}],  # Initial guesses
)
# Limit concurrent trials for Bayesian methods
search_alg = ConcurrencyLimiter(searcher, max_concurrent=4)
```

Optuna provides Tree-structured Parzen Estimator (TPE) by default, effective for most hyperparameter optimization tasks. Support multi-objective optimization by passing lists to `metric` and `mode`:

```python
searcher = OptunaSearch(
    metric=["loss", "accuracy"],
    mode=["min", "max"],
)
```

**BayesOpt Integration:**

```python
from ray.tune.search.bayesopt import BayesOptSearch

searcher = BayesOptSearch(
    metric="loss",
    mode="min",
    utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
)
```

BayesOptSearch uses Gaussian Process regression. Configure the acquisition function through `utility_kwargs`: use "ucb" (Upper Confidence Bound) for exploration-exploitation balance, "ei" (Expected Improvement) for exploitation-focused search.

**HyperOpt Integration:**

```python
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

# HyperOpt native search space (alternative to Tune primitives)
hyperopt_space = {
    "lr": hp.loguniform("lr", -5, -1),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
}

searcher = HyperOptSearch(
    space=hyperopt_space,
    metric="loss",
    mode="min",
    n_initial_points=20,
)
```

### Schedulers

Schedulers control trial execution by implementing early stopping and resource allocation strategies. Combine schedulers with search algorithms for efficient optimization.

**ASHA (Asynchronous Successive Halving Algorithm):**

```python
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=100,           # Maximum training iterations
    grace_period=10,     # Minimum iterations before stopping
    reduction_factor=2,  # Halving factor for successive rounds
    brackets=1,          # Number of brackets (1 for standard SHA)
)
```

ASHA aggressively stops underperforming trials early, allocating resources to promising configurations. Set `grace_period` high enough to allow meaningful performance differentiation.

**Population Based Training (PBT):**

```python
from ray.tune.schedulers import PopulationBasedTraining

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="loss",
    mode="min",
    perturbation_interval=5,  # Iterations between perturbations
    hyperparam_mutations={
        "lr": tune.uniform(0.0001, 0.1),
        "batch_size": [16, 32, 64, 128],
    },
    quantile_fraction=0.25,   # Bottom 25% replaced
    resample_probability=0.25,
)
```

PBT evolves hyperparameters during training by copying weights from top performers to underperformers and perturbing their hyperparameters. Requires the trainable to support checkpointing. Effective for long-running training where hyperparameters benefit from adaptation (e.g., learning rate schedules).

**BOHB (Bayesian Optimization HyperBand):**

```python
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

# BOHB requires paired scheduler and search algorithm
search_alg = TuneBOHB(metric="loss", mode="min")
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)

scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=100,
    reduction_factor=4,
    stop_last_trials=False,
)
```

BOHB combines HyperBand scheduling with Bayesian optimization for sample-efficient search. Always use `HyperBandForBOHB` scheduler with `TuneBOHB` search algorithm together.

**Median Stopping Rule:**

```python
from ray.tune.schedulers import MedianStoppingRule

scheduler = MedianStoppingRule(
    metric="loss",
    mode="min",
    grace_period=10,
    min_samples_required=3,
    hard_stop=True,
)
```

Simple early stopping: terminate trials performing below the median of all trials at the same iteration. Use for quick exploration when sophisticated scheduling is unnecessary.

### Metric Reporting with tune.report()

Report metrics from the training function to enable scheduler decisions and result tracking:

```python
def training_function(config):
    model = create_model(config)

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, config)
        val_loss, val_acc = evaluate(model)

        # Report metrics for this iteration
        tune.report({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": val_acc,
            "epoch": epoch,
        })
```

Report all metrics of interest, not just the optimization target. Secondary metrics aid in result analysis and debugging. The metric specified in `TuneConfig` determines optimization direction; other metrics are tracked for analysis.

### Result Analysis with ResultGrid

After experiment completion, analyze results through the `ResultGrid` object:

```python
results = tuner.fit()

# Get the best result based on configured metric
best_result = results.get_best_result(metric="loss", mode="min")
print(f"Best config: {best_result.config}")
print(f"Best loss: {best_result.metrics['loss']}")

# Access the best checkpoint
best_checkpoint = best_result.checkpoint
model = load_model_from_checkpoint(best_checkpoint)

# Convert all results to DataFrame for analysis
results_df = results.get_dataframe()
print(results_df[["config/lr", "config/batch_size", "loss", "accuracy"]])

# Iterate over all trial results
for result in results:
    print(f"Trial {result.metrics['trial_id']}: loss={result.metrics['loss']}")
```

Filter and sort results programmatically:

```python
# Get top 5 results
top_results = results.get_dataframe().nsmallest(5, "loss")

# Filter by configuration criteria
filtered = results_df[results_df["config/batch_size"] >= 64]
```

### Resource Specification per Trial

Control CPU, GPU, and custom resource allocation for each trial:

```python
tuner = Tuner(
    tune.with_resources(
        training_function,
        resources={"cpu": 2, "gpu": 0.5},  # Fractional GPU for multiple trials per GPU
    ),
    tune_config=TuneConfig(num_samples=50),
    param_space=param_space,
)
```

For trainables requiring variable resources based on configuration:

```python
def resource_request(config):
    return {"cpu": config.get("num_workers", 1) + 1, "gpu": 1}

tuner = Tuner(
    tune.with_resources(training_function, resources=resource_request),
    param_space={"num_workers": tune.choice([1, 2, 4])},
)
```

Specify placement groups for co-located resources:

```python
from ray.tune import PlacementGroupFactory

tuner = Tuner(
    tune.with_resources(
        training_function,
        resources=PlacementGroupFactory([
            {"CPU": 1, "GPU": 1},  # Head bundle
            {"CPU": 2},            # Worker bundles
            {"CPU": 2},
        ]),
    ),
)
```

## Common Pitfalls

- **Mismatched scheduler and search algorithm**: BOHB requires `HyperBandForBOHB` scheduler paired with `TuneBOHB` search algorithm. Using incompatible combinations causes silent failures or suboptimal behavior.

- **Missing ConcurrencyLimiter for Bayesian methods**: Bayesian optimization algorithms (Optuna, BayesOpt, HyperOpt) perform poorly with high concurrency because they need completed trials to update their models. Wrap with `ConcurrencyLimiter(searcher, max_concurrent=4)` for effective sequential suggestions.

- **Insufficient grace_period for ASHA**: Setting `grace_period` too low terminates trials before meaningful performance differentiation, leading to random selection. Allow enough iterations for the training signal to stabilize.

- **Forgetting checkpoints for PBT**: Population Based Training copies model weights between trials. Without proper checkpoint implementation in the trainable, PBT degrades to random perturbation without weight transfer.

- **Overlooking tune.report() frequency**: Schedulers make decisions based on reported metrics. Infrequent reporting delays early stopping decisions and wastes resources on underperforming trials. Report at least once per epoch or training iteration.

## Additional Resources

Consult `references/search-schedulers.md` for detailed algorithm comparison tables, scheduler selection flowcharts, distributed Tune configuration with resource budgets, Ray Train integration patterns, custom stopper implementations, experiment visualization techniques, and checkpoint management strategies.
