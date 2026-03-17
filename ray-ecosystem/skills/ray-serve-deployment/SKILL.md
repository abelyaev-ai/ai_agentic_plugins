---
name: ray-serve-deployment
description: >
  This skill should be used when the user asks to "deploy a model with Ray Serve",
  "set up HTTP endpoints", "configure A/B testing", "scale a serving deployment",
  or works with Ray Serve.
---

## Purpose

Provide patterns, best practices, and implementation guidance for deploying and scaling model serving infrastructure with Ray Serve. This skill covers the full lifecycle of serving machine learning models in production, including deployment creation, endpoint configuration, request handling, model composition, autoscaling, and traffic management.

## Prerequisites

Before implementing Ray Serve deployments, resolve up-to-date documentation using context7:

1. Use `resolve-library-id` with query "ray" to obtain the Context7-compatible library ID
2. Use `query-docs` with the resolved library ID to fetch current Ray Serve API documentation
3. Verify the Ray Serve version matches the target deployment environment

Required packages:
- `ray[serve]` - Core Ray Serve functionality
- `fastapi` - For HTTP endpoint integration (optional but recommended)
- `starlette` - Request/response handling (included with FastAPI)

## Core Workflow

### Deployment Creation with @serve.deployment

Create deployments using the `@serve.deployment` decorator. Each deployment represents a scalable unit that handles requests independently.

```python
from ray import serve

@serve.deployment
class ModelDeployment:
    def __init__(self, model_path: str):
        # Load model during initialization
        self.model = self._load_model(model_path)

    def _load_model(self, path: str):
        # Model loading logic
        import torch
        return torch.load(path)

    async def __call__(self, request):
        # Handle incoming requests
        data = await request.json()
        result = self.model.predict(data["input"])
        return {"prediction": result}
```

Configure deployment options for resource allocation and scaling:

```python
@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.5},
    max_ongoing_requests=100,
)
class ConfiguredDeployment:
    pass
```

### Binding Deployments

Bind deployments with constructor arguments using the `.bind()` method. This creates a deployable application handle.

```python
# Bind with initialization arguments
deployment = ModelDeployment.bind(model_path="/models/classifier.pt")

# Run the deployment
serve.run(deployment)
```

For multiple deployments, bind them together to create a deployment graph:

```python
preprocessor = Preprocessor.bind()
model = Model.bind()
postprocessor = Postprocessor.bind(model)

# Create pipeline
app = Pipeline.bind(preprocessor, model, postprocessor)
serve.run(app)
```

### HTTP Endpoint Configuration

Configure HTTP routes using the `route_prefix` parameter:

```python
@serve.deployment(route_prefix="/api/v1/predict")
class PredictionEndpoint:
    async def __call__(self, request):
        return {"status": "ok"}
```

For more sophisticated routing, integrate with FastAPI:

```python
from fastapi import FastAPI
from ray import serve

app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class APIGateway:
    def __init__(self, model_deployment):
        self.model = model_deployment

    @app.get("/health")
    async def health_check(self):
        return {"status": "healthy"}

    @app.post("/predict")
    async def predict(self, item: dict):
        result = await self.model.predict.remote(item)
        return {"prediction": result}
```

### Request Handling: Synchronous and Asynchronous

Ray Serve supports both synchronous and asynchronous request handling. Prefer async handlers for I/O-bound operations:

```python
@serve.deployment
class AsyncHandler:
    async def __call__(self, request):
        # Async handling for non-blocking I/O
        data = await request.json()
        result = await self._async_inference(data)
        return result

    async def _async_inference(self, data):
        # Async model inference or external API calls
        import asyncio
        await asyncio.sleep(0.01)  # Simulate async work
        return {"result": data}
```

For CPU-bound synchronous operations:

```python
@serve.deployment
class SyncHandler:
    def __call__(self, request):
        # Synchronous handling for CPU-bound work
        data = request.json()
        return self._compute_heavy_inference(data)
```

### Model Composition with Deployment Graphs

Build complex inference pipelines by composing multiple deployments:

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Preprocessor:
    def preprocess(self, data: dict) -> dict:
        # Normalize, tokenize, transform
        return {"processed": data["raw"]}

@serve.deployment
class Classifier:
    def __init__(self):
        self.model = self._load_model()

    def classify(self, data: dict) -> dict:
        return {"class": "positive", "confidence": 0.95}

@serve.deployment
class Pipeline:
    def __init__(self, preprocessor: DeploymentHandle, classifier: DeploymentHandle):
        self.preprocessor = preprocessor
        self.classifier = classifier

    async def __call__(self, request):
        data = await request.json()
        processed = await self.preprocessor.preprocess.remote(data)
        result = await self.classifier.classify.remote(processed)
        return result

# Bind the deployment graph
preprocessor = Preprocessor.bind()
classifier = Classifier.bind()
pipeline = Pipeline.bind(preprocessor, classifier)

serve.run(pipeline)
```

### Scaling Configuration

Configure static replica count or dynamic autoscaling:

**Static scaling:**

```python
@serve.deployment(num_replicas=4)
class StaticScaled:
    pass
```

**Autoscaling configuration:**

```python
from ray.serve.config import AutoscalingConfig

@serve.deployment(
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=10,
        target_ongoing_requests=5,
        upscale_delay_s=10,
        downscale_delay_s=30,
    )
)
class AutoscaledDeployment:
    pass
```

Key autoscaling parameters:
- `min_replicas`: Minimum number of replicas (set to 0 for scale-to-zero)
- `max_replicas`: Maximum number of replicas
- `target_ongoing_requests`: Target concurrent requests per replica
- `upscale_delay_s`: Delay before scaling up
- `downscale_delay_s`: Delay before scaling down
- `metrics_interval_s`: How often to check scaling metrics

### Traffic Splitting for A/B Testing

Implement traffic splitting using deployment handles with weighted routing:

```python
from ray import serve
import random

@serve.deployment
class ModelV1:
    def predict(self, data):
        return {"version": "v1", "result": "prediction_v1"}

@serve.deployment
class ModelV2:
    def predict(self, data):
        return {"version": "v2", "result": "prediction_v2"}

@serve.deployment
class TrafficRouter:
    def __init__(self, model_v1, model_v2, v2_traffic_ratio: float = 0.1):
        self.model_v1 = model_v1
        self.model_v2 = model_v2
        self.v2_ratio = v2_traffic_ratio

    async def __call__(self, request):
        data = await request.json()
        if random.random() < self.v2_ratio:
            return await self.model_v2.predict.remote(data)
        return await self.model_v1.predict.remote(data)

# Deploy with 10% traffic to v2
model_v1 = ModelV1.bind()
model_v2 = ModelV2.bind()
router = TrafficRouter.bind(model_v1, model_v2, v2_traffic_ratio=0.1)

serve.run(router)
```

### FastAPI Integration

Leverage FastAPI for type validation, automatic documentation, and middleware support:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve

app = FastAPI(
    title="ML Model API",
    description="Production ML serving endpoint",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str
    options: dict = {}

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class MLService:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        # Load model
        return None

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        result = self._run_inference(request.text)
        return PredictionResponse(
            prediction=result["label"],
            confidence=result["score"]
        )

    @app.get("/health")
    async def health(self):
        return {"status": "healthy"}

    def _run_inference(self, text: str) -> dict:
        return {"label": "positive", "score": 0.95}

# Deploy
service = MLService.bind()
serve.run(service)
```

## Common Pitfalls

- **Blocking the event loop**: Avoid synchronous blocking calls inside async handlers. Use `asyncio.to_thread()` or `loop.run_in_executor()` for CPU-bound operations within async contexts.

- **Memory leaks from model reloading**: Load models once in `__init__`, not per-request. Store models as instance attributes to ensure they persist across requests and are not garbage collected prematurely.

- **Incorrect autoscaling configuration**: Setting `target_ongoing_requests` too low causes excessive scaling; too high causes request queuing. Profile actual workload latency to determine optimal values.

- **Missing health checks in production**: Always implement health check endpoints for load balancers and orchestrators. Use the `@app.get("/health")` pattern with FastAPI integration.

- **Forgetting to handle deployment handle exceptions**: Wrap `await handle.method.remote()` calls in try-except blocks to handle replica failures gracefully. Implement retry logic for transient failures.

## Additional Resources

For detailed reference documentation including deployment graphs, batching patterns, streaming responses, and production deployment with KubeRay, see `references/deployment-guide.md`.

Key topics covered in the reference guide:
- Advanced deployment graph patterns
- Model multiplexing for multi-tenant serving
- Dynamic request batching with `@serve.batch`
- Streaming response implementation
- KubeRay production deployment
- Health checks and liveness probes
- Logging and metrics collection
- gRPC service configuration
- Model warm-up patterns
