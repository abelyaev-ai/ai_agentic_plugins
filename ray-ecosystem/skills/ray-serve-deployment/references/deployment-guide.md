# Ray Serve Deployment Reference Guide

This reference guide provides comprehensive documentation for advanced Ray Serve deployment patterns, production configurations, and operational best practices.

## Deployment Graph Patterns

### Sequential Pipeline Pattern

Chain multiple deployments for multi-stage inference pipelines where each stage processes the output of the previous stage:

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class TextPreprocessor:
    def process(self, text: str) -> dict:
        # Tokenization, normalization, cleaning
        tokens = text.lower().split()
        return {"tokens": tokens, "length": len(tokens)}

@serve.deployment
class FeatureExtractor:
    def __init__(self):
        self.embedder = self._load_embedder()

    def _load_embedder(self):
        # Load embedding model
        return None

    def extract(self, data: dict) -> dict:
        # Generate embeddings from tokens
        embeddings = [0.1] * 768  # Placeholder
        return {"embeddings": embeddings, "metadata": data}

@serve.deployment
class Classifier:
    def __init__(self):
        self.model = self._load_classifier()

    def _load_classifier(self):
        return None

    def classify(self, data: dict) -> dict:
        # Run classification on embeddings
        return {"label": "positive", "confidence": 0.92}

@serve.deployment
class SequentialPipeline:
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        extractor: DeploymentHandle,
        classifier: DeploymentHandle
    ):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.classifier = classifier

    async def __call__(self, request):
        text = (await request.json())["text"]

        # Sequential processing
        preprocessed = await self.preprocessor.process.remote(text)
        features = await self.extractor.extract.remote(preprocessed)
        result = await self.classifier.classify.remote(features)

        return result

# Build and deploy the graph
preprocessor = TextPreprocessor.bind()
extractor = FeatureExtractor.bind()
classifier = Classifier.bind()
pipeline = SequentialPipeline.bind(preprocessor, extractor, classifier)

serve.run(pipeline)
```

### Ensemble Pattern

Combine predictions from multiple models for improved accuracy:

```python
from ray import serve
from ray.serve.handle import DeploymentHandle
import asyncio

@serve.deployment
class ModelA:
    def predict(self, data: dict) -> dict:
        return {"prediction": 0.7, "model": "A"}

@serve.deployment
class ModelB:
    def predict(self, data: dict) -> dict:
        return {"prediction": 0.8, "model": "B"}

@serve.deployment
class ModelC:
    def predict(self, data: dict) -> dict:
        return {"prediction": 0.75, "model": "C"}

@serve.deployment
class EnsembleAggregator:
    def __init__(
        self,
        model_a: DeploymentHandle,
        model_b: DeploymentHandle,
        model_c: DeploymentHandle,
        weights: list = None
    ):
        self.models = [model_a, model_b, model_c]
        self.weights = weights or [0.33, 0.34, 0.33]

    async def __call__(self, request):
        data = await request.json()

        # Run all models in parallel
        tasks = [model.predict.remote(data) for model in self.models]
        results = await asyncio.gather(*tasks)

        # Weighted average
        weighted_sum = sum(
            r["prediction"] * w for r, w in zip(results, self.weights)
        )

        return {
            "ensemble_prediction": weighted_sum,
            "individual_predictions": results
        }

# Deploy ensemble
model_a = ModelA.bind()
model_b = ModelB.bind()
model_c = ModelC.bind()
ensemble = EnsembleAggregator.bind(model_a, model_b, model_c)

serve.run(ensemble)
```

### Conditional Routing Pattern

Route requests to different models based on input characteristics:

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class LightweightModel:
    def predict(self, data: dict) -> dict:
        return {"result": "light", "latency_ms": 10}

@serve.deployment
class HeavyModel:
    def predict(self, data: dict) -> dict:
        return {"result": "heavy", "latency_ms": 100}

@serve.deployment
class ConditionalRouter:
    def __init__(
        self,
        light_model: DeploymentHandle,
        heavy_model: DeploymentHandle,
        complexity_threshold: int = 100
    ):
        self.light_model = light_model
        self.heavy_model = heavy_model
        self.threshold = complexity_threshold

    async def __call__(self, request):
        data = await request.json()

        # Route based on input complexity
        complexity = self._compute_complexity(data)

        if complexity < self.threshold:
            return await self.light_model.predict.remote(data)
        else:
            return await self.heavy_model.predict.remote(data)

    def _compute_complexity(self, data: dict) -> int:
        # Compute input complexity score
        return len(str(data))

light = LightweightModel.bind()
heavy = HeavyModel.bind()
router = ConditionalRouter.bind(light, heavy, complexity_threshold=100)

serve.run(router)
```

## Model Multiplexing

Serve multiple model variants from a single deployment using the `@serve.multiplexed` decorator. This pattern is efficient for multi-tenant serving or A/B testing with many model versions:

```python
from ray import serve

@serve.deployment
class MultiplexedModelServer:
    def __init__(self):
        self.model_cache = {}

    @serve.multiplexed(max_num_models_per_replica=10)
    async def get_model(self, model_id: str):
        """Load and cache model by ID."""
        if model_id not in self.model_cache:
            # Load model on demand
            self.model_cache[model_id] = self._load_model(model_id)
        return self.model_cache[model_id]

    def _load_model(self, model_id: str):
        # Load model from storage
        print(f"Loading model: {model_id}")
        return f"Model_{model_id}"

    async def __call__(self, request):
        data = await request.json()
        model_id = serve.get_multiplexed_model_id()

        model = await self.get_model(model_id)
        result = self._run_inference(model, data)

        return {"model_id": model_id, "result": result}

    def _run_inference(self, model, data):
        return f"Prediction from {model}"

# Deploy
server = MultiplexedModelServer.bind()
serve.run(server)
```

Client-side usage with model ID in headers:

```python
import requests

response = requests.post(
    "http://localhost:8000/",
    json={"input": "data"},
    headers={"serve_multiplexed_model_id": "model_v2"}
)
```

## Dynamic Request Batching with @serve.batch

The `@serve.batch` decorator automatically batches incoming requests for improved throughput, especially beneficial for GPU inference:

### Basic Batching

```python
from typing import List
from ray import serve
import numpy as np

@serve.deployment
class BatchedInference:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        # Load model optimized for batch inference
        return None

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def predict_batch(self, inputs: List[np.ndarray]) -> List[dict]:
        """
        Process a batch of inputs.

        Args:
            inputs: List of individual inputs, automatically batched

        Returns:
            List of results, one per input
        """
        # Stack inputs into batch tensor
        batch = np.stack(inputs)

        # Run batch inference
        batch_results = self._run_batch_inference(batch)

        # Return list of individual results
        return [{"prediction": r} for r in batch_results]

    def _run_batch_inference(self, batch: np.ndarray) -> np.ndarray:
        # Vectorized inference
        return batch.mean(axis=1)

    async def __call__(self, request):
        data = await request.json()
        input_array = np.array(data["input"])
        return await self.predict_batch(input_array)

batched = BatchedInference.bind()
serve.run(batched)
```

### Streaming Batched Responses

Combine batching with streaming for real-time token generation in LLM serving:

```python
from typing import List, AsyncGenerator, Union
from ray import serve
import asyncio

@serve.deployment
class StreamingBatchedLLM:
    def __init__(self):
        self.model = self._load_llm()

    def _load_llm(self):
        return None

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.5)
    async def generate_stream(
        self, prompts: List[str]
    ) -> AsyncGenerator[List[Union[str, StopIteration]], None]:
        """
        Generate tokens for a batch of prompts.

        Yields a list of tokens (or StopIteration for completed sequences)
        for each iteration.
        """
        max_tokens = 100
        active_sequences = [True] * len(prompts)

        for token_idx in range(max_tokens):
            outputs = []
            for i, (prompt, active) in enumerate(zip(prompts, active_sequences)):
                if not active:
                    outputs.append(StopIteration)
                else:
                    token = self._generate_next_token(prompt, token_idx)
                    if token == "<EOS>":
                        active_sequences[i] = False
                        outputs.append(StopIteration)
                    else:
                        outputs.append(token)

            yield outputs

            # Check if all sequences are done
            if not any(active_sequences):
                break

            await asyncio.sleep(0.01)  # Simulate generation time

    def _generate_next_token(self, prompt: str, idx: int) -> str:
        # Simulate token generation
        if idx > 10:
            return "<EOS>"
        return f"token_{idx}"

    async def __call__(self, request):
        from starlette.responses import StreamingResponse

        prompt = (await request.json())["prompt"]

        async def token_stream():
            async for token in self.generate_stream(prompt):
                yield token + " "

        return StreamingResponse(token_stream(), media_type="text/plain")

llm = StreamingBatchedLLM.bind()
serve.run(llm)
```

### Batching Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_batch_size` | Maximum requests per batch | 10 |
| `batch_wait_timeout_s` | Maximum wait time to fill batch | 0.0 |

Tune these based on:
- Model batch inference latency
- Acceptable request latency
- Request arrival rate
- GPU memory capacity

## Streaming Responses

Implement streaming responses for long-running inference or token-by-token LLM generation:

```python
from ray import serve
from starlette.responses import StreamingResponse
import asyncio

@serve.deployment
class StreamingEndpoint:
    async def generate_tokens(self, prompt: str):
        """Async generator yielding tokens."""
        words = prompt.split()
        for i, word in enumerate(words):
            processed = f"[{i}] {word.upper()} "
            yield processed
            await asyncio.sleep(0.1)  # Simulate processing time

    async def __call__(self, request):
        data = await request.json()
        prompt = data.get("prompt", "")

        return StreamingResponse(
            self.generate_tokens(prompt),
            media_type="text/plain"
        )

streaming = StreamingEndpoint.bind()
serve.run(streaming)
```

Client-side consumption:

```python
import requests

response = requests.post(
    "http://localhost:8000/",
    json={"prompt": "hello world from ray serve"},
    stream=True
)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end="", flush=True)
```

## Production Deployment with KubeRay

### RayService Custom Resource

Deploy Ray Serve applications on Kubernetes using the RayService CRD:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: ml-serving-service
  namespace: ray-system
spec:
  serviceUnhealthySecondThreshold: 900
  deploymentUnhealthySecondThreshold: 300
  serveConfigV2: |
    applications:
      - name: ml-app
        import_path: serve_app:deployment
        route_prefix: /
        runtime_env:
          pip:
            - torch>=2.0.0
            - transformers>=4.30.0
        deployments:
          - name: ModelDeployment
            num_replicas: 2
            ray_actor_options:
              num_cpus: 1
              num_gpus: 1
            autoscaling_config:
              min_replicas: 1
              max_replicas: 10
              target_ongoing_requests: 5
  rayClusterConfig:
    rayVersion: '2.9.0'
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray-ml:2.9.0-py310-gpu
              resources:
                limits:
                  cpu: "4"
                  memory: "8Gi"
                requests:
                  cpu: "2"
                  memory: "4Gi"
              ports:
                - containerPort: 6379
                  name: gcs
                - containerPort: 8265
                  name: dashboard
                - containerPort: 8000
                  name: serve
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 2
        minReplicas: 1
        maxReplicas: 5
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray-ml:2.9.0-py310-gpu
                resources:
                  limits:
                    cpu: "4"
                    memory: "16Gi"
                    nvidia.com/gpu: "1"
                  requests:
                    cpu: "2"
                    memory: "8Gi"
                    nvidia.com/gpu: "1"
```

### Serve Config File Format

For complex deployments, use a YAML configuration file:

```yaml
# serve_config.yaml
proxy_location: EveryNode
http_options:
  host: 0.0.0.0
  port: 8000

applications:
  - name: prediction-service
    import_path: app.main:app
    route_prefix: /api/v1
    runtime_env:
      working_dir: ./
      pip:
        - torch==2.0.0
        - numpy>=1.24.0
    deployments:
      - name: Preprocessor
        num_replicas: 2
        max_ongoing_requests: 100
        ray_actor_options:
          num_cpus: 1

      - name: Model
        num_replicas: 4
        ray_actor_options:
          num_cpus: 2
          num_gpus: 0.5
        autoscaling_config:
          min_replicas: 2
          max_replicas: 10
          target_ongoing_requests: 3
          upscale_delay_s: 5
          downscale_delay_s: 30

      - name: Postprocessor
        num_replicas: 2
        ray_actor_options:
          num_cpus: 1
```

Deploy using the CLI:

```bash
serve deploy serve_config.yaml
```

## Health Checks and Liveness Probes

### Application-Level Health Checks

Implement health check endpoints for load balancers and Kubernetes probes:

```python
from fastapi import FastAPI
from ray import serve
import time

app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class HealthAwareService:
    def __init__(self):
        self.model = self._load_model()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

    def _load_model(self):
        return None

    @app.get("/health")
    async def health_check(self):
        """Basic health check."""
        return {"status": "healthy"}

    @app.get("/health/live")
    async def liveness_check(self):
        """Kubernetes liveness probe."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness_check(self):
        """Kubernetes readiness probe."""
        if self.model is None:
            return {"status": "not ready", "reason": "model not loaded"}, 503
        return {"status": "ready"}

    @app.get("/health/detailed")
    async def detailed_health(self):
        """Detailed health metrics."""
        uptime = time.time() - self.start_time
        error_rate = self.error_count / max(self.request_count, 1)

        return {
            "status": "healthy" if error_rate < 0.1 else "degraded",
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate
        }

    @app.post("/predict")
    async def predict(self, item: dict):
        self.request_count += 1
        try:
            result = self._run_inference(item)
            return {"prediction": result}
        except Exception as e:
            self.error_count += 1
            raise

    def _run_inference(self, item):
        return "prediction"

service = HealthAwareService.bind()
serve.run(service)
```

### Deployment-Level Health Checks

Configure health check parameters in the deployment:

```python
@serve.deployment(
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class HealthCheckedDeployment:
    def __init__(self):
        self.healthy = True

    def check_health(self):
        """Custom health check method called by Ray Serve."""
        if not self.healthy:
            raise RuntimeError("Deployment unhealthy")
        return True
```

## Logging and Metrics

### Structured Logging

Implement structured logging for observability:

```python
from ray import serve
import logging
import json
import time
import uuid

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("ray.serve")

@serve.deployment
class LoggingService:
    def __init__(self):
        self.deployment_id = str(uuid.uuid4())[:8]

    def _log_structured(self, level: str, message: str, **kwargs):
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "deployment_id": self.deployment_id,
            "message": message,
            **kwargs
        }
        logger.info(json.dumps(log_entry))

    async def __call__(self, request):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        self._log_structured(
            "INFO",
            "Request received",
            request_id=request_id,
            path=request.url.path
        )

        try:
            data = await request.json()
            result = self._process(data)

            latency_ms = (time.time() - start_time) * 1000
            self._log_structured(
                "INFO",
                "Request completed",
                request_id=request_id,
                latency_ms=latency_ms,
                status="success"
            )

            return result
        except Exception as e:
            self._log_structured(
                "ERROR",
                "Request failed",
                request_id=request_id,
                error=str(e)
            )
            raise

    def _process(self, data):
        return {"result": "processed"}
```

### Custom Metrics with Ray Metrics

Export custom metrics to Prometheus:

```python
from ray import serve
from ray.util.metrics import Counter, Histogram, Gauge

@serve.deployment
class MetricsService:
    def __init__(self):
        # Define custom metrics
        self.request_counter = Counter(
            "serve_requests_total",
            description="Total number of requests",
            tag_keys=("deployment", "status")
        )
        self.latency_histogram = Histogram(
            "serve_request_latency_seconds",
            description="Request latency in seconds",
            boundaries=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            tag_keys=("deployment",)
        )
        self.active_requests_gauge = Gauge(
            "serve_active_requests",
            description="Number of active requests",
            tag_keys=("deployment",)
        )

    async def __call__(self, request):
        import time

        self.active_requests_gauge.set(1, tags={"deployment": "MetricsService"})
        start_time = time.time()

        try:
            result = await self._handle_request(request)
            self.request_counter.inc(tags={
                "deployment": "MetricsService",
                "status": "success"
            })
            return result
        except Exception as e:
            self.request_counter.inc(tags={
                "deployment": "MetricsService",
                "status": "error"
            })
            raise
        finally:
            latency = time.time() - start_time
            self.latency_histogram.observe(
                latency,
                tags={"deployment": "MetricsService"}
            )
            self.active_requests_gauge.set(0, tags={"deployment": "MetricsService"})

    async def _handle_request(self, request):
        return {"status": "ok"}
```

## gRPC Service Configuration

### Define gRPC Service

Create gRPC services for high-performance binary communication:

```python
# user_service.proto
"""
syntax = "proto3";

service UserService {
    rpc Predict (PredictRequest) returns (PredictResponse);
    rpc StreamPredict (PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
    string input = 1;
    int32 batch_size = 2;
}

message PredictResponse {
    string output = 1;
    float confidence = 2;
}
"""

# After generating Python files with protoc:
from ray import serve
from typing import Generator

# Import generated protobuf classes
# from user_service_pb2 import PredictRequest, PredictResponse

@serve.deployment
class GRPCModelService:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        return None

    def __call__(self, request) -> dict:
        """Handle unary gRPC requests."""
        # request is the deserialized protobuf message
        result = self._run_inference(request.input)
        return {
            "output": result,
            "confidence": 0.95
        }

    def StreamPredict(self, request) -> Generator:
        """Handle streaming gRPC requests."""
        for i in range(10):
            yield {
                "output": f"token_{i}",
                "confidence": 0.9 + i * 0.01
            }

    def _run_inference(self, input_text: str) -> str:
        return f"processed: {input_text}"

grpc_service = GRPCModelService.bind()
```

### gRPC Server Configuration

Configure the gRPC proxy in the serve config:

```yaml
grpc_options:
  port: 9000
  grpc_servicer_functions:
    - "user_service_pb2_grpc.add_UserServiceServicer_to_server"

applications:
  - name: grpc-app
    import_path: grpc_service:grpc_service
    route_prefix: /
```

## Model Warm-up Patterns

### Initialization Warm-up

Warm up models during deployment initialization:

```python
from ray import serve
import numpy as np

@serve.deployment
class WarmupDeployment:
    def __init__(self, warmup_samples: int = 10):
        self.model = self._load_model()
        self._warmup(warmup_samples)

    def _load_model(self):
        # Load model
        return None

    def _warmup(self, num_samples: int):
        """Run warmup inference to initialize CUDA kernels and caches."""
        print(f"Running {num_samples} warmup iterations...")

        for i in range(num_samples):
            # Create dummy input matching production shape
            dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
            _ = self._run_inference(dummy_input)

        print("Warmup complete")

    def _run_inference(self, input_data):
        # Run inference
        return {"result": "prediction"}

    async def __call__(self, request):
        data = await request.json()
        return self._run_inference(data["input"])

deployment = WarmupDeployment.bind(warmup_samples=10)
serve.run(deployment)
```

### Reconfigurable Warm-up

Use the `reconfigure` method for dynamic warm-up after configuration updates:

```python
from ray import serve

@serve.deployment
class ReconfigurableDeployment:
    def __init__(self):
        self.model = None
        self.config = {}

    def reconfigure(self, config: dict):
        """Called when deployment configuration changes."""
        self.config = config

        # Reload model if path changed
        if "model_path" in config:
            self.model = self._load_model(config["model_path"])
            self._warmup()

    def _load_model(self, path: str):
        print(f"Loading model from {path}")
        return None

    def _warmup(self):
        """Warmup after model reload."""
        print("Running post-reconfigure warmup...")
        # Warmup logic
        print("Warmup complete")

    async def __call__(self, request):
        if self.model is None:
            return {"error": "Model not loaded"}, 503
        return {"prediction": "result"}
```

### Graceful Shutdown

Handle graceful shutdown for cleanup operations:

```python
from ray import serve
import signal

@serve.deployment(
    graceful_shutdown_timeout_s=30,
    graceful_shutdown_wait_loop_s=2,
)
class GracefulDeployment:
    def __init__(self):
        self.model = self._load_model()
        self.active_requests = 0

    def _load_model(self):
        return None

    async def __call__(self, request):
        self.active_requests += 1
        try:
            result = await self._process(request)
            return result
        finally:
            self.active_requests -= 1

    async def _process(self, request):
        return {"result": "processed"}

    def __del__(self):
        """Cleanup on shutdown."""
        print(f"Shutting down with {self.active_requests} active requests")
        # Perform cleanup: close connections, flush buffers, etc.

deployment = GracefulDeployment.bind()
serve.run(deployment)
```

## Resource Allocation Best Practices

### Fractional GPU Allocation

Share GPUs across multiple replicas for smaller models:

```python
@serve.deployment(
    num_replicas=4,
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0.25,  # 4 replicas share 1 GPU
    }
)
class FractionalGPUDeployment:
    pass
```

### Memory Management

Configure memory limits to prevent OOM:

```python
@serve.deployment(
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 1,
        "memory": 8 * 1024 * 1024 * 1024,  # 8 GB
    }
)
class MemoryManagedDeployment:
    pass
```

### Custom Resources

Use custom resources for specialized hardware:

```python
@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "resources": {"TPU": 1, "special_hardware": 0.5}
    }
)
class CustomResourceDeployment:
    pass
```
