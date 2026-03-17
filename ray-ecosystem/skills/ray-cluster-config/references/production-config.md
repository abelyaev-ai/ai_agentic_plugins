# Production Ray Cluster Configuration Reference

This reference provides comprehensive guidance for deploying Ray clusters in production environments, covering Kubernetes deployment with KubeRay, cloud-specific configurations, monitoring, security, and advanced scheduling patterns.

## KubeRay Operator Setup

KubeRay provides Kubernetes-native management of Ray clusters through Custom Resource Definitions (CRDs). The operator handles cluster lifecycle, scaling, and fault tolerance.

### Installing KubeRay Operator

**Helm Installation (Recommended)**

```bash
# Add the KubeRay Helm repository
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install the KubeRay operator
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace ray-system \
  --create-namespace \
  --version 1.1.0

# Verify installation
kubectl get pods -n ray-system
```

**Customized Helm Installation**

```bash
# Install with custom values
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace ray-system \
  --create-namespace \
  --set image.repository=rayproject/kuberay-operator \
  --set image.tag=v1.1.0 \
  --set resources.limits.cpu=500m \
  --set resources.limits.memory=512Mi \
  --set watchNamespace="" \  # Watch all namespaces
  --set batchScheduler.enabled=true
```

**Kubectl Installation**

```bash
# Apply CRDs and operator manifests
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.1.0"
```

### KubeRay Custom Resource Definitions

KubeRay introduces three primary CRDs for Ray cluster management:

#### RayCluster CRD

The RayCluster resource defines a complete Ray cluster deployment:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: production-cluster
  namespace: ray-workloads
spec:
  rayVersion: "2.9.0"
  enableInTreeAutoscaling: true

  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: "0.0.0.0"
      block: "true"
      num-cpus: "0"  # Reserve head for management
    template:
      metadata:
        labels:
          ray.io/cluster: production-cluster
          ray.io/node-type: head
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-py310
            ports:
              - containerPort: 6379
                name: gcs
              - containerPort: 8265
                name: dashboard
              - containerPort: 10001
                name: client
            resources:
              limits:
                cpu: "4"
                memory: "8Gi"
              requests:
                cpu: "2"
                memory: "4Gi"
            volumeMounts:
              - name: ray-logs
                mountPath: /tmp/ray
        volumes:
          - name: ray-logs
            emptyDir: {}

  workerGroupSpecs:
    - groupName: cpu-workers
      replicas: 3
      minReplicas: 1
      maxReplicas: 10
      rayStartParams:
        block: "true"
      template:
        metadata:
          labels:
            ray.io/cluster: production-cluster
            ray.io/node-type: worker
            ray.io/worker-group: cpu-workers
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray:2.9.0-py310
              resources:
                limits:
                  cpu: "8"
                  memory: "16Gi"
                requests:
                  cpu: "4"
                  memory: "8Gi"

    - groupName: gpu-workers
      replicas: 0
      minReplicas: 0
      maxReplicas: 4
      rayStartParams:
        num-gpus: "1"
        block: "true"
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray-ml:2.9.0-py310-gpu
              resources:
                limits:
                  cpu: "8"
                  memory: "32Gi"
                  nvidia.com/gpu: "1"
                requests:
                  cpu: "4"
                  memory: "16Gi"
                  nvidia.com/gpu: "1"
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

#### RayJob CRD

RayJob manages batch job execution on Ray clusters:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: training-job
  namespace: ray-workloads
spec:
  entrypoint: python /home/ray/train.py --epochs 100 --batch-size 32
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 300

  runtimeEnvYAML: |
    pip:
      - torch>=2.0.0
      - transformers>=4.30.0
      - datasets
    env_vars:
      CUDA_VISIBLE_DEVICES: "0"
      TOKENIZERS_PARALLELISM: "false"
    working_dir: "https://github.com/org/repo/archive/main.zip"

  rayClusterSpec:
    rayVersion: "2.9.0"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.9.0-py310
              resources:
                limits:
                  cpu: "2"
                  memory: "4Gi"

    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 2
        minReplicas: 2
        maxReplicas: 2
        rayStartParams:
          num-gpus: "1"
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray-ml:2.9.0-py310-gpu
                resources:
                  limits:
                    cpu: "8"
                    memory: "32Gi"
                    nvidia.com/gpu: "1"
```

#### RayService CRD

RayService manages long-running Ray Serve deployments with automatic updates:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: ml-inference-service
  namespace: ray-workloads
spec:
  serviceUnhealthySecondThreshold: 300
  deploymentUnhealthySecondThreshold: 300

  serveConfigV2: |
    applications:
      - name: text-classification
        route_prefix: /classify
        import_path: serve_app:deployment
        runtime_env:
          pip:
            - transformers
            - torch
        deployments:
          - name: TextClassifier
            num_replicas: 2
            max_replicas_per_node: 1
            ray_actor_options:
              num_cpus: 2
              num_gpus: 0.5

  rayClusterSpec:
    rayVersion: "2.9.0"
    headGroupSpec:
      serviceType: LoadBalancer
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.9.0-py310
              ports:
                - containerPort: 8000
                  name: serve
              resources:
                limits:
                  cpu: "4"
                  memory: "8Gi"

    workerGroupSpecs:
      - groupName: serve-workers
        replicas: 2
        minReplicas: 2
        maxReplicas: 6
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray-ml:2.9.0-py310-gpu
                resources:
                  limits:
                    cpu: "8"
                    memory: "16Gi"
                    nvidia.com/gpu: "1"
```

## Helm Chart Configuration

### RayCluster Helm Chart

Deploy RayCluster using the official Helm chart:

```bash
# Install RayCluster
helm install my-ray-cluster kuberay/ray-cluster \
  --namespace ray-workloads \
  --create-namespace \
  -f values.yaml
```

**values.yaml Example**

```yaml
image:
  repository: rayproject/ray
  tag: 2.9.0-py310

head:
  rayStartParams:
    dashboard-host: "0.0.0.0"
    num-cpus: "0"
  resources:
    limits:
      cpu: "4"
      memory: "8Gi"
    requests:
      cpu: "2"
      memory: "4Gi"
  serviceType: ClusterIP
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"

worker:
  groupName: default-workers
  replicas: 3
  minReplicas: 1
  maxReplicas: 10
  rayStartParams: {}
  resources:
    limits:
      cpu: "8"
      memory: "16Gi"
    requests:
      cpu: "4"
      memory: "8Gi"

additionalWorkerGroups:
  - groupName: gpu-workers
    replicas: 0
    minReplicas: 0
    maxReplicas: 4
    rayStartParams:
      num-gpus: "1"
    resources:
      limits:
        cpu: "8"
        memory: "32Gi"
        nvidia.com/gpu: "1"
    nodeSelector:
      nvidia.com/gpu.present: "true"
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

## Cloud-Specific Deployment

### Amazon Web Services (AWS)

**EKS Cluster Setup for Ray**

```bash
# Create EKS cluster with GPU node group
eksctl create cluster \
  --name ray-cluster \
  --region us-west-2 \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10

# Add GPU node group
eksctl create nodegroup \
  --cluster ray-cluster \
  --name gpu-workers \
  --node-type p3.2xlarge \
  --nodes 0 \
  --nodes-min 0 \
  --nodes-max 4 \
  --node-labels "nvidia.com/gpu=true"
```

**AWS-Specific RayCluster Configuration**

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: aws-ray-cluster
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    template:
      spec:
        serviceAccountName: ray-head-sa
        containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-py310
            env:
              - name: AWS_REGION
                value: us-west-2
            volumeMounts:
              - name: s3-credentials
                mountPath: /root/.aws
                readOnly: true
        volumes:
          - name: s3-credentials
            secret:
              secretName: aws-credentials

  workerGroupSpecs:
    - groupName: spot-workers
      replicas: 5
      template:
        spec:
          nodeSelector:
            eks.amazonaws.com/capacityType: SPOT
          tolerations:
            - key: "eks.amazonaws.com/capacityType"
              operator: "Equal"
              value: "SPOT"
              effect: "NoSchedule"
```

### Google Cloud Platform (GCP)

**GKE Cluster Setup**

```bash
# Create GKE cluster with autoscaling
gcloud container clusters create ray-cluster \
  --zone us-central1-a \
  --machine-type n2-standard-8 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster ray-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 4
```

**GCS Integration for Checkpointing**

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: gcp-ray-cluster
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    template:
      spec:
        serviceAccountName: ray-gcs-sa
        containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-py310
            env:
              - name: RAY_external_storage_namespace
                value: ray-checkpoints
        nodeSelector:
          cloud.google.com/gke-nodepool: cpu-pool
```

### Microsoft Azure

**AKS Cluster Setup**

```bash
# Create AKS cluster
az aks create \
  --resource-group ray-rg \
  --name ray-cluster \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Add GPU node pool
az aks nodepool add \
  --resource-group ray-rg \
  --cluster-name ray-cluster \
  --name gpupool \
  --node-count 0 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 4
```

## Monitoring with Prometheus and Grafana

### Ray Metrics Export Configuration

Enable Prometheus metrics export in RayCluster:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: monitored-cluster
  annotations:
    ray.io/enable-metrics: "true"
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    rayStartParams:
      metrics-export-port: "8080"
    template:
      metadata:
        annotations:
          prometheus.io/scrape: "true"
          prometheus.io/port: "8080"
          prometheus.io/path: "/metrics"
      spec:
        containers:
          - name: ray-head
            ports:
              - containerPort: 8080
                name: metrics
```

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ray-cluster-monitor
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      ray.io/cluster: monitored-cluster
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
  namespaceSelector:
    matchNames:
      - ray-workloads
```

### Grafana Dashboard Configuration

Import the Ray dashboard or configure custom panels:

```json
{
  "dashboard": {
    "title": "Ray Cluster Metrics",
    "panels": [
      {
        "title": "Active Tasks",
        "targets": [
          {
            "expr": "sum(ray_tasks{State=\"RUNNING\"})",
            "legendFormat": "Running Tasks"
          }
        ]
      },
      {
        "title": "Object Store Memory",
        "targets": [
          {
            "expr": "ray_object_store_memory_used_bytes / ray_object_store_memory_total_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "CPU Utilization by Node",
        "targets": [
          {
            "expr": "ray_node_cpu_utilization",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "ray_node_gpu_utilization",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      }
    ]
  }
}
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `ray_object_store_memory_used_bytes` | Object store memory consumption | > 80% capacity |
| `ray_tasks{State="PENDING"}` | Tasks waiting for resources | > 1000 for > 5 min |
| `ray_node_cpu_utilization` | CPU usage per node | > 90% sustained |
| `ray_node_gpu_utilization` | GPU usage per node | < 50% (underutilization) |
| `ray_gcs_update_resource_usage_time_ms` | GCS latency | > 100ms |

## Logging Configuration

### Persistent Log Collection

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: logged-cluster
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    template:
      spec:
        containers:
          - name: ray-head
            volumeMounts:
              - name: ray-logs
                mountPath: /tmp/ray
          - name: fluentbit
            image: fluent/fluent-bit:latest
            volumeMounts:
              - name: ray-logs
                mountPath: /tmp/ray
              - name: fluentbit-config
                mountPath: /fluent-bit/etc/
        volumes:
          - name: ray-logs
            emptyDir: {}
          - name: fluentbit-config
            configMap:
              name: fluentbit-config
```

**Fluent Bit Configuration**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentbit-config
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info

    [INPUT]
        Name          tail
        Path          /tmp/ray/session_*/logs/*.log
        Tag           ray.*

    [OUTPUT]
        Name          es
        Match         ray.*
        Host          elasticsearch.monitoring
        Port          9200
        Index         ray-logs
```

## Security Configuration

### TLS Authentication

Enable TLS for secure cluster communication:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ray-tls-secret
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
  ca.crt: <base64-encoded-ca>
---
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: secure-cluster
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    rayStartParams:
      object-manager-port: "8076"
      node-manager-port: "8077"
    template:
      spec:
        containers:
          - name: ray-head
            env:
              - name: RAY_USE_TLS
                value: "1"
              - name: RAY_TLS_SERVER_CERT
                value: /etc/ray/tls/tls.crt
              - name: RAY_TLS_SERVER_KEY
                value: /etc/ray/tls/tls.key
              - name: RAY_TLS_CA_CERT
                value: /etc/ray/tls/ca.crt
            volumeMounts:
              - name: tls-certs
                mountPath: /etc/ray/tls
                readOnly: true
        volumes:
          - name: tls-certs
            secret:
              secretName: ray-tls-secret
```

### RBAC Configuration

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ray-operator-sa
  namespace: ray-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ray-operator-role
rules:
  - apiGroups: ["ray.io"]
    resources: ["rayclusters", "rayjobs", "rayservices"]
    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "secrets"]
    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["create", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ray-operator-binding
subjects:
  - kind: ServiceAccount
    name: ray-operator-sa
    namespace: ray-system
roleRef:
  kind: ClusterRole
  name: ray-operator-role
  apiGroup: rbac.authorization.k8s.io
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ray-cluster-policy
  namespace: ray-workloads
spec:
  podSelector:
    matchLabels:
      ray.io/cluster: production-cluster
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              ray.io/cluster: production-cluster
      ports:
        - port: 6379  # GCS
        - port: 8265  # Dashboard
        - port: 10001 # Client
        - port: 8076  # Object manager
        - port: 8077  # Node manager
  egress:
    - to:
        - podSelector:
            matchLabels:
              ray.io/cluster: production-cluster
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              app: kube-dns
      ports:
        - port: 53
          protocol: UDP
```

## Resource Quotas and Limits

### Namespace Resource Quota

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ray-workloads-quota
  namespace: ray-workloads
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    limits.cpu: "200"
    limits.memory: "400Gi"
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    pods: "100"
```

### LimitRange for Default Resources

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ray-limit-range
  namespace: ray-workloads
spec:
  limits:
    - type: Container
      default:
        cpu: "2"
        memory: "4Gi"
      defaultRequest:
        cpu: "1"
        memory: "2Gi"
      max:
        cpu: "16"
        memory: "64Gi"
        nvidia.com/gpu: "4"
```

## Node Affinity and GPU Scheduling

### Node Affinity Configuration

```yaml
workerGroupSpecs:
  - groupName: high-memory-workers
    template:
      spec:
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: node-type
                      operator: In
                      values:
                        - high-memory
            preferredDuringSchedulingIgnoredDuringExecution:
              - weight: 100
                preference:
                  matchExpressions:
                    - key: topology.kubernetes.io/zone
                      operator: In
                      values:
                        - us-west-2a
```

### GPU Scheduling Patterns

**Dedicated GPU Nodes**

```yaml
workerGroupSpecs:
  - groupName: gpu-workers
    template:
      spec:
        nodeSelector:
          nvidia.com/gpu.product: Tesla-V100-SXM2-16GB
        tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
        containers:
          - name: ray-worker
            resources:
              limits:
                nvidia.com/gpu: "1"
```

**GPU Sharing with MIG**

```yaml
workerGroupSpecs:
  - groupName: mig-workers
    template:
      spec:
        nodeSelector:
          nvidia.com/mig.strategy: single
        containers:
          - name: ray-worker
            resources:
              limits:
                nvidia.com/mig-1g.5gb: "1"
```

**Fractional GPU Allocation**

```python
# In Ray code, request fractional GPU
@ray.remote(num_gpus=0.25)
class InferenceWorker:
    def __init__(self):
        import torch
        self.device = torch.device("cuda")

    def predict(self, data):
        # Uses 1/4 of a GPU
        pass
```

### Pod Priority and Preemption

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ray-high-priority
value: 1000000
globalDefault: false
description: "High priority for critical Ray workloads"
---
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: priority-cluster
spec:
  headGroupSpec:
    template:
      spec:
        priorityClassName: ray-high-priority
```

## GCS Fault Tolerance

### External Redis for GCS Persistence

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ha-cluster
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    rayStartParams:
      redis-password: $REDIS_PASSWORD
    template:
      spec:
        containers:
          - name: ray-head
            env:
              - name: RAY_REDIS_ADDRESS
                value: redis-master.redis:6379
              - name: REDIS_PASSWORD
                valueFrom:
                  secretKeyRef:
                    name: redis-secret
                    key: password
```

### Redis Tuning for Production

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    maxmemory 4gb
    maxmemory-policy allkeys-lru
    appendonly yes
    appendfsync everysec
    tcp-keepalive 300
    timeout 0
```

This reference provides the foundation for deploying production-grade Ray clusters. Adapt configurations based on specific workload requirements, compliance needs, and infrastructure constraints.
