# VisionPilot Technical Report

## OpenCV AI Competition 2026 Entry

---

## 1. Problem Statement

**Problem:** Real-time visual monitoring systems often lack the ability to adapt their processing pipeline based on what they actually observe. Traditional systems run fixed pipelines regardless of image quality, detection confidence, or scene complexity, leading to:
- Wasted compute on low-quality inputs
- Missed critical detections when confidence is low but not flagged
- No mechanism for human-in-the-loop when automated analysis is uncertain
- Inability to provide audit trails for decisions

**Target Users:**
- Security operations centers requiring auditability
- Industrial inspection systems needing adaptive quality control
- Autonomous systems requiring explainable vision decisions
- Researchers building trustworthy computer vision pipelines

**Value Proposition:** VisionPilot implements a genuine agentic vision loop where OpenCV 5 perception results directly drive subsequent decisions, tool selection, and human approval requests — creating a trustworthy, auditable computer vision system.

---

## 2. Technical Approach

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VISIONPILOT ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────┐  │
│  │  INPUT   │───▶│  VALIDATION  │───▶│    OPENCV 5   │───▶│  DECISION  │  │
│  │  IMAGE   │    │  & QUALITY   │    │  PERCEPTION   │    │  ENGINE    │  │
│  └──────────┘    └──────────────┘    └───────────────┘    └──────┬─────┘  │
│         │                              │                        │        │
│         │                              ▼                        ▼        │
│         │                     ┌────────────────┐     ┌────────────────┐  │
│         │                     │  DETECTIONS    │     │   AGENT STATE  │  │
│         │                     │  • Contours    │     │  • IDLE        │  │
│         │                     │  • Colors      │     │  • ANALYZING   │  │
│         │                     │  • Synthetic*  │     │  • DECIDING    │  │
│         │                     └────────────────┘     │  • APPROVAL    │  │
│         │                              │            │  • COMPLETED   │  │
│         │                              ▼            └────────┬─────────┘  │
│         │                     ┌────────────────┐            │            │
│         │                     │  MEASUREMENTS  │            │            │
│         │                     │  • Dimensions  │            ▼            │
│         │                     │  • Distances   │     ┌────────────────┐  │
│         │                     └────────────────┘     │  TOOL EXECUTION│  │
│         │                                             │  • STORE       │  │
│         │                              │            │  • ALERT       │  │
│         │                              ▼            │  • APPROVAL    │  │
│         │                     ┌────────────────┐    │  • RETRY       │  │
│         │                     │  ANNOTATED IMG │    └───────┬────────┘  │
│         │                     └────────────────┘            │            │
│         │                                                   ▼            │
│         │                                         ┌────────────────┐    │
│         └────────────────────────────────────────▶│  STRUCTURED    │    │
│                                                   │  TRACE LOG     │    │
│                                                   └────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 OpenCV 5 Implementation

**Core Computer Vision Operations:**

1. **Image Validation & Quality Assessment**
   - Laplacian variance for blur detection (OpenCV `cv2.Laplacian`)
   - Brightness analysis using channel means
   - Minimum dimension enforcement
   - Decoding verification with `cv2.imdecode`

2. **Object Detection (Multi-method)**
   - **Edge Detection + Contours**: Canny edges (`cv2.Canny`) + `findContours` with area filtering
   - **Color Segmentation**: HSV conversion (`cv2.cvtColor`) + `inRange` masking + morphology
   - **Synthetic Fallback**: For demo purposes when real detection yields no results

3. **Geometric Measurements**
   - Bounding box extraction (`cv2.boundingRect`)
   - Centroid calculation from moments
   - Euclidean distance between detection centers
   - Real-world scale estimation (when reference provided)

4. **Visualization**
   - Bounding box drawing (`cv2.rectangle`)
   - Label rendering (`cv2.putText`)
   - Center point marking (`cv2.circle`)
   - Base64 encoding for API response

**OpenCV 5 Specific Features Used:**
- Version 5.0.0 verified at runtime
- Optimized NEON/ARM64 dispatch on Apple Silicon
- DNN module available for future model integration
- Modern Python bindings with type hints

### 2.3 Agent Orchestration

**Finite State Machine:**

| State | Description | Transitions |
|-------|-------------|-------------|
| IDLE | Waiting for input | → RECEIVED |
| RECEIVED | Input accepted | → VALIDATING |
| VALIDATING | Quality checks | → ANALYZING / REJECTED |
| ANALYZING | OpenCV perception | → INTERPRETING |
| INTERPRETING | Result analysis | → DECIDING |
| DECIDING | Rule evaluation | → COMPLETED / NEEDS_APPROVAL / REJECTED / RESTARTED |
| NEEDS_APPROVAL | Awaiting human | → COMPLETED / REJECTED |
| COMPLETED | Success | → IDLE |
| REJECTED | Quality/confidence failure | → IDLE |
| FAILED | System error | → IDLE |

**Decision Rules (Priority Order):**

1. **Reject Invalid** (Priority 100): Validation failure → REJECTED
2. **Reject Blurry** (Priority 95): Quality < 0.3 → REJECTED
3. **No Detections** (Priority 90): Zero detections → NEEDS_APPROVAL (if quality good) or COMPLETED (empty scene)
4. **Low Confidence** (Priority 80): Max confidence < 0.9 → NEEDS_APPROVAL
5. **Good Detections** (Priority 70): All confidences > 0.3 → COMPLETED
6. **Mixed Quality** (Priority 60): Some low confidence → COMPLETED with approval flag
7. **Crowded Scene** (Priority 50): >3 detections → COMPLETED with density metric

**Available Tools:**
- `STORE_RESULT`: Persist successful analysis
- `REQUEST_HUMAN_APPROVAL`: Escalate uncertain decisions
- `SEND_ALERT`: Notify operators of events
- `UPLOAD_TO_S3`: Cloud storage integration
- `RETRY_IMAGE`: Reprocess with adjusted parameters

### 2.4 AWS Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS CLOUD                                │
│  ┌─────────────┐                                                │
│  │   USERS     │                                                │
│  └──────┬──────┘                                                │
│         │ HTTPS                                                  │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │    ALB      │  Application Load Balancer                      │
│  │  (Port 80)  │  Health checks on /health                       │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────┐               │
│  │           ECS FARGATE SERVICE                │               │
│  │  ┌─────────────────────────────────────┐   │               │
│  │  │      VISIONPILOT CONTAINER          │   │               │
│  │  │  • FastAPI + Uvicorn                │   │               │
│  │  │  • OpenCV 5 + NumPy                 │   │               │
│  │  │  • Agent Orchestrator               │   │               │
│  │  │  • 1 vCPU, 2 GB RAM                 │   │               │
│  │  └─────────────────────────────────────┘   │               │
│  └─────────────────────────────────────────────┘               │
│         │                                                        │
│         ├──▶ CloudWatch Logs                                     │
│         ├──▶ CloudWatch Metrics                                  │
│         └──▶ S3 (optional uploads)                               │
└─────────────────────────────────────────────────────────────────┘
```

**Infrastructure as Code:** CloudFormation template provides:
- VPC with public subnets across 2 AZs
- ALB with health checks
- ECS Fargate service with auto-scaling
- IAM roles with least-privilege S3 access
- CloudWatch logging and monitoring

---

## 3. Evaluation

### 3.1 Test Results

| Test Category | Tests | Passed | Failed |
|---------------|-------|--------|--------|
| Vision Pipeline | 9 | 9 | 0 |
| Agentic Vision | 2 | 2 | 0 |
| **Total** | **11** | **11** | **0** |

### 3.2 Key Metrics

| Metric | Value | Method |
|--------|-------|--------|
| OpenCV Version | 5.0.0 | Runtime verification |
| Pipeline Latency | ~60-100ms | Measured per request |
| Quality Detection Accuracy | 100% | Unit tests on synthetic images |
| Agent Decision Variance | Verified | Different inputs → different outcomes |
| Trace Completeness | 100% | Every request produces structured traces |

### 3.3 Behavioral Verification

**Requirement:** *Agent must react differently to changing visual evidence*

**Verification:**
```python
# Good quality image
good_result = orchestrator.process_image(good_image, 'good.jpg')
# Decision: COMPLETED, Outcome: success

# Blurry image
blur_result = orchestrator.process_image(blurry_image, 'blur.jpg')
# Decision: REJECTED, Outcome: rejected

# Dark image
dark_result = orchestrator.process_image(dark_image, 'dark.jpg')
# Decision: REJECTED (blurry), Outcome: rejected

# Different outcomes confirmed ✓
assert good_result['decision']['state'] != blur_result['decision']['state']
```

### 3.4 Failure Cases Documented

1. **Invalid image data**: Graceful rejection with error message
2. **Sub-minimum dimensions**: Rejected with quality=POOR
3. **Severe blur**: Detected via Laplacian variance < 50
4. **Extreme darkness**: Brightness < 0.2 normalized → quality=DARK
5. **Low detection confidence**: Triggers human approval workflow
6. **No detections in good image**: Completes with "empty scene" note

---

## 4. Limitations & Known Issues

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No pre-trained Haar cascades in opencv-python | Person/vehicle detection uses contours/colors | Document synthetic fallback; integrate DNN models for production |
| Synthetic detections for demo | Not real object detection | Clearly labeled; replace with YOLO/MobileNet for production |
| Single-threaded processing | Limited throughput | Deploy multiple ECS tasks behind ALB |
| In-memory result storage | Not persistent | Add DynamoDB/RDS for production |
| No authentication on API | Security risk for public deploy | Add API Gateway + Cognito for production |

---

## 5. Responsible Use Considerations

- **No facial recognition**: System detects generic contours/colors, not identities
- **No biometric data**: No PII extracted from images
- **Human-in-the-loop**: Low-confidence results require approval
- **Audit trail**: Every decision logged with reasoning
- **Data retention**: Configurable; default in-memory only
- **No surveillance claims**: Demo system for competition purposes

---

## 6. Reproducibility

### Local Development
```bash
# Clone and setup
git clone <repo>
cd VisionPilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
PYTHONPATH=src pytest src/visionpilot/tests/ -v

# Run server
PYTHONPATH=src python -m visionpilot.main
```

### Docker
```bash
docker build -t visionpilot .
docker run -p 8000:8000 visionpilot
```

### Docker Compose
```bash
docker-compose up -d
```

### AWS Deployment
```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t visionpilot .
docker tag visionpilot:latest <account>.dkr.ecr.us-east-1.amazonaws.com/visionpilot:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/visionpilot:latest

# Deploy CloudFormation
aws cloudformation deploy --template-file deploy/aws/cloudformation.yaml \
  --stack-name visionpilot --parameter-overrides ImageUri=<account>.dkr.ecr.us-east-1.amazonaws.com/visionpilot:latest
```

---

## 7. Future Work

- [ ] Integrate YOLOv8/YOLO11 for real object detection
- [ ] Add COOL (Cloud-Optimized OpenCV Library) benchmarking on Graviton
- [ ] Implement video stream processing with temporal analysis
- [ ] Add Prometheus metrics endpoint
- [ ] Build React frontend for visualization
- [ ] Add WebSocket support for real-time results
- [ ] Implement model versioning and A/B testing