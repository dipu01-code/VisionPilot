# VisionPilot

**An Agentic Computer Vision System for the OpenCV AI Competition 2026**

VisionPilot implements a genuine **perception-decision-action loop** where OpenCV 5 analysis results directly drive subsequent agent decisions, tool selection, and human approval requests — creating a trustworthy, auditable computer vision system.

---

## 🎯 Competition Targets

| Award | Status | Evidence |
|-------|--------|----------|
| **Agentic Vision Award** (Primary) | ✅ Implemented | Finite state machine, decision engine, structured traces, visual evidence changes decisions |
| **Overall Competition** | ✅ Implemented | Technical execution, innovation, real-world impact, UX, documentation, cloud delivery |
| **Best Use of COOL** | 🔄 Optional | Architecture prepared for Graviton benchmarking |

---

## ✨ Key Features

### Agentic Vision Loop
- **OpenCV 5 Perception**: Image validation, quality assessment, multi-method detection (contours + color segmentation), geometric measurements
- **Deterministic Decision Engine**: 7 priority rules evaluating visual evidence → actions
- **Finite State Machine**: 9 states (IDLE → RECEIVED → VALIDATING → ANALYZING → DECIDING → COMPLETED/NEEDS_APPROVAL/REJECTED)
- **Tool System**: STORE_RESULT, REQUEST_HUMAN_APPROVAL, SEND_ALERT, UPLOAD_TO_S3, RETRY_IMAGE
- **Structured Traces**: Every decision logged with reasoning, confidence, and tool results

### Computer Vision (OpenCV 5)
- ✅ OpenCV 5.0.0 verified at runtime
- ✅ Image validation & quality scoring (Laplacian variance + brightness)
- ✅ Edge detection + contour analysis (`cv2.Canny`, `findContours`)
- ✅ Color segmentation in HSV space (`cv2.inRange`, morphology)
- ✅ Geometric measurements (bounding boxes, centroids, distances)
- ✅ Annotated output images (base64 encoded)

### API & Interface
- **FastAPI** server with automatic OpenAPI docs at `/docs`
- **REST endpoints**: `/analyze` (base64), `/analyze/upload` (multipart), `/health`, `/traces`, `/approve`
- **CORS enabled** for web frontend integration
- **In-memory result storage** (replaceable with DB)

### AWS Deployment Ready
- **Docker** multi-stage build (builder → runtime)
- **docker-compose** for local development
- **ECS Fargate** CloudFormation template
- **ALB** with health checks
- **IAM least-privilege** roles
- **CloudWatch** logging & metrics

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Local Development
```bash
# Clone
git clone <repo-url>
cd VisionPilot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
PYTHONPATH=src pytest src/visionpilot/tests/ -v

# Start API server
PYTHONPATH=src python -m visionpilot.main
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Docker
```bash
# Build
docker build -t visionpilot .

# Run
docker run -p 8000:8000 visionpilot

# Or with docker-compose
docker-compose up -d
```

### Run Demo
```bash
PYTHONPATH=src python demo.py
```

---

## 📡 API Usage

### Analyze Image (Base64)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "/9j/4AAQSkZJRgABAQ...",
    "filename": "test.jpg"
  }'
```

### Upload Image File
```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F "file=@image.jpg"
```

### Health Check
```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"0.1.0","timestamp":"...","opencv_version":"5.0.0"}
```

### Get Traces
```bash
curl http://localhost:8000/traces?count=10
```

---

## 🧪 Example Output

```json
{
  "outcome": "success",
  "decision": {
    "state": "completed",
    "reason": "Good detections found. Avg confidence: 0.82",
    "confidence": 0.82,
    "tool": "store_result",
    "needs_human_approval": false
  },
  "response": {
    "validation": {"is_valid": true, "quality": "good", "quality_score": 0.67},
    "detections": [
      {"label": "contour", "confidence": 0.90, "bounding_box": [260, 180, 121, 121], "center": [320.5, 240.5], "area_pixels": 14641},
      {"label": "color_blue", "confidence": 0.85, "bounding_box": [199, 49, 84, 74], "center": [241.0, 86.0], "area_pixels": 6216}
    ],
    "measurements": [...],
    "annoted_image_base64": "iVBORw0KGgoAAAANSUhEUg..."
  },
  "traces": [...],
  "processing_time_ms": 62.7
}
```

---

## 📁 Project Structure

```
VisionPilot/
├── src/visionpilot/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── core/
│   │   └── pipeline.py         # OpenCV 5 vision pipeline
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── types.py            # Agent state, decisions, traces
│   │   ├── decision_engine.py  # Deterministic decision rules
│   │   └── orchestrator.py     # Perception-decision-action loop
│   ├── api/
│   │   ├── server.py           # FastAPI endpoints
│   │   └── schemas.py          # Pydantic models
│   └── tests/
│       └── test_pipeline.py    # Unit + agentic tests
├── deploy/aws/
│   ├── README.md
│   ├── cloudformation.yaml     # Full AWS infrastructure
│   └── ecs-task-definition.json
├── demo.py                     # Demonstration script
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # Local development
├── requirements.txt            # Pinned dependencies
├── TECHNICAL_REPORT.md         # Competition technical report
├── ARCHITECTURE.md             # Architecture diagrams
└── README.md                   # This file
```

---

## 🏗️ Architecture

```
Input Image
     │
     ▼
┌────────────────┐
│  VALIDATION    │ ── Invalid? ──▶ REJECTED
│  (quality,     │
│   dimensions)  │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  OPENCV 5      │
│  PERCEPTION    │
│  • Contours    │
│  • Colors      │
│  • Measurements│
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  DECISION      │
│  ENGINE        │
│  (7 rules,     │
│   priority)    │
└───────┬────────┘
        │
    ┌───┴───┐
    ▼       ▼
COMPLETED  NEEDS_APPROVAL
    │           │
    ▼           ▼
TOOL:      TOOL:
STORE      REQUEST_
RESULT     APPROVAL
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams.

---

## 🧪 Testing

```bash
# All tests
PYTHONPATH=src pytest src/visionpilot/tests/ -v

# Specific test
PYTHONPATH=src pytest src/visionpilot/tests/test_pipeline.py::TestAgenticVision::test_agent_makes_different_decisions_for_different_images -v
```

**Test Results:** 11/11 passing
- 9 vision pipeline tests
- 2 agentic vision tests (proves visual evidence changes decisions)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Full competition technical report |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture diagrams & component details |
| [deploy/aws/README.md](deploy/aws/README.md) | AWS deployment guide |
| API Docs | `/docs` when server running |

---

## ⚠️ Limitations & Future Work

| Limitation | Status |
|------------|--------|
| No pre-trained Haar cascades in opencv-python | Uses contour/color detection; DNN integration planned |
| Synthetic fallback detections for demo | Clearly documented; replace with YOLO for production |
| In-memory storage | Add DynamoDB/RDS for persistence |
| No authentication | Add API Gateway + Cognito for production |
| Single-threaded | Scale with multiple ECS tasks |

---

## 🔐 Responsible Use

- **No facial recognition** — detects generic contours/colors only
- **No biometric data** — no PII extracted
- **Human-in-the-loop** — low-confidence results require approval
- **Audit trail** — every decision logged with reasoning
- **Data retention** — configurable, default in-memory only

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Team

VisionPilot Team — OpenCV AI Competition 2026

---

**Built with OpenCV 5, FastAPI, and AWS** 🚀