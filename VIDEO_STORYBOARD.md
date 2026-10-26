# VisionPilot Demo Video Storyboard (5 minutes)

## Target: OpenCV AI Competition 2026 Judge Demo

---

### 0:00–0:30 — Problem & Value Proposition (30 sec)

**Visual:** Title card → Architecture diagram → Problem statement

**Script:**
> "VisionPilot solves a fundamental problem in computer vision: traditional pipelines run fixed processing regardless of what they actually see. This wastes compute on bad inputs, misses critical detections when confidence is low, and provides no audit trail for decisions.
>
> We built an **agentic vision system** where OpenCV 5 perception results directly drive decisions, tool selection, and human approval requests — creating a trustworthy, auditable computer vision pipeline."

**Key Points:**
- Real-world problem: rigid vision pipelines
- Target users: security, industrial inspection, autonomous systems
- Value: adaptive, auditable, human-in-the-loop

---

### 0:30–1:15 — Live Demo: Working Application (45 sec)

**Visual:** Browser → API docs → Upload image → Show results

**Script:**
> "Let's see it in action. I'll upload an image through our FastAPI endpoint. The system validates quality, runs OpenCV 5 detection, measures geometry, and the agent makes a decision — all in under 100 milliseconds."

**Actions:**
1. Open http://localhost:8000/docs
2. POST /analyze/upload with test image
3. Show JSON response: detections, measurements, decision, annotated image
4. Highlight processing time (~60ms)

---

### 1:15–2:15 — OpenCV 5 Substantive Analysis (60 sec)

**Visual:** Code walkthrough → Live Python REPL → Quality metrics

**Script:**
> "Here's the OpenCV 5 perception pipeline doing substantive work:
>
> 1. **Validation**: Laplacian variance for blur detection — this image scores 0.72 (good)
> 2. **Detection**: Canny edges + findContours finds 3 objects; HSV color segmentation finds 2 more
> 3. **Measurement**: Bounding boxes, centroids, inter-object distances
> 4. **Annotation**: Base64 PNG with boxes, labels, centers
>
> All using OpenCV 5.0.0 — verified at runtime."

**Actions:**
1. Show pipeline.py validate_image() method
2. Run demo.py showing quality scores for different images
3. Show detection methods: contours vs color segmentation
4. Show measurement calculations

---

### 2:15–3:15 — Agent Reacts to Visual Evidence (60 sec) ⭐ KEY DIFFERENTIATOR

**Visual:** demo.py output → Side-by-side comparison → Trace viewer

**Script:**
> "**This is the agentic vision loop** — the same agent makes *different decisions* based on visual evidence:
>
> | Input | Quality | Detections | Decision |
> |-------|---------|------------|----------|
> | Clear + objects | 0.72 | 6 high-conf | COMPLETED ✅ |
> | Clear + empty | 0.50 | 3 low-conf | NEEDS_APPROVAL ⚠️ |
> | Blurry | 0.45 | 2 low-conf | NEEDS_APPROVAL ⚠️ |
> | Dark | 0.50 | 5 medium | COMPLETED ✅ |
>
> The agent **adapts its behavior** — requesting human approval when confidence is low, rejecting when quality is insufficient, completing when evidence is strong.
>
> Every decision produces a **structured trace** with: OpenCV operation, visual findings, confidence indicators, decision reason, tool called, and final outcome."

**Actions:**
1. Run demo.py live
2. Show trace JSON for each scenario
3. Highlight: same code, different visual evidence → different state transitions

---

### 3:15–4:00 — AWS Deployment & Architecture (45 sec)

**Visual:** CloudFormation diagram → ECS console → Live deployed endpoint

**Script:**
> "VisionPilot runs on AWS using a minimal, production-ready architecture:
>
> - **ECS Fargate**: Serverless containers, no EC2 management
> - **ALB**: Health checks on `/health`, auto-scaling
> - **CloudFormation**: Full IaC — VPC, subnets, IAM, logging
> - **CloudWatch**: Logs, metrics, 14-day retention
> - **IAM**: Least-privilege roles for S3 uploads
>
> The container includes OpenCV 5, FastAPI, and the agent — 2GB RAM, 1 vCPU. Deploy with one CloudFormation command."

**Actions:**
1. Show cloudformation.yaml
2. Show ECS task definition
3. If deployed: curl live endpoint
4. Show cost controls: desired count = 1, max 10

---

### 4:00–4:40 — Evaluation, Failure Cases, Limitations (40 sec)

**Visual:** Test results table → Failure case examples → Limitations slide

**Script:**
> "Rigorous evaluation:
>
> - **11/11 tests passing** — pipeline + agentic behavior
> - **Agentic test proves** visual evidence changes decisions
> - **Failure cases handled**: invalid data, too small, blurry, dark, low confidence
> - **Limitations documented**: no Haar cascades in pip package (uses contours/colors), synthetic fallback for demo, in-memory storage
>
> Responsible use: no facial recognition, no biometrics, human-in-the-loop for uncertainty, full audit trail."

**Actions:**
1. Show pytest output
2. Show test_agent_makes_different_decisions
3. List failure cases from TECHNICAL_REPORT.md

---

### 4:40–5:00 — Summary & Impact (20 sec)

**Visual:** Title card with key achievements

**Script:**
> "VisionPilot delivers:
>
> ✅ **OpenCV 5** substantive perception pipeline
> ✅ **Genuine agentic loop** — perception → decision → action
> ✅ **Visual evidence changes behavior** — proven by tests
> ✅ **AWS deployment** — ECS Fargate, CloudFormation, CloudWatch
> ✅ **Complete documentation** — technical report, architecture, traces
> ✅ **Reproducible** — Docker, pinned deps, test suite
>
> This is a working, measurable, auditable computer vision system — not just a polished interface. Thank you."

**End Card:** GitHub URL | Demo Video | Contact

---

## Production Notes

### Recording Setup
- Screen: 1920x1080, 30fps
- Audio: Clear narration, minimal background noise
- Terminal: Large font (16pt), high contrast theme
- Browser: Chrome, dev tools closed

### Demo Assets Needed
- [ ] 4 test images (good, empty, blurry, dark)
- [ ] Pre-recorded AWS console if not live
- [ ] Architecture diagram PNG for slides

### Timing Buffer
- Target: 4:45 actual content
- Buffer: 15 seconds for transitions
- Hard limit: 5:00

---

## Alternative: If AWS Not Deployed

Replace 3:15–4:00 with:
- Show CloudFormation template
- Explain deployment steps
- Show docker-compose running locally
- Emphasize "deployment-ready configuration"

---

## Judges' Checklist Alignment

| Competition Requirement | Covered At |
|------------------------|------------|
| OpenCV 5 substantive use | 1:15–2:15 |
| AWS meaningful component | 3:15–4:00 |
| Real-world problem | 0:00–0:30 |
| Working endpoint | 0:30–1:15 |
| Technical report | Referenced |
| Architecture diagram | 3:15–4:00 |
| Pinned dependencies | requirements.txt |
| Setup instructions | README |
| Evaluation evidence | 4:00–4:40 |
| Limitations/failures | 4:00–4:40 |
| Responsible use | 4:00–4:40 |
| Judge-accessible repo | GitHub URL |
| ≤5 min video | This storyboard |
| Agentic vision loop | 2:15–3:15 ⭐ |