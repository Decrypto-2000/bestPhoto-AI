# BestPhoto AI 📸

An intelligent image ranking API for photographers — especially wedding and event shooters.
Send a batch of image paths, get back a ranked list with per-dimension quality scores.

**100% open-source. No cloud API required.**

---

## Scoring Dimensions

| Dimension | What it measures |
|---|---|
| **Face Clarity** | Sharpness of detected faces |
| **Eyes Open** | Are eyes open across all faces? |
| **Smile** | Presence of genuine, warm smiles |
| **Emotion** | Emotional resonance of the moment |
| **Technical Quality** | Sharpness, exposure, and noise |
| **Composition** | Rule-of-thirds framing quality |
| **Cinematic** | Cinematic lighting and composition |
| **Invitation Suitability** | Suitable for formal print / invitation cards |
| **Storytelling** | Narrative and documentary depth |

Images with no faces (landscapes, venues, food) are fairly ranked — face-related
weights are automatically redistributed to technical and semantic dimensions.

---

## Scoring Modes

| Mode | Optimised for |
|---|---|
| `wedding` | Wedding ceremonies and receptions |
| `portrait` | Studio and outdoor portrait sessions |
| `event` | Corporate and social events |
| `general` | Mixed or unspecified photo types |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy env config
cp .env.example .env

# 3. Start the API (first run downloads CLIP ~600 MB)
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Open interactive docs
# http://localhost:8000/docs
```

---

## API Usage

### Rank specific image paths

```bash
curl -X POST http://localhost:8000/api/v1/rank \
  -H "Content-Type: application/json" \
  -d '{
    "paths": [
      "/mnt/nas/wedding/DSC_0421.jpg",
      "/mnt/nas/wedding/DSC_0422.jpg"
    ],
    "mode": "wedding",
    "top_n": 10
  }'
```

### Rank all images in a folder

```bash
curl -X POST http://localhost:8000/api/v1/rank \
  -H "Content-Type: application/json" \
  -d '{
    "folder": "/mnt/nas/wedding_2024",
    "recursive": true,
    "mode": "wedding"
  }'
```

### Mix paths and folder (both are merged)

```bash
curl -X POST http://localhost:8000/api/v1/rank \
  -d '{
    "paths": ["/extra/DSC_9999.jpg"],
    "folder": "/mnt/nas/wedding_2024",
    "mode": "wedding"
  }'
```

### Restrict to specific file types

```bash
curl -X POST http://localhost:8000/api/v1/rank \
  -d '{
    "folder": "/mnt/nas/session",
    "extensions": ["jpg", "jpeg"],
    "mode": "portrait"
  }'
```

### Large batch — async mode

```bash
# Submit job
curl -X POST http://localhost:8000/api/v1/rank \
  -d '{"folder": "/mnt/nas/event_2024", "mode": "event", "async": true}'
# → { "job_id": "abc-123", "poll_url": "/api/v1/jobs/abc-123" }

# Poll for results
curl http://localhost:8000/api/v1/jobs/abc-123
# → { "status": "processing", "progress": 45, ... }
# → { "status": "completed", "result": { "ranked": [...] } }
```

### Windows / UNC paths

```json
{
  "paths": ["C:\\Photos\\DSC001.jpg"],
  "folder": "\\\\NAS-SERVER\\Photos\\Wedding2024",
  "mode": "wedding"
}
```

---

## Response Format

```json
{
  "status": "success",
  "mode": "wedding",
  "total_images": 312,
  "processing_time_ms": 24850.3,
  "ranked": [
    {
      "path": "/mnt/nas/wedding/DSC_0421.jpg",
      "rank": 1,
      "total_score": 8.74,
      "scores": {
        "face_clarity":     9.20,
        "eyes_open":        9.80,
        "smile":            8.10,
        "emotion":          8.50,
        "technical_quality":8.90,
        "composition":      8.30,
        "cinematic":        7.90,
        "invitation_suit":  9.10,
        "storytelling":     7.60
      },
      "meta": {
        "has_faces":    true,
        "face_count":   2,
        "is_duplicate": false,
        "duplicate_of": null
      }
    }
  ]
}
```

**Non-portrait image** (no faces detected):
```json
{
  "scores": {
    "face_clarity":     null,
    "eyes_open":        null,
    "smile":            null,
    "emotion":          6.20,
    "technical_quality":8.50,
    "composition":      7.80,
    "cinematic":        7.10,
    "invitation_suit":  6.90,
    "storytelling":     6.40
  },
  "meta": { "has_faces": false, "face_count": 0 }
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cpu` | `cuda` for GPU acceleration |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | CLIP model variant |
| `EAR_THRESHOLD` | `0.20` | Eye Aspect Ratio threshold (open vs closed) |
| `DUPLICATE_HASH_THRESHOLD` | `8` | pHash Hamming distance for duplicate grouping |
| `SYNC_BATCH_LIMIT` | `200` | Images above this → async job |
| `JOB_TTL_SECONDS` | `3600` | In-memory job expiry time |

---

## Tech Stack

| Library | Purpose |
|---|---|
| **FastAPI** | Async REST framework |
| **MediaPipe** | Face detection, eye landmark analysis |
| **OpenCV** | Blur, exposure, and noise scoring |
| **CLIP (HuggingFace)** | Zero-shot semantic scoring |
| **imagehash** | Perceptual duplicate detection |
| **Pydantic v2** | Request/response validation |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/rank` | Rank a batch of images |
| `GET` | `/api/v1/jobs/{job_id}` | Poll async job status |
| `GET` | `/api/v1/health` | Service health & model status |
| `GET` | `/docs` | Interactive Swagger UI |
