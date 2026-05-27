# BestPhoto AI — Developer Notes for Claude

## What This Is
A FastAPI REST API that ranks images by photographic quality across 9 scoring dimensions.
Designed for wedding and event photographers processing large NAS-stored batches.

## Running

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

First run downloads the CLIP model (~600 MB) to `~/.cache/huggingface/`.

## Architecture

```
main.py
└── src/api/app.py                   FastAPI lifespan (loads CLIP at startup)
      ├── src/api/routes/health.py   GET  /api/v1/health
      └── src/api/routes/rank.py     POST /api/v1/rank
                                     GET  /api/v1/jobs/{job_id}
            └── src/core/pipeline.py  orchestrator
                  ├── src/scorers/face_scorer.py       MediaPipe
                  ├── src/scorers/technical_scorer.py  OpenCV
                  ├── src/scorers/duplicate_scorer.py  imagehash pHash
                  └── src/scorers/semantic_scorer.py   CLIP (HuggingFace)
```

## Key Files

| File | Purpose |
|---|---|
| `src/config/weights.py` | Mode-specific scoring weights (tune here) |
| `src/config/settings.py` | All env-var backed settings |
| `src/scorers/semantic_scorer.py` | CLIP prompts + scoring logic |
| `src/core/pipeline.py` | Combines all scorer outputs |
| `src/models/request.py` | API request schema |
| `src/models/response.py` | API response schema |

## Scoring Dimensions (0-10 each)

| Dimension | Scorer | Details |
|---|---|---|
| `face_clarity` | FaceScorer | Laplacian variance of face crop |
| `eyes_open` | FaceScorer | Eye Aspect Ratio (EAR) via face mesh landmarks |
| `smile` | SemanticScorer | CLIP zero-shot: smiling vs neutral prompts |
| `emotion` | SemanticScorer | CLIP zero-shot: emotional vs posed prompts |
| `technical_quality` | TechnicalScorer | blur×0.5 + exposure×0.3 + noise×0.2 |
| `composition` | FaceScorer | Rule-of-thirds alignment of face centroid |
| `cinematic` | SemanticScorer | CLIP zero-shot: cinematic vs amateur prompts |
| `invitation_suit` | SemanticScorer | CLIP zero-shot: formal vs casual prompts |
| `storytelling` | SemanticScorer | CLIP zero-shot: narrative vs generic prompts |

## No-Face Images (landscapes, food, venues)
When MediaPipe detects no faces: `face_clarity`, `eyes_open`, `smile` → `None`.
The ranker redistributes their weights proportionally to remaining dimensions.
Controlled by `NO_FACE_REDISTRIBUTE` in `src/config/weights.py`.

## Duplicate Detection
pHash Hamming distance ≤ `DUPLICATE_HASH_THRESHOLD` (default 8) → duplicate.
Union-Find groups duplicates. Representative = first node in group.
Duplicates receive a `DUPLICATE_PENALTY_FACTOR` (0.35×) score multiplier.

## Async Jobs
- ≤ `SYNC_BATCH_LIMIT` images (default 200) OR `"async": false` → synchronous response
- Above limit OR `"async": true` → returns `job_id`, poll `/api/v1/jobs/{id}`
- Job store is in-memory (`src/core/job_store.py`).
- **For production scale:** replace with Redis + Celery workers.

## Adding a New Scoring Dimension
1. Add prompts to `CRITERION_PROMPTS` in `src/scorers/semantic_scorer.py`
2. Add the key + weights to all modes in `src/config/weights.py`
3. Wire it up in `src/core/pipeline.py → scores_raw`
4. Add the field to `src/models/response.py → ScoreBreakdown`

## Adjusting Weights
Edit `src/config/weights.py`. Weights per mode must sum to 1.0.
The `NO_FACE_REDISTRIBUTE` list controls which keys get redistributed away for non-portrait images.

## Cross-Platform Paths
`src/core/image_loader.py` uses `pathlib.Path` throughout.
- Linux NFS: `/mnt/nas/wedding/photo.jpg`
- Windows drive: `C:\Photos\DSC001.jpg`
- Windows UNC: `\\NAS\Photos\Wedding`
- Folder scanning with optional `recursive=true`
