"""
In-memory async job store.

For production at scale, replace with Redis-backed store.
Jobs expire after JOB_TTL_SECONDS (default 1 h).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from src.config.settings import settings


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: str = "pending"     # pending | processing | completed | failed
        self.progress: int = 0           # 0-100
        self.total_images: Optional[int] = None
        self.processed_images: int = 0
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at: float = time.time()

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Job:
        async with self._lock:
            job = Job(job_id=str(uuid4()))
            self._jobs[job.job_id] = job
            return job

    async def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs):
        job = self._jobs.get(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)

    async def cleanup_expired(self):
        """Remove jobs older than JOB_TTL_SECONDS."""
        now = time.time()
        expired = [
            jid for jid, job in self._jobs.items()
            if (now - job.created_at) > settings.JOB_TTL_SECONDS
        ]
        for jid in expired:
            del self._jobs[jid]


# Singleton
job_store = JobStore()
