from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .urls import normalize_url


@dataclass
class CrawlState:
    path: Path
    visited_urls: set[str] = field(default_factory=set)
    transcript_urls: set[str] = field(default_factory=set)
    content_hashes: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "CrawlState":
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            visited_urls=set(data.get("visited_urls", [])),
            transcript_urls=set(data.get("transcript_urls", [])),
            content_hashes=set(data.get("content_hashes", [])),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "visited_urls": sorted(self.visited_urls),
                    "transcript_urls": sorted(self.transcript_urls),
                    "content_hashes": sorted(self.content_hashes),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def has_visited(self, url: str) -> bool:
        return normalize_url(url) in self.visited_urls

    def mark_visited(self, url: str) -> None:
        self.visited_urls.add(normalize_url(url))

    def has_transcript_url(self, url: str) -> bool:
        return normalize_url(url) in self.transcript_urls

    def mark_transcript_url(self, url: str) -> None:
        self.transcript_urls.add(normalize_url(url))

    def has_content_hash(self, content_hash: str) -> bool:
        return content_hash in self.content_hashes

    def mark_content_hash(self, content_hash: str) -> None:
        self.content_hashes.add(content_hash)

