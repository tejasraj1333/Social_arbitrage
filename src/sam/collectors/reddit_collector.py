"""Reddit recon collector (needs credentials).

Read-only PRAW over the configured subreddits. Credentials come from settings
(SAM_REDDIT__CLIENT_ID / __CLIENT_SECRET / __USER_AGENT). When absent, `fetch`
raises CredentialsMissing and `run()` reports status="needs_credentials" rather
than crashing — so the collector is built and tested now, and proven live the
moment creds are added to .env.

Record schema: title, body (selftext), score, comments, created_utc, subreddit, author.
`body` is intentionally not a required field (link posts have empty selftext).
"""

from __future__ import annotations

import math
from typing import Any

from sam.core.config import get_settings
from sam.core.errors import CredentialsMissing
from sam.recon.collector_base import ReconCollector
from sam.recon.sources import load_sources


class RedditCollector(ReconCollector):
    source_name = "reddit"
    required_fields = ["title", "score", "comments", "created_utc", "subreddit", "author"]
    timestamp_field = "created_utc"
    sample_size = 100

    def __init__(
        self,
        subreddits: list[str] | None = None,
        post_limit: int | None = None,
    ) -> None:
        super().__init__()
        cfg = load_sources().get("reddit", {})
        self.subreddits = subreddits if subreddits is not None else cfg.get("subreddits", [])
        self.post_limit = post_limit if post_limit is not None else cfg.get("post_limit", 200)

    def fetch(self) -> list[dict[str, Any]]:
        creds = get_settings().reddit
        if not creds.client_id or not creds.client_secret:
            raise CredentialsMissing(
                "Reddit credentials missing - set SAM_REDDIT__CLIENT_ID and "
                "SAM_REDDIT__CLIENT_SECRET (register a 'script' app at "
                "https://www.reddit.com/prefs/apps)."
            )
        reddit = self._client(creds)
        n_subs = max(1, len(self.subreddits))
        per_sub = max(1, math.ceil(self.post_limit / n_subs))
        records: list[dict[str, Any]] = []
        for sub in self.subreddits:
            for submission in reddit.subreddit(sub).hot(limit=per_sub):
                records.append(self._to_record(submission, sub))
        return records

    def _client(self, creds: Any) -> Any:
        import praw  # lazy: keep module importable without praw installed

        return praw.Reddit(
            client_id=creds.client_id,
            client_secret=creds.client_secret,
            user_agent=creds.user_agent,
            check_for_async=False,
        )

    @staticmethod
    def _to_record(submission: Any, subreddit: str) -> dict[str, Any]:
        author = getattr(submission.author, "name", None) if submission.author else None
        return {
            "id": submission.id,
            "title": submission.title,
            "body": submission.selftext,
            "score": submission.score,
            "comments": submission.num_comments,
            "created_utc": submission.created_utc,
            "subreddit": subreddit,
            "author": author,
        }
