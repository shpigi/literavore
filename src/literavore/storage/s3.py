from __future__ import annotations

from pathlib import Path


class S3Storage:
    """S3-compatible storage backend (stub — not yet implemented).

    Args:
        endpoint: S3 endpoint URL (e.g. ``"https://s3.amazonaws.com"``).
        bucket: Target bucket name.
        access_key: AWS / S3-compatible access key ID.
        secret_key: AWS / S3-compatible secret access key.
    """

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key

    def put(self, key: str, data: bytes) -> None:
        raise NotImplementedError("S3Storage is not yet implemented")

    def get(self, key: str) -> bytes:
        raise NotImplementedError("S3Storage is not yet implemented")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("S3Storage is not yet implemented")

    def list_keys(self, prefix: str = "") -> list[str]:
        raise NotImplementedError("S3Storage is not yet implemented")

    def delete(self, key: str) -> None:
        raise NotImplementedError("S3Storage is not yet implemented")

    def get_local_path(self, key: str) -> Path | None:
        """S3 objects have no local filesystem path; always returns None."""
        return None
