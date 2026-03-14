from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the interface for all storage backends."""

    def put(self, key: str, data: bytes) -> None:
        """Store data under the given key."""
        ...

    def get(self, key: str) -> bytes:
        """Retrieve data for the given key.

        Raises:
            FileNotFoundError: if the key does not exist.
        """
        ...

    def exists(self, key: str) -> bool:
        """Return True if the key exists in storage."""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all keys that start with the given prefix."""
        ...

    def delete(self, key: str) -> None:
        """Delete the data stored under the given key.

        Raises:
            FileNotFoundError: if the key does not exist.
        """
        ...

    def get_local_path(self, key: str) -> Path | None:
        """Return the local filesystem path for the key, if available.

        Returns None for remote backends (e.g. S3).
        """
        ...
