from __future__ import annotations

from pathlib import Path

from literavore.storage.base import StorageBackend


class LocalStorage:
    """Local filesystem storage backend.

    Keys map directly to file paths relative to ``base_dir``.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, key: str) -> Path:
        """Resolve a key to an absolute filesystem path."""
        return self.base_dir / key

    # ------------------------------------------------------------------
    # StorageBackend implementation
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes) -> None:
        """Write *data* to the file at ``base_dir / key``.

        Parent directories are created automatically.
        """
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        """Read and return the bytes stored under *key*.

        Raises:
            FileNotFoundError: if the key does not exist.
        """
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Key not found in local storage: {key!r}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        """Return True if the file for *key* exists."""
        return self._resolve(key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all keys (relative paths) whose string representation starts with *prefix*."""
        results: list[str] = []
        search_root = self.base_dir
        for path in search_root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(self.base_dir).as_posix()
                if relative.startswith(prefix):
                    results.append(relative)
        return sorted(results)

    def delete(self, key: str) -> None:
        """Delete the file stored under *key*.

        Raises:
            FileNotFoundError: if the key does not exist.
        """
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Key not found in local storage: {key!r}")
        path.unlink()

    def get_local_path(self, key: str) -> Path | None:
        """Return the filesystem path for *key* (whether or not it currently exists)."""
        return self._resolve(key)


# Verify structural compatibility at import time (caught by type checkers too).
_: StorageBackend = LocalStorage.__new__(LocalStorage)  # type: ignore[assignment]
