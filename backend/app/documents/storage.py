"""
File storage abstraction.

One Protocol, one implementation for now (LocalFilesystemStorage). A future
S3 backend implements the same three methods and is wired in at one place
(get_storage_backend, below) — no document-service code changes required.

Path traversal is prevented structurally: every key this module is ever
called with is server-generated ("{organization_id}/{document_id}.pdf",
both UUIDs — see app/documents/service.py), never derived from a client-
supplied filename. The resolved-path assertion in _resolve() is defense in
depth on top of that guarantee, not the guarantee itself.
"""

from pathlib import Path
from typing import Protocol


class StorageError(Exception):
    """
    Raised for any storage backend failure (write/read/delete). Callers
    catch this one exception type, never a backend-specific exception —
    that's what keeps backends swappable without touching call sites.
    """


class StorageBackend(Protocol):
    def save(self, key: str, content: bytes) -> None: ...
    def read(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


class LocalFilesystemStorage:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self.root_dir / key).resolve()
        # Defense in depth: even though every key we're ever called with is
        # server-generated (never client-controlled), assert the resolved
        # path is still under root_dir before touching the filesystem.
        if self.root_dir not in path.parents and path != self.root_dir:
            raise StorageError("Resolved storage path escapes the storage root")
        return path

    def save(self, key: str, content: bytes) -> None:
        path = self._resolve(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        except OSError as exc:
            raise StorageError("Failed to write file to storage") from exc

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise StorageError("Failed to read file from storage") from exc

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError("Failed to delete file from storage") from exc


_storage_backend_instance: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """
    FastAPI dependency. A module-level singleton so every request reuses the
    same backend instance instead of re-resolving the storage root each time.
    Tests override this via app.dependency_overrides to inject a fake backend
    for failure-mode tests (storage write failure, etc.) without touching the
    real filesystem.
    """
    global _storage_backend_instance
    if _storage_backend_instance is None:
        from app.core.config import settings

        _storage_backend_instance = LocalFilesystemStorage(settings.document_storage_root)
    return _storage_backend_instance
