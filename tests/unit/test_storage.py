"""Tests for literavore.storage."""

from pathlib import Path

import pytest

from literavore.storage.base import StorageBackend
from literavore.storage.local import LocalStorage


class TestProtocol:
    def test_local_storage_is_storage_backend(self, tmp_path: Path):
        storage = LocalStorage(tmp_path)
        assert isinstance(storage, StorageBackend)


class TestPutGet:
    def test_roundtrip(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("test.txt", b"hello")
        assert s.get("test.txt") == b"hello"

    def test_overwrite(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("test.txt", b"first")
        s.put("test.txt", b"second")
        assert s.get("test.txt") == b"second"

    def test_empty_bytes(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("empty.bin", b"")
        assert s.get("empty.bin") == b""

    def test_missing_key_raises(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        with pytest.raises(FileNotFoundError):
            s.get("nonexistent")


class TestExists:
    def test_false_before_put(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        assert s.exists("test.txt") is False

    def test_true_after_put(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("test.txt", b"data")
        assert s.exists("test.txt") is True

    def test_false_after_delete(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("test.txt", b"data")
        s.delete("test.txt")
        assert s.exists("test.txt") is False


class TestDelete:
    def test_delete_existing(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("test.txt", b"data")
        s.delete("test.txt")
        assert not s.exists("test.txt")

    def test_delete_missing_raises(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        with pytest.raises(FileNotFoundError):
            s.delete("nope")


class TestListKeys:
    def test_empty(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        assert s.list_keys() == []

    def test_lists_all(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("a.txt", b"1")
        s.put("b.txt", b"2")
        keys = s.list_keys()
        assert sorted(keys) == ["a.txt", "b.txt"]

    def test_prefix_filter(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("fetch/p1.json", b"1")
        s.put("fetch/p2.json", b"2")
        s.put("extract/p1.json", b"3")
        keys = s.list_keys("fetch/")
        assert len(keys) == 2
        assert all(k.startswith("fetch/") for k in keys)

    def test_no_match_prefix(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("fetch/p1.json", b"1")
        assert s.list_keys("nope/") == []


class TestGetLocalPath:
    def test_returns_path(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("test.txt", b"data")
        p = s.get_local_path("test.txt")
        assert isinstance(p, Path)
        assert p.read_bytes() == b"data"

    def test_returns_path_for_nonexistent(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        p = s.get_local_path("nope.txt")
        assert isinstance(p, Path)


class TestNestedKeys:
    def test_nested_put_get(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("a/b/c.txt", b"deep")
        assert s.get("a/b/c.txt") == b"deep"

    def test_nested_list(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.put("a/b/c.txt", b"1")
        s.put("a/d.txt", b"2")
        keys = s.list_keys("a/")
        assert len(keys) == 2

    def test_base_dir_created(self, tmp_path: Path):
        deep = tmp_path / "deep" / "nested"
        LocalStorage(deep)
        assert deep.exists()
