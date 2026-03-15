# Phase 8 Post: Docker Security, API Polish

**Date:** 2026-03-15

## Changes

### Docker: non-root user mapping
- **Dockerfile**: added `groupadd`/`useradd` for `appuser` (UID/GID 1000) in final stage
- **docker-compose.yml**: both `api` and `ui` services now run with `user: "${UID:-1000}:${GID:-1000}"` so volume-mounted files remain owned by the host user
- **docker-compose.yml** (`ui`): added `HOME=/tmp` so Streamlit can write its config without a writable home directory

### API: `/conferences` endpoint
- **serve/api.py**: added `GET /conferences` returning distinct conference names from the database
- **README.md**: added `curl localhost:8000/conferences` example to the API section

### CLI: Streamlit file-watcher disabled
- **cli.py**: added `--server.fileWatcherType none` to the Streamlit subprocess launch to suppress inotify errors in containerized environments

### Config: default `max_papers` reduced
- **config/default.yml**: CoRL-2025 `max_papers` lowered from 2000 → 50 for development; commented-out conference entries unchanged
