# Agent Swarm Coordination

## Roles

### Orchestrator (Opus)
- Plans work, identifies parallelizable tasks
- Spawns Sonnet workers via Agent tool with `isolation: "worktree"`
- Reviews worker output, merges branches, resolves conflicts
- Commits with `phase-N.M: description` format
- Writes dev log entries after each phase
- Runs `pytest` gate after merging phase work

### Workers (Sonnet)
- Implement one task in an isolated worktree
- Only modify files listed in their prompt
- Follow ruff formatting (line-length 100)
- Add type hints to all public functions
- Write corresponding unit tests
- Run relevant tests before finishing

## Phase Dependencies

Phases are sequential gates. All tasks in a phase must complete before the next phase starts.
Within a phase, tasks with no file overlap run in parallel (same parallel group).

## Commit Format

```
phase-N.M: <description>
```

Builds a readable narrative in git log.

## Dev Log Format

After each phase, the orchestrator writes:

```
dev_log/YYYY-MM-DD_HH-MM_phase-N-M_<slug>.md
```

Content: what was built, decisions made, how agents collaborated.

## File Ownership

Within a parallel group, each worker has exclusive ownership of its listed files.
No two workers in the same group may modify the same file.

## Testing

- Workers run `pytest tests/unit/<relevant_test>.py` before finishing
- Orchestrator runs full `pytest` after merging a phase's work
- Tests must pass before advancing to the next phase
