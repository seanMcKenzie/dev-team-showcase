# OpenClaw Config Snapshots

Sanitized snapshots of `openclaw.json` (tokens redacted).

## Change Log

| Date | Change |
|------|--------|
| 2026-03-13 | Initial memory optimizations: hybrid search, embedding cache, contextPruning TTL 5m, memoryFlush |
| 2026-03-16 | Compaction tuning: maxHistoryShare 0.6, compaction model → haiku |
| 2026-03-16 | Blocker-watch cron: 30m → 2h, quiet hours 10PM-8AM CST, 4h cooldown |
