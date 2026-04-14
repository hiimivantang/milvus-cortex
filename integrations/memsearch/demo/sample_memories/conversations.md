# Conversation Summaries

## Meeting with Sarah Chen (2026-04-10)

Discussed the Q2 roadmap for the platform team. Sarah wants to prioritize
the authentication migration from JWT to OAuth2 PKCE flow. Timeline is
6 weeks, starting April 15. The mobile team (led by Marcus Rivera) needs
the new auth endpoints by May 20 for the iOS 18 release.

Key decision: we'll use Keycloak as the identity provider instead of
building custom. Sarah approved the $2,400/month Keycloak Cloud budget.

## Debugging Session: Memory Leak in Worker Pool (2026-04-12)

Found the root cause of the memory leak in the background worker pool.
The `TaskExecutor` class was holding references to completed futures
in `self._completed_tasks` without ever pruning them. Fix: added a
`max_completed_history=1000` parameter and a pruning sweep every 60s.

The leak was causing OOM kills after ~48 hours in production. After the
fix, memory usage stabilized at ~800MB (down from 4.2GB before crash).
Deployed to staging on April 12, production rollout scheduled for April 14.

## Code Review with Alex Kim (2026-04-13)

Reviewed Alex's PR #847 for the new caching layer. Main feedback:
1. The Redis TTL should be configurable, not hardcoded to 3600s
2. Cache invalidation on write needs to handle race conditions
3. The `CacheMiddleware` class is doing too much — split into
   `CacheReader` and `CacheWriter` following SRP
4. Missing tests for cache miss scenarios

Alex agreed to address all points. Follow-up review scheduled for April 15.
