---
name: change-request
description: End-to-end feature implementation workflow for codebases. Use when a user asks to "implement a feature", "add a feature", "build feature", "extend functionality", "feature request", or "end-to-end feature".
metadata:
  short-description: End-to-end feature implementation (plan -> code -> tests)
---

# Feature Dev

Follow this workflow exactly and keep it concise.

## Workflow

1) Read before write: inspect relevant files and existing patterns; identify entry points and dependencies.
2) Produce a short plan (max 10 lines) listing files to touch and why.
3) Implement the feature with minimal coherent diffs, respecting repo conventions.
4) Add/adjust tests for behavior changes; include exact commands to run tests/lint/build.
5) Update docs if user-visible behavior changes.
6) Provide risk notes: edge cases, migration/rollback notes, and what to monitor.
7) Final output format must always be:
   - Plan
   - Changes (by file)
   - Tests (commands)
   - Docs (if any)
   - Risks/edge cases
   - Summary (5 lines max)

## Output Template

Plan
[max 10 lines; list files and why]

Changes (by file)
[file path: change summary]

Tests (commands)
[exact commands to run]

Docs (if any)
[doc updates or "None"]

Risks/edge cases
[edge cases, migration/rollback notes, monitoring]

Summary
[5 lines max]
