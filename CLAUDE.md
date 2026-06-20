# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

These rules are **MANDATORY**. They are not suggestions. Any change that violates
them is incorrect and must be fixed before it is considered done.

## Coding standards (STRICT — non-negotiable)

- **`mypy --strict` MUST pass with zero errors.** All code MUST be fully typed.
  `from __future__ import annotations` MUST be at the top of every module. NEVER use
  `Any` to silence the type checker; NEVER add `# type: ignore` without a specific
  error code and a one-line justification comment.
- **`ruff check .` MUST pass with zero warnings.** NEVER disable a lint rule inline to
  get green; fix the underlying issue.
- **NEVER import or call the Anthropic SDK outside the provider/judge layer.** ALL model
  access MUST go through the `LLMProvider` Protocol and the normalized
  `ProviderResponse`/`ToolCall` models in `providers/base.py`.
- **Eval cases are data, not code** — `evalsets/*.yaml`, validated by the
  `EvalCase`/`EvalSuite` pydantic models in `eval/case.py`. NEVER hardcode cases.
- **Schema sync is CI-enforced and MUST stay in sync.** If you change the
  `EvalCase`/`EvalSuite` models you MUST regenerate the JSON Schema in the same change:
  `poetry run python scripts/export_schema.py` → `evalsets/schema/eval_suite.schema.json`.
- **Temperature handling MUST be preserved.** Opus 4.7+/Fable 5 reject sampling params
  (400). `AnthropicProvider._supports_temperature` drops `temperature` for those
  prefixes. Default agent = **Haiku 4.5** (accepts temperature, needed to elicit
  non-determinism); judge = **Opus 4.8**. Overridable via `AGENT_EVAL_AGENT_MODEL` /
  `AGENT_EVAL_JUDGE_MODEL`. Do not regress this.
- **Every API-calling path MUST retry with `tenacity`** (exponential backoff, 4
  attempts) on `RateLimitError`/`APIStatusError` ONLY. Other errors MUST surface.
- **`eval/stats.py` MUST stay hand-rolled and exhaustively unit-tested.** NEVER pull in
  an external metrics library for Wilson CI / flake rate / pass^k. New statistical
  behavior REQUIRES new tests.
- **Tests MUST spend zero tokens.** Use the mock provider. A real `ANTHROPIC_API_KEY`
  MUST never be required to run the test suite.
- **The intentional `pylint` exceptions in `pyproject.toml` are deliberate**
  (`too-few-public-methods`, raised `max-args`/`max-locals`). NEVER refactor solely to
  satisfy them, and NEVER add new blanket disables.

## SonarQube compliance (STRICT — zero tolerance)

All code MUST be SonarQube-clean before it is considered done: **zero bugs, zero
vulnerabilities, zero code smells, zero security hotspots, and no new duplication.**
Treat every Sonar issue as a blocker, not a warning.

- **Reliability (bugs):** NEVER leave a raised exception unhandled where it can crash a
  run mid-suite; NEVER ignore a returned value that signals failure; NEVER compare
  floats for exact equality in statistical code; ALWAYS close/await async resources.
- **Security (vulnerabilities & hotspots):** NEVER hardcode secrets, API keys, or tokens
  — they come from the environment (`ANTHROPIC_API_KEY`) only. Use `yaml.safe_load`
  (NEVER `yaml.load`). NEVER build shell/SQL/HTML by string concatenation of untrusted
  input; the HTML report MUST escape interpolated values (`jinja2` autoescape).
- **Maintainability (code smells):** Keep **cognitive complexity low** — extract helpers
  rather than nesting; NEVER duplicate a block (DRY — factor it out); remove dead code,
  unused imports, and commented-out code; NO magic numbers (name them as constants);
  every function MUST have a single clear responsibility.
- **No new duplication.** Sonar flags copy-paste; reuse existing helpers/models instead.
- **Self-verify before finishing.** Run `make check` (lint + type + test) and resolve
  every finding. If a Sonar rule genuinely does not apply, justify it explicitly in a
  comment rather than ignoring it silently — but prefer fixing over suppressing.
