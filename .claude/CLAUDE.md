# CLAUDE.md

> **Purpose:** Coding standards, behavioral guidelines, and project-specific contracts for AI-assisted development.
> Merge project-specific sections as needed. Universal rules always apply.

---

## Part 1: Universal Coding Guidelines

These apply to every project regardless of domain or stack.

---

### 1.1 Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly before implementing. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

> **Analogy:** Think of this like a surgeon's pre-op checklist — rushing saves no time if you cut the wrong thing.

---

### 1.2 Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

> **Litmus test:** Would a senior engineer say this is overcomplicated? If yes, simplify.

---

### 1.3 Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

> **The test:** Every changed line should trace directly to the user's request.

---

### 1.4 Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

| Vague Request | Concrete Goal |
|---|---|
| "Add validation" | Write tests for invalid inputs, then make them pass |
| "Fix the bug" | Write a test that reproduces it, then make it pass |
| "Refactor X" | Ensure tests pass before and after |

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

> Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

### 1.5 Success Signal

These guidelines are working if:
- Fewer unnecessary changes in diffs.
- Fewer rewrites due to overcomplication.
- Clarifying questions come before implementation, not after mistakes.

---

## Part 2: Repository Standards

Standard conventions for project structure, tooling, and developer experience.

---

### 2.1 Project Structure

```
project-root/
├── CLAUDE.md              # This file (always at root)
├── README.md              # Single README at root
├── Makefile               # One-line developer commands
├── settings.py            # Pydantic-based configuration
├── .env                   # Secrets only (API keys, passwords)
├── .gitignore             # Must ignore .venv, .env, logs/
├── docs/
│   ├── blog.md            # Blog-style project explanation
│   └── demo_narrative.md  # Stakeholder demo narration
├── scripts/               # Educational standalone scripts
├── logs/                  # Structured JSON logs (max 5 files)
└── src/                   # Application code (modular)
```

---

### 2.2 Developer Experience

- Use `make` for all common workflows (start, test, lint, check).
- Prefer simple, predictable developer workflows.

```makefile
# Example Makefile targets
run:        ## Start the application
test:       ## Run all tests
lint:       ## Run linter
check:      ## Health check
```

---

### 2.3 Python and Environment

- Use `uv` for dependency and environment management.
- **Do not** use `pip`, `poetry`, or other package managers.
- `.env` is for secrets only (API keys, passwords). No non-secret config.
- `.venv` and `.env` must be in `.gitignore`.

---

### 2.4 Configuration

- Centralize all config in `settings.py` using Pydantic `BaseSettings`.
- Access configuration through settings objects.
- **Do not** scatter config across files or read env vars directly in app code.

> **Analogy:** Settings is the single reception desk. Don't let every module wander outside to check the mailbox.

---

### 2.5 Logging

- Use `loguru` for all logging. **No `print()` for application logging.**
- Terminal logs for development/debugging.
- Persist structured JSON logs in `logs/` (max 5 files via rotation/retention).
- LLM tracing must include: input, output, latency.

---

### 2.6 Architecture

- Maintain modular design: separate orchestration, integrations, business logic, config, and utilities.
- Design the LLM layer for easy provider swaps (Ollama ↔ Anthropic ↔ OpenAI ↔ Gemini).
- No provider-specific logic leaking across the codebase.
- Use Google-style function docstrings.

> **Analogy:** The LLM provider is like a power outlet — the appliance (your app) shouldn't care whether it's plugged into a wall socket or a generator.

---

### 2.7 Documentation and Maintainability

- Keep code concise, readable, and production-oriented.
- Prefer replaceable modules over tightly coupled implementations.
- Write code so future agents (or engineers) can understand and modify it safely.
- New features must fit existing modular structure — no shortcuts or duplication.
- `scripts/` must contain educational scripts for each core module used in the codebase.

---
