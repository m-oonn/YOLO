# Enterprise-Level Project Optimization Plan

## Context

The project is **YOLO 课程设计 — 实时目标检测与行为分析系统** (Real-time Object Detection & Behavior Analysis System), located at `e:/projects-YOLO/course-design/`. It uses FastAPI (Python 3.10) + Vue 3 + Ultralytics YOLO with Docker deployment. The goal is to upgrade it to open-source enterprise quality for use as a portfolio showcase in internship applications.

**Current gaps:**
- No CI/CD (no `.github/` directory at all)
- No ESLint/Prettier config (mentioned in docs but never configured)
- No Python linting/formatting config
- No `LICENSE` file (currently "educational use only")
- No `CONTRIBUTING.md`, Issue templates, or PR template

**All tools used are free** (GitHub Actions free tier, Codecov free for public repos, all packages are OSS).

---

## Files to Create / Modify

### Phase 1 — Open Source Essentials (no tooling prereqs)

**`LICENSE`** [NEW]
```
MIT License

Copyright (c) 2025 YOUR_NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**`.editorconfig`** [NEW]
```ini
root = true

[*]
charset = utf-8
end_of_line = lf
indent_style = space
indent_size = 4
trim_trailing_whitespace = true
insert_final_newline = true

[*.{js,mjs,vue,ts,tsx,css,html,json,yml,yaml}]
indent_size = 2

[*.md]
trim_trailing_whitespace = false
indent_size = 2

[*.py]
indent_size = 4
max_line_length = 88
```

**`.github/ISSUE_TEMPLATE/bug_report.md`** [NEW]
```markdown
---
name: 🐛 Bug Report
about: Report a reproducible bug or unexpected behavior
title: "[Bug] "
labels: ["bug", "needs-triage"]
---

## Bug Description

## Steps to Reproduce
1. 
2. 

## Expected Behavior

## Actual Behavior
<!-- Include full error traceback if available -->

## Environment
| Item | Value |
|---|---|
| OS | e.g., Ubuntu 22.04 |
| Python version | e.g., 3.10.12 |
| YOLO version | e.g., ultralytics 8.x.x |
| GPU | e.g., RTX 3060 / CPU only |
| Browser (if frontend) | e.g., Chrome 124 |

## Logs
```
paste logs here
```
```

**`.github/ISSUE_TEMPLATE/feature_request.md`** [NEW]
```markdown
---
name: ✨ Feature Request
about: Suggest an enhancement or new capability
title: "[Feature] "
labels: ["enhancement"]
---

## Problem Statement

## Proposed Solution

## Alternatives Considered

## Use Case
```

**`.github/ISSUE_TEMPLATE/config.yml`** [NEW]
```yaml
blank_issues_enabled: false
contact_links:
  - name: 📖 Documentation
    url: https://github.com/YOUR_USERNAME/YOUR_REPO/tree/main/docs
    about: Read the docs first.
```

**`.github/PULL_REQUEST_TEMPLATE.md`** [NEW]
```markdown
## Summary
- 

## Type of Change
- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 📝 Documentation update
- [ ] 🔧 Chore / Refactor

## Related Issue
Closes #

## Testing
- [ ] Added/updated unit tests
- [ ] `pytest tests/ -v` — all pass
- [ ] `npm run build` in `frontend/` — succeeds
- [ ] Tested manually

## Checklist
- [ ] Code follows project style (Ruff + ESLint/Prettier)
- [ ] Documentation updated if needed
- [ ] Pre-commit hooks pass locally
```

**`CONTRIBUTING.md`** [NEW] — Key sections:
- Development setup (venv + npm install + pre-commit install)
- Branching strategy (main / feat/* / fix/* / docs/* / chore/*)
- Commit message convention (Conventional Commits)
- Code style: Ruff for Python, ESLint + Prettier for Vue
- PR process (branch → CI pass → review → merge)

---

### Phase 2 — Python Toolchain

**`requirements-dev.txt`** [NEW]
```
ruff>=0.4.0
black>=24.4.0
pytest>=8.0.0
pytest-cov>=5.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
pre-commit>=3.7.0
```

**`pyproject.toml`** [NEW] — Full content:
```toml
[tool.ruff]
line-length = 88
target-version = "py310"
exclude = [".git", ".venv", "__pycache__", "dist"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "SIM", "C4"]
ignore = ["E501", "B008", "UP007"]
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "PLR2004"]

[tool.ruff.lint.isort]
known-first-party = ["core", "backend"]

[tool.ruff.lint.flake8-bugbear]
extend-immutable-calls = ["fastapi.Depends", "fastapi.Query", "fastapi.Body"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "lf"

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "--verbose",
    "--tb=short",
    "--strict-markers",
    "--cov=core",
    "--cov=backend",
    "--cov-report=term-missing",
    "--cov-report=xml:coverage.xml",
    "--cov-fail-under=70",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests requiring running services",
    "gpu: marks tests requiring NVIDIA GPU",
]

[tool.coverage.run]
source = ["core", "backend"]
omit = ["*/tests/*", "*/scripts/*", "*/__init__.py", "backend/main.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
show_missing = true
```

After creating `pyproject.toml`, run to fix existing violations:
```bash
pip install -r requirements-dev.txt
ruff check . --fix
ruff format .
```

---

### Phase 3 — Frontend Toolchain

Install inside `frontend/` (compatible with Vue 3.4 + Vite 5, ESLint 9 flat config):
```bash
npm install -D \
  eslint@^9.13.0 \
  eslint-plugin-vue@^9.30.0 \
  @vue/eslint-config-prettier@^10.1.0 \
  prettier@^3.3.3 \
  @eslint/js@^9.13.0 \
  globals@^15.0.0
```

**`frontend/eslint.config.js`** [NEW] — Uses ESLint 9 flat config format:
```javascript
import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import prettierConfig from '@vue/eslint-config-prettier'
import globals from 'globals'

export default [
  { name: 'app/files-to-ignore', ignores: ['**/dist/**', '**/coverage/**'] },
  { name: 'app/files-to-lint', files: ['**/*.{js,mjs,vue}'] },
  js.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  {
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.es2022 },
    },
  },
  {
    rules: {
      'vue/multi-word-component-names': ['error', { ignores: ['index', 'default'] }],
      'vue/no-v-html': 'warn',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
  prettierConfig,  // MUST be last — disables rules conflicting with Prettier
]
```

**`frontend/.prettierrc.json`** [NEW]:
```json
{
  "semi": false,
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "trailingComma": "es5",
  "endOfLine": "lf",
  "htmlWhitespaceSensitivity": "ignore"
}
```

**`frontend/.prettierignore`** [NEW]:
```
dist/
coverage/
node_modules/
*.min.js
```

**`frontend/package.json`** [MODIFY] — Add to `"scripts"`:
```json
"lint": "eslint . --report-unused-disable-directives",
"lint:fix": "eslint . --fix --report-unused-disable-directives",
"format": "prettier --write \"src/**/*.{js,vue,css,json}\"",
"format:check": "prettier --check \"src/**/*.{js,vue,css,json}\""
```

After setup, fix existing violations:
```bash
npx eslint . --fix
npx prettier --write "src/**/*.{js,vue,css,json}"
npm run build  # verify build still works
```

---

### Phase 4 — Pre-commit Hooks

**`.pre-commit-config.yaml`** [NEW]:
```yaml
default_stages: [pre-commit]
fail_fast: false

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
        args: [--markdown-linebreak-ext=md]
      - id: end-of-file-fixer
      - id: mixed-line-ending
        args: [--fix=lf]
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: check-merge-conflict
      - id: no-commit-to-branch
        args: [--branch, main]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: prettier
        name: prettier (frontend)
        entry: bash -c 'cd frontend && npx prettier --write'
        language: system
        files: ^frontend/src/.*\.(js|vue|css|json)$
        pass_filenames: true
      - id: eslint
        name: eslint (frontend)
        entry: bash -c 'cd frontend && npx eslint --fix'
        language: system
        files: ^frontend/src/.*\.(js|vue)$
        pass_filenames: true
```

Then:
```bash
pre-commit install
pre-commit run --all-files  # must pass clean before CI
```

---

### Phase 5 — GitHub Actions CI/CD

**`.github/workflows/ci.yml`** [NEW]:
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint-python:
    name: Lint Python (Ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"
          cache-dependency-path: |
            requirements.txt
            requirements-dev.txt
      - run: pip install ruff
      - run: ruff check . --output-format=github
      - run: ruff format . --check

  lint-frontend:
    name: Lint Frontend (ESLint + Prettier)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run lint
      - run: npm run format:check

  test:
    name: Test (pytest + coverage)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"
          cache-dependency-path: |
            requirements.txt
            requirements-dev.txt
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ -v
      - if: always()
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: false
          token: ${{ secrets.CODECOV_TOKEN }}

  build-frontend:
    name: Build Frontend (Vite)
    runs-on: ubuntu-latest
    needs: [test, lint-frontend]
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run build
        env:
          NODE_ENV: production
      - uses: actions/upload-artifact@v4
        with:
          name: frontend-dist-${{ github.sha }}
          path: frontend/dist/
          retention-days: 7
```

> **Note**: After pushing, register the repo on [codecov.io](https://codecov.io) (free for public repos), then add `CODECOV_TOKEN` to GitHub repo Secrets.

---

### Phase 6 — README Enhancement

**`README.md`** [MODIFY — complete rewrite] with:

1. **Header**: Project title + bilingual subtitle + 7 badges (CI status, Codecov, License, Python, FastAPI, Vue, Docker, Ruff)
2. **Language toggle**: `[中文](#chinese) · [English](#english)`
3. **Features list**: 6 bullet points with emoji icons
4. **ASCII architecture diagram**: Shows Frontend → FastAPI → Core Engine + SQLite
5. **Quick Start**: Docker one-liner + local dev commands (5 steps each)
6. **Project structure**: Tree diagram matching actual layout
7. **Test coverage table**: 3 rows for test_config, test_geometry, test_rules
8. **Contributing section**: Links to CONTRIBUTING.md
9. **License section**: MIT badge + link
10. **English summary**: ~5 sentences for international audience

Badge URLs (replace `YOUR_USERNAME/YOUR_REPO`):
```markdown
[![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)](...)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO/branch/main/graph/badge.svg)](...)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](...)
[![Vue.js](https://img.shields.io/badge/Vue.js-3.4+-4FC08D?logo=vue.js&logoColor=white)](...)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](...)
```

---

## Implementation Order

```
Phase 1 (no prereqs):
  LICENSE → .editorconfig → .github/ISSUE_TEMPLATE/ → .github/PULL_REQUEST_TEMPLATE.md → CONTRIBUTING.md

Phase 2 (Python toolchain):
  requirements-dev.txt → pyproject.toml → pip install → ruff check . --fix → ruff format .

Phase 3 (Frontend toolchain):
  npm install (ESLint/Prettier packages) → eslint.config.js → .prettierrc.json → .prettierignore
  → update package.json scripts → eslint --fix → prettier --write → npm run build

Phase 4 (Pre-commit):
  .pre-commit-config.yaml → pre-commit install → pre-commit run --all-files (must pass clean)

Phase 5 (CI/CD):
  .github/workflows/ci.yml → push to GitHub → verify CI passes → register Codecov

Phase 6 (Documentation):
  README.md rewrite (with actual CI badge URLs once pipeline is green)
```

---

## Verification Checklist

| Check | Command | Expected |
|---|---|---|
| Python linting | `ruff check .` | 0 errors |
| Python formatting | `ruff format . --check` | 0 diffs |
| Frontend linting | `cd frontend && npm run lint` | 0 errors |
| Frontend formatting | `cd frontend && npm run format:check` | 0 diffs |
| All tests pass | `pytest tests/ -v` | All green |
| Coverage threshold | auto-checked by pytest-cov | ≥70% |
| Frontend build | `cd frontend && npm run build` | Exit 0 |
| Pre-commit | `pre-commit run --all-files` | All passed |
| CI badge | GitHub Actions UI | Green checkmark |
