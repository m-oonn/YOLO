# Contributing to YOLO Course Design

Thank you for your interest in contributing to the YOLO Course Design project!

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/m-oonn/YOLO.git
   cd YOUR_REPO/course-design
   ```

2. **Setup Python Environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate

   pip install -r requirements.txt -r requirements-dev.txt
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Running Locally

**Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

**Run Tests:**
```bash
pytest tests/ -v
```

## Branching Strategy

We follow a simplified Git Flow:

- `main` - Production-ready code
- `develop` - Development branch
- `feat/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates
- `chore/*` - Maintenance tasks

```bash
# Create a feature branch
git checkout -b feat/new-detection-rule

# Make changes and commit
git add .
git commit -m "feat: add new detection rule"

# Push and create PR
git push origin feat/new-detection-rule
```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Code style (formatting)
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Maintenance

**Examples:**
```
feat(rules): add crowd detection rule
fix(pipeline): resolve memory leak in frame processing
docs(readme): update installation instructions
test(api): add tests for event endpoints
```

## Code Style

### Python

We use Ruff for linting and formatting:

```bash
# Check for issues
ruff check .

# Format code
ruff format .
```

### Frontend (Vue.js / JavaScript)

We use ESLint + Prettier:

```bash
# Check for issues
cd frontend
npm run lint

# Format code
npm run format
```

### Pre-commit Hooks

Before each commit, the following checks run automatically:

- Ruff linting and formatting (Python)
- ESLint and Prettier (Frontend)
- Trailing whitespace removal
- YAML/JSON validation

To run manually:
```bash
pre-commit run --all-files
```

## Pull Request Process

1. **Create PR** from your branch to `develop`
2. **Fill PR Template** completely
3. **Ensure CI Passes** - All checks must green
4. **Review** - At least one approval required
5. **Merge** - Squash and merge to `develop`

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated if needed
- [ ] All tests pass locally
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts

## Reporting Issues

See [Bug Report Template](../.github/ISSUE_TEMPLATE/bug_report.md) for detailed instructions.

## Questions?

- Open a [GitHub Discussion](https://github.com/YOUR_USERNAME/YOUR_REPO/discussions)
- Check [Documentation](./docs/)
