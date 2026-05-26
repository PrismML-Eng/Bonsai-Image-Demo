# CI Workflows

## Workflow

| Workflow | Auto-runs | PR label | Manual trigger |
|----------|-----------|----------|----------------|
| **Platform smoke tests** | No | `smoke-test` | Actions › Platform smoke tests › Run workflow |

## PR Labels

Add `smoke-test` to a PR to fire the matrix below. Only triggers on label-add
(not on push). To re-run after new commits, remove and re-add the label.

## Jobs

| Job | Runner | What it covers |
|-----|--------|----------------|
| `macos-apple-silicon` | `macos-14` (GitHub-hosted) | Full e2e: `setup.sh` → download ternary model → `generate.sh` small image → `serve.sh` smoke (`/backends` + frontend root, then stop). |
| `linux-x86-setup` | `ubuntu-latest` (GitHub-hosted) | Setup-only: verifies `setup.sh` succeeds, `nodejs-wheel-binaries` lands `node`/`npm` in `.venv/bin`, Python imports resolve. No generation (mflux needs MLX, MLX needs Apple Silicon or CUDA — neither here). |
