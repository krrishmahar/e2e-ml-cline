# Architecture Rules

## 1. Project Layout
- Use clean folder structure:
  - src/
  - src/models/
  - src/data/
  - src/pipelines/
  - src/services/
  - configs/
  - tests/

## 2. Environment
- `.env` → config only
- DO NOT hardcode secrets.

## 3. Scaling
- Pipelines must be stateless.
- Use message queues for async jobs.

## 4. Deployment
- Oumi Deployment must expose:
  - health check endpoint
  - metrics endpoint
  - logs in JSON

