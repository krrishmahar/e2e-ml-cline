# Security Rules

## 1. Secrets
- NEVER commit secrets.
- Use environment variables or secret stores.

## 2. Dependencies
- Run `pip audit` / `npm audit` pre-commit.
- Update outdated dependencies weekly.

## 3. Auth
- All services must use token-based auth.
- Check for any API Token explicitly written
