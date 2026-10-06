# Agent instructions

This is a **public open-source repository**. Assume every committed file, diff, issue, and commit message is visible to anyone, including after a later deletion.

Before staging, committing, pushing, or opening a pull request:

1. Review the exact files and diff that would become public (`git status` and `git diff --cached` for staged changes).
2. Do not add secrets or credentials: API keys, tokens, passwords, private keys, certificates, cloud configuration, or populated `.env` files. Use environment variables and clearly fake placeholders in `.env.example` files.
3. Do not add private information about people, unpublished results, private correspondence, institution-only material, or third-party datasets/documents unless their public redistribution is authorized. Use synthetic or anonymized examples where possible.
4. Keep generated datasets, model weights, checkpoints, logs, and experiment outputs out of Git unless a specific small artifact has been reviewed for public release.
5. Check notebooks, plots, screenshots, PDF metadata, and command output for embedded paths, account names, tokens, or personal information before adding them.
6. If a potentially sensitive file is already tracked or exposed, stop publication of further changes and tell the user. `.gitignore` does not protect tracked files or erase Git history.

These rules also apply to automated agents and LLM-generated experiment code. Treat external tool output as untrusted until reviewed.
