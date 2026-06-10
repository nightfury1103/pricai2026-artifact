# Anonymous Review Checklist

Use this checklist before creating the anonymous repository mirror.

## Repository Content

- No author names, emails, personal usernames, affiliations, acknowledgements, or lab names.
- No local paths that reveal identity or machine names.
- No `.git` history copied from the development repository.
- No generated files with author metadata, editor state, or local review comments.
- No private datasets, checkpoints, API responses, credentials, or API keys.
- No personal GitHub URL in the submitted paper.

## Anonymous GitHub Mirror

1. Push this clean repository to GitHub.
2. Open `https://anonymous.4open.science/`.
3. Paste the GitHub repository URL.
4. Add identity terms to anonymize, including:

```text
<author full name>
<author GitHub username>
<author email>
<institution or lab name>
<internal project names that reveal identity>
```

5. Use only the generated anonymous mirror URL in the PRICAI submission.
6. After acceptance, replace the anonymous mirror URL with the final public GitHub URL.

## Suggested Paper Wording

```text
An anonymized implementation and reproduction instructions are available at:
<anonymous repository URL>
```

