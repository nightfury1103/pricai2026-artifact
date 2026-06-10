# GitHub and Anonymous Mirror Release Steps

Use these steps after reviewing the clean local artifact.

## 1. Create a GitHub Repository

Create a new empty GitHub repository. Use a neutral name, for example:

```text
pricai2026-artifact
```

Do not initialize it with a README, license, or `.gitignore`; this local artifact already contains those files.

## 2. Push This Clean Repository

From this directory:

```bash
git remote add origin git@github.com:<github-username>/pricai2026-artifact.git
git push -u origin main
```

The local Git commit author is already set to:

```text
Anonymous Authors <anonymous@example.com>
```

The normal GitHub URL will still reveal the account owner, so do not put the normal GitHub URL in the PRICAI submission.

## 3. Create Anonymous GitHub Mirror

Open:

```text
https://anonymous.4open.science/
```

Paste the normal GitHub repository URL and add identity terms to redact:

```text
<author full name>
<author GitHub username>
<author emails>
<institution or lab name>
<company name, if any>
<internal project names that reveal identity>
```

Use the generated `anonymous.4open.science` URL in the submitted paper and supplementary material.

## 4. Suggested PRICAI Paper Text

```text
An anonymized implementation and reproduction instructions are available at:
<anonymous repository URL>
```

After acceptance, replace the anonymous URL with the final public GitHub URL.

