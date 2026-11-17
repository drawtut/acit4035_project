# Update an existing GitHub main branch safely

The supplied development ZIP contained its own `.git` directory, old scripts, data, and experimental assets. **This GitHub-ready ZIP contains no `.git` directory intentionally.** Do not copy the old `.git` folder into the new package and do not use `git push --force`.

The cleanest safe path is a fresh clone of the intended remote repository, retaining existing history:

1. `git clone <YOUR_REPO_URL> acit4035-main`
2. `cd acit4035-main`
3. `git checkout main && git pull --ff-only origin main`
4. Make a local safety branch: `git branch backup/pre-cleanup-2026-11-17`
5. Copy this package's contents into your checkout, and remove the old development files **after comparing the tree**. Keep `.git` untouched.
6. Review: `git status --short`, `git diff --stat`, `git diff --cached --name-status`, and `git ls-files` for old sessions, videos or other private paths.
7. Stage intended source-only changes: `git add -A`, inspect them, then `git commit -m "Consolidate FPA and bilateral CRP implementation for Windows and Pi"`.
8. `git push origin main` (no force).

**Important:** `.gitignore` will not remove anything already tracked. If the history already includes large recordings or other personally identifying files, simply removing them in the latest commit does NOT remove them from prior history. Before publishing a previously private repository, stop and evaluate whether those historical Git objects require a separately approved history rewrite or a completely new, clean repository. Do not automatically rewrite history in a shared group project.

If clean new-history publication is approved instead, start a new empty Git repository from this package, `git init -b main`, `git add .`, `git commit`, then add a remote and push. Never overwrite an existing remote main without confirming the history strategy with collaborators.
