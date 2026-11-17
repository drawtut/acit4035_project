# Research data and repository hygiene

Do not commit raw participant sessions or any identifying information to this repository. This includes `left_raw.csv`, `right_raw.csv`, processed gait traces, participant ZIPs, painted-footprint photographs, videos, raw sensor metadata tied to trial dates and identifiable filenames. **Even five anonymous trial curves can sometimes be identifying in context.** Keep private experimental data in an approved research storage location, subject to your study consent and university policy.

- `.gitignore` excludes common data/results/media patterns, but it **does not undo previously tracked history**. Always inspect `git status` and `git ls-files` before pushing.
- Saved calibration JSON in the repository is a demonstration of the original frozen test setup. The results assume the physical mount and calibration correspond to the sensor/shoe pair; calibration must be validated again if the shoe or mount changes.
- No participant videos, data sessions or their previously committed Git objects are included in the prepared source package.
- Do not upload unpublished or participant-identifiable measurements into GitHub Issues/PRs.
- Do not describe CRP as independently validated; no optical/synchronized reference phases were available in the existing archive.
