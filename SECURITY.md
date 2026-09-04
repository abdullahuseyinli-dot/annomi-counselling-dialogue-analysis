# Security policy

## Scope

Security reports are welcome for the Python package, command-line interface, dependency handling,
download verification, and accidental exposure of restricted or local data. Model-quality disputes
and requests for clinical deployment are research questions, not security vulnerabilities.

This software is not a clinical system and must not be used for diagnosis, triage, treatment
selection, or live intervention. No trained checkpoint is released for deployment.

## Reporting

Use a private security advisory on the repository host when that facility is available. If no
private channel is available, open a minimal issue requesting maintainer contact. Do not include
dialogue text, access tokens, local paths, exploit details, or other sensitive material in a public
issue.

Include the affected revision, environment, reproduction steps, impact, and the smallest safe proof
of concept. The maintainer should acknowledge the report, reproduce it where possible, prepare a
fix, and coordinate disclosure. No response-time guarantee is made for this research project.

## Data exposure

Raw AnnoMI and MI-TAGS files belong under ignored `data/raw/` paths. Checkpoints and embeddings
belong under ignored artifact directories. If such material is committed or published, stop using
the affected revision, preserve the incident evidence, and contact the maintainer privately. Git
history must be treated as already disclosed until any hosting and cache remediation is complete.

Dependency vulnerabilities should identify the package, resolved version from `uv.lock`, advisory,
and whether the affected code path is exercised here.
