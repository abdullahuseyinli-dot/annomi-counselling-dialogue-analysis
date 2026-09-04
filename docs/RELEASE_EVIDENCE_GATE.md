# Release evidence gate

A release is accepted only when every required item is verified against the exact candidate commit.
Existing research outputs are immutable inputs to this process; release preparation must not delete
or rewrite them.

## Present in the development candidate

- Commit-pinned, checksum-verified manifests for AnnoMI-simple and AnnoMI-full.
- Source-disjoint folds, machine-readable protocols, row-level probability ledgers, selections,
  calibration records, paired inference, failure records, and deterministic publication manifests.
- Explicit separation of supported, numerical-only, exploratory, post-hoc, and failed-gate claims.
- Raw-dialogue and checkpoint exclusions documented in repository policy.
- Tests, repository validation, packaging metadata, citation metadata, and third-party notices.

Presence does not substitute for rerunning the checks at the final release commit.

## Required at the candidate commit

- [ ] The working tree is clean and the release commit is identified.
- [ ] Package, citation, archive, and changelog versions agree; development suffixes are removed only
      for a real release.
- [ ] Tests, lint, formatting, type checks, repository validation, and package build checks pass in
      the supported environments.
- [ ] Both pinned AnnoMI downloads validate and `annomi-research validate` reconstructs all retained
      evidence.
- [ ] Publication builders reproduce byte-identical tracked tables, figures, and manifests.
- [ ] The built source and wheel archives contain no raw dialogue, checkpoints, caches, secrets,
      machine-specific paths, or unlisted top-level packages.
- [ ] Every manuscript number maps to a tracked artifact and retains its interval, gate, and
      exploratory/confirmatory qualification.
- [ ] Third-party notices, data-use terms, model revisions, and citations have been reviewed.
- [ ] A fresh clone follows the documented software and data-validation path.
- [ ] Hosted checks pass on the release commit.

## External and archival gates

- [ ] Create a signed or annotated version tag only after the candidate checks pass.
- [ ] Publish the exact tagged source archive and record its checksum.
- [ ] Add a DOI only after the archive service has issued it; do not reserve or guess one in advance.
- [ ] Record the release URL and archive identifier in citation metadata and the changelog.

Full MI-TAGS evaluation is not required to release the AnnoMI evidence package, but its absence must
remain explicit. If claimed as external confirmation, it becomes a separate mandatory gate: obtain
authorized full data, apply the frozen overlap quarantine and source split, meet the minimum group
count, make no held-out retuning, and retain the complete external evidence lineage.

## Current verdict

Research evidence is present; formal release and archive gates are pending. Full-corpus external
confirmation is blocked. Therefore the current state should be described as a development or
release candidate, not as an archived publication release.
