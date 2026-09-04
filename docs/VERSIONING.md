# Versioning and evidence lineage

## Software versions

The Python package follows semantic versioning while it remains pre-1.0:

- patch: compatible corrections that do not change a recorded result or public contract;
- minor: new commands, analyses, or compatible evidence readers;
- major: incompatible command, schema, or interpretation changes.

Development candidates should use an explicit development or prerelease suffix. A final version is
written to package and citation metadata only when a matching tag and release evidence package are
ready. `date-released` and DOI fields must describe an event that has actually occurred.

## Protocol versions

Experiment protocols have independent identifiers such as `protocol_v1`, `safe_mi_v2`, and
`safe_mi_v2_1`. Their version communicates a scientific design boundary, not the installable package
version. A protocol is frozen before the outcomes it governs are inspected. Post-hoc work receives
a new identifier and is labeled accordingly.

## Result versions

Completed result directories are immutable once written. If code, dependencies, data, split, seed
policy, or hardware changes produce a different payload, retain the old output and create a new lineage. Each
new lineage should record:

- governing protocol and configuration hashes;
- source-data manifests and split hash;
- code commit and dependency lock hash;
- runtime environment and fixed seeds;
- row-level predictions, selection trace, aggregate summary, and comparison to the prior lineage;
- failures, fallbacks, calibration, and acceptance-gate outcome.

Publication tables and figures are derived artifacts. Their manifests bind source inputs, output
hashes, and builder code; they do not create a new scientific result by themselves.

## Tags and releases

Commit `e3ff100` marks the earlier portfolio state, and `dash-mi-protocol-v1` marks the DASH-MI
protocol commit. Neither is a tag for the current software
candidate. A future release tag must point to the exact verified commit and must not be used to hide
or replace those historical lineages.
