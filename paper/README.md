# Paper package

This directory is a publication-facing index for the completed AnnoMI experiments. It is a
manuscript scaffold, not evidence that a paper has been submitted, accepted, or archived. The
current full narrative is the [study report](../docs/research/STUDY_REPORT.md); the files here keep
the proposed paper structure and its claims tied to machine-readable evidence.

## Contents

- [OUTLINE.md](OUTLINE.md) gives a lean manuscript structure and assigns each table and analysis
  to a section.
- [CLAIM_EVIDENCE_CROSSWALK.md](CLAIM_EVIDENCE_CROSSWALK.md) records the exact result behind each
  candidate claim and the status of that evidence.
- [references.bib](references.bib) contains the core dataset, modelling, disagreement, transition,
  and uncertainty references.
- The [literature matrix](../docs/LITERATURE_MATRIX.md) explains how those references position this
  study without asserting priority.

## Evidence packages

The manuscript should report values from the compact publication tables, not from copied prose or
training logs:

- [source-disjoint classification and label distributions](../results/research/publication_v1/)
- [early-quality and next-action study](../results/research/publication_ac_v1/)
- [exploratory and post-hoc Task A/C extensions](../results/research/publication_safe_mi_v2/)

Each directory contains a manifest that binds its tables to source-result and builder hashes. The
underlying out-of-source prediction ledgers remain in `results/research/` and are checked by the
repository validator.

## Reporting discipline

Use the following distinctions consistently:

- **commit-locked** means that a local Git commit fixed the stated protocol before the relevant
  evaluation. It does not mean public preregistration.
- **development-consumed** means that earlier outcomes had already influenced the question,
  architecture, or analysis.
- **exploratory** identifies the SAFE-MI v2 search conducted after all AnnoMI outer-fold outcomes
  had been inspected.
- **post-hoc** identifies the separately recorded SAFE-MI v2.1 extension.
- **quarantined** identifies external sample records excluded by the fixed overlap rules.

The paper may claim supported improvements only where the paired interval and the prespecified gate
support them. Numerical leaders, secondary metrics, failed gates, and cross-run differences must
retain those labels. Results concern public demonstration dialogue under the recorded split design;
they do not establish clinical validity, treatment quality, safe intervention choice, or external
generalization.

Before submission, authors should add the final venue format, authorship and affiliation details,
an availability statement consistent with the upstream data terms, and any disclosure required by
the venue. Repository citation metadata is maintained separately in [CITATION.cff](../CITATION.cff).
