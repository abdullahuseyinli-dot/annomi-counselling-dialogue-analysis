# Project status

## Current assessment

The repository includes the result summaries and supporting files needed to verify the reported
AnnoMI experiments. It has not yet been archived or assigned a DOI, and full MI-TAGS evaluation
requires access to the complete corpus.

| Area | Status | Evidence or next condition |
|---|---|---|
| Source-disjoint sparse and RoBERTa benchmark | Complete | Out-of-source ledgers, paired inference, and publication manifest are tracked |
| DASH-MI context experiment | Complete negative result | DASH-MI did not meet the primary or ablation criteria fixed before evaluation |
| Seven-transcript vote-distribution study | Complete | Soft-linear result supported; PANEL-MI primary gate failed |
| Task A/C and Q-TRACE-MI study | Complete negative joint result | The protocol was fixed before evaluation; Task A gain is uncertain and Task C degrades versus C-only |
| SAFE-MI staged and post-hoc experiments | Complete exploratory result | Task A numerical gain and Task C calibration trade-offs retained without superiority claims |
| MI-TAGS public-sample overlap audit | Complete | 9 of 12 records quarantined as possible overlap; no performance result permitted |
| Full MI-TAGS confirmation | Blocked | Requires authorized complete corpus and the frozen external protocol |
| Formal software release | Pending | Rerun release gates at the candidate commit, assign matching version/tag, and publish an archive |
| Archival DOI | Pending | Must come from an actual completed deposit; none is claimed here |

## Main findings

The strongest supported findings are the source-disjoint target-only RoBERTa improvement over
TF-IDF and soft-linear vote-distribution improvement over the transcript prior. The numerical
classification maximum, exploratory Task A result, calibrated Task C trade-offs, and unsuccessful
candidate gates are reported with their uncertainty and calibration qualifications.

The study uses source-level leakage control, nested selection, fixed-seed neural ensembles, cluster
inference, full probability ledgers, disagreement modelling, and a pre-evaluation overlap
quarantine. These measures improve traceability but do not establish clinical validity or external
generalisability.

## Before submission or release

Complete the [release evidence gate](RELEASE_EVIDENCE_GATE.md), freeze the claim-to-artifact mapping
at the submission revision, review third-party data terms, and record the actual archive identifier
only after deposit. Any external result must use independent, non-quarantined data without retuning
on the held-out partition.
