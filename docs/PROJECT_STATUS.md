# Project status

## Current assessment

The repository is a local publication candidate with complete retained AnnoMI experiment evidence.
It is not yet a formal archived release: no matching release tag, DOI, or public archive deposit is
recorded, and full external MI-TAGS confirmation is blocked by data access.

| Area | Status | Evidence or next condition |
|---|---|---|
| Source-disjoint sparse and RoBERTa benchmark | Complete | Out-of-source ledgers, paired inference, and publication manifest are tracked |
| DASH-MI context experiment | Complete negative result | Commit-locked primary and ablation gates did not establish improvement |
| Seven-transcript vote-distribution study | Complete | Soft-linear result supported; PANEL-MI primary gate failed |
| Commit-locked Task A/C and Q-TRACE-MI study | Complete negative joint result | Task A gain uncertain; Task C degrades versus C-only |
| SAFE-MI staged and post-hoc experiments | Complete exploratory result | Task A numerical gain and Task C calibration trade-offs retained without superiority claims |
| MI-TAGS public-sample overlap audit | Complete | 9 of 12 records quarantined as possible overlap; no performance result permitted |
| Full MI-TAGS confirmation | Blocked | Requires authorized complete corpus and the frozen external protocol |
| Formal software release | Pending | Rerun release gates at the candidate commit, assign matching version/tag, and publish an archive |
| Archival DOI | Pending | Must come from an actual completed deposit; none is claimed here |

## Defensible contribution

The strongest supported findings are the source-disjoint target-only RoBERTa improvement over
TF-IDF and soft-linear vote-distribution improvement over the transcript prior. The numerical
classification maximum, exploratory Task A result, calibrated Task C trade-offs, and unsuccessful
candidate gates are reported with their uncertainty and calibration qualifications.

The methodological package adds source-level leakage control, nested selection, fixed-seed neural
ensembles, cluster inference, full probability ledgers, disagreement modelling, explicit negative
results, and a pre-evaluation overlap quarantine. That combination makes the work auditable; it does
not establish novelty priority, state of the art, clinical validity, or transportability.

## Before submission or release

Complete the [release evidence gate](RELEASE_EVIDENCE_GATE.md), freeze the claim-to-artifact mapping
at the submission revision, review third-party data terms, and record the actual archive identifier
only after deposit. Any external result must use independent, non-quarantined data without retuning
on the held-out partition.
