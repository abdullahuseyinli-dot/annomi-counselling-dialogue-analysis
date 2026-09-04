# Limitations

## Data validity

- AnnoMI contains public demonstration dialogues, not a representative sample of routine clinical
  care. Performance on its labels does not establish therapeutic benefit or safety.
- The 133 transcripts reduce to 119 normalized video sources. Source-disjoint evaluation addresses
  direct source reuse but cannot remove all speaker, production, transcription, or topic effects.
- The multi-annotator study has only seven transcript clusters. Cluster-level effects and intervals
  are more informative than small rank differences.
- No demographic metadata supports a meaningful subgroup-fairness analysis.
- Raw dialogues are not redistributed, and the current upstream data and source-media terms must be
  reviewed independently before reuse.

## Target validity

- Therapist-behaviour codes are corpus annotations, not diagnoses or outcome measures.
- Task A predicts uploader-designated high/low demonstration metadata. It is not an independent
  MITI score or a measure of a therapist's real-world quality.
- Task C predicts the next observed therapist action. It does not identify an appropriate, safe, or
  optimal intervention.
- Prediction sets quantify uncertainty among four recorded action labels only.

## Statistical and modelling limits

- The highest observed macro-F1 is not automatically a supported pairwise improvement. Causal
  RoBERTa, DASH-MI, PANEL-MI, Q-TRACE-MI, and SAFE-MI results retain their recorded uncertainty and
  gate outcomes.
- SAFE-MI and its extension were designed after earlier AnnoMI outcomes were available. Their Task
  A/C results are exploratory or post-hoc and need genuinely independent confirmation.
- Multiple secondary metrics describe calibration and error shape; they are not independent
  discoveries.
- Source-cluster bootstrap intervals describe variation under the recorded resampling design and
  do not prove transportability to a new population.
- A single recorded GPU environment establishes one successful execution path, not hardware
  portability or deterministic equality across accelerator stacks.

## External validity and use

The MI-TAGS public samples are insufficient for performance evaluation: 9 of 12 records trigger the
locked possible-overlap quarantine. Full-corpus confirmation remains blocked pending authorized
access and may still fail the minimum independent-group requirement.

No result supports clinical diagnosis, triage, treatment selection, therapist ranking, employment
decisions, live feedback, or autonomous conversational control. The repository makes no
state-of-the-art or clinical-deployment claim. See the [model card](MODEL_CARD.md),
[data card](DATA_CARD.md), and [project status](PROJECT_STATUS.md) for the corresponding usage and
release boundaries.
