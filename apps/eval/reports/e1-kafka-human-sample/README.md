# E1 human sanity sample

100 items = top-scored rule-negative events under R5 at 2 % (spec sheet A); 100 items = label-definition audit stratified over 6 disagreement cells (sheet B). Rows are shuffled; `annotation_sheet.csv` hides scores and labels. Each annotator fills their column with `yes` / `no` / `unsure` for: *Would you, as a Kafka committer, want to be interrupted for this event right now?* Score with `annotation_key.csv`: Cohen κ between annotators, then κ of the majority human label against `reg_model`, `ge2_outcome`, `ge_half` (sheet B) and the yes-rate by R5 score bucket (sheet A).

Disagreement cells (reg_model, ge2_outcome, ge_half) and their population sizes: {'000': 358366, '001': 11067, '010': 8465, '011': 2695, '111': 811, '101': 4}
