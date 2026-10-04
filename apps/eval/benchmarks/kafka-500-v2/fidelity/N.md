# Arm `N` — closed-book floor

Source: Zhou et al. 2607.26637 §3 "Memory variants": *"Closed-book answers each question directly,
with no store and no retrieval, measuring the floor the model reaches on its own knowledge."*
Shared decisions: see [README.md](README.md).

| paper says | what the code does | deviation (and why) |
|---|---|---|
| No store, no retrieval. | `arms.yaml:12` `{id: N, access: none}`; `runner.py:195-196` refuses a `store` for `access: none`; no `/context`, `/store` or `/sources` mount (`container.py:117-124`). | none |
| The model answers from its own knowledge. | Prompt paragraph `runner.py:52-53`: *"No project-history context is provided … Work from the task statement{workspace} and your own knowledge; there is no network access."* | none |
| (Paper's tasks are conversational; no code.) | Coding tasks: `N` still gets the pruned git workspace and `git log/blame/show` (`runner.py:67-70`, identical wording for every arm) and offline Gradle. | Deliberate: the workspace is part of the task, not of the store. `N` is therefore "no memory", not "no information" — git history up to the base commit is visible to it. |
| Floor is a measurement, not a comparator. | E2-V2-DESIGN "Genuine baseline": `N` is *"a diagnostic, not the baseline the stores are compared against. Primary contrasts are store arm − S-tools."* | none — but note the paper's primary contrast IS against closed-book; this benchmark's is against `S-tools`. |
| Contamination is not discussed (chat data is private). | Kafka is public: JIRA/GitHub up to the reader model's training cutoff may be memorised. `N` is what exposes this: a high `N` score on a historical family means the family is answerable from pretraining. | Reason the arm is kept at all. Live github.com / issues.apache.org are blocked for every arm (`container.py:46-50`), so the only leak path is the model's weights. |

## What an examiner should know

- `N` runs on every task (500 × agent), the same as any store arm; nothing is subsampled.
- Grading is identical to the other arms; a `null` value is allowed by the answer contract
  (`runner.py:71-74`) and scores as wrong, so `N` cannot gain by abstaining.
- Because `N` sees the repository at the base commit, coding tasks whose fix is inferable from the
  code alone (typos, obvious null checks) will be solved by `N`; the store arms' lift is only
  measured on what git + knowledge do not give. That is intended.
- A per-family `N` score near a store arm's score is the contamination flag; the report should
  show `N` per family before any store-vs-store claim.
- `N` has no store manifest, no builder cost, no checkpoint; `preflight` only checks the workspace
  for it.
