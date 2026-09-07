# R2 focus items (from the orchestrator's own inspection of run 20260830T193119Z)

Beyond re-checking your module against the spec and verifying the R1 "Fixes applied" sections are real, specifically investigate:

1. E1: `random_vus_at_prevalence` check fails — is the VUS-PR implementation correct, or is the check itself wrong at smoke sample sizes? R5 recall=1.0 while R0–R4,R6 all 0.0 on the smoke corpus: verify this is a legitimate consequence of the injected archetypes aligning with guards (and not gold leaking into R5's features or the situation-dedup path seeing labels). CIs are NaN at n=120 — confirm they become finite at the default 2000-event profile and that NaN is labelled, not silently dropped.
2. E5: confirm the validity gating (`invalid_reasons`) clears once E2's fixed checks pass on cadence stores, on the default profile; confirm `real_parquet` (pyarrow) is either added as a dependency or the JSON fallback is declared in the manifest.
3. E3: confirm its results are written and rendered in the report for the smoke profile (results file layout differed from e1/e2), and that S3−S4 now goes through the vector arm.
4. Everywhere: run `make eval-smoke` yourself and check the checks table in report.html — any check that is False must be either a real failure (report it) or an explicitly declared not-applicable-in-smoke with a reason in the results.
