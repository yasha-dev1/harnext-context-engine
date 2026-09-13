# Native Codex: historical context versus no MCP

The same eight development questions ran through Codex CLI 0.154.0 with
`gpt-5.6-luna`, medium effort. This adds the missing no-context anchor to the
earlier MCP pilot. Both arms completed all eight tasks.

| Task family | Harnext MCP correct | No MCP correct |
| --- | ---: | ---: |
| Assignee at snapshot | Yes | No |
| Assignment history | Yes | No |
| Closed at snapshot | Yes | Yes |
| Priority at snapshot | Yes | No |
| Resolution at snapshot | Yes | No |
| Resolution history | Yes | No |
| Status at snapshot | Yes | No |
| Status history | Yes | No |
| Total | 8/8 | 1/8 |

Harnext helped on seven tasks, harmed on none, and tied on one. The no-MCP arm's
correct value was `false` for the binary Closed question, with no evidence.
Its other values were null, empty histories, Major, Fixed, and Open. These are
scored according to the unchanged typed schema; null/empty arrays are not a
separate abstention category. This tiny result cannot establish a population
effect or exclude lucky guesses or training-data knowledge.

| Observed resource metric, excluding capability probes | MCP | No MCP |
| --- | ---: | ---: |
| MCP calls | 25 | 0 |
| Returned context bytes | 134,759 | 0 |
| Required evidence exposure | 100% on all tasks | 0% |
| Reader wall time, summed | 256.9 s | 99.0 s |
| Native input tokens | 467,895 | 54,890 |
| Cached input tokens, already included in input | 253,952 | 0 |
| Native output tokens | 3,491 | 3,686 |

Reader time includes startup. Native token sums span multiple reader turns;
they are not unique context bytes. Cache use differs between sequential runs,
so this pilot is not a controlled cold/warm latency or pricing experiment.
Retrieval-algorithm recall/precision for the no-MCP arm is not applicable.

## Matching and isolation checks

The task IDs, frozen benchmark hash, question core/cutoff, answer schemas,
model, effort, and CLI version match. The no-MCP prompt replaces only the MCP
availability instruction with a question/existing-knowledge-only instruction.
The actual prompts and their hashes are retained. These were sequential arms
with one trial per task; the expanded study must block/randomize condition order
and repeat trials before confirmatory inference.

The no-MCP launcher uses `--ignore-user-config`, registers no MCP servers,
disables native tools, and rejects any observed MCP/native tool attempt. Every
recorded native event passed that allowlist. The negative native-file canary
also passed. This is configuration/event enforcement plus a behavioral test,
not a formal operating-system isolation proof. No store was built for this arm.

- [Audited paired results and exact answers](../reports/e2-native-context-baseline-20260912.json)
- [No-MCP YAML](../configs/benchmark-codex-no-mcp-pilot.yaml)
- [Full experimental map](../E2-EXPERIMENT-MAP.md)
- Local native transcripts: `apps/eval/out/benchmark-pilot/codex-luna-medium-no-mcp-001`

Validation: 55 E2 tests passed; targeted Ruff and Pyright checks passed. New
tests verify prompt preservation, refusal of MCP calls in the no-context arm,
and absence of MCP registration/config access in its native launch.

Next scientific anchor: raw searchable historical records with the same cutoff
and response allowance. N versus H demonstrates access value on these tasks;
R versus H is needed to evaluate Harnext organization beyond raw data access.
Coding no-MCP must keep the same source workspace, visible tests, and isolated
shell as its MCP counterpart. Coding execution admission remains pending.
