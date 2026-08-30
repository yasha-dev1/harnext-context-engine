# E6 execution profiles

The default registered E6 run uses `BenchmarkConfig.smoke()`: deterministic,
offline, in-process, short, and backed by the fake constant-time harness. Its
`support_status.json` marks real-corpus scale, full wall-clock duration, a
separate load host, and Kafka as `supported-not-run`.

The full path is explicit and never selected accidentally:

```bash
uv run python -m harnext_eval.e6 \
  --profile research \
  --config apps/eval/configs/e6-twolane.yaml \
  --replay /data/kafka-H1.jsonl \
  --situations-json /data/orgforge-situations.json \
  --out /results/e6 \
  --seed 1 \
  --kafka-bootstrap-servers kafka.internal:9092 \
  --kafka-output-topic cms.e6.telemetry.v1 \
  --load-generator-host loadgen-02.internal \
  --kafka-telemetry-path /results/broker-telemetry.jsonl
```

`BenchmarkConfig.research()` selects the full `{8,32}` partitions × `{1,4}`
workers matrix, all shapes and both knee loads, cardinalities `{8,32}`, 20-minute
runs with a 10-minute burst, three repetitions, and 10,000 entity-clustered BCa
resamples. Duration, repetitions, and resamples are also explicit CLI overrides
for staged deployments.

The output topic must emit JSON objects keyed by unique `event_id` with `lane`,
`agent_start_ts`, optional `service_start_ts` / `service_end_ts`, `partition`,
`partition_lag`, and, for batch commits, `window_close_ts` and `commit_ts`.
Duplicate records remain duplicate observations and cannot terminate collection;
the runner waits for every unique expected ID or the configured timeout.
Broker telemetry is JSONL with timestamped `cpu_pct`, `disk_util_pct`, and
`disk_io_bytes` samples. Research mode rejects localhost as a load-generator
host and requires telemetry.

Each input record also carries Kafka headers `e6-lane-design`, `e6-partitions`,
`e6-workers`, `e6-budget-pct`, and the three effective guard settings. The
deployed consumer/router must apply these cell settings and echo the resulting
lane in its output record.

Construction gold belongs in the situations sidecar under
`injected_situations`; it is never copied into event payloads. When no sidecar
is supplied, the deterministic smoke catalogue is used.
