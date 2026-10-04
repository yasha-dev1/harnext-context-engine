# E2 engine: gpt-oss-20b on local Ollama

The small agent model for the context-engine evaluation. It runs locally on the Titan RTX (24 GB) through Ollama's
OpenAI-compatible endpoint. Everything below was measured on 2026-09-28 on this machine with Ollama 0.20.5 and
driver 595.84 / CUDA 13.2. The scripts and raw outputs are in the session scratchpad (`.../scratchpad/ollama/`).

## Model

| | |
|---|---|
| Tag pulled | `gpt-oss:20b` (official library tag) |
| Manifest / ID | `17052f91a42e` |
| Weights blob | `sha256:e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb` (13 793 422 144 B) |
| Template blob | `sha256:fa6710a93d78da62641e192361344be7a8c0a1c3737f139cf89f20ce1626b99c` (Harmony format, 172 lines) |
| Arch / params / quant | gptoss, 20.9B, MXFP4 |
| Native context | 131 072 |
| Capabilities | `completion`, `tools`, `thinking` |
| Default params | `temperature 1` |
| **Model to use** | **`gpt-oss-20b-ctx32k`**: a derived model (`FROM gpt-oss:20b` + `PARAMETER num_ctx 32768`). It shares the weights blob, so it adds no disk use. Its ID is `65086a4a68ab`. |

The template always adds its own system header ("You are ChatGPT… Knowledge cutoff: 2024-06, Current date: <today>,
Reasoning: <level>"). Our system prompt goes in as the Harmony *developer* message. Because the date is filled in
from the server clock, prompts that depend on "today" are not time-frozen.

## Endpoint and request parameters

- Base URL from the host: `http://localhost:11434/v1`. From a trial container: `http://172.19.0.1:11434/v1` (see
  Wiring). The API key can be any non-empty string.
- Recommended body: `model: "gpt-oss-20b-ctx32k"`, `reasoning_effort: "low"` (or `"medium"`), `temperature: 0`,
  `seed: <trial seed>`, `max_tokens: 4096` (always set it; see Quirks), `tools: [...]`. Do **not** use
  `response_format` (see Quirks).
- Streaming works. Add `stream_options: {"include_usage": true}` to get usage in the final chunk.

## Context window

- The effective default here is **32 768**. Ollama 0.20 picks the default from VRAM ("vram-based default context",
  24 GiB -> 32768) unless `OLLAMA_CONTEXT_LENGTH` is set. On a machine with less VRAM it would silently be smaller.
- `/v1/chat/completions` **ignores** `options.num_ctx` (tested: the loaded context stayed at 32768). The only
  per-request control is the native `/api/chat` with `options.num_ctx`, and every change of `num_ctx` reloads the
  model (~9.5 s). This is why we pin the value in a derived model:
  ```
  printf 'FROM gpt-oss:20b\nPARAMETER num_ctx 32768\n' > Modelfile.ctx32k
  ollama create gpt-oss-20b-ctx32k -f Modelfile.ctx32k
  ```
- gpt-oss uses sliding-window layers, so its KV cache is small. The whole model fits on the GPU at every size we tried:

| num_ctx | KV cache (Ollama) | GPU used (nvidia-smi, incl. ~0.8 GB desktop) | prompt fill | prompt eval | generation |
|---|---|---|---|---|---|
| 8 192 | 300 MiB | 13.75 GB | 6 928 tok | 1 743 tok/s | 84.0 tok/s |
| 16 384 | 492 MiB | 13.95 GB | 14 236 tok | 1 716 tok/s | 83.3 tok/s |
| 32 768 | 876 MiB | 14.39 GB | 28 343 tok | 1 565 tok/s | 77.5 tok/s |
| 65 536 | 1.6 GiB | 15.22 GB | 1 097 tok | – | 91 tok/s |
| 131 072 | 3.1 GiB | 16.94 GB | 1 097 tok | – | 99 tok/s |

The model found a needle placed at the start of the prompt at every fill level. The log reports "offloaded 25/25
layers to GPU" and `ollama ps` shows "100% GPU". About 1.1 GiB of weights (the token embedding table) stay in
system RAM by design.

**Recommendation:** use `num_ctx 32768` (through `gpt-oss-20b-ctx32k`) with `OLLAMA_NUM_PARALLEL=4`. The KV cache
is allocated as num_ctx × parallel. With 4 slots at 32k it is 3.4 GiB, 16.7 GiB in total and 17.1 GB on
nvidia-smi, which leaves about 7 GB of headroom.

## Throughput and concurrency

- Cold load: `load_duration` was 9.2–9.9 s, with the weights already in the page cache. The very first request after
  the server started took 23.7 s in total, 14 s of it one-off CUDA warm-up in prompt eval.
- A single stream with a warm model gets about 1 550–1 750 tok/s prompt eval and 77–99 tok/s generation. Time to
  first token (the first reasoning token) was 1.55 s for a 2.3k-token prompt.
- Concurrency test: a 2.3k-token prompt, low effort, max 1024 output tokens.

| server `OLLAMA_NUM_PARALLEL` | N concurrent | aggregate output tok/s | TTFT per request (s) | per-stream tok/s |
|---|---|---|---|---|
| 1 (default) | 1 / 2 / 4 | 75 / 79 / 78 | 1.6 / 0.3–13.4 / 0.4–39.6 (queued) | ~75 |
| 4 | 2 / 4 | 84 / 93–118 | 0.9–1.7 / 3.2–5.1 | ~29–42 |
| 8 | 8 | 130 | 2.6–8.3 | ~16 |

- **Practical limit: 4 parallel slots.** The default (1) queues requests: every extra concurrent request adds a full
  request's latency to TTFT. 8 slots × 32k does fit (KV 6.8 GiB, 21.0 GB on nvidia-smi, ~3.5 GB free), but it only
  raises aggregate throughput from ~118 to 130 tok/s while cutting each stream to ~16 tok/s, and it leaves little room
  for desktop GPU use. Requests beyond the slot count wait in Ollama's queue (`OLLAMA_MAX_QUEUE` 512), not fail. The
  harness should still cap in-flight model calls at about 4–8 so its timeouts measure the model, not the queue.

## Reasoning

- The reasoning text comes back in `choices[0].message.reasoning` (non-standard field; `thinking` on native
  `/api/chat`). It is never mixed into `content`: 0 leaks in 467 tool-loop responses.
- It is controlled by `reasoning_effort: "low"|"medium"|"high"`. The nested `reasoning: {"effort": ...}` form is
  also honoured, and the two gave identical results. Omitting the field means **medium**. `"none"` is accepted but
  reasoning still happens (gpt-oss cannot turn it off). `"minimal"` returns HTTP 400.
- `completion_tokens` **includes** reasoning tokens. The same arithmetic question (answer "184") cost:
  - low: 47–54 completion tokens, 0.8 s
  - medium: 152 tokens, 1.9 s
  - high: 292 tokens, 3.3 s
- High effort can run away. One trivial lookup question at high effort produced 7 868 tokens in 88.8 s. Use low
  (tool loops, median 49–58 completion tokens and 0.9–1.1 s per turn) or medium (median 95 tokens, 1.5 s, p90 3.0 s
  per turn).

## Tool calling (the `find` / `sql` loop)

The test script defined two tools, `find(need: string, values: string[])` and `sql(query: string)`. It asked 24
varied Kafka questions (PR/issue/reviewer/commit lookups) and returned tool results, allowing up to 6 turns per
question. That makes 5 configurations × 24 = 120 conversations, 467 responses and 371 tool calls.

- First turn: **120/120 produced a well-formed `tool_calls` entry** (`type: function`, an id, arguments as a JSON
  string). There were 0 answers given without calling a tool, and no tool calls leaked into the text.
- Arguments: **0 malformed JSON in 371 calls.** 8 calls (2.2 %) broke the schema: integers inside the
  `values: string[]` array (e.g. `[21002]`). Tool schemas are **not enforced**, so coerce values with `str()`
  before use.
- It never made parallel tool calls in one turn; it mostly chose `sql` over `find`.
- Continuing after a tool result, when the result settles the question ("no matching records"): the model gave a
  final answer in 24/24 conversations at low effort and 22/24 at medium. With canned results that did not fit the
  question it kept querying: low effort finished within 6 turns in 21/24 and 19/24 runs, medium in 10/24. The
  harness needs a turn cap.
- **One HTTP 500** in about 480 requests (medium effort, turn 4). The body was not captured and it did not happen
  again in 3 replays. Retry 5xx once.

## Structured output

`response_format: {"type": "json_schema", ...}` returned schema-valid `{"value", "evidence"}` in 20/20 cases, and
`json_object` works too. **However**, with a format set, Ollama runs the reasoning pass first and then a constrained
pass. The reported `usage` covers only the second pass: the reasoning comes back as **prompt** tokens and the
reasoning pass's completion tokens are dropped. Measured on the same question:

| effort | free-form usage | with `json_schema` |
|---|---|---|
| low | 172 in / 72 out | 197 in / 47 out |
| medium | 172 in / 324 out | 500 in / 47 out |
| high | 172 in / 7 868 out | 517 in / 18 out |

With a format set, output tokens are therefore **undercounted**. For exact costs, have the model give its final
answer as plain JSON text (or through an `answer` tool) and parse it, rather than using `response_format`.

## Token accounting and cost

- `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`) was present on **every** response: 467/467 in the
  tool loops, all plain and streaming calls (with `include_usage`), and the container test. There are no
  `completion_tokens_details` or `prompt_tokens_details`, so reasoning tokens are not broken out; they are included
  in `completion_tokens`.
- Counts are exact counts from the model's tokenizer, including Ollama's template header (about 70 tokens on a
  one-line prompt).
- API-equivalent cost is computed from these counts at the OpenRouter list price of `openai/gpt-oss-20b`:
  **$0.018 per million input tokens and $0.09 per million output tokens** (as of 2026-09-28):
  `cost = prompt_tokens × 0.018e-6 + completion_tokens × 0.09e-6`.

## Determinism

With `temperature: 0` + `seed`, 5 repeated calls returned identical content, reasoning and token counts. Across
separate batches it was **near** but not bit-exact deterministic: one of 3 low-effort repeats used 47 instead of 54
tokens. Batching with other parallel requests will also change results. With `temperature: 1` and no seed, 5 of 5
answers differed. Report results over several trials rather than relying on replay.

## Wiring from trial containers (for the integration stage)

Tested with throwaway `harnext-e2-agent` containers on `harnext-e2-net`. The proxy and the network were not changed.

- `harnext-e2-net` is `--internal`, subnet 172.19.0.0/16. The host is at the gateway address **172.19.0.1**
  (bridge `br-0c01ac641830`). A container on it **can reach a host process listening on 172.19.0.1** directly: this
  was tested with the openai client inside the agent image against a temporary forwarder
  172.19.0.1:11434 -> 127.0.0.1:11434, and `curl --noproxy '*' http://172.19.0.1:11434/api/version` also worked.
  ufw is active but did not block it.
- **Going through the proxy does not work and is not needed.** tinyproxy is default-deny and returns
  `403 Filtered` for `http://172.17.0.1:...`. Its `ConnectPort 443` also only allows TLS tunnels. Do not add
  Ollama to `allowlist.filter`.
- Required changes:
  1. Ollama must listen on the bridge. Use `OLLAMA_HOST=0.0.0.0:11434` (then block it on the LAN, see below), or
     `OLLAMA_HOST=172.19.0.1:11434`, in which case host-side scripts must also use 172.19.0.1 and the service must
     start after Docker has created the network.
  2. In `container.py`, `proxy_env()` and `mcp_env()`: add `172.19.0.1` to `NO_PROXY`/`no_proxy`, e.g.
     `"localhost,127.0.0.1,172.19.0.1"`. Without it the openai/httpx client sends the request to the proxy and
     fails. This was tested: `APIConnectionError` without the entry, a successful completion with it.
  3. Pass the agent `OPENAI_BASE_URL=http://172.19.0.1:11434/v1` and `OPENAI_API_KEY=ollama` (or whatever the agent
     CLI's config expects), plus model `gpt-oss-20b-ctx32k`. Derive the gateway from
     `docker network inspect -f '{{(index .IPAM.Config 0).Gateway}}' harnext-e2-net`; do not hard-code it, because
     it changes if the network is recreated. Adding `--add-host ollama:<gateway>` gives the containers a stable
     name.
  4. `verify_egress` / `verify-egress.sh` could gain one positive check: `curl --noproxy '*' <gateway>:11434/api/version`.
- Boundary note: any host process listening on 0.0.0.0 or 172.19.0.1 is reachable from trials. The Docker-published
  ports we tried (5432, 6379, 6333, 5173) were **not** reachable.

## Service state and host setup (needs root, not done)

At setup time the systemd unit `ollama.service` was **crash-looping** (restart counter above 49 000). It runs as
user `ollama`, whose home `/usr/share/ollama` does not exist ("mkdir /usr/share/ollama: permission denied"). The
models live in `/home/yasha/.ollama/models`, owned by yasha. For these measurements a **user-level** `ollama serve`
was started instead (`OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=4`). It will not survive a reboot. To make it
permanent, stop the user process and run as root:

```
sudo systemctl edit ollama      # creates /etc/systemd/system/ollama.service.d/override.conf
[Service]
User=yasha
Group=yasha
Environment="HOME=/home/yasha"
Environment="OLLAMA_MODELS=/home/yasha/.ollama/models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_CONTEXT_LENGTH=32768"
Environment="OLLAMA_KEEP_ALIVE=30m"
# then:
pkill -u yasha -f '^/usr/local/bin/ollama serve$'
sudo systemctl daemon-reload && sudo systemctl restart ollama
sudo ufw deny in on enp0s31f6 to any port 11434 proto tcp    # keep it off the LAN if binding 0.0.0.0
```

The alternative is to leave the unit alone and run the user-level server, after stopping the crash loop with
`sudo systemctl disable --now ollama`.

## Known quirks (summary)

1. `/v1` ignores `options.num_ctx`; pin the context with the derived model or `OLLAMA_CONTEXT_LENGTH`.
2. **Silent truncation.** A prompt longer than num_ctx is cut from the front (log: `truncating input prompt …
   keep=4`), and `usage.prompt_tokens` then reports the truncated count (32768), not what was sent. With the
   question kept but the context gone, the model then produced an empty answer. Without `max_tokens`, one such
   request ran for over 10 minutes. Always set `max_tokens` and treat `prompt_tokens >= num_ctx` as an overflow
   error. Native `/api/chat` with `"truncate": false` returns HTTP 400 "the input length exceeds the context
   length"; `/v1` ignores `truncate`.
3. `response_format` hides the reasoning tokens from `completion_tokens` (see Structured output).
4. Tool argument schemas are not enforced: integers can appear in string arrays.
5. `reasoning_effort` high can run away. `"minimal"` returns 400; `"none"` still reasons.
6. There was one unexplained HTTP 500 in about 480 requests.
7. Each change of `num_ctx` or parallel slots reloads the model (~9.5 s). Keep one model name and one server config
   for the whole run.
