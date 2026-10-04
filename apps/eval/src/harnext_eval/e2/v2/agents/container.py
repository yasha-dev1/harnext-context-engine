"""The trial boundary: one container per agent run on an internal Docker network whose only
way out is an allowlisting forward proxy. The agent CLI's own sandbox is off inside it.

Gradle: the shared dependency cache is mounted READ-ONLY at /gradle-ro and consumed through
Gradle's read-only dependency cache (GRADLE_RO_DEP_CACHE); GRADLE_USER_HOME=/gradle is a
per-trial writable directory seeded with only the wrapper distribution the base commit needs.
An agent therefore cannot poison the cache other arms and the grader resolve from, trials
can run concurrently, and nothing multi-gigabyte is copied per trial.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from harnext_eval.e2.benchmark import ROOT
from harnext_eval.e2.v2.errors import PauseRun
from harnext_eval.e2.v2.mcp_server import ENGINE_TOOL_TIMEOUT_S, READER_TIMEOUT

IMAGE = "harnext-e2-agent"
PROXY_IMAGE = "harnext-e2-proxy"
NETWORK = "harnext-e2-net"
PROXY = "harnext-e2-proxy"
PROXY_URL = f"http://{PROXY}:8888"
DOCKER_DIR = ROOT / "docker/e2-agent"
GRADLE_CACHES = ROOT / "out/kafka-admission-v2"  # gradle-cache-<slot>, written by admission


def gradle_cache(dist=None, root=None):
    """The admission cache to hand out read-only: among the per-slot user homes that hold the
    wrapper distribution, the fullest dependency cache. Read-only mounts make sharing safe
    (Gradle's cross-process locks do not span containers, so a WRITABLE home is never shared)."""
    homes = [h for h in sorted((root or GRADLE_CACHES).glob("gradle-cache*")) if h.is_dir()]
    if dist:
        homes = [h for h in homes if any(h.glob(f"wrapper/dists/{dist}-*/*/{dist}"))]
    size = lambda h: sum(1 for _ in (h / "caches/modules-2/files-2.1").glob("*"))  # noqa: E731
    return max(homes, key=size, default=None)


MODELS = ROOT.parents[1] / ".harnext/models"  # fastembed ONNX models the rag/graphiti readers load
CLAUDE_AUTH = Path.home() / ".claude/.credentials.json"  # the Linux CLI's OAuth login
CLAUDE_AUTH_IN = "/home/agent/.claude/.credentials.json"
MCP_TOOL_TIMEOUT_MS = (READER_TIMEOUT + 60) * 1000  # one context_research call can run that long
ENGINE_TOOL_TIMEOUT_MS = ENGINE_TOOL_TIMEOUT_S * 1000  # kind engine: every redo + health waits (mcp_server)


def tool_timeout_ms(kind, preset=None):
    """The agent CLI's MCP tool timeout: research kinds 240 s; kind engine from the preset's research
    deadline (mcp_server.engine_policy): 678 s at 120 s (openrouter, ollama), 674 s at 240 s (-b8)."""
    if kind != "engine":
        return MCP_TOOL_TIMEOUT_MS
    if preset in (None, "openrouter"):
        return ENGINE_TOOL_TIMEOUT_MS
    from harnext_eval.e2.v2.engine.config import for_endpoint
    from harnext_eval.e2.v2.mcp_server import engine_policy

    return engine_policy(for_endpoint(preset).deadline_s)["tool_timeout_s"] * 1000
LIMITS = ["--memory", "10g", "--cpus", "4", "--pids-limit", "4096"]
# openrouter.ai (2026-09-28): kind engine's models (Jev decisions + gpt-oss-20b chat). The proxy
# allowlist is ONE list for every trial container, so every arm can reach it (with no key).
ALLOWED = ("https://api.openai.com/", "https://api.anthropic.com/", "https://openrouter.ai/")
# Kind engine: the export's scratch database lives on a size-capped tmpfs (>= 2x the export); the
# provider key reaches the MCP server as a per-trial file the server deletes once read.
ENGINE_SCRATCH = ("--tmpfs", "/scratch:rw,nosuid,nodev,size=512m,mode=1777")
ENGINE_KEY_DIR = "/run/harnext-engine"
ENGINE_KEY_ENV = "OPENROUTER_API_KEY"
OLLAMA_PORT = 11434
ENGINE_KEY_GRACE_S = 60  # the key file goes this long after the container starts, server or not
PINNED = {"image": None}  # the image ID a live run resolved at preflight; trials start from it
BLOCKED = (
    "https://github.com/",
    "https://raw.githubusercontent.com/",
    "https://issues.apache.org/",
)


def docker(*args, check=True, timeout=120):
    try:
        done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PauseRun(f"docker {args[0]} failed: {exc}") from exc
    if check and done.returncode:
        raise PauseRun(f"docker {' '.join(args[:3])} failed: {done.stderr.strip()[:500]}")
    return done


def image_present(image=IMAGE):
    return docker("image", "inspect", image, check=False).returncode == 0


def image_id(image=IMAGE):
    """The content ID (sha256:...) the tag points to now."""
    return docker("image", "inspect", "-f", "{{.Id}}", image).stdout.strip()


def pin_image():
    """Resolve the tag once per run: every trial of this process starts from that exact image."""
    PINNED["image"] = image_id()
    return PINNED["image"]


def image_ref():
    return PINNED["image"] or IMAGE


def image_identity(image, presets):
    """engine_pin.identity measured inside `image` (no network, no mounts)."""
    script = (Path(__file__).resolve().parents[1] / "engine_pin.py").read_text()
    try:  # the host's hashing script, the image's code
        done = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", image, "/opt/venv/bin/python",
                               "-", ",".join(presets)], input=script, capture_output=True, text=True, timeout=300)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PauseRun(f"docker run (engine identity) failed: {exc}") from exc
    if done.returncode:
        raise PauseRun(f"engine identity inside {image} failed: {done.stderr.strip()[-500:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def proxy_allowlist():
    """The allowlist the running proxy enforces (read from inside it)."""
    ensure_network()
    done = docker("exec", PROXY, "cat", "/etc/tinyproxy/allowlist.filter")
    return [line for line in done.stdout.splitlines() if line.strip()]


def ensure_network():
    """Idempotent: internal network (no gateway egress) + proxy attached to it and to bridge."""
    if docker("network", "inspect", NETWORK, check=False).returncode:
        docker("network", "create", "--internal", NETWORK)
    state = docker("inspect", "-f", "{{.State.Running}}", PROXY, check=False)
    if state.returncode == 0 and state.stdout.strip() == "true":
        return
    docker("rm", "-f", PROXY, check=False)
    docker(
        "run", "-d", "--name", PROXY, "--restart", "unless-stopped", "--network", "bridge",
        "--memory", "256m", "--cap-drop", "ALL", PROXY_IMAGE,
    )  # fmt: skip
    docker("network", "connect", NETWORK, PROXY)


@functools.cache
def gateway():
    """The trial network's gateway: the host address a container reaches directly (NO_PROXY)."""
    ensure_network()
    done = docker("network", "inspect", "-f", "{{(index .IPAM.Config 0).Gateway}}", NETWORK)
    return done.stdout.strip()


def local_model_unreachable(timeout_s=5.0):
    """Why the host's local model server (Ollama) is not reachable on the gateway, or None."""
    import httpx

    url = f"http://{gateway()}:{OLLAMA_PORT}/v1/models"
    try:
        httpx.get(url, timeout=timeout_s, trust_env=False)
    except httpx.HTTPError as exc:
        return (f"engine preset ollama: {url} unreachable ({type(exc).__name__}); Ollama must listen "
                "on the docker gateway (E2-ENGINE-OLLAMA.md, 'Service state and host setup')")  # fmt: skip
    return None


def jev_unhealthy(calls=8, spacing_s=2.0):
    """Why the Jev endpoint is not fit to start engine trials now, or None: `calls` sequential
    decisions calls from the host (invented state, no benchmark content). 28 Sep: stalls of 23-27 s
    were measured upstream (host and trial paths alike); a call at or past the engine's own request
    timeout, or any error, means trials would end in the infra pause. Cost ~ $0.0005."""
    import asyncio

    import httpx

    from harnext_eval.e2.v2.engine.clients import load_key
    from harnext_eval.e2.v2.engine.config import for_endpoint

    config, key = for_endpoint("openrouter"), load_key(ENGINE_KEY_ENV)
    if not key:
        return f"{ENGINE_KEY_ENV} missing"
    from harnext_eval.e2.v2.mcp_server import jev_probe_body

    body = jev_probe_body(config.jev_model)
    limit = config.jev_timeout_s

    async def probe():
        seconds = []
        async with httpx.AsyncClient() as http:
            for i in range(calls):
                started = time.monotonic()
                try:
                    reply = await asyncio.wait_for(http.post(config.jev_url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=3 * limit), 3 * limit)  # fmt: skip
                    reply.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    return f"Jev call {i + 1}/{calls} failed after {time.monotonic() - started:.1f} s ({type(exc).__name__})", seconds
                seconds.append(round(time.monotonic() - started, 2))
                if seconds[-1] >= limit:
                    return f"Jev call {i + 1}/{calls} took {seconds[-1]} s (request timeout {limit:g} s)", seconds
                await asyncio.sleep(spacing_s)
        return None, seconds

    why, seconds = asyncio.run(probe())
    return f"provider unhealthy: {why}; latencies {seconds}" if why else None


CHAT_PROBE_CALLS, CHAT_PROBE_LIMIT_S = 2, 30.0  # the chat model is slower (DeepSeek: p95 27 s per turn)


def chat_unhealthy(preset, calls=CHAT_PROBE_CALLS, limit_s=CHAT_PROBE_LIMIT_S):
    """Why the chat model of an engine preset is not fit to start trials, or None: `calls` short chat
    calls from the host, each must answer (HTTP 200) within `limit_s`. Local presets are skipped
    (preflight checks their endpoint). Invented prompt; cost well under $0.001."""
    import asyncio

    import httpx

    from harnext_eval.e2.v2.engine.clients import load_key
    from harnext_eval.e2.v2.engine.config import for_endpoint

    config = for_endpoint(preset)
    if not config.chat_key_env:
        return None
    key = load_key(config.chat_key_env)
    if not key:
        return f"{config.chat_key_env} missing"
    body = {"model": config.chat_model, "messages": [{"role": "user", "content": "Health probe: reply with the single word OK."}],
            "max_tokens": 256, **config.chat_options}  # fmt: skip
    url = config.chat_base_url.rstrip("/") + "/chat/completions"

    async def probe():
        seconds = []
        async with httpx.AsyncClient() as http:
            for i in range(calls):
                started = time.monotonic()
                try:
                    reply = await asyncio.wait_for(http.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=limit_s), limit_s)  # fmt: skip
                    reply.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    return f"{config.chat_model} call {i + 1}/{calls} failed after {time.monotonic() - started:.1f} s ({type(exc).__name__})", seconds
                seconds.append(round(time.monotonic() - started, 2))
        return None, seconds

    why, seconds = asyncio.run(probe())
    return f"chat model unhealthy: {why}; latencies {seconds}" if why else None


def engine_key(trial):
    """Per-trial directory holding the provider key (0600), mounted for the engine MCP server,
    which deletes the file once read. The value never enters argv, env or an archived file."""
    from harnext_eval.e2.v2.engine.clients import load_key

    key = load_key(ENGINE_KEY_ENV)
    if not key:
        raise PauseRun(f"{ENGINE_KEY_ENV} missing (environment or repo .env): kind engine needs it")
    folder = trial.out / "engine-key"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(folder / "key", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(key)
    # A server that never starts never deletes it: gone after the grace period regardless (the
    # runner also removes the folder when the agent returns, on every path).
    timer = threading.Timer(ENGINE_KEY_GRACE_S, lambda: (folder / "key").unlink(missing_ok=True))
    timer.daemon = True
    timer.start()
    return folder


def proxy_env():
    env = {"HTTPS_PROXY": PROXY_URL, "HTTP_PROXY": PROXY_URL, "NO_PROXY": "localhost,127.0.0.1"}
    return [x for k, v in env.items() for key in (k, k.lower()) for x in ("-e", f"{key}={v}")]


def egress_ok():
    """True only if, from inside the trial network, the model API answers through the proxy
    and the leak hosts do not answer by any route (proxy or direct)."""
    ensure_network()
    script = DOCKER_DIR / "verify-egress.sh"
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=300)
    return done.returncode == 0, (done.stdout + done.stderr)[-2000:]


def seed_gradle_home(home, dist, cache=None):
    """Per-trial GRADLE_USER_HOME holding just the unpacked wrapper distribution."""
    home.mkdir(parents=True, exist_ok=True)
    cache = cache or gradle_cache(dist)
    if not dist or not cache:
        return
    for source in (cache / "wrapper/dists").glob(f"{dist}-*/*"):
        target = home / source.relative_to(cache)
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_dir():
                shutil.copytree(item, target / item.name, symlinks=True, dirs_exist_ok=True)
            elif item.suffix == ".ok":
                shutil.copy2(item, target / item.name)


def run_argv(trial, name, command, *, auth, home, cache=None, passthrough=()):
    """docker run argv. `auth` = [(host file, container path)], mounted read-only; `home` is a
    per-trial host directory mounted as the agent's HOME (the CLIs need a writable home);
    `passthrough` names environment variables forwarded by name, so values never reach argv."""
    mounts = ["-v", f"{trial.workspace}:/workspace", "-v", f"{home}:/home/agent"]
    if trial.context:
        mounts += ["-v", f"{trial.context}:/context:ro"]
    if trial.mcp:
        mounts += ["-v", f"{trial.mcp.store}:/store:ro", "-v", f"{trial.out / 'mcp'}:/mcp-log"]
        if trial.mcp.kind in ("research", "research_search", "engine"):  # the Claude Code CLI's MCP client timeout (ms)
            mounts += ["-e", f"MCP_TOOL_TIMEOUT={tool_timeout_ms(trial.mcp.kind, trial.mcp.engine)}"]
        if trial.mcp.kind == "engine":
            mounts += list(ENGINE_SCRATCH)
            if trial.mcp.memory:
                mounts += ["-v", f"{trial.mcp.memory}:/memory:ro"]
            if trial.mcp.reader != "fake":
                mounts += ["-v", f"{engine_key(trial)}:{ENGINE_KEY_DIR}"]
        if MODELS.exists():  # in the image the store modules resolve the cache at /.harnext/models
            mounts += ["-v", f"{MODELS}:/.harnext/models:ro"]
    if trial.sources:  # the file's directory: SQLite opens it immutable, nothing is written
        mounts += ["-v", f"{trial.sources.parent}:/sources:ro", "-e", f"HARNEXT_SOURCES=/sources/{trial.sources.name}"]  # fmt: skip
    if trial.kind == "coding":
        # Claude Code backgrounds a shell command after 2 minutes; a Kafka test run takes 3-4.
        # Without this the agent loses dozens of turns polling its own test runs.
        mounts += ["-e", "BASH_DEFAULT_TIMEOUT_MS=600000", "-e", "BASH_MAX_TIMEOUT_MS=900000"]
        mounts += ["-v", f"{home / '.gradle-home'}:/gradle"]
        cache = cache or gradle_cache(trial.gradle_dist)
        if cache and (cache / "caches").exists():
            mounts += ["-v", f"{cache}:/gradle-ro:ro"]
    for source, target in auth:
        mounts += ["-v", f"{source}:{target}:ro"]
    return [
        "docker", "run", "--rm", "-i", "--name", name, "--network", NETWORK, *LIMITS,
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}", *proxy_env(), *mounts,
        *(x for key in passthrough for x in ("-e", key)), "-w", "/workspace", image_ref(), *command,
    ]  # fmt: skip


def reader_auth(trial):
    """The research reader (kind `research`) is the Claude Code harness inside the container, for
    EVERY agent: its login is mounted read-only unless ANTHROPIC_API_KEY is passed through."""
    needs = trial.mcp and trial.mcp.kind in ("research", "research_search") and trial.mcp.reader == "claude"
    if not needs or os.environ.get("ANTHROPIC_API_KEY"):
        return []
    if not CLAUDE_AUTH.exists():
        raise PauseRun(f"{CLAUDE_AUTH} missing: the research reader needs a Claude Code login")
    return [(CLAUDE_AUTH, CLAUDE_AUTH_IN)]


def prepare_home(trial, auth):
    """Writable HOME with empty mountpoints so Docker never creates root-owned paths in it."""
    home = trial.out / "home"
    for _, target in auth:
        point = home / Path(target).relative_to("/home/agent")
        point.parent.mkdir(parents=True, exist_ok=True)
        point.touch()
    home.mkdir(parents=True, exist_ok=True)
    if trial.instructions:  # both CLIs load their user-level instruction file on start
        for name in (".claude/CLAUDE.md", ".codex/AGENTS.md"):
            (home / name).parent.mkdir(parents=True, exist_ok=True)
            (home / name).write_text(trial.instructions)
        (trial.out / "instructions.md").write_text(trial.instructions)
    if trial.mcp:
        (trial.out / "mcp").mkdir(parents=True, exist_ok=True)
    if trial.kind == "coding":
        seed_gradle_home(home / ".gradle-home", trial.gradle_dist)
    return home


def execute(argv, name, *, stdin, stdout, stderr, timeout, turn_limit=None):
    """Run to completion or kill the container at the wall-clock limit, or - with
    `turn_limit=(n, is_turn)` - as soon as the live trace holds more than n turns.
    Returns (returncode|None, seconds, stopped) with stopped in {False, "time", "turns"}."""
    started = time.monotonic()
    with stdout.open("w") as out, stderr.open("w") as err:
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                start_new_session=True,
            )  # fmt: skip
        except OSError as exc:
            raise PauseRun(f"docker could not start: {exc}") from exc

        def stop(reason):
            subprocess.run(["docker", "kill", name], capture_output=True)
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.communicate(timeout=30)
                else:  # the turn-limit path already handed over and closed stdin
                    proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return None, time.monotonic() - started, reason

        try:
            if turn_limit is None:
                proc.communicate(stdin, timeout=timeout)
                return proc.returncode, time.monotonic() - started, False
            limit, is_turn = turn_limit
            proc.stdin.write(stdin)
            proc.stdin.close()
            turns, offset = 0, 0
            while True:
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                with stdout.open() as seen:  # the trace so far, complete lines only
                    seen.seek(offset)
                    chunk = seen.read()
                done = chunk.rfind("\n") + 1
                offset += len(chunk[:done].encode())
                for line in chunk[:done].splitlines():
                    try:
                        turns += bool(is_turn(json.loads(line)))
                    except ValueError:
                        continue
                if turns > limit:
                    return stop("turns")
                if proc.poll() is not None:
                    return proc.returncode, time.monotonic() - started, False
                if time.monotonic() - started > timeout:
                    return stop("time")
        except subprocess.TimeoutExpired:
            return stop("time")
        except BaseException:
            subprocess.run(["docker", "kill", name], capture_output=True)
            raise


def mcp_env(spec):
    """Environment of the MCP server process. Agent CLIs start MCP servers with a scrubbed
    environment (Codex forwards almost nothing), and the research reader is itself a model
    client: it needs the proxy, the HOME holding the Claude login and the image's CLI switches."""
    env = {"PYTHONPATH": "/opt/harnext/src"}
    if spec.kind in ("research", "research_search"):
        proxy = {
            "HTTPS_PROXY": PROXY_URL,
            "HTTP_PROXY": PROXY_URL,
            "NO_PROXY": "localhost,127.0.0.1",
        }
        env |= proxy | {k.lower(): v for k, v in proxy.items()} | {
            "HOME": "/home/agent", "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }  # fmt: skip
    if spec.kind == "engine":  # model clients over the proxy; a host model server directly
        proxy = {"HTTPS_PROXY": PROXY_URL, "HTTP_PROXY": PROXY_URL,
                 "NO_PROXY": f"localhost,127.0.0.1,{gateway()}"}  # fmt: skip
        env |= proxy | {k.lower(): v for k, v in proxy.items()} | {"HOME": "/home/agent", "TZ": "UTC"}
    return env


def mcp_command(spec):
    """How the CLI inside the container launches the store's MCP server (stdio)."""
    args = [
        "-m", "harnext_eval.e2.v2.mcp_server", "--kind", spec.kind,
        "--store", "/store", "--log", "/mcp-log/receipts.jsonl",
    ]  # fmt: skip
    if spec.kind in ("research", "research_search"):
        args += ["--reader", spec.reader]
    if spec.kind == "engine":
        preset = spec.engine or "openrouter"
        args += ["--preset", preset, "--scratch", "/scratch"]
        args += ["--clients", "fake"] if spec.reader == "fake" else ["--key-file", f"{ENGINE_KEY_DIR}/key"]
        if spec.memory:
            args += ["--memory", "/memory"]
        if preset == "ollama":
            args += ["--chat-base-url", f"http://{gateway()}:{OLLAMA_PORT}/v1"]
    return "/opt/venv/bin/python", args
