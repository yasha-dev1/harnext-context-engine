# harnext-e2-agent: the trial boundary for E2 v2 (see apps/eval/E2-V2-DESIGN.md).
# Build from the REPO ROOT (Dockerfile.dockerignore allowlists the context):
#   docker build -f apps/eval/docker/e2-agent/Dockerfile -t harnext-e2-agent .
# Never add benchmarks/, out/, private tasks or the Kafka clone to this image.
FROM eclipse-temurin:17-jdk

ARG CODEX_VERSION=0.154.0
ARG CLAUDE_CODE_VERSION=2.1.278
ARG NODE_MAJOR=22

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ripgrep python3 python3-venv patch less procps unzip \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @openai/codex@${CODEX_VERSION} @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    && npm cache clean --force && rm -rf /var/lib/apt/lists/*

# MCP server runtime: workspace packages + the v2 module tree only. harnext_eval's real
# package __init__ files import the whole evaluation stack, so the image gets empty ones.
COPY packages /opt/harnext/packages
COPY apps/mcp /opt/harnext/apps/mcp
COPY apps/builder /opt/harnext/apps/builder
COPY apps/eval/src/harnext_eval/e2/v2 /opt/harnext/src/harnext_eval/e2/v2
COPY apps/eval/docker/e2-agent/constraints.txt /opt/harnext/constraints.txt
RUN touch /opt/harnext/src/harnext_eval/__init__.py /opt/harnext/src/harnext_eval/e2/__init__.py \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -c /opt/harnext/constraints.txt \
        /opt/harnext/packages/shared /opt/harnext/apps/builder /opt/harnext/apps/mcp \
        "fastmcp>=2.0" "rank-bm25>=0.2" "numpy>=1.26" "scipy==1.18.1" "pyyaml>=6.0" "fastembed==0.7.4" tokenizers \
        "graphiti-core[kuzu]==0.30.2" "kuzu==0.11.3" \
        "chdb==4.4.0" "pandas==3.0.5" "pyarrow==25.0.1" \
    && find /opt/venv -name __pycache__ -prune -exec rm -rf {} +

# gh / jira / mail: time-frozen replicas (v2/sources/cli.py, standard library only). They are on
# PATH in EVERY arm; only S-tools trials mount a database (HARNEXT_SOURCES), elsewhere they
# print "not available in this trial" and exit 2. The script's own directory, not the cwd,
# heads sys.path, so nothing in /workspace can shadow the module.
RUN for tool in gh jira mail; do \
        printf '#!/usr/bin/python3\nimport sys\nsys.path.insert(0, "/opt/harnext/src")\nfrom harnext_eval.e2.v2.sources.cli import main\nsys.exit(main("%s"))\n' "$tool" > /usr/local/bin/$tool \
        && chmod 0755 /usr/local/bin/$tool; \
    done

ENV PYTHONPATH=/opt/harnext/src \
    HARNEXT_PYTHON=/opt/venv/bin/python \
    HOME=/home/agent \
    GRADLE_USER_HOME=/gradle \
    GRADLE_RO_DEP_CACHE=/gradle-ro/caches \
    HF_HUB_OFFLINE=1 \
    DISABLE_AUTOUPDATER=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
RUN mkdir -p /home/agent /workspace /gradle && chmod 1777 /home/agent /workspace /gradle
WORKDIR /workspace
