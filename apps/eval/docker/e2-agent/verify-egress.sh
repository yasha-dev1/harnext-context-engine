#!/usr/bin/env bash
# Proves, from INSIDE a container on the trial network, that the model APIs answer through the
# allowlisting proxy while the leak hosts (code host, raw files, issue tracker) are unreachable
# both through the proxy and directly. Exit 0 = boundary holds. No credentials are mounted.
set -u
NETWORK=${NETWORK:-harnext-e2-net}
PROXY=${PROXY:-harnext-e2-proxy}
IMAGE=${IMAGE:-harnext-e2-agent}
PROXY_IMAGE=${PROXY_IMAGE:-harnext-e2-proxy}

docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create --internal "$NETWORK" >/dev/null
if [ "$(docker inspect -f '{{.State.Running}}' "$PROXY" 2>/dev/null)" != "true" ]; then
  docker rm -f "$PROXY" >/dev/null 2>&1
  docker run -d --name "$PROXY" --restart unless-stopped --network bridge --memory 256m \
    --cap-drop ALL "$PROXY_IMAGE" >/dev/null || { echo "FAIL proxy did not start"; exit 2; }
  docker network connect "$NETWORK" "$PROXY" || { echo "FAIL proxy not on $NETWORK"; exit 2; }
  sleep 1
fi
[ "$(docker network inspect -f '{{.Internal}}' "$NETWORK")" = "true" ] || { echo "FAIL $NETWORK is not internal"; exit 2; }

docker run --rm -i --network "$NETWORK" --cap-drop ALL --user "$(id -u):$(id -g)" \
  -e HTTPS_PROXY="http://$PROXY:8888" -e HTTP_PROXY="http://$PROXY:8888" \
  -e https_proxy="http://$PROXY:8888" -e http_proxy="http://$PROXY:8888" \
  "$IMAGE" bash -s <<'INSIDE'
fail=0
code() { curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$@" 2>/dev/null; }
# openrouter.ai: kind engine's models (Jev + gpt-oss-20b); the allowlist is shared by every arm.
for url in https://api.openai.com/v1/models https://chatgpt.com/backend-api/codex/models https://api.anthropic.com/v1/models https://openrouter.ai/api/v1/models; do
  c=$(code "$url")
  if [ "${c:-000}" != "000" ]; then echo "ok    allowed  via-proxy $url -> HTTP $c"; else echo "FAIL  allowed host unreachable: $url"; fail=1; fi
done
for url in https://github.com/ https://raw.githubusercontent.com/apache/kafka/trunk/README.md \
           https://issues.apache.org/jira/browse/KAFKA-1 https://api.github.com/ https://repo.maven.apache.org/maven2/ \
           http://github.com/ https://pypi.org/simple/; do
  c=$(code "$url")
  case "${c:-000}" in 000|403) echo "ok    blocked  via-proxy $url -> ${c:-000}";; *) echo "FAIL  leak via proxy: $url -> HTTP $c"; fail=1;; esac
  c=$(code --noproxy '*' --max-time 6 "$url")
  if [ "${c:-000}" = "000" ]; then echo "ok    blocked  direct    $url"; else echo "FAIL  leak direct: $url -> HTTP $c"; fail=1; fi
done
for ip in 1.1.1.1 140.82.121.4; do
  if curl -s --noproxy '*' --max-time 5 -o /dev/null "http://$ip/"; then echo "FAIL  direct IP egress to $ip"; fail=1; else echo "ok    blocked  direct    ip $ip"; fi
done
# look-alike hosts of the new entry must stay blocked (the filter is anchored)
for url in https://openrouter.ai.evil.example/ https://api.openrouter.ai.example/; do
  c=$(code "$url")
  case "${c:-000}" in 000|403) echo "ok    blocked  via-proxy $url -> ${c:-000}";; *) echo "FAIL  leak via proxy: $url -> HTTP $c"; fail=1;; esac
done
# a CONNECT to a non-TLS port on an allowed host must be refused too
c=$(code -x "$HTTPS_PROXY" https://api.openai.com:22/); [ "${c:-000}" = "000" ] && echo "ok    blocked  via-proxy api.openai.com:22" || { echo "FAIL  CONNECT to port 22 allowed"; fail=1; }
exit $fail
INSIDE
status=$?
[ $status -eq 0 ] && echo "EGRESS BOUNDARY HOLDS" || echo "EGRESS BOUNDARY BROKEN"
exit $status
