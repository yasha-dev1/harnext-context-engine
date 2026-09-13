"""Reference temporal graph: controlled methods, not Graphiti/LightRAG replicas."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from .config import GraphSchema, Resolution, StrategyConfig, Traversal


def build_graph(records: list[dict], config: StrategyConfig) -> tuple[list[dict], list[dict]]:
    graph, decisions = [], []
    canonical = {
        value
        for record in records
        for value in [record["subject"], record["object"]]
        if re.match(r"^[a-z_]+:", value)
    }
    aliases: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for alias in record["aliases"]:
            if alias not in record["quote"]:
                decisions.append(
                    {"alias": alias, "canonical": record["subject"], "decision": "unsupported"}
                )
            elif alias != record["subject"] and (
                alias in canonical or re.match(r"^[a-z_]+:", alias)
            ):
                decisions.append(
                    {
                        "alias": alias,
                        "canonical": record["subject"],
                        "decision": "canonical_conflict",
                    }
                )
            else:
                aliases[alias].add(record["subject"])
    accepted = {}
    for alias, targets in sorted(aliases.items()):
        valid = len(targets) == 1
        decisions.append(
            {
                "alias": alias,
                "targets": sorted(targets),
                "decision": "accepted" if valid else "ambiguous",
                "applied": valid and config.graph.resolution == Resolution.ALIASES,
            }
        )
        if valid and config.graph.resolution == Resolution.ALIASES:
            accepted[alias] = next(iter(targets))
    for record in records:
        if (
            config.graph.extraction == "domain_predicates"
            and record["predicate"] not in config.files.predicates
        ):
            continue
        if config.graph.schema_name == GraphSchema.TYPED:
            if record["predicate"] not in config.files.predicates:
                raise ValueError(f"typed graph rejects predicate {record['predicate']}")
            if record["subject"].split(":", 1)[0] not in {
                "issue",
                "pr",
                "person",
                "file",
                "thread",
                "version",
            }:
                raise ValueError(f"typed graph rejects entity {record['subject']}")
            object_types = {
                "assigned_to": "person:",
                "targets_version": "version:",
                "references_issue": "issue:",
                "changes_file": "file:",
                "discussed_in": "thread:",
            }
            prefix = object_types.get(record["predicate"])
            target = accepted.get(record["object"], record["object"])
            if prefix and not target.startswith(prefix):
                raise ValueError(f"typed graph rejects object {target} for {record['predicate']}")
        graph.append(
            {
                **record,
                "subject": accepted.get(record["subject"], record["subject"]),
                "object": accepted.get(record["object"], record["object"]),
            }
        )
    return graph, decisions


def eligible(
    records: list[dict], valid: datetime, known: datetime, functional: list[str]
) -> list[dict]:
    def date(value):
        return datetime.fromisoformat(value)

    rows = [
        r
        for r in records
        if date(r["valid_from"]) <= valid
        and date(r["known_from"]) <= known
        and (r["valid_to"] is None or valid < date(r["valid_to"]))
        and (r["known_to"] is None or known < date(r["known_to"]))
    ]
    latest = {}
    for row in rows:
        if row["predicate"] in functional:
            key = (row["subject"], row["predicate"])
            previous = latest.get(key)
            if previous is None or (date(row["valid_from"]), date(row["known_from"]), row["id"]) > (
                date(previous["valid_from"]),
                date(previous["known_from"]),
                previous["id"],
            ):
                latest[key] = row
    return [
        r
        for r in rows
        if r["predicate"] not in functional
        or latest[(r["subject"], r["predicate"])]["id"] == r["id"]
    ]


def traverse(
    rows: list[dict], seeds: list[str], config: StrategyConfig, target: str | None = None
) -> list[dict]:
    adjacency = defaultdict(list)
    for row in rows:
        adjacency[row["subject"]].append((row["object"], row))
        adjacency[row["object"]].append((row["subject"], row))
    nodes = set(seeds)
    chosen = {}
    paths = [(seed, [seed], []) for seed in seeds]
    expansions = 0
    for _ in range(config.graph.hops):
        following = []
        for node, visited, edges in paths:
            for neighbor, edge in sorted(adjacency[node], key=lambda item: item[1]["id"]):
                if expansions >= config.graph.max_expansions:
                    break
                expansions += 1
                if neighbor in visited:
                    continue
                next_edges = [*edges, edge]
                following.append((neighbor, [*visited, neighbor], next_edges))
                nodes.add(neighbor)
                if (
                    config.graph.traversal != Traversal.PATHS
                    or target is None
                    or neighbor == target
                ):
                    for item in next_edges:
                        chosen[item["id"]] = item
        paths = following
        if expansions >= config.graph.max_expansions:
            break
    if config.graph.traversal == Traversal.PAGERANK:
        # PPR over the configured hop-bounded subgraph, with explicit teleport seeds.
        scores = {node: 1 / len(seeds) if node in seeds else 0.0 for node in nodes} if seeds else {}
        teleport = dict(scores)
        damping = config.graph.damping
        for _ in range(config.graph.iterations):
            updated = {node: (1 - damping) * teleport[node] for node in nodes}
            for node, value in scores.items():
                neighbors = {n for n, _ in adjacency[node] if n in nodes}
                if neighbors:
                    for neighbor in neighbors:
                        updated[neighbor] += damping * value / len(neighbors)
                else:
                    for neighbor in nodes:
                        updated[neighbor] += damping * value * teleport[neighbor]
            scores = updated
        ordered = sorted(
            chosen.values(), key=lambda r: (-(scores[r["subject"]] + scores[r["object"]]), r["id"])
        )
    else:
        ordered = list(chosen.values())
    return ordered[: config.graph.max_edges]
