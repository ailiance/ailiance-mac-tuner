#!/usr/bin/env python3
"""Cross-domain deduplication v2 for Micro_KIKI pipeline.

Improvements over v1:
  - Hashes FULL user+assistant content (not just first 500 chars)
  - Secondary near-duplicate detection: normalized hash (lowercase, strip
    whitespace/punctuation) catches reformulations with minor formatting diffs
  - Merges spice + spice-sim into unified "spice" domain before dedup
  - --dry-run flag for impact analysis without writing files
  - Per-domain logging with counts

Usage:
    python scripts/micro_kiki/deduplicate_v2.py \
        --classified-dir data/micro-kiki/classified \
        --generated-dir data/micro-kiki/generated \
        --output-dir data/micro-kiki/deduped \
        --config configs/micro_kiki/domains.yaml

    # Dry run (report only, no writes):
    python scripts/micro_kiki/deduplicate_v2.py --dry-run
"""

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dedup_v2")

# Domains to merge: source -> target
DOMAIN_MERGES = {
    "spice-sim": "spice",
}

# Regex: strip all punctuation and collapse whitespace
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def _extract_content(example: dict) -> tuple[str, str]:
    """Extract the last user and last assistant content from a chat example."""
    messages = example.get("messages", [])
    user_content = ""
    assistant_content = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            user_content = msg.get("content", "")
        elif role == "assistant":
            assistant_content = msg.get("content", "")
    return user_content, assistant_content


def exact_hash(example: dict) -> str:
    """SHA-256 of full user + assistant content (exact match)."""
    user, assistant = _extract_content(example)
    text = user.strip() + "\n###\n" + assistant.strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_hash(example: dict) -> str:
    """SHA-256 of normalized user content only.

    Normalization: lowercase, strip accents, remove punctuation, collapse
    whitespace. This catches near-duplicates where only the user prompt
    matters (same question asked with minor formatting differences).
    """
    user, _ = _extract_content(example)
    text = user.strip().lower()
    # Remove accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Remove punctuation
    text = _PUNCT_RE.sub("", text)
    # Collapse whitespace
    text = _SPACE_RE.sub(" ", text).strip()

    if not text:
        # Fallback: use full content to avoid false collisions on empty
        _, assistant = _extract_content(example)
        text = assistant.strip().lower()[:200]

    return "norm:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_domain_jsonl(filepath: Path) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    if not filepath.exists():
        return examples
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def merge_domains(domain_data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Merge configured domain pairs (e.g. spice-sim -> spice)."""
    merged = {}
    for name, examples in domain_data.items():
        target = DOMAIN_MERGES.get(name, name)
        if target not in merged:
            merged[target] = []
        merged[target].extend(examples)
    return merged


def dedup_cross_domain(
    domain_data: dict[str, list[dict]],
    domain_phases: dict[str, int],
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """Remove exact and near-duplicates across domains.

    Priority: lower phase number first, then alphabetical within phase.

    Returns:
        (deduped_data, stats) where stats[domain] = {
            "before": int, "after": int,
            "exact_dupes": int, "near_dupes": int
        }
    """
    seen_exact: set[str] = set()
    seen_norm: set[str] = set()
    result: dict[str, list[dict]] = {}
    stats: dict[str, dict[str, int]] = {}

    # Sort by (phase, name) for priority
    sorted_names = sorted(
        domain_data.keys(),
        key=lambda n: (domain_phases.get(n, 99), n),
    )

    for domain_name in sorted_names:
        examples = domain_data[domain_name]
        deduped = []
        exact_dupes = 0
        near_dupes = 0

        for example in examples:
            h_exact = exact_hash(example)

            # 1. Exact duplicate check
            if h_exact in seen_exact:
                exact_dupes += 1
                continue

            # 2. Near-duplicate check (normalized user prompt)
            h_norm = normalized_hash(example)
            if h_norm in seen_norm:
                near_dupes += 1
                continue

            seen_exact.add(h_exact)
            seen_norm.add(h_norm)
            deduped.append(example)

        result[domain_name] = deduped
        stats[domain_name] = {
            "before": len(examples),
            "after": len(deduped),
            "exact_dupes": exact_dupes,
            "near_dupes": near_dupes,
        }

    return result, stats


def run_dedup(
    config_path: str,
    classified_dir: str,
    generated_dir: str,
    output_dir: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Run full deduplication pipeline.

    Merges classified + generated data per domain, applies domain merges
    (spice-sim -> spice), then deduplicates cross-domain.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    out_path = Path(output_dir)

    # Build phase lookup (including merged domains)
    domain_phases: dict[str, int] = {}
    for name, cfg in domains.items():
        target = DOMAIN_MERGES.get(name, name)
        phase = cfg["phase"]
        # For merged domains, keep the lowest phase
        if target not in domain_phases or phase < domain_phases[target]:
            domain_phases[target] = phase

    # Load all domain data (classified + generated)
    raw_domain_data: dict[str, list[dict]] = {}

    sorted_domains = sorted(
        domains.keys(),
        key=lambda n: (domains[n]["phase"], n),
    )

    for name in sorted_domains:
        examples = []
        classified_file = Path(classified_dir) / f"{name}.jsonl"
        examples.extend(load_domain_jsonl(classified_file))
        generated_file = Path(generated_dir) / f"{name}.jsonl"
        examples.extend(load_domain_jsonl(generated_file))
        raw_domain_data[name] = examples

    total_raw = sum(len(v) for v in raw_domain_data.values())
    log.info("Loaded %d total examples from %d domains", total_raw, len(raw_domain_data))

    # Merge spice-sim -> spice
    domain_data = merge_domains(raw_domain_data)
    merged_names = [k for k in DOMAIN_MERGES if k in raw_domain_data and len(raw_domain_data[k]) > 0]
    if merged_names:
        for src in merged_names:
            tgt = DOMAIN_MERGES[src]
            log.info(
                "Merged %s (%d) into %s (now %d total)",
                src, len(raw_domain_data[src]), tgt, len(domain_data[tgt]),
            )

    total_merged = sum(len(v) for v in domain_data.values())
    log.info("After domain merges: %d examples in %d domains", total_merged, len(domain_data))

    # Deduplicate
    deduped, stats = dedup_cross_domain(domain_data, domain_phases)

    # Report
    print("\n" + "=" * 72)
    print(f"{'Domain':<18} {'Phase':>5} {'Before':>7} {'After':>7} {'Exact':>7} {'Near':>7} {'Removed':>8}")
    print("-" * 72)

    total_before = 0
    total_after = 0
    total_exact = 0
    total_near = 0
    counts: dict[str, int] = {}

    # Sort output by phase then name
    for name in sorted(deduped.keys(), key=lambda n: (domain_phases.get(n, 99), n)):
        s = stats[name]
        removed = s["before"] - s["after"]
        phase = domain_phases.get(name, 99)
        print(
            f"  {name:<16} {phase:>5} {s['before']:>7} {s['after']:>7} "
            f"{s['exact_dupes']:>7} {s['near_dupes']:>7} {removed:>8}"
        )
        total_before += s["before"]
        total_after += s["after"]
        total_exact += s["exact_dupes"]
        total_near += s["near_dupes"]
        counts[name] = s["after"]

    print("-" * 72)
    total_removed = total_before - total_after
    print(
        f"  {'TOTAL':<16} {'':>5} {total_before:>7} {total_after:>7} "
        f"{total_exact:>7} {total_near:>7} {total_removed:>8}"
    )
    print("=" * 72)

    if dry_run:
        log.info("DRY RUN — no files written")
        return counts

    # Write output
    out_path.mkdir(parents=True, exist_ok=True)
    for name in sorted(deduped.keys()):
        outfile = out_path / f"{name}.jsonl"
        with open(outfile, "w") as f:
            for ex in deduped[name]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("Wrote %s: %d examples", outfile.name, len(deduped[name]))

    # Remove any stale files from merged-away domains
    for src in DOMAIN_MERGES:
        stale = out_path / f"{src}.jsonl"
        if stale.exists():
            stale.unlink()
            log.info("Removed stale merged file: %s", stale.name)

    log.info("Deduplication complete: %d → %d (%d removed)", total_before, total_after, total_removed)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Cross-domain deduplication v2 for Micro_KIKI")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml")
    parser.add_argument("--classified-dir", default="data/micro-kiki/classified")
    parser.add_argument("--generated-dir", default="data/micro-kiki/generated")
    parser.add_argument("--output-dir", default="data/micro-kiki/deduped")
    parser.add_argument("--dry-run", action="store_true", help="Report impact without writing files")
    args = parser.parse_args()

    run_dedup(
        args.config,
        args.classified_dir,
        args.generated_dir,
        args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
