"""Per-dataset finalize: per_video.jsonl + summary.csv from the cache.
F1 = harmonic mean of precision (1-HR) and recall (1-OR). Systems without full
cache for a dataset are skipped."""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache as cachemod
import systems as sysmod


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _datasets(config):
    """dataset -> list of (split, vid), from the enumerated clips."""
    by_ds = {}
    for split, vid in sysmod.list_clips(config):
        ds = config["splits"][split]["dataset"]
        by_ds.setdefault(ds, []).append((split, vid))
    return by_ds


def _row(name, recs):
    n = len(recs)
    hr = _mean([r["faithfulness"]["hallucination_rate"] for r in recs])
    nv_rate = _mean([
        (r["faithfulness"]["not_verifiable"] / r["faithfulness"]["n_pos_claims"])
        if r["faithfulness"]["n_pos_claims"] else None for r in recs])
    orate = _mean([r["omission"]["omission_rate"] for r in recs])

    SUP = sum(r["faithfulness"]["supported"] for r in recs)
    CON = sum(r["faithfulness"]["contradicted"] for r in recs)
    NCOV = sum(r["omission"]["n_covered"] for r in recs)
    NREF = sum(r["omission"]["n_reference"] for r in recs)
    precision = SUP / (SUP + CON) if (SUP + CON) else None
    recall = NCOV / NREF if NREF else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    avg_len = _mean([r["richness"]["word_count"] for r in recs])
    below = _mean([1.0 if r["richness"]["below_floor"] else 0.0 for r in recs])
    ttr = _mean([r["richness"]["ttr"] for r in recs])
    vocab = set()
    tot_tok = tot_n = tot_v = tot_a = 0
    for r in recs:
        vocab.update(r["richness"]["types"])
        tot_tok += r["richness"]["n_tokens"]
        tot_n += r["richness"]["pos"]["noun"]
        tot_v += r["richness"]["pos"]["verb"]
        tot_a += r["richness"]["pos"]["adj"]

    return {
        "system": name,
        "n_clips": n,
        "F1": round(f1, 4) if f1 is not None else "",
        "precision": round(precision, 4) if precision is not None else "",
        "recall": round(recall, 4) if recall is not None else "",
        "sup_per_video": round(SUP / n, 2) if n else "",
        "con_per_video": round(CON / n, 2) if n else "",
        "HR_faithfulness": round(hr, 4) if hr is not None else "",
        "not_verifiable_rate": round(nv_rate, 4) if nv_rate is not None else "",
        "OR_omission": round(orate, 4) if orate is not None else "",
        "avg_length_words": round(avg_len, 2) if avg_len is not None else "",
        "below_floor_ratio": round(below, 4) if below is not None else "",
        "corpus_vocab_size": len(vocab),
        "noun_ratio": round(tot_n / tot_tok, 4) if tot_tok else "",
        "verb_ratio": round(tot_v / tot_tok, 4) if tot_tok else "",
        "adj_ratio": round(tot_a / tot_tok, 4) if tot_tok else "",
        "avg_ttr": round(ttr, 4) if ttr is not None else "",
    }


def finalize(config):
    """Assemble per-dataset per_video.jsonl from cache + write per-dataset summary.csv."""
    root = config["output_root"]
    paths = []
    for dataset, clips in _datasets(config).items():
        mdir = os.path.join(root, dataset, "metrics")
        rows = []
        for sy in config["systems"]:
            recs = []
            for split, vid in clips:
                rec = cachemod.load_valid(
                    cachemod.result_path(config["cache_dir"], sy["name"], split, vid))
                if rec is None:
                    recs = None
                    break
                recs.append(rec)
            if recs is None:
                continue  # system not fully computed for this dataset -> skip
            sdir = os.path.join(mdir, sy["name"])
            os.makedirs(sdir, exist_ok=True)
            with open(os.path.join(sdir, "per_video.jsonl"), "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows.append(_row(sy["name"], recs))
        if not rows:
            continue
        spath = os.path.join(mdir, "summary.csv")
        with open(spath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        paths.append(spath)
    return paths


# Back-compat alias (run_eval calls summarize()).
def summarize(config):
    return finalize(config)


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1
                              else os.path.join(os.path.dirname(__file__), "config.yaml")))
    for p in finalize(cfg):
        print(p)
