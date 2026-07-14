"""Omission rate: a text LLM checks which judge-listed salient items the caption
mentions. OR = fraction not covered."""

from llm_client import parse_json_strict

_PROMPT = """A reference list of salient items (entities/events) was extracted from a
video clip, independent of any caption. Given a CAPTION, decide for EACH reference
item whether the caption mentions it (allowing paraphrase / synonyms).

Output ONLY a bare JSON array (NOT wrapped in an object), one object per reference
item, in the same order, with EXACTLY one entry for EVERY item (use "covered": true
or false). Use double quotes.
[{{"index": 0, "covered": true}}, {{"index": 1, "covered": false}}, ...]

REFERENCE ITEMS:
{items}

CAPTION:
\"\"\"{caption}\"\"\"
"""


def coverage(caption: str, salient: list[str], client, config: dict) -> dict:
    if not salient:
        return {"n_reference": 0, "n_covered": 0, "omission_rate": 0.0, "missed": []}
    items = "\n".join(f"{i}. {s}" for i, s in enumerate(salient))
    retries = config.get("claim_retries", 4)
    base_t = config.get("claim_temperature", 0.0)
    last = None
    covered_flags = None
    for attempt in range(retries):
        resp = client.chat.completions.create(
            model=config["claim_model"],
            messages=[{"role": "user", "content": _PROMPT.format(items=items, caption=caption)}],
            temperature=base_t if attempt == 0 else 0.4,
            max_tokens=config.get("claim_max_tokens", 4096),
            extra_body={"top_k": config.get("claim_top_k", 20)},  # Gemma3 has no thinking mode
        )
        try:
            d = parse_json_strict(resp.choices[0].message.content)
            if not isinstance(d, list):
                raise ValueError("omission coverage did not return a list")
            flags = {int(x["index"]): bool(x["covered"]) for x in d
                     if isinstance(x, dict) and "index" in x and "covered" in x}
            if flags:                 # accept partial; missing items filled below
                covered_flags = flags
                break
            raise ValueError("omission coverage had no usable items")
        except (ValueError, KeyError, TypeError) as e:
            last = e
    if covered_flags is None:
        covered_flags = {}
    # Items the LLM didn't judge default to not covered.
    for i in range(len(salient)):
        covered_flags.setdefault(i, False)
    n_cov = sum(1 for v in covered_flags.values() if v)
    missed = [salient[i] for i in range(len(salient)) if not covered_flags[i]]
    return {
        "n_reference": len(salient),
        "n_covered": n_cov,
        "omission_rate": (len(salient) - n_cov) / len(salient),
        "missed": missed,
    }
