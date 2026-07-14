"""Visual judge (Gemma3-27B-it, independent family; frames as image_url items).
Ops on the shared frame batch: verify_claims, list_salient, verify_negations."""

from llm_client import image_content, parse_json_strict

VERDICTS = ["supported", "contradicted", "not_verifiable"]

_VERIFY_PROMPT = """You are a strict visual fact-checker. You are given {n} frames
uniformly sampled across one short video clip, followed by a list of CLAIMS about
that clip. Judge each claim using ONLY what is visible in the frames.

For each claim return:
- verdict: "supported" (frames clearly show it), "contradicted" (frames clearly
  show the opposite), or "not_verifiable" (frames neither confirm nor deny).
- evidence: one short phrase citing what you actually see (or its absence).

Be conservative: if you cannot clearly see it, use not_verifiable, not supported.

Output ONLY a bare JSON array (NOT wrapped in any object), with EXACTLY one object
per claim (same ids, same order). The "verdict" value MUST be one of exactly:
supported, contradicted, not_verifiable (a bare double-quoted string, no extra quotes).
[{{"claim_id": 1, "verdict": "supported", "evidence": "..."}}, ...]

CLAIMS:
{claims}
"""

_SALIENT_PROMPT = """You are given {n} frames uniformly sampled across one short
video clip. WITHOUT any external description, list the salient, clearly-visible
entities and events in this clip — the things a complete caption ought to mention.
Include distinct people/vehicles/objects and any notable actions or events. Exclude
trivial background. Keep each item a short noun/verb phrase.

List the 10 to 25 MOST salient items only. Do NOT repeat the same item.

Output ONLY a JSON array of short strings:
["a man in a black jacket", "a white sedan entering from the left", ...]
"""

_NEGATION_PROMPT = """You are given {n} frames densely sampled across one short
video clip. Each claim below asserts an ABSENCE (e.g. "no people are visible",
"the area stays empty"). Carefully check the WHOLE set of frames.

For each claim return verdict:
- "supported": the asserted absence truly holds across all frames.
- "contradicted": the thing claimed absent actually appears in at least one frame.
- "not_verifiable": cannot tell.
Plus a short evidence phrase.

Output ONLY a JSON array: [{{"claim_id": 1, "verdict": "...", "evidence": "..."}}, ...]

CLAIMS:
{claims}
"""


def _call(client, config, frames: list[bytes], text: str, temperature: float):
    resp = client.chat.completions.create(
        model=config["judge_model"],
        messages=[{"role": "user", "content": [*image_content(frames), {"type": "text", "text": text}]}],
        temperature=temperature,
        frequency_penalty=config.get("judge_frequency_penalty", 0.0),
        max_tokens=config.get("judge_max_tokens", 4096),
        extra_body=config.get("judge_extra_body") or None,
    )
    return resp.choices[0].message.content


def _call_parse(client, config, frames, text, parse_fn):
    """Call the judge and parse; retry on unparseable JSON (greedy can emit
    truncated/looping JSON). Temperature is bumped on retries to break the loop.
    Raises only after all retries fail — no silent fallback."""
    retries = config.get("judge_retries", 4)
    base_t = config.get("judge_temperature", 0.0)
    last = None
    for attempt in range(retries):
        temp = base_t if attempt == 0 else 0.4
        raw = _call(client, config, frames, text, temperature=temp)
        try:
            return parse_fn(raw)
        except ValueError as e:  # json.JSONDecodeError is a ValueError subclass
            last = e
    raise RuntimeError(f"judge JSON unparseable after {retries} attempts: {last}") from last


def _fmt_claims(claims: list[dict]) -> str:
    return "\n".join(f'{c["claim_id"]}. {c["text"]}' for c in claims)


def _coerce_to_list(data):
    """Gemma3 sometimes wraps the array in an object ({"verdicts": [...]}) or emits
    a single object instead of a one-element array. Recover the intended list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "verdict" in data or "claim_id" in data:   # a lone verdict object
            return [data]
        list_vals = [v for v in data.values() if isinstance(v, list)]
        if len(list_vals) == 1:                        # {"verdicts": [...]}
            return list_vals[0]
    raise ValueError(f"expected a JSON list, got {type(data).__name__}")


def _norm_verdict(v):
    """Gemma3 sometimes wraps/decorates the verdict ('"not_verifiable"', '_supported_',
    'not verifiable'). Normalize to the canonical token, or None if unrecognizable."""
    if not isinstance(v, str):
        return None
    v = v.strip().strip("\"'").strip("_").strip().lower().replace(" ", "_")
    return v if v in VERDICTS else None


def _parse_verdicts(raw: str) -> dict:
    """Return whatever valid {claim_id: verdict} pairs the judge produced. Does NOT
    require all ids or reject odd items — missing/invalid ones are filled by the
    caller as not_verifiable (per-claim, bounded). Only a wholly-unparseable response
    raises (via parse_json_strict), which triggers a retry."""
    data = _coerce_to_list(parse_json_strict(raw))
    out = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        cid = item.get("claim_id")
        v = _norm_verdict(item.get("verdict"))
        if cid is None or v is None:
            continue
        try:
            out[int(cid)] = {"verdict": v, "evidence": str(item.get("evidence", ""))[:200]}
        except (ValueError, TypeError):
            continue
    return out


def _verify_batched(client, config, frames, claims, prompt_tmpl):
    """Verify claims in small batches. A single call with many claims makes Gemma3
    drop verdicts; small batches stay reliable, and the (identical) image prefix is
    reused across batches via prefix caching, so frames are encoded once per clip."""
    batch = config.get("judge_claim_batch", 5)
    out = {}
    for i in range(0, len(claims), batch):
        chunk = claims[i:i + batch]
        ids = {c["claim_id"] for c in chunk}
        try:
            verds = _call_parse(client, config, frames,
                                prompt_tmpl.format(n=len(frames), claims=_fmt_claims(chunk)),
                                _parse_verdicts)
        except RuntimeError:
            verds = {}  # wholly unparseable after retries -> all default below
        # Per-claim bounded degradation: any claim the judge didn't return a usable
        # verdict for defaults to not_verifiable (excluded from HR), instead of
        # discarding the whole batch. Keeps the verdicts it DID return.
        for cid in ids:
            out[cid] = verds.get(cid) or {"verdict": "not_verifiable", "evidence": "not returned by judge"}
    return out


def verify_claims(client, config, frames, claims):
    if not claims:
        return {}
    return _verify_batched(client, config, frames, claims, _VERIFY_PROMPT)


def verify_negations(client, config, dense_frames, neg_claims):
    if not neg_claims:
        return {}
    return _verify_batched(client, config, dense_frames, neg_claims, _NEGATION_PROMPT)


def _parse_salient(raw):
    data = _coerce_to_list(parse_json_strict(raw))
    return [str(x) for x in data]


def list_salient(client, config, frames) -> list[str]:
    return _call_parse(client, config, frames, _SALIENT_PROMPT.format(n=len(frames)), _parse_salient)
