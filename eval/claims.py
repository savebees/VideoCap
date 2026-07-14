"""Claim decomposition (text LLM): split a caption into atomic typed assertions.
Negation claims are flagged and kept out of the positive-claim HR."""

from llm_client import parse_json_strict

CLAIM_TYPES = ["entity", "attribute", "action", "count", "spatial", "negation"]

_PROMPT = """You decompose a video caption into ATOMIC claims for fact-checking.

Split the caption below into the smallest independently-verifiable assertions. Each
claim must be checkable against the video on its own. Assign exactly one type:
- entity: a thing/person/object is present (e.g. "a man is present")
- attribute: a property of something (color, clothing, size, state)
- action: something moving or doing (e.g. "the man walks left")
- count: a specific number/quantity (e.g. "three people")
- spatial: a spatial relation/location (e.g. "a booth on the left")
- negation: an explicit absence or "nothing happens" (e.g. "no people are visible",
  "the walkway remains empty")

Rules:
- Decompose compound sentences into separate claims.
- Keep each claim a short, self-contained statement; resolve pronouns to the noun.
- Do NOT invent claims not stated in the caption. Do NOT editorialize.
- Output ONLY a JSON array, nothing else:
[{{"claim_id": 1, "text": "...", "type": "entity"}}, ...]

CAPTION:
\"\"\"{caption}\"\"\"
"""


def _parse_claims(text):
    claims = parse_json_strict(text)
    if not isinstance(claims, list):
        raise ValueError("claim decomposition did not return a list")
    return claims


def decompose(caption: str, client, config: dict) -> list[dict]:
    if not caption.strip():
        return []
    retries = config.get("claim_retries", 4)
    base_t = config.get("claim_temperature", 0.0)
    last = None
    claims = None
    for attempt in range(retries):
        resp = client.chat.completions.create(
            model=config["claim_model"],
            messages=[{"role": "user", "content": _PROMPT.format(caption=caption)}],
            temperature=base_t if attempt == 0 else 0.4,
            max_tokens=config.get("claim_max_tokens", 4096),
            extra_body={"top_k": config.get("claim_top_k", 20)},  # Gemma3 has no thinking mode
        )
        try:
            claims = _parse_claims(resp.choices[0].message.content)
            break
        except ValueError as e:
            last = e
    if claims is None:
        # No parseable list after all retries: treat as no checkable claims.
        import sys as _sys
        print(f"[claims] decomposition unrecoverable, using empty claim set: {last}",
              file=_sys.stderr, flush=True)
        return []
    out = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        text = c.get("text")
        t = c.get("type")
        if not text:
            continue                      # drop malformed claim (bounded)
        if t not in CLAIM_TYPES:
            t = "entity"                  # default odd/missing type to a positive claim
        out.append({"claim_id": len(out) + 1, "text": text, "type": t})
    return out
