from openai import OpenAI


def get_vlm_client(config: dict) -> OpenAI:
    return OpenAI(
        base_url=config.get("vllm_base_url", "http://localhost:8000/v1"),
        api_key="not-needed",
    )


def get_extra_body(config: dict) -> dict | None:
    extra = {}
    if not config.get("enable_thinking", False):
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    top_k = config.get("vlm_top_k")
    if top_k is not None:
        extra["top_k"] = top_k
    return extra if extra else None
