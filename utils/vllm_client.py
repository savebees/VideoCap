from openai import OpenAI


def get_vlm_client(config: dict) -> OpenAI:
    return OpenAI(
        base_url=config.get("vllm_base_url", "http://localhost:8000/v1"),
        api_key="not-needed",
    )
