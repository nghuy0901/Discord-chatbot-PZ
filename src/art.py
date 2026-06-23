import os

from openai import AsyncOpenAI


openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_KEY") or os.getenv("LLM_API_KEY"))


async def draw(model: str, prompt: str) -> str:
    if not (os.getenv("OPENAI_KEY") or os.getenv("LLM_API_KEY")):
        raise RuntimeError("Image generation requires OPENAI_KEY or LLM_API_KEY.")

    response = await openai_client.images.generate(
        model=model or "gpt-image-1",
        prompt=prompt,
        size="1792x1024",
        quality="auto",
        n=1,
    )
    return response.data[0].url


