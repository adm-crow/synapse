"""Provider-agnostic LLM answer generation for the CLI --ai flag."""

import os
import urllib.request
import urllib.error
import json
from typing import Optional

PROVIDERS = ("anthropic", "openai", "ollama")

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-4o",
    "ollama": "llama3",
}

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "Be concise and direct. "
    "If the answer is not in the context, say so clearly.\n\n"
    "CONTEXT:\n{context}"
)


def detect_provider() -> Optional[str]:
    """Return the first available provider based on env vars / local services."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # Ollama: no key needed, just needs a local server
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return "ollama"
    except Exception:
        pass
    return None


def generate_answer(
    question: str,
    context: str,
    provider: str,
    model: Optional[str] = None,
) -> str:
    """Generate an answer using the specified provider and model."""
    model = model or DEFAULT_MODELS.get(provider, "")
    system = _SYSTEM_PROMPT.format(context=context)

    if provider == "anthropic":
        return _answer_anthropic(question, system, model)
    if provider == "openai":
        return _answer_openai(question, system, model)
    if provider == "ollama":
        return _answer_ollama(question, system, model)

    raise ValueError(
        f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}"
    )


# ── providers ────────────────────────────────────────────────────────────────

def _answer_anthropic(question: str, system: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK not installed. Run: pip install anthropic"
        )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def _answer_openai(question: str, system: str, model: str) -> str:
    try:
        import openai
    except ImportError:
        raise ImportError(
            "OpenAI SDK not installed. Run: pip install openai"
        )
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


def _answer_ollama(question: str, system: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": f"{system}\n\nQuestion: {question}",
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["response"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama request failed: {e}. Is 'ollama serve' running?"
        )
