"""Minimal OpenAI-compatible endpoint for the E4 rhetoric model (merged QLoRA weights or base + adapter).

    python training/serve_rhetoric.py --model models/rhetoric/merged --port 8001
    python training/serve_rhetoric.py --model meta-llama/Llama-3.2-3B-Instruct --adapter models/rhetoric/adapter
    set VAKTA_LLM_URL=http://127.0.0.1:8001   (content.py then calls POST /v1/chat/completions)

Needs: pip install transformers accelerate (+ peft for --adapter, bitsandbytes for --4bit on CUDA).
On CPU a 3B model takes roughly 1-3 minutes per speech; use a GPU or Ollama for live demos.
"""
from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str = "vakta-rhetoric"
    messages: list[dict[str, str]]
    temperature: float = 0.2
    max_tokens: int = 1200
    response_format: dict[str, Any] | None = None


def load(model_path: str, adapter: str | None, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"device_map": "auto"} if torch.cuda.is_available() else {"torch_dtype": torch.float32}
    if four_bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
    tok = AutoTokenizer.from_pretrained(adapter or model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


def build_app(tok, model, name: str) -> FastAPI:
    import torch

    app = FastAPI(title="VaktaAI rhetoric LLM")

    @app.post("/v1/chat/completions")
    def complete(req: ChatRequest) -> dict[str, Any]:
        ids = tok.apply_chat_template(req.messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=req.max_tokens, do_sample=req.temperature > 0,
                                 temperature=max(req.temperature, 1e-3), pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": name, "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": text}}]}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": name}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    tok, model = load(args.model, args.adapter, args.four_bit)
    uvicorn.run(build_app(tok, model, "vakta-rhetoric"), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
