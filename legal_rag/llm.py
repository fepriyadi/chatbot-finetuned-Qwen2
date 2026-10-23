"""Klien vLLM pengganti `generate_chat` notebook (Unsloth `llm.generate` -> HTTP ke server vLLM).

Prompt dirender secara lokal dengan chat template tokenizer, lalu dikirim ke `/v1/completions`
(bukan `/v1/chat/completions`) agar prefill `<think>\\n` setelah header assistant tetap bisa dipakai.
"""
from openai import OpenAI
from transformers import AutoTokenizer

from legal_rag.config import HF_TOKEN, LLM_MODEL, THINK_CLOSE, THINK_OPEN, VLLM_API_KEY, VLLM_BASE_URL


class VLLMClient:
    def __init__(self, base_url=VLLM_BASE_URL, model=LLM_MODEL, api_key=VLLM_API_KEY, hf_token=HF_TOKEN):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
        # hanya tokenizer (chat template), bobot model tetap di server vLLM
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)

    def is_ready(self):
        try:
            return any(m.id == self.model for m in self.client.models.list().data)
        except Exception:
            return False

    def generate_chat(self, messages, max_new_tokens=512, temperature=0.2, num_return_sequences=1, assistant_prefix=""):
        """Generate dari model fine-tuned. `assistant_prefix` ditempel setelah header assistant (prefill)."""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + assistant_prefix
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,   # 0 -> greedy, sama dengan do_sample=False
            top_p=0.9,
            n=num_return_sequences,
            extra_body={"repetition_penalty": 1.1, "add_special_tokens": False, "skip_special_tokens": True},
        )
        choices = sorted(response.choices, key=lambda c: c.index)
        return [assistant_prefix + c.text for c in choices]


def extract_thinking(text):
    """Isi <think>...</think>. Toleran terhadap penalaran yang terpotong max_new_tokens (tanpa </think>)."""
    if THINK_OPEN not in text:
        return ""
    after_open = text.split(THINK_OPEN, 1)[1]
    return after_open.split(THINK_CLOSE, 1)[0].strip()


def extract_final_answer(text):
    """Jawaban akhir = teks setelah </think> terakhir.
    Ada <think> tapi tidak ditutup -> model belum selesai berpikir -> belum ada jawaban akhir.
    Tidak ada tag sama sekali -> seluruh teks dianggap jawaban."""
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1].strip()
    if THINK_OPEN in text:
        return ""
    return text.strip()
