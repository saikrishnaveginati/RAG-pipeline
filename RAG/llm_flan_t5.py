from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_answer(
    query: str,
    contexts: list[str],
    max_new_tokens: int = 200
) -> str:
    """
    Generate an answer using retrieved context chunks.
    
    """

    context_text = "\n\n".join(contexts)

    prompt = f"""
Answer the question using ONLY the context below.


Context:
{context_text}

Question:
{query}

Answer:
"""
    print(context_text)
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0
        )

    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()
