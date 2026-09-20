from transformers import PreTrainedTokenizerBase


def tokenize_text_prompt(
    tokenizer: PreTrainedTokenizerBase | None,
    prompt: str | list[int],
) -> list[int]:
    """Validate and tokenize the text-only engine input contract."""
    if isinstance(prompt, str):
        if tokenizer is None:
            raise RuntimeError("A tokenizer is required for string prompts.")
        add_special_tokens = (
            tokenizer.bos_token is not None
            and not prompt.startswith(tokenizer.bos_token)
        )
        return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    if not isinstance(prompt, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in prompt
    ):
        raise TypeError(
            "SparseEngine accepts text prompts only: prompt must be a string "
            "or a flat list of integer token IDs; structured image, video, "
            "or MTP request objects are unsupported."
        )
    if not prompt:
        raise ValueError("prompt token IDs must not be empty.")
    return list(prompt)
