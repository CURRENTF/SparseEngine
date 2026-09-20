from __future__ import annotations

from collections.abc import Sequence


def get_sparseengine_generate_api(
    model_path: str,
    infer_config: dict | None,
    *,
    deltakv_checkpoint_path: str | None = None,
    sparse_method: str | None = None,
    use_cache: bool = True,
):
    """Build the benchmark generation adapter for the native SparseEngine runtime."""
    from sparseengine import LLM, SamplingParams

    if not use_cache:
        raise ValueError("SparseEngine benchmark generation requires use_cache=True.")

    public_infer_config = dict(infer_config or {})
    if sparse_method is not None:
        configured_method = public_infer_config.get("sparse_method")
        if configured_method is not None and configured_method != sparse_method:
            raise ValueError(
                f"Conflicting sparse_method values: argument={sparse_method!r}, "
                f"infer_config={configured_method!r}."
            )
        public_infer_config["sparse_method"] = sparse_method
    if deltakv_checkpoint_path is not None:
        configured_checkpoint = public_infer_config.get("deltakv_checkpoint_path")
        if configured_checkpoint is not None and configured_checkpoint != deltakv_checkpoint_path:
            raise ValueError(
                "Conflicting deltakv_checkpoint_path values: "
                f"argument={deltakv_checkpoint_path!r}, "
                f"infer_config={configured_checkpoint!r}."
            )
        public_infer_config["deltakv_checkpoint_path"] = deltakv_checkpoint_path

    llm = LLM(model_path, **public_infer_config)

    def generate(
        prompt: str | list[int] | Sequence[str | list[int]],
        **kwargs,
    ):
        if kwargs.get("past_key_values") is not None:
            raise ValueError(
                "The native SparseEngine benchmark adapter does not accept "
                "external past_key_values."
            )
        if isinstance(prompt, (str, list)) and (
            isinstance(prompt, str) or all(isinstance(token_id, int) for token_id in prompt)
        ):
            prompts = [prompt]
            is_single = True
        else:
            prompts = list(prompt)
            is_single = False

        max_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 128))
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 1.0)
        top_k = kwargs.get("top_k", 0)
        if top_k < 0:
            top_k = 0
        if not kwargs.get("do_sample", True):
            temperature = 0.0
        elif temperature < 1e-5:
            temperature = 1e-5

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            eos_token_ids=kwargs.get("eos_token_id"),
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        results = [output["text"] for output in outputs]
        return results[0] if is_single else results

    generate._sparseengine_llm = llm
    generate._sparseengine_infer_config = dict(public_infer_config)
    return generate
