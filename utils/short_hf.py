"""
ShortHF: HuggingFace model wrapper for layer pruning (ShortGPT-style).

Supports computing layer-wise importance via block influence and removing
layers for model shortening. Works with LLaMA, BERT, ChatGLM, and other
HuggingFace causal/encoder models.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from metrics import block_influence

# Optional: use HF mirror when default endpoint is unreachable (e.g. in some regions)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class ShortHFModel:
    """
    HuggingFace model wrapper for layer pruning and importance evaluation.

    Resolves the transformer layers via a dot-notation path (e.g. "model.layers")
    and supports removal of the least important layers based on block influence.
    """

    def __init__(
        self,
        model_name: str,
        layers_path: str,
        n_prune_layers: Optional[int] = None,
        mode: str = "hf",
        tokenizer_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model name or local path.
            layers_path: Dot-notation path to the module list of layers,
                e.g. "model.layers" for LLaMA or "bert.encoder.layer" for BERT.
            n_prune_layers: Number of layers to prune. Required when calling
                remove_layers without explicit layer indices.
            mode: One of "hf", "diy", "glm". Loads tokenizer/model accordingly.
            tokenizer_path: Override tokenizer path (used in "diy" mode).
            cache_dir: Optional cache directory for downloads (e.g. for "glm").
        """
        self.n_prune_layers = n_prune_layers
        self._cache_dir = cache_dir

        if mode == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                local_files_only=True,
            )

        elif mode == "diy":
            path = tokenizer_path or model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                model_max_length=256,
            )
            self.tokenizer.pad_token_id = 0
            self.model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )

        elif mode == "glm":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "THUDM/chatglm-6b",
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                "THUDM/chatglm-6b",
                trust_remote_code=True,
                cache_dir=cache_dir or "./cache",
                torch_dtype=torch.float16,
            )

        if "bert" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = (
                    self.tokenizer.eos_token_id or 0
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            self.layers = self.model.bert.encoder.layer
        else:
            modules = layers_path.split(".")
            mod = self.model
            for m in modules:
                mod = getattr(mod, m)
            self.layers = mod

        self.model.to("cuda")
        self.importances = [0.0] * len(self.layers)

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = None,
        angular: bool = False,
    ) -> List[int]:
        """
        Remove specified layers or the least important consecutive/global layers.

        Args:
            layers_to_remove: Indices of layers to remove. If None and
                n_prune_layers is set, layers are chosen by importance.
            angular: If True, remove n_prune_layers consecutive layers starting
                from the least important (angular block influence).

        Returns:
            List of layer indices that were removed.
        """
        if layers_to_remove is None:
            layers_to_remove = []

        if angular:
            assert self.importances, (
                "Compute importances first via eval_importance()."
            )
            assert self.n_prune_layers is not None, (
                "Set n_prune_layers when using angular removal."
            )
            n = self.n_prune_layers
            start_layer = int(
                np.argsort(np.array(self.importances[: -n + 1]))[0]
            )
            layers_to_remove = list(range(start_layer, start_layer + n))
        elif not layers_to_remove and self.n_prune_layers is not None:
            assert self.importances, (
                "Compute importances first via eval_importance()."
            )
            layers_to_remove = (
                np.argsort(np.array(self.importances))[: self.n_prune_layers]
                .tolist()
            )

        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.layers[layer_idx]
            except IndexError:
                print(
                    f"Layer {layer_idx} does not exist; "
                    "remove_layers may have been called already."
                )
                return []
        return layers_to_remove

    def _compute_bi(
        self,
        hiddens: List[torch.Tensor],
        angular: bool,
        n: int,
    ) -> None:
        """Accumulate block influence from a forward pass into layer importances."""
        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i + n]
            if angular:
                # Use only last token for angular distance (ShortGPT §3.2)
                # https://arxiv.org/abs/2403.17887
                in_hidden = in_hidden[:, -1:]
                out_hidden = out_hidden[:, -1:]
            self.importances[i] += (
                block_influence(
                    in_hidden,
                    out_hidden,
                    angular=angular,
                )
                .mean()
                .cpu()
                .item()
            )

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9,
        angular: bool = False,
        n: int = 1,
    ) -> int:
        """
        Compute layer-wise importance over prompts using block influence.

        Uses a sliding window over the tokenized prompts. ShortGPT uses
        max_gen_len=0 (no generation) during importance computation.

        Args:
            prompts: Input texts to evaluate.
            max_seq_len: Maximum sequence length (window size).
            stride: Token stride between windows.
            max_gen_len: Max new tokens to generate; 0 for forward-only.
            temperature: Sampling temperature when max_gen_len > 0.
            top_p: Nucleus sampling top-p when max_gen_len > 0.
            angular: Use angular distance for block influence.
            n: Block span for influence (input layer i vs output layer i+n).

        Returns:
            Number of windows (forward passes) used.
        """
        prompt_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_seq_len,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = input_ids.shape[1]
        calc_times = 0

        for start in range(0, max_prompt_len, stride):
            inputs = input_ids[0:1, start : start + max_seq_len]
            attn = attn_mask[0:1, start : start + max_seq_len]

            if max_gen_len == 0:
                outputs = self.model(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    output_hidden_states=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    max_new_tokens=max_gen_len,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            self._compute_bi(outputs.hidden_states, angular=angular, n=n)
            calc_times += 1

        return calc_times
