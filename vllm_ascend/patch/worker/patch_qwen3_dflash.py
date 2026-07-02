from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_dflash import (
    AutoWeightsLoader,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    process_eagle_weight,
)
from vllm.model_executor.models.utils import maybe_prefix

class DominoHead(nn.Module):
    def __init__(self, *, config, vllm_config: VllmConfig, quant_config, prefix: str = ""):
        super().__init__()
        dflash_config = getattr(config, "dflash_config", {}) or {}

        self.gru_hidden_dim = int(dflash_config["gru_hidden_dim"])
        self.emb_dim = int(dflash_config["emb_dim"])

        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        ).to(dtype=torch.float16)

        self.embed_proj = nn.Sequential(
            ReplicatedLinear(
                input_size=config.hidden_size + self.gru_hidden_dim,
                output_size=self.emb_dim,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "embed_proj.0"),
                return_bias=False,
            ),
            nn.SiLU(),
            ReplicatedLinear(
                input_size=self.emb_dim,
                output_size=config.vocab_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "embed_proj.2"),
                return_bias=False,
            ),
        )

    def init_state(self, embed_input_ids, prefix_token_ids: torch.Tensor):
        prefix_embeds = embed_input_ids(prefix_token_ids).to(torch.float16).contiguous()
        _, gru_hidden = self.prefix_gru(prefix_embeds)
        return gru_hidden

    def advance_state(
        self,
        embed_input_ids,
        token_ids: torch.Tensor,
        gru_hidden: torch.Tensor,
    ):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)
        token_embeds = embed_input_ids(token_ids)
        token_embeds = embed_input_ids(token_ids).to(torch.float16).contiguous()
        gru_hidden = gru_hidden.to(torch.float16).contiguous()
        _, gru_hidden = self.prefix_gru(token_embeds, gru_hidden)
        return gru_hidden

    def compute_logits(
        self,
        parallel_hidden: torch.Tensor,
        gru_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ):
        squeeze_time_dim = parallel_hidden.dim() == 2
        if squeeze_time_dim:
            parallel_hidden = parallel_hidden.unsqueeze(1)

        if base_logits.dim() == 2:
            base_logits = base_logits.unsqueeze(1)

        state = gru_hidden.transpose(0, 1).to(dtype=parallel_hidden.dtype)
        correction_input = torch.cat([parallel_hidden, state], dim=-1)
        correction_bias = self.embed_proj(correction_input)

        logits = base_logits + correction_bias
        if squeeze_time_dim:
            logits = logits.squeeze(1)
        return logits


#TODO: For adding the domino head as upstream vllm wouldn't support the domino head right now. Remove once the vllm support the feature 
_origin_init = DFlashQwen3ForCausalLM.__init__

def patched_init(self, *args, **kwargs):
    vllm_config = kwargs.get("vllm_config") or get_current_vllm_config()
    prefix = kwargs.get("prefix", "")

    kwargs["vllm_config"] = vllm_config
    _origin_init(self, *args, **kwargs)

    dflash_config = getattr(self.config, "dflash_config", {}) or {}
    self.projector_type = dflash_config.get("projector_type")
    self.is_domino = self.projector_type == "domino"
    self.domino_head = None

    if self.is_domino:
        self.domino_head = DominoHead(
            config=self.config,
            vllm_config=vllm_config,
            quant_config=self.model.quant_config,
            prefix=maybe_prefix(prefix, "domino_head"),
        )

def load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
):
    model_weights = {}
    includes_draft_id_mapping = False
    includes_embed_tokens = False
    domino_weights = []

    for name, loaded_weight in weights:
        assert "mask_hidden" not in name, (
            "DFlash should use mask_token_id to embed the padding hidden state"
        )

        if getattr(self, "is_domino", False):
            if name.startswith("prefix_gru.") or name.startswith("embed_proj."):
                domino_weights.append((f"domino_head.{name}", loaded_weight))
                continue

        if "t2d" in name:
            continue
        if "d2t" in name:
            name = name.replace("d2t", "draft_id_to_target_id")
            includes_draft_id_mapping = True
        elif "lm_head" not in name:
            name = "model." + name

        if "embed_tokens" in name:
            includes_embed_tokens = True

        model_weights[name] = loaded_weight
        process_eagle_weight(self, name)

    skip_substrs = []
    if not includes_draft_id_mapping:
        skip_substrs.append("draft_id_to_target_id")
    if not includes_embed_tokens:
        skip_substrs.append("embed_tokens")
    if not self.model.use_aux_hidden_state:
        skip_substrs.append("fc.")

    loader = AutoWeightsLoader(
        self,
        skip_prefixes=None,
        skip_substrs=skip_substrs,
    )
    loader.load_weights(model_weights.items())

    if getattr(self, "is_domino", False):
        loader = AutoWeightsLoader(self)
        loader.load_weights(domino_weights)

        domino_param_names = {
            name
            for name, _ in self.named_parameters()
            if name.startswith("domino_head.")
        }
        loaded_domino_names = {name for name, _ in domino_weights}
        missing_domino_names = domino_param_names - loaded_domino_names

        if missing_domino_names:
            raise RuntimeError(
                "Domino weight loading is incomplete. Missing: "
                f"{sorted(missing_domino_names)}"
            )

    self.model._build_fused_kv_buffers()

def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    for i in range(L):
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            context_slot_mapping,
        )

def init_domino_state(self, prefix_token_ids: torch.Tensor) -> torch.Tensor:
    if self.domino_head is None:
        raise RuntimeError("Domino head is not enabled.")
    return self.domino_head.init_state(self.embed_input_ids, prefix_token_ids)

def advance_domino_state(
    self,
    token_ids: torch.Tensor,
    gru_hidden: torch.Tensor,
) -> torch.Tensor:
    if self.domino_head is None:
        raise RuntimeError("Domino head is not enabled.")
    return self.domino_head.advance_state(
        self.embed_input_ids,
        token_ids,
        gru_hidden,
    )

def compute_domino_logits(
    self,
    parallel_hidden: torch.Tensor,
    gru_hidden: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    if self.domino_head is None:
        raise RuntimeError("Domino head is not enabled.")
    return self.domino_head.compute_logits(
        parallel_hidden,
        gru_hidden,
        base_logits,
    )




DFlashQwen3ForCausalLM.__init__ = patched_init
DFlashQwen3ForCausalLM.load_weights = load_weights
DFlashQwen3ForCausalLM.init_domino_state = init_domino_state
DFlashQwen3ForCausalLM.advance_domino_state = advance_domino_state
DFlashQwen3ForCausalLM.compute_domino_logits = compute_domino_logits

DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv