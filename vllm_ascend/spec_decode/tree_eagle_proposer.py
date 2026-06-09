import os

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.tree_types import AscendSpecTree
from vllm_ascend.utils import lmhead_tp_enable


class AscendTreeEagleProposer(AscendEagleProposer):
    def __init__(self, *args, tree_branch: int = 2, tree_depth: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree_branch = tree_branch
        self.tree_depth = tree_depth
        self.last_tree: AscendSpecTree | None = None
        self.is_tree_proposer = True

        self._tree_dump_tokenizer = None
        if os.getenv("VLLM_ASCEND_TREE_DUMP", "0") == "1":
            try:
                from transformers import AutoTokenizer

                tokenizer_name = (
                    getattr(vllm_config.model_config, "tokenizer", None)
                    or vllm_config.model_config.model
                )
                self._tree_dump_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(
                    f"[TreeEagle3Dump] failed to load tokenizer: {e}",
                    flush=True,
                )

    def _dump_tree_proposal(self, draft_token_ids: torch.Tensor) -> None:
        import os
        from vllm.distributed import get_world_group

        if get_world_group().rank != 0:
            return

        if os.getenv("VLLM_ASCEND_TREE_DUMP", "0") != "1":
            return

        # Avoid dumping every step forever.
        dump_limit = int(os.getenv("VLLM_ASCEND_TREE_DUMP_LIMIT", "20"))
        dump_count = getattr(self, "_tree_dump_count", 0)
        if dump_count >= dump_limit:
            return
        self._tree_dump_count = dump_count + 1

        ids_cpu = draft_token_ids.detach().cpu().tolist()

        parent_indices = getattr(self, "last_tree", None)
        if parent_indices is not None:
            parents = self.last_tree.parent_indices.detach().cpu().tolist()
            depths = self.last_tree.depths.detach().cpu().tolist()
        else:
            parents = [-1] * draft_token_ids.shape[-1]
            depths = [1] * draft_token_ids.shape[-1]

        token_strings = None
        tokenizer = getattr(self, "_tree_dump_tokenizer", None)
        if tokenizer is not None:
            token_strings = [
                [
                    tokenizer.convert_ids_to_tokens(int(tok_id))
                    for tok_id in row
                ]
                for row in ids_cpu
            ]

        print(
            "[TreeEagle3Dump] "
            f"count={dump_count} "
            f"shape={list(draft_token_ids.shape)} "
            f"branch={getattr(self, 'tree_branch', draft_token_ids.shape[-1])} "
            f"depth=1 "
            f"parents={parents} "
            f"depths={depths}",
            flush=True,
        )

        for req_idx, row in enumerate(ids_cpu):
            print(
                f"[TreeEagle3Dump] req={req_idx} tree:",
                flush=True,
            )
            print(
                f"[TreeEagle3Dump] req={req_idx} root",
                flush=True,
            )

            for node_idx, tok_id in enumerate(row):
                connector = "└─" if node_idx == len(row) - 1 else "├─"

                token_repr = ""
                if token_strings is not None:
                    token_repr = f" token={token_strings[req_idx][node_idx]!r}"

                parent = parents[node_idx] if node_idx < len(parents) else -1
                depth = depths[node_idx] if node_idx < len(depths) else 1

                print(
                    "[TreeEagle3Dump] "
                    f"req={req_idx}   {connector} "
                    f"node={node_idx} "
                    f"parent={parent} "
                    f"depth={depth} "
                    f"id={int(tok_id)}"
                    f"{token_repr}",
                    flush=True,
                )



    def sample_draft_token_ids_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        draft_token_ids = torch.topk(
            logits,
            k=self.tree_branch,
            dim=-1,
        ).indices.to(torch.int32)

        self.last_tree = AscendSpecTree(
            token_ids=draft_token_ids,
            parent_indices=torch.full(
                (self.tree_branch,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            depths=torch.ones(
                (self.tree_branch,),
                dtype=torch.int32,
                device=self.device,
            ),
            branch=self.tree_branch,
            depth=1,
        )

        self._dump_tree_proposal(draft_token_ids)
        return draft_token_ids

    # def _propose(self, *args, **kwargs):
    #     if self.tree_depth == 1:
    #         return self._propose_depth1_tree(*args, **kwargs)
    #     return self._propose_depthN_tree_slow(*args, **kwargs)

    # def _propose_depthN_tree_slow(self, *args, **kwargs):
    #     raise NotImplementedError

    # def _propose_depth1_tree(
    #     self,
    #     target_token_ids,
    #     target_positions,
    #     target_hidden_states,
    #     next_token_ids,
    #     token_indices_to_sample,
    #     common_attn_metadata,
    #     target_model_batch_desc,
    #     sampling_metadata,
    #     mm_embed_inputs=None,
    #     req_scheduled_tokens=None,
    #     long_seq_metadata=None,
    #     num_prefill_reqs=0,
    #     num_decode_reqs=0,
    #     scheduler_output=None,
    #     num_scheduled_tokens=0,
    #     num_rejected_tokens_gpu=None,
    # ):
    #     """
    #     Build a depth=1 tree proposal for Eagle3.

    #     Current normal Eagle3 proposer generates a linear draft:

    #         req -> [t1, t2, t3]

    #     This demo tree proposer only generates sibling candidates at depth=1:

    #         req -> [t1, t2, t3, t4]

    #     It does NOT generate children of t1/t2/t3/t4 yet.

    #     Return value:
    #         draft_token_ids: Tensor[int32], shape [batch_size, tree_branch]

    #     This is intentionally compatible with the current vLLM draft-token protocol,
    #     which expects something shaped like [batch_size, num_speculative_tokens].
    #     The difference is semantic:
    #         old: [t1, t2, t3] means a chain
    #         new: [t1, t2, t3] means sibling candidates
    #     """

    #     # Number of active requests in the current draft batch.
    #     batch_size = common_attn_metadata.batch_size()

    #     # token_indices_to_sample tells us which output positions from the draft
    #     # model should be sampled with lm_head.
    #     #
    #     # If caller does not provide it, use the last token position of each request.
    #     #
    #     # query_start_loc looks like:
    #     #   [0, q1, q1 + q2, q1 + q2 + q3]
    #     #
    #     # query_start_loc[1:] - 1 gives:
    #     #   [last index of req0, last index of req1, last index of req2]
    #     if token_indices_to_sample is None:
    #         token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

    #     # Eagle3 is different from plain draft_model/MTP:
    #     # it consumes target model hidden states, often multiple auxiliary hidden
    #     # states, and combines them into the hidden state expected by the Eagle3
    #     # draft model.
    #     #
    #     # In normal Eagle3 path this is also done before feeding the draft model.
    #     target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
    #     assert target_hidden_states.shape[-1] == self.hidden_size

    #     # set_inputs_first_pass prepares the first draft-model input.
    #     #
    #     # Conceptually, target model has just sampled/verified some token:
    #     #
    #     #   next_token_ids
    #     #
    #     # The draft model now predicts candidates after that token.
    #     #
    #     # This helper:
    #     #   1. writes next_token_ids into draft input buffer
    #     #   2. prepares draft positions
    #     #   3. prepares draft hidden_states
    #     #   4. adjusts attention metadata / slot_mapping for draft model
    #     #   5. returns token_indices_to_sample for lm_head
    #     (
    #         num_tokens,
    #         token_indices_to_sample,
    #         common_attn_metadata,
    #         long_seq_args,
    #     ) = self.set_inputs_first_pass(
    #         target_token_ids=target_token_ids,
    #         next_token_ids=next_token_ids,
    #         target_positions=target_positions,
    #         target_hidden_states=target_hidden_states,
    #         token_indices_to_sample=token_indices_to_sample,
    #         cad=common_attn_metadata,
    #         num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    #         req_scheduled_tokens=req_scheduled_tokens,
    #         long_seq_metadata=long_seq_metadata,
    #         num_prefill_reqs=num_prefill_reqs,
    #         num_decode_reqs=num_decode_reqs,
    #     )

    #     # Demo simplification:
    #     # tree proposer first version only supports the simplest path.
    #     #
    #     # PCP/DCP requires extra token reordering and context-parallel metadata.
    #     # For tree decode, that would make the first demo much harder to reason about.
    #     if self.pcp_size * self.dcp_size > 1:
    #         raise NotImplementedError(
    #             "Tree Eagle3 proposer demo does not support PCP/DCP yet."
    #         )

    #     assert self.runner is not None

    #     # Whether LoRA is active affects graph dispatch shape/runtime mode.
    #     has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0

    #     # Whether this is uniform decode batch. Existing Ascend graph dispatcher
    #     # needs this to choose correct graph/eager path.
    #     uniform_decode = target_model_batch_desc.uniform

    #     # Decide padded token count for draft model.
    #     #
    #     # In eager mode:
    #     #   num_input_tokens = num_tokens
    #     #
    #     # In graph mode:
    #     #   num_input_tokens may be padded to graph batch shape.
    #     if self.use_cuda_graph:
    #         _, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
    #             num_tokens=num_tokens,
    #             uniform_decode=uniform_decode,
    #             has_lora=has_lora,
    #         )
    #         num_input_tokens = batch_descriptor.num_tokens
    #     else:
    #         num_input_tokens = num_tokens

    #     # Sync token counts across DP ranks.
    #     #
    #     # Some distributed paths require every rank to agree on batch/token shapes.
    #     (
    #         num_input_tokens,
    #         num_tokens_across_dp,
    #         _,
    #     ) = self.runner._sync_metadata_across_dp(
    #         num_input_tokens,
    #         is_draft_model=True,
    #     )

    #     # Re-dispatch after DP sync because num_input_tokens may have changed.
    #     if self.use_cuda_graph:
    #         aclgraph_runtime_mode, batch_descriptor = (
    #             self.runner.cudagraph_dispatcher.dispatch(
    #                 num_tokens=num_input_tokens,
    #                 uniform_decode=uniform_decode,
    #                 has_lora=has_lora,
    #             )
    #         )
    #         num_input_tokens = batch_descriptor.num_tokens
    #     else:
    #         aclgraph_runtime_mode = CUDAGraphMode.NONE
    #         batch_descriptor = None

    #     # If running full graph mode, metadata must be padded to graph shape.
    #     #
    #     # For a minimal demo, you can disable graph mode and this block will not
    #     # matter. It is kept here because normal Eagle3 proposer has to handle it.
    #     if aclgraph_runtime_mode == CUDAGraphMode.FULL:
    #         num_reqs_padded = self.runner._pad_query_start_loc_for_fia(
    #             num_input_tokens,
    #             batch_descriptor.num_reqs
    #             if batch_descriptor.num_reqs is not None
    #             else common_attn_metadata.num_reqs,
    #             common_attn_metadata.num_reqs,
    #             aclgraph_runtime_mode,
    #             batch_descriptor.num_reqs,
    #         )

    #         common_attn_metadata.num_reqs = num_reqs_padded
    #         common_attn_metadata.query_start_loc = (
    #             self.runner.query_start_loc.gpu[: num_reqs_padded + 1]
    #         )
    #         common_attn_metadata.query_start_loc_cpu = (
    #             self.runner.query_start_loc.cpu[: num_reqs_padded + 1]
    #         )

    #         common_attn_metadata.block_table_tensor = self._adjust_tensor(
    #             common_attn_metadata.block_table_tensor,
    #             num_reqs_padded,
    #         )

    #         common_attn_metadata.seq_lens = self._adjust_tensor(
    #             self.runner.seq_lens,
    #             num_reqs_padded,
    #         )
    #         common_attn_metadata.seq_lens_cpu = self._adjust_tensor(
    #             self.runner.optimistic_seq_lens_cpu,
    #             num_reqs_padded,
    #         )

    #         # Keep upstream mirror aligned if it exists.
    #         if common_attn_metadata._seq_lens_cpu is not None:
    #             common_attn_metadata._seq_lens_cpu = (
    #                 common_attn_metadata.seq_lens_cpu.clone()
    #             )

    #         if common_attn_metadata.num_computed_tokens_cpu is not None:
    #             common_attn_metadata.num_computed_tokens_cpu = self._adjust_tensor(
    #                 common_attn_metadata.num_computed_tokens_cpu,
    #                 num_reqs_padded,
    #             )
    #     else:
    #         # Eager mode path.
    #         num_reqs_padded = common_attn_metadata.num_reqs

    #         # For non-MLA attention, keep block table shape aligned with req count.
    #         if not self.vllm_config.model_config.use_mla:
    #             common_attn_metadata.block_table_tensor = self._adjust_tensor(
    #                 common_attn_metadata.block_table_tensor,
    #                 num_reqs_padded,
    #             )

    #     # Multimodal models may need inputs_embeds instead of pure input_ids.
    #     # For text-only models, inputs_embeds is None.
    #     if self.supports_mm_inputs:
    #         mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
    #         inputs_embeds = self.model.embed_input_ids(
    #             self.input_ids[:num_tokens],
    #             multimodal_embeddings=mm_embeds,
    #             is_multimodal=is_mm_embed,
    #         )
    #         self.inputs_embeds[:num_tokens] = inputs_embeds
    #         inputs_embeds = self.inputs_embeds[:num_input_tokens]
    #     else:
    #         inputs_embeds = None

    #     # Copy slot_mapping into private draft buffers.
    #     #
    #     # The target model runner and draft proposer share some metadata buffers.
    #     # Existing Eagle path snapshots slot_mapping/seq_lens/query_start_loc so
    #     # draft forward does not get corrupted by later metadata changes.
    #     slot_mapping_lens = common_attn_metadata.slot_mapping.shape[0]
    #     self.slot_mapping_group[0][:slot_mapping_lens].copy_(
    #         common_attn_metadata.slot_mapping
    #     )
    #     self.slot_mapping_group[0][slot_mapping_lens:].fill_(-1)
    #     common_attn_metadata.slot_mapping = self.slot_mapping_group[0]

    #     self.seq_lens_group[0][:num_reqs_padded].copy_(
    #         common_attn_metadata.seq_lens
    #     )
    #     self.seq_lens_group[0][num_reqs_padded:].fill_(0)
    #     common_attn_metadata.seq_lens = self.seq_lens_group[0][:num_reqs_padded]

    #     self.query_start_loc_group[0][: num_reqs_padded + 1].copy_(
    #         common_attn_metadata.query_start_loc
    #     )
    #     self.query_start_loc_group[0][num_reqs_padded + 1 :].fill_(0)
    #     common_attn_metadata.query_start_loc = (
    #         self.query_start_loc_group[0][: num_reqs_padded + 1]
    #     )

    #     common_attn_metadata.num_input_tokens = num_input_tokens

    #     # Build attention metadata for draft model.
    #     #
    #     # This is the metadata consumed by Ascend attention kernels during
    #     # draft model forward.
    #     assert len(self.draft_attn_groups) > 0
    #     builder = self.draft_attn_groups[0].get_metadata_builder()

    #     extra_attn_metadata_args = {}
    #     if self.use_compress:
    #         extra_attn_metadata_args = dict(
    #             prefill_ratio_to_sas_metadata=dict(),
    #             decode_ratio_to_sas_metadata=dict(),
    #             common_ratio_to_sas_metadata=dict(),
    #             block_size=self.draft_attn_groups[0].kv_cache_spec.block_size,
    #         )

    #     attn_metadata = builder.build(
    #         0,
    #         common_attn_metadata,
    #         self.runner.get_model(),
    #         **extra_attn_metadata_args,
    #     )

    #     # Some attention backends may mark metadata as non-causal.
    #     # If non-causal, clear the mask as existing path does.
    #     if hasattr(attn_metadata, "causal") and not attn_metadata.causal:
    #         attn_metadata.attn_mask = None

    #     # ForwardContext expects per-layer attention metadata.
    #     # All draft attention layers use the same metadata object here.
    #     per_layer_attn_metadata = {}
    #     for layer_name in self.attn_layer_names:
    #         per_layer_attn_metadata[layer_name] = attn_metadata

    #     forward_context = get_forward_context()
    #     if forward_context is not None:
    #         forward_context.attn_metadata = per_layer_attn_metadata
    #         forward_context.moe_layer_index = 0

    #     # Extra context used by some Ascend kernels / attention paths.
    #     _EXTRA_CTX.num_tokens = num_input_tokens
    #     _EXTRA_CTX.num_accept_tokens = batch_size

    #     # Prepare draft model inputs.
    #     model_input_ids = self.input_ids[:num_input_tokens]
    #     model_positions = self._get_positions(num_input_tokens)
    #     model_hidden_states = self.hidden_states[:num_input_tokens]

    #     # Eagle3 passes hidden_states to draft model.
    #     # maybe_pad_and_reduce handles internal padding / tensor-parallel reduction.
    #     if self.pass_hidden_states_to_model:
    #         model_hidden_states = self.hidden_states[:num_input_tokens]
    #         model_hidden_states, model_positions = self.maybe_pad_and_reduce(
    #             model_hidden_states,
    #             model_positions,
    #         )

    #     model_kwargs = {
    #         "input_ids": model_input_ids,
    #         "positions": model_positions,
    #         "inputs_embeds": inputs_embeds,
    #     }
    #     if self.pass_hidden_states_to_model:
    #         model_kwargs["hidden_states"] = model_hidden_states

    #     # This is the actual Eagle3 draft model forward.
    #     ret_hidden_states = self.model(**model_kwargs)

    #     # Different draft models return either:
    #     #   hidden_states
    #     # or:
    #     #   (last_hidden_states, hidden_states)
    #     if not self.model_returns_tuple():
    #         last_hidden_states = ret_hidden_states
    #         hidden_states = last_hidden_states
    #     else:
    #         last_hidden_states, hidden_states = ret_hidden_states

    #     # Handle tensor parallel / DP unpadding / all-gather.
    #     last_hidden_states, model_positions, hidden_states = (
    #         self.maybe_all_gather_and_unpad(
    #             last_hidden_states,
    #             model_positions,
    #             hidden_states,
    #         )
    #     )

    #     # token_indices_to_sample selects which hidden states feed lm_head.
    #     # For normal decode, this is one index per request.
    #     num_indices = token_indices_to_sample.shape[0]

    #     if lmhead_tp_enable():
    #         max_num_reqs_across_dp = (
    #             self.vllm_config.scheduler_config.max_num_seqs
    #             * self.runner.uniform_decode_query_len
    #         )
    #         token_indices_to_sample = nn.functional.pad(
    #             token_indices_to_sample,
    #             (0, max_num_reqs_across_dp - num_indices),
    #         )

    #     sample_hidden_states = last_hidden_states[token_indices_to_sample]

    #     # For tree top-k we need full logits.
    #     # Existing reduce_sample path may only return sampled ids, which is not
    #     # enough to construct sibling candidates.
    #     if get_ascend_config().enable_reduce_sample:
    #         raise NotImplementedError(
    #             "Tree Eagle3 proposer needs full logits for top-k; "
    #             "disable enable_reduce_sample for this demo."
    #         )

    #     # Compute draft logits from selected hidden states.
    #     logits = self.model.compute_logits(sample_hidden_states)

    #     if lmhead_tp_enable() and num_indices < logits.shape[0]:
    #         logits = logits[:num_indices]
    #         token_indices_to_sample = token_indices_to_sample[:num_indices]

    #     # This is the only conceptual difference from linear Eagle3 proposer.
    #     #
    #     # Linear proposer:
    #     #     draft_token_ids = logits.argmax(dim=-1)
    #     #
    #     # Depth=1 tree proposer:
    #     #     draft_token_ids = top-k sibling candidates
    #     #
    #     # Shape:
    #     #     logits:          [batch_size, vocab_size]
    #     #     draft_token_ids: [batch_size, tree_branch]
    #     draft_token_ids = torch.topk(
    #         logits,
    #         k=self.tree_branch,
    #         dim=-1,
    #     ).indices.to(torch.int32)

    #     # Tree topology for depth=1:
    #     #
    #     #   root
    #     #    ├── candidate 0
    #     #    ├── candidate 1
    #     #    └── candidate k-1
    #     #
    #     # Every node is a root child:
    #     #     parent = -1
    #     #
    #     # Every node is at depth 1:
    #     #     depth = 1
    #     parent_indices = torch.full(
    #         (self.tree_branch,),
    #         -1,
    #         dtype=torch.int32,
    #         device=self.device,
    #     )
    #     depths = torch.ones(
    #         (self.tree_branch,),
    #         dtype=torch.int32,
    #         device=self.device,
    #     )

    #     # Save sidecar tree metadata inside vllm-ascend.
    #     #
    #     # The returned draft_token_ids is still just a tensor, because current
    #     # vLLM protocol only understands draft token tensors/lists.
    #     #
    #     # last_tree tells vllm-ascend that the returned tokens are siblings,
    #     # not a linear chain.
    #     self.last_tree = AscendSpecTree(
    #         token_ids=draft_token_ids,
    #         parent_indices=parent_indices,
    #         depths=depths,
    #         branch=self.tree_branch,
    #         depth=1,
    #     )


    #     self._dump_tree_proposal(draft_token_ids)

    #     # Return shape [batch_size, tree_branch].
    #     #
    #     # This is compatible with existing draft-token transfer code, but semantic
    #     # meaning is different:
    #     #
    #     #   old: each row is a chain
    #     #   new: each row is depth=1 sibling candidates
    #     return draft_token_ids