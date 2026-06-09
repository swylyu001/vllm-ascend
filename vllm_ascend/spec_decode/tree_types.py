from dataclasses import dataclass
import torch

@dataclass
class AscendSpecTree:
    token_ids: torch.Tensor        # [batch, num_nodes]
    parent_indices: torch.Tensor   # [num_nodes]
    depths: torch.Tensor           # [num_nodes]
    branch: int
    depth: int
    
@dataclass
class AscendTreeVerifyResult:
    accepted_token_ids: list[int]
    accepted_node_ids: list[int]
    replacement_token_id: int | None

    @property
    def output_token_ids(self) -> list[int]:
        if self.replacement_token_id is None:
            return self.accepted_token_ids
        return self.accepted_token_ids + [self.replacement_token_id]