class AscendTreeProposer:
    def __init__(self, branch: int = 2):
        self.branch = branch

    def propose_from_logits(self, logits):
        topk = torch.topk(logits, self.branch, dim=-1).indices
        return topk