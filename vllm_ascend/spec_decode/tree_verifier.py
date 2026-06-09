

class AscendTreeVerifier:
    def verify_depth1(self, logits, trees):
        target_tokens = logits.argmax(dim=-1)

        output_tokens = []
        for i, tree in enumerate(trees):
            target = target_tokens[i].item()
            candidates = tree.token_ids.tolist()

            if target in candidates:
                output_tokens.append(target)
            else:
                output_tokens.append(target)

        return output_tokens