from __future__ import annotations

import subprocess
from pathlib import Path

import torch


def shift_logits_dream(logits: torch.Tensor) -> torch.Tensor:
    """Align Dream's model logits with the positions they predict."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def verify_dawn_checkout(repo: Path, expected_revision: str) -> str:
    """Verify the pinned official DAWN checkout used by DAWN strategies."""
    repo = repo.resolve()
    if not (repo / ".git").is_dir():
        raise FileNotFoundError(
            f"DAWN checkout not found at {repo}; run scripts/setup_vast.sh"
        )
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise RuntimeError(
            f"DAWN revision mismatch: expected {expected_revision}, found {revision}"
        )
    return revision


class DreamModelAdapter:
    """Small compatibility layer for public Dream and the DAWN Dream fork."""

    def forward_logits(self, model, tokens: torch.Tensor) -> torch.Tensor:
        if model.__class__.__module__ == "model.modeling_dream":
            outputs = model(
                tokens,
                attention_mask="full",
                output_attentions=False,
                return_dict=True,
            )
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return shift_logits_dream(outputs.logits)
        outputs = model.model(tokens, output_attentions=False, return_dict=True)
        return shift_logits_dream(model.lm_head(outputs.last_hidden_state))

    def forward_with_dawn_attention(
        self,
        model,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and DAWN's late-layer averaged attention in one NFE."""
        result = model(
            tokens,
            attention_mask="full",
            output_attentions=False,
            return_attn_scores=True,
            return_dict=True,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "DAWN strategy requires the pinned official DAWN Dream model fork"
            )
        outputs, attention = result
        if attention is None:
            raise RuntimeError("Official DAWN model returned no attention scores")
        return shift_logits_dream(outputs.logits), attention
