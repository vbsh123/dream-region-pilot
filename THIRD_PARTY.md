# Third-party source notes

This repository does not vendor Dream.

- Dream/DAWN's confidence and filtering equations in `commit.py` are a small
  adaptation of `lizhuo-luo/DAWN@19c32c28b5bf0475ccdfad853c74fc885f6410cd`,
  `dream/model/generation_utils.py`, distributed under Apache-2.0. The
  original copyright notice identifies NVIDIA Corporation & Affiliates.
- Dream model weights and remote model code are loaded from
  `Dream-org/Dream-v0-Instruct-7B@05334cb9faaf763692dcf9d8737c642be2b2a6ae`
  and remain subject to their upstream terms.
- Official FlowBlock is checked out source-only from
  `Red-EAD/FlowBlock@8f730a2173140792a4324736efdcba27a2bdee75` under its
  Apache-2.0 license. Its code is not imported by the Dream harness.
- OpenAI HumanEval is imported from a source-only checkout at
  `openai/human-eval@6d43fb980f9fee3c892a914eda09951f772ad10d` under its MIT
  license. Its legacy packaging script is not installed.

The `mean_field_repro` implementation follows equations from, but does not
copy source code from:

- *Mean-Field Parallel Decoding for Discrete Diffusion Language Models*, arXiv
  `2606.15805`: Algorithm 1 and normalized predictive overlap
  `1 - JSD(p_i, p_j) / ln(2)`. Its linked `AmeenAli/MDP` repository returned
  404 when pinned sources were prepared, so no official implementation could
  be imported.
- *FlowBlock*, arXiv `2607.17652`, and the pinned official checkout: the
  distinction between maximum active window `W`, per-token confidence
  threshold, and the frontier acceptance-ratio spawn threshold. No FlowBlock
  source is imported or vendored.
