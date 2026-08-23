# Third-party source notes

This repository does not vendor DAPD or Dream.

- DAPD is imported at runtime from
  `quasar529/DAPD@05727b08da4cb4008a275123d7d9885dd5714f7c` and is MIT licensed.
- Dream/DAWN's confidence and filtering equations in `commit.py` are a small
  adaptation of `lizhuo-luo/DAWN@19c32c28b5bf0475ccdfad853c74fc885f6410cd`,
  `dream/model/generation_utils.py`, distributed under Apache-2.0. The
  original copyright notice identifies NVIDIA Corporation & Affiliates.
- Dream model weights and remote model code are loaded from
  `Dream-org/Dream-v0-Instruct-7B@05334cb9faaf763692dcf9d8737c642be2b2a6ae`
  and remain subject to their upstream terms.

The Vast dependency versions follow DAPD's pinned public `env.yml` where they
overlap this minimal pilot.
