# External source checkouts

`scripts/setup_vast.sh` creates ignored, detached source checkouts here:

- `DAPD` at `05727b08da4cb4008a275123d7d9885dd5714f7c`.
- official `Red-EAD/FlowBlock` at
  `8f730a2173140792a4324736efdcba27a2bdee75`.
- official `lizhuo-luo/DAWN` at
  `19c32c28b5bf0475ccdfad853c74fc885f6410cd`.

FlowBlock is checked out for source comparison only and is not installed into
the Dream environment. Its official LLaDA-2.1/SGLang environment has different
dependency pins and documents roughly 80 GB GPU memory.

The Mean-Field paper links `github.com/AmeenAli/MDP`, but that repository
currently returns 404. `mean_field_repro` therefore implements the published
Algorithm 1 and is explicitly not labelled an official-code result.

DAWN's Dream model fork is imported source-only by the regional DAWN
strategies so logits and late-layer attention are produced in one forward.
Its repository is MIT licensed; the modified Dream model files retain their
upstream Apache-2.0 notices.
