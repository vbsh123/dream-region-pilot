# External source checkouts

`scripts/setup_vast.sh` creates ignored, detached source checkouts here:

- `DAPD` at `05727b08da4cb4008a275123d7d9885dd5714f7c`.
- official `Red-EAD/FlowBlock` at
  `8f730a2173140792a4324736efdcba27a2bdee75`.

FlowBlock is checked out for source comparison only and is not installed into
the Dream environment. Its official LLaDA-2.1/SGLang environment has different
dependency pins and documents roughly 80 GB GPU memory.

The Mean-Field paper links `github.com/AmeenAli/MDP`, but that repository
currently returns 404. `mean_field_repro` therefore implements the published
Algorithm 1 and is explicitly not labelled an official-code result.
