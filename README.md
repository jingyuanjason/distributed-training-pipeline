# Distributed MoE Training Pipeline

A distributed training implementation for a **BF16 Mixtral 8×7B MoE model**, built on [Stanford CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems) with substantial extensions.

## Implementation

- **Fully Sharded Data Parallel (FSDP):** custom parameter sharding and gradient synchronization in [distributed wrappers](implementation/distributed/wrappers.py).
- **Data Parallel (DP):** data distribution and multidimensional communication-group setup in [training orchestration](train.py).
- **Expert Parallel (EP):** distributed experts and all-to-all token routing in [MoE layers](implementation/layers.py).
- **Pipeline Parallel (PP):** microbatch scheduling and communication overlap in [pipeline training](implementation/pipeline/pipelined_train_overlap.py).

## Beyond the Assignment

The project adds PP and EP, detailed **NVIDIA Nsight Systems profiling**, training outside Modal, and large-scale experiment support with multidimensional communication groups—targeting up to **32 NVIDIA B200/B300 GPUs**. Much larger models and context lengths require distributing training across GPUs.

**Scope note:** the bundled configs are reduced-size examples, not the full 8×7B setup; the current bundled launchers enforce single-node execution. The 32-GPU scale and non-Modal training describe the broader project scope.

## Ongoing Work / Future Directions

- **Training checkpoints:** improve distributed checkpoint saving and restoration for reliable training resumption.
- **Tensor Parallelism (TP):** shard computation within layers across GPUs, complementing the existing PP, DP, FSDP, and EP setup.

## AI-Generated Content

AI generated portions of the training startup scripts, profiling setup, MoE **loss collection** (not the MoE module itself), and project-structure cleanup/refactoring and some local test cases (not included in the repo). AI was used for these routine, relatively trivial support tasks rather than the core implementation.
