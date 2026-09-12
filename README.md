# Distributed MoE Training Pipeline

A distributed training implementation for a **BF16 Mixtral 8×7B MoE model**, inspired by [Stanford CS336 LLM System Project](https://github.com/stanford-cs336/assignment2-systems) with substantial extensions.

## Implementation

- **Fully Sharded Data Parallel (FSDP):** custom parameter sharding and gradient synchronization in [distributed wrappers](implementation/distributed/wrappers.py).
- **Data Parallel (DP):** data distribution and multidimensional communication-group setup in [training orchestration](train.py).
- **Expert Parallel (EP):** distributed experts and all-to-all token routing in [MoE layers](implementation/layers.py).
- **Pipeline Parallel (PP):** microbatch scheduling and communication overlap in [pipeline training](implementation/pipeline/pipelined_train_overlap.py).

## Topology

The current multi-node topology is designed to match the communication characteristics of each parallelism strategy with the available hardware interconnect:

- **Intra-node: Data Parallel (DP) and Expert Parallel (EP)** groups are placed **within the same node**, where high-speed NVLink is available. Both DP (gradient synchronization) and EP (all-to-all token routing) are bandwidth-intensive, so they benefit from NVLink's high intra-node bandwidth.
- **Inter-node: Pipeline Parallel (PP)** stages are placed **across different nodes**, since PP communication is limited to sending activations forward and gradients backward between adjacent stages—point-to-point transfers that are small enough to tolerate slower inter-node networking.

## Beyond the CS336 Project

The project adds PP and EP, detailed NVIDIA Nsight Systems profiling, training outside Modal, and large-scale experiment support with multidimensional communication groups—targeting up to 32 NVIDIA B200/B300 GPUs. Much larger models and context lengths require distributing training across GPUs.

## Ongoing Work / Future Directions

- **Training checkpoints:** improve distributed checkpoint saving and restoration for reliable training resumption.
- **Tensor Parallelism (TP):** shard computation within layers across GPUs, complementing the existing PP, DP, FSDP, and EP setup.

## AI-Generated Content

AI generated portions of the training startup scripts, profiling setup, MoE loss collection, and project-structure cleanup/refactoring and some local test cases (not included in the repo).
