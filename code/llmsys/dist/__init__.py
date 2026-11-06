"""Distributed-training building blocks written from scratch (Chapter 22).

Modules
-------
launch          spawn N CPU processes with the gloo backend (works on any laptop)
collectives     naive and ring all-reduce / reduce-scatter / all-gather from send/recv
costs           alpha-beta communication model and ZeRO/FSDP memory formulas
ddp             data parallelism with gradient bucketing and comm/compute overlap
zero            ZeRO stage-1 optimizer-state sharding
tensor_parallel Megatron-style column/row parallel linear layers
pipeline        GPipe / 1F1B schedule generator and bubble analysis
"""
