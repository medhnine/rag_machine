# vLLM Server Guide

## Loading LoRA Adapters

vLLM supports LoRA adapters for serving customized models.
A LoRA adapter can be loaded dynamically while the server is running.
The server exposes the `/v1/load_lora_adapter` endpoint for loading
a new adapter without restarting the model server.

The request includes the LoRA adapter name and the path to the adapter.

## GPU Memory

vLLM uses GPU memory to store model weights and the KV cache.
The gpu_memory_utilization parameter controls how much GPU memory
the engine is allowed to use.

Reducing this value can help avoid out-of-memory errors.

## Tensor Parallelism

Tensor parallelism allows model execution to be distributed across
multiple GPUs. The tensor_parallel_size configuration specifies
the number of GPUs used for tensor parallel execution.

This is useful when a model does not fit on a single GPU.

## Prefix Caching

Prefix caching allows vLLM to reuse KV cache blocks for requests
that share the same prompt prefix.

This can reduce redundant computation when many requests begin
with identical content.