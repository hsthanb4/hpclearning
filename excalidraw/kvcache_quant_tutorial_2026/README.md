# KV Cache 量化教程：TurboQuant 与常用方法

更新时间：2026-05-25

本目录包含 3 张标准 `.excalidraw` 图稿，配合仓库根目录的 `kv_cache_quantization_tutorial_2026.html` 使用。

## 文件

1. `00_kvcache_quant_overview.excalidraw`：KV cache 内存公式、读写路径、量化插入点和常见误区。
2. `01_turboquant_pipeline.excalidraw`：TurboQuant 论文算法和 vLLM 工程实现的差异。
3. `02_kvcache_quant_methods_comparison.excalidraw`：FP8、HF QuantizedCache、KIVI、KVQuant、GEAR/SKVQ/WKVQuant、TurboQuant 的对比和选型。
4. `generate_kvcache_quant_tutorial.py`：重新生成 HTML 与 Excalidraw 的脚本。

## 主要资料源

- TurboQuant paper: https://arxiv.org/abs/2504.19874
- TurboQuant OpenReview PDF: https://openreview.net/pdf?id=tO3ASKZlok
- vLLM TurboQuant study: https://vllm.ai/blog/2026-05-11-turboquant
- vLLM quantized KV cache docs: https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/
- vLLM TurboQuant docs: https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/
- Hugging Face KV cache docs: https://huggingface.co/docs/transformers/kv_cache
- TensorRT-LLM GPT attention docs: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
- KIVI: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079
- GEAR: https://arxiv.org/abs/2403.05527
- SKVQ: https://arxiv.org/abs/2405.06219
- WKVQuant: https://arxiv.org/abs/2402.12065
- Quantize What Counts: https://arxiv.org/abs/2502.15075
