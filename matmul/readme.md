### 参考链接：
https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE


### docker启动：
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace gemm-cuda:12.2.2 /bin/bash