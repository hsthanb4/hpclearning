//练习顺序
// vector_add
// reduce_sum
// gemv
// rmsnorm
// layernorm
// softmax naive
// online softmax
// transpose naive
// transpose tiled
// gemm tiled


for(int i = 0; i<n ;i+= gridsize){
    c[i] = a[i] + b[i];
}



template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
    for(int offset =  16; offset>0; offset>>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
        
    }
    return val;
}



template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val){
    constexpr int NUM_WARPS = (BLOCK_SIZE+31)/32;
    __shared__ float smem[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warp_reduce_sum<BLOCK_SIZE>(val);
    if(lane == 0){
        smem[warp] = val;
    }
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem[lane]: 0.0f;
    if(warp == 0){
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }
}



template<int BLOCK_SIZE>
__global__ __forceinline__ float block_reduce_sum(foat val){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >>5;
    val = warp_reduce_sum<BLOCK_SIZE>(val);
    if(lane == 0){
        smem[warp] = val;
    }
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS)? smem[lane] : 0.0f;
    if(warp == 0){
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }
}






