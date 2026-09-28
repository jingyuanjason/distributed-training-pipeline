// flash_attention.cu
// FlashAttention CUDA Kernel Submission
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <vector>

using namespace nvcuda;
using namespace cute;

// Constants
#define WARP_SIZE 32
#define NUM_WARP 4
#define THREAD_PER_BLOCK 128
#define SMEM_STRIDE 32
#define QUERY_PER_BLOCK 32
#define FRAG_SIZE 16

#define HALF_TILE_STRIDE 40   
#define Q_TILE_STRIDE 136     
#define KQ_TILE_STRIDE 36     
constexpr int STAGES = 2;

using QTileShape = Shape<Int<32>, Int<128>>;

using MmaAtomPV = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMmaPV = TiledMMA<MmaAtomPV, Layout<Shape<_2, _2, _1>>,
                            Tile<_32, _32, _32>>;
// ------------------------------------------------------------------------
// CUDA Kernel Implementation
// ------------------------------------------------------------------------


template <int HEAD_DIM>
__global__ void flash_attention_kernel_v1(
    // add shared memory for calculation
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,

    const float scale_log2e) {
    static_assert(HEAD_DIM == 128,
                  "tile stride Q_TILE_STRIDE assumes head_dim 128");
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        STAGES
    > pipeline_state;
    auto block = cooperative_groups::this_thread_block();

    cuda::pipeline<cuda::thread_scope_block> pipeline =
        cuda::make_pipeline(block, &pipeline_state);

    int threadId = (size_t)threadIdx.x;

    int headStartSeqIdx = blockIdx.y * seq_len;
    int blockStartSeqIdx = headStartSeqIdx + blockIdx.x * QUERY_PER_BLOCK;
    int threadIdxInWarp = threadId % WARP_SIZE;
    const int num_seq_const = batch_size * num_heads * seq_len;


    const int warpId  = threadIdx.x / 32;
    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    auto tileKnV_smem_layout = make_layout(make_shape(Int<STAGES>{}, Int<QUERY_PER_BLOCK>{}, Int<HALF_TILE_STRIDE>{}), LayoutRight{});

    auto tileQ_smem_layout = make_layout(make_shape(Int<QUERY_PER_BLOCK>{}, Int<128>{}), make_stride(Int<Q_TILE_STRIDE>{}, Int<1>{}));
    auto tileKQ_smem_layout = make_layout(make_shape(Int<QUERY_PER_BLOCK>{}, Int<KQ_TILE_STRIDE>{}), LayoutRight{});
    auto tileKQHalf_smem_layout = make_layout(make_shape(Int<QUERY_PER_BLOCK>{}, Int<HALF_TILE_STRIDE>{}), LayoutRight{});
    auto rowStats_smem_layout = make_layout(make_shape(Int<QUERY_PER_BLOCK>{}));
    auto global_Qlayout = make_layout(make_shape(num_seq_const, Int<HEAD_DIM>{}), LayoutRight{});

    __shared__ __align__(32) half  tileKnV[cosize_v<decltype(tileKnV_smem_layout)>];
    __shared__ __align__(32) half  tileQ[cosize_v<decltype(tileQ_smem_layout)>];
    __shared__ __align__(32) float tileKQ[cosize_v<decltype(tileKQ_smem_layout)>];
    __shared__ __align__(32) half  tileKQHalf[cosize_v<decltype(tileKQHalf_smem_layout)>];
    __shared__ float sumSeq[cosize_v<decltype(rowStats_smem_layout)>];
    __shared__ float maxExpPrev[cosize_v<decltype(rowStats_smem_layout)>];
    __shared__ float maxExpSeq[cosize_v<decltype(rowStats_smem_layout)>];
    __shared__ float diffExp[cosize_v<decltype(rowStats_smem_layout)>];

    Tensor tKnV = make_tensor(make_smem_ptr(tileKnV), tileKnV_smem_layout);
    Tensor tQ = make_tensor(make_smem_ptr(tileQ), tileQ_smem_layout);
    Tensor tKQ = make_tensor(make_smem_ptr(tileKQ), tileKQ_smem_layout);
    Tensor tKQHalf = make_tensor(make_smem_ptr(tileKQHalf), tileKQHalf_smem_layout);
    Tensor tSumSeq = make_tensor(make_smem_ptr(sumSeq), rowStats_smem_layout);
    Tensor tMaxExpPrev = make_tensor(make_smem_ptr(maxExpPrev), rowStats_smem_layout);
    Tensor tMaxExpSeq = make_tensor(make_smem_ptr(maxExpSeq), rowStats_smem_layout);
    Tensor tDiffExp = make_tensor(make_smem_ptr(diffExp), rowStats_smem_layout);
    Tensor globQ = make_tensor(make_gmem_ptr(Q), global_Qlayout);

    tSumSeq(threadIdxInWarp) = 0.f;
    tMaxExpPrev(threadIdxInWarp) = -INFINITY;
    tMaxExpSeq(threadIdxInWarp) = -INFINITY;

    TiledMmaPV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(threadId);
    Tensor gO = make_tensor(make_gmem_ptr(O + (size_t)blockStartSeqIdx * HEAD_DIM),
                            make_shape(Int<QUERY_PER_BLOCK>{}, Int<HEAD_DIM>{}),
                            LayoutRight{});
    Tensor cChunk = make_identity_tensor(make_shape(Int<QUERY_PER_BLOCK>{}, Int<SMEM_STRIDE>{}));
    using OFragT = decltype(thr_mma_pv.partition_fragment_C(
        local_tile(gO, Shape<Int<QUERY_PER_BLOCK>, Int<SMEM_STRIDE>>{}, make_coord(0, 0))));
    using OCoordT = decltype(thr_mma_pv.partition_C(
        local_tile(cChunk, Shape<Int<QUERY_PER_BLOCK>, Int<SMEM_STRIDE>>{}, make_coord(0, 0))));
    OFragT tCrO[HEAD_DIM / SMEM_STRIDE];
    OCoordT tCcO[HEAD_DIM / SMEM_STRIDE];
    #pragma unroll
    for(int c = 0; c < HEAD_DIM / SMEM_STRIDE; ++c){
      tCrO[c] = thr_mma_pv.partition_fragment_C(local_tile(gO, Shape<Int<QUERY_PER_BLOCK>, Int<SMEM_STRIDE>>{}, make_coord(0, c)));
      clear(tCrO[c]);
      tCcO[c] = thr_mma_pv.partition_C(local_tile(cChunk, Shape<Int<QUERY_PER_BLOCK>, Int<SMEM_STRIDE>>{}, make_coord(0, c)));
    }

    auto s2r_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma_pv);
    auto s2r_thr_A  = s2r_copy_A.get_thread_slice(threadId);
    auto s2r_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, half>{}, tiled_mma_pv);
    auto s2r_thr_B  = s2r_copy_B.get_thread_slice(threadId);

    const int currentRow = warpId * (QUERY_PER_BLOCK / NUM_WARP) + (threadIdxInWarp >> 2);
    const int currentCol = (threadIdxInWarp & 3) * 8;  // in halves (8 halves = 16 B)

    wmma::fragment<wmma::matrix_a, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE,
                   half, wmma::row_major> a_frags[HEAD_DIM / FRAG_SIZE];

    wmma::fragment<wmma::matrix_b, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE,
                   half, wmma::col_major> b_frag;

    wmma::fragment<wmma::accumulator, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE,
                   float> acc_frag;

    {
        auto block_tile = local_tile(
            globQ,
            QTileShape{},
            make_coord(blockIdx.y * (seq_len / QUERY_PER_BLOCK) + blockIdx.x, 0)
        );
        auto tiled_copy = make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half>{},
            Layout<Shape<_8, _16>, Stride<_16, _1>>{},  // 128 threads: 16 along contiguous dim
            Layout<Shape<_1,  _8>>{}                    // 8 contiguous halves (16B) per thread
        );

        auto thread_copy =
            tiled_copy.get_thread_slice(threadIdx.x);

        Tensor thread_global_source =
            thread_copy.partition_S(block_tile);

        Tensor thread_shared_destination =
            thread_copy.partition_D(tQ);

        pipeline.producer_acquire();
        copy(
            tiled_copy,
            thread_global_source,
            thread_shared_destination
        );
        pipeline.producer_commit();
        pipeline.consumer_wait();
        #pragma unroll
        for (int f = 0; f < HEAD_DIM / FRAG_SIZE; ++f) {
          wmma::load_matrix_sync(a_frags[f],
                                 &tQ(warpRow * FRAG_SIZE, f * FRAG_SIZE),
                                 Q_TILE_STRIDE);
        }
        pipeline.consumer_release();
    }

    for(int kvOffset=0;kvOffset<seq_len; kvOffset += QUERY_PER_BLOCK){


      wmma::fill_fragment(acc_frag, 0.0f);


      pipeline.producer_acquire();

      cuda::memcpy_async(
      &tKnV(0, currentRow, currentCol),
      &K[(size_t)(headStartSeqIdx + kvOffset + currentRow) * HEAD_DIM + currentCol],
      cuda::aligned_size_t<16>(16),
      pipeline);

      pipeline.producer_commit();

      int k = 1;
      int next_slot, use_slot;
      #pragma unroll
      for(int i=SMEM_STRIDE;i<HEAD_DIM;i+= SMEM_STRIDE){
        next_slot = k & 1;
        use_slot  = (k - 1) & 1;
        pipeline.producer_acquire();

        cuda::memcpy_async(
        &tKnV(next_slot, currentRow, currentCol),
        &K[(size_t)(headStartSeqIdx + kvOffset + currentRow) * HEAD_DIM + i + currentCol],
        cuda::aligned_size_t<16>(16),
        pipeline);
        pipeline.producer_commit();

        pipeline.consumer_wait();

        wmma::load_matrix_sync(b_frag, &tKnV(use_slot, warpCol*FRAG_SIZE, 0), HALF_TILE_STRIDE);
        wmma::mma_sync(acc_frag, a_frags[(i - SMEM_STRIDE) / FRAG_SIZE], b_frag, acc_frag);
        wmma::load_matrix_sync(b_frag, &tKnV(use_slot, warpCol*FRAG_SIZE, FRAG_SIZE), HALF_TILE_STRIDE);
        wmma::mma_sync(acc_frag, a_frags[(i - SMEM_STRIDE) / FRAG_SIZE + 1], b_frag, acc_frag);
        pipeline.consumer_release();
        k += 1;
      }

      use_slot  = (k - 1) & 1;
      pipeline.consumer_wait();
      wmma::load_matrix_sync(b_frag, &tKnV(use_slot, warpCol*FRAG_SIZE, 0), HALF_TILE_STRIDE);
      wmma::mma_sync(acc_frag, a_frags[(HEAD_DIM - SMEM_STRIDE) / FRAG_SIZE], b_frag, acc_frag);
      wmma::load_matrix_sync(b_frag, &tKnV(use_slot, warpCol*FRAG_SIZE, FRAG_SIZE), HALF_TILE_STRIDE);
      wmma::mma_sync(acc_frag, a_frags[(HEAD_DIM - SMEM_STRIDE) / FRAG_SIZE + 1], b_frag, acc_frag);
      pipeline.consumer_release();


      wmma::store_matrix_sync(&tKQ(warpRow*FRAG_SIZE, warpCol*FRAG_SIZE), acc_frag, KQ_TILE_STRIDE, wmma::mem_row_major);
      __syncthreads();


      if(warpCol == 0 && threadIdxInWarp < FRAG_SIZE){
        int r = warpRow * FRAG_SIZE + threadIdxInWarp;
        float m = tMaxExpPrev(r);
        for(int i=0; i< QUERY_PER_BLOCK;i++){
          m = max(m, tKQ(r, i));
        }
        tMaxExpSeq(r) = m;
        tDiffExp(r) = exp2f((tMaxExpPrev(r)-m) * scale_log2e);   // 0 on 1st iter (m_prev = -inf)
        tSumSeq(r) = tSumSeq(r) * tDiffExp(r);
      }

      __syncthreads();

      #pragma unroll
      for(int c = 0; c < HEAD_DIM / SMEM_STRIDE; ++c){
        #pragma unroll
        for(int i = 0; i < size(tCrO[c]); ++i){
          tCrO[c](i) *= tDiffExp(get<0>(tCcO[c](i)));
        }
      }

      for(int idx=threadId; idx< QUERY_PER_BLOCK * QUERY_PER_BLOCK;idx += THREAD_PER_BLOCK){
        int r = idx / QUERY_PER_BLOCK;
        int c = idx % QUERY_PER_BLOCK;
        tKQ(r, c) = exp2f((tKQ(r, c) - tMaxExpSeq(r)) * scale_log2e);
      }

      __syncthreads();


      if(warpCol == 0 && threadIdxInWarp < FRAG_SIZE){
        int r = warpRow * FRAG_SIZE + threadIdxInWarp;
        float s = 0.f;
        for(int i=0; i< QUERY_PER_BLOCK;i++){
          s = s + tKQ(r, i);
        }
        tSumSeq(r) = tSumSeq(r) + s;
      }

      for(int idx=threadId; idx< QUERY_PER_BLOCK * QUERY_PER_BLOCK;idx += THREAD_PER_BLOCK){
        int r = idx / QUERY_PER_BLOCK;
        int c = idx % QUERY_PER_BLOCK;
        tKQHalf(r, c) = __float2half_rn(tKQ(r, c));
      }
      __syncthreads();

      Tensor tP = make_tensor(make_smem_ptr(tileKQHalf),
                              make_shape(Int<QUERY_PER_BLOCK>{}, Int<SMEM_STRIDE>{}),
                              make_stride(Int<HALF_TILE_STRIDE>{}, Int<1>{}));
      Tensor tSrP = s2r_thr_A.partition_S(tP);
      Tensor tCrP = thr_mma_pv.partition_fragment_A(tP);
      Tensor tCrP_view = s2r_thr_A.retile_D(tCrP);
      copy(s2r_copy_A, tSrP, tCrP_view);

      pipeline.producer_acquire();

      cuda::memcpy_async(
      &tKnV(0, currentRow, currentCol),
      &V[(size_t)(headStartSeqIdx + kvOffset + currentRow) * HEAD_DIM + currentCol],
      cuda::aligned_size_t<16>(16),
      pipeline);

      pipeline.producer_commit();

      int vslot = 0;
      #pragma unroll
      for(int i=0;i<HEAD_DIM;i+= SMEM_STRIDE){

        use_slot = vslot;
        if(i + SMEM_STRIDE < HEAD_DIM){

          pipeline.producer_acquire();
          cuda::memcpy_async(
          &tKnV(use_slot ^ 1, currentRow, currentCol),
          &V[(size_t)(headStartSeqIdx + kvOffset + currentRow) * HEAD_DIM + i + SMEM_STRIDE + currentCol],
          cuda::aligned_size_t<16>(16),
          pipeline);
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();


        Tensor tSrV = make_tensor(tKnV(use_slot, _, _).data(),
                                  make_shape(Int<SMEM_STRIDE>{}, Int<SMEM_STRIDE>{}),
                                  make_stride(Int<1>{}, Int<HALF_TILE_STRIDE>{}));
        Tensor tCrV = thr_mma_pv.partition_fragment_B(tSrV);
        Tensor tCrV_view = s2r_thr_B.retile_D(tCrV);
        copy(s2r_copy_B, s2r_thr_B.partition_S(tSrV), tCrV_view);

        // O_chunk += P x V_chunk, accumulating in registers
        gemm(tiled_mma_pv, tCrP, tCrV, tCrO[i / SMEM_STRIDE]);

        pipeline.consumer_release();

        vslot ^= 1;
      }


      if(warpCol == 0 && threadIdxInWarp < FRAG_SIZE){
        int r = warpRow * FRAG_SIZE + threadIdxInWarp;
        tMaxExpPrev(r) = tMaxExpSeq(r);
      }
    }

    #pragma unroll
    for(int c = 0; c < HEAD_DIM / SMEM_STRIDE; ++c){
      #pragma unroll
      for(int i = 0; i < size(tCrO[c]); ++i){
        auto coord = tCcO[c](i);
        int r = get<0>(coord);
        int col = c * SMEM_STRIDE + get<1>(coord);
        O[(size_t)(blockStartSeqIdx + r) * HEAD_DIM + col] =
            __float2half_rn(tCrO[c](i) / tSumSeq(r));
      }
    }




}

// ------------------------------------------------------------------------
// Naive fallback (correct for any shape; used only off the fast path)
// ------------------------------------------------------------------------
__global__ void naive_flash_attention_kernel(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O, const int seq_len,
    const int head_dim, const float scale) {
  const int q_idx = blockIdx.x;  // one block per (batch, head, token)
  const int token_idx = q_idx % seq_len;
  const size_t base = (size_t)(q_idx / seq_len) * seq_len * head_dim;
  const half* q_vec = Q + base + (size_t)token_idx * head_dim;
  half* o_vec = O + base + (size_t)token_idx * head_dim;

  float m_i = -INFINITY;
  float l_i = 0.f;
  for (int d = 0; d < head_dim; ++d) o_vec[d] = __float2half(0.f);

  for (int j = 0; j < seq_len; ++j) {
    const half* k_vec = K + base + (size_t)j * head_dim;
    const half* v_vec = V + base + (size_t)j * head_dim;
    float score = 0.f;
    for (int d = 0; d < head_dim; ++d)
      score += __half2float(q_vec[d]) * __half2float(k_vec[d]);
    score *= scale;
    const float m_prev = m_i;
    m_i = fmaxf(m_i, score);
    const float alpha = expf(m_prev - m_i);
    const float beta = expf(score - m_i);
    l_i = l_i * alpha + beta;
    for (int d = 0; d < head_dim; ++d)
      o_vec[d] = __float2half(__half2float(o_vec[d]) * alpha +
                              __half2float(v_vec[d]) * beta);
  }
  for (int d = 0; d < head_dim; ++d)
    o_vec[d] = __float2half(__half2float(o_vec[d]) / l_i);
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {

  c10::cuda::CUDAGuard device_guard(Q.device());

  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(Q.get_device());

  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
              "Q, K, and V must be CUDA tensors");
  TORCH_CHECK(Q.scalar_type() == torch::kFloat16 &&
                  K.scalar_type() == torch::kFloat16 &&
                  V.scalar_type() == torch::kFloat16,
              "Q, K, and V must be float16 tensors");
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
              "Q, K, and V must be contiguous");

  // 1. Setup Output Tensor
  auto O = torch::empty_like(Q);

  // 2. Extract Dimensions
  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  const float scale_log2e = scale * 1.4426950408889634f;


  if (seq_len % QUERY_PER_BLOCK == 0 && head_dim == 128) {
    TORCH_CHECK(1LL * batch_size * num_heads <= 65535LL,
                "batch_size * num_heads exceeds the gridDim.y limit");
    dim3 blocks((unsigned int)(seq_len / QUERY_PER_BLOCK),
                (unsigned int)(batch_size * num_heads));
    dim3 threads(THREAD_PER_BLOCK);
    flash_attention_kernel_v1<128><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()), batch_size, num_heads,
        seq_len, scale_log2e);
  } else {
    long long total_blocks = 1LL * batch_size * num_heads * seq_len;
    TORCH_CHECK(total_blocks <= 0x7fffffffLL, "grid too large");
    dim3 blocks((unsigned int)total_blocks);
    dim3 threads(1);
    naive_flash_attention_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()), seq_len, head_dim,
        scale);
  }

  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  return O;
}
