/*!
 * Copyright (c) 2016 by Contributors
 * \file postprocess_rpn.cu
 * \brief post process of rpn operator
 * \author Ming Zhang
*/
#include "./postprocess_rpn-inl.h"
#include "./mshadow_op.h"


namespace mshadow {

namespace cuda {

__global__ void PostProcessRPNForwardKernel(
                int count,
                const float *pfCls, const float *pfReg, const float *pfAnchor, int dwAnchorNum, int dwFeatH, int dwFeatW, 
                float *pfBBs, int dwMaxBBNum, float clsthreshold, int originalH, int originalW, int *pdwbb_num_now) {
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (index < count)
  {
    int nownum = *pdwbb_num_now;
    if (nownum > 0 && nownum < dwMaxBBNum)
    {
      int dwFeatSize = dwFeatH * dwFeatW;
      int dwAnchorI = index / dwFeatSize;
      int dwRI = (index - dwAnchorI * dwFeatSize) / dwFeatW;
      int dwCI = (index - dwAnchorI * dwFeatSize) % dwFeatW;
      int dwOft = dwRI * dwFeatW + dwCI;
      int dwAnchorOft = dwAnchorI * dwFeatSize;
      const float *pfNowAnchor = pfAnchor + dwAnchorI * 2;
      if (pfCls[dwOft + dwAnchorOft] > clsthreshold)
      {
        float fCY = pfReg[dwAnchorOft * 4 + 0 * dwFeatSize + dwOft];
        float fCX = pfReg[dwAnchorOft * 4 + 1 * dwFeatSize + dwOft];
        float fH = pfReg[dwAnchorOft * 4 + 2 * dwFeatSize + dwOft];
        float fW = pfReg[dwAnchorOft * 4 + 3 * dwFeatSize + dwOft];
        fCY = fCY * pfNowAnchor[0] + ((float)(dwRI) * originalH) / dwFeatH;
        fCX = fCX * pfNowAnchor[1] + ((float)(dwCI) * originalW) / dwFeatW;
        fH = expf(fH) * pfNowAnchor[0];
        fW = expf(fW) * pfNowAnchor[1];
        atomicInc((unsigned int*)pdwbb_num_now, dwMaxBBNum);
        nownum = *pdwbb_num_now;
        if (nownum > 0)
        {
          pfBBs[(nownum-1) * 4 + 0] = fCY;
          pfBBs[(nownum-1) * 4 + 1] = fCX;
          pfBBs[(nownum-1) * 4 + 2] = fH;
          pfBBs[(nownum-1) * 4 + 3] = fW;
        }
      }
    }
  }
}


inline void PostProcessRPNForward(const Tensor<gpu, 4> &datacls_in,
                           const Tensor<gpu, 4> &datareg_in,
                           const Tensor<gpu, 2> &anchorinfo_in,
                           const Tensor<gpu, 3> &bb_out,
                           const float clsthreshold,
                           const int originalH, const int originalW) {
  CHECK_EQ(datacls_in.size(0), datareg_in.size(0));
  
  int dwBatchNum = datacls_in.size(0);
  int dwAnchorNum = anchorinfo_in.size(0);
  int bb_maxnum_per_batch = bb_out.size(1);
  
  int dwFeatH = datacls_in.size(2);
  int dwFeatW = datacls_in.size(3);
  int dwBBMemLen = bb_out.MemSize<0>();
  memset(bb_out.dptr_, 0, dwBBMemLen * sizeof(float));
  
  int count = dwFeatH * dwFeatW * dwAnchorNum;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);
  
  CheckLaunchParam(dimGrid, dimBlock, "PostProcessRPN Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(bb_out.stream_);
  
  for (int bi = 0; bi < dwBatchNum; bi++) {
    const Tensor<gpu, 3> &datacls_onebatch = datacls_in[bi];
    const Tensor<gpu, 3> &datareg_onebatch = datareg_in[bi];
    Tensor<gpu, 2> bb_onebatch = bb_out[bi];
    int bb_num_now = 0;
    PostProcessRPNForwardKernel<<<dimGrid, dimBlock, 0, stream>>>(
            count, 
            datacls_onebatch.dptr_, datareg_onebatch.dptr_, datareg_onebatch.dptr_, dwAnchorNum, dwFeatH, dwFeatW, 
            bb_onebatch.dptr_, bb_maxnum_per_batch, clsthreshold, originalH, originalW, &bb_num_now);
  }
}
  
} // namespace cuda

inline void PostProcessRPNForward(const Tensor<gpu, 4> &datacls_in,
                           const Tensor<gpu, 4> &datareg_in,
                           const Tensor<gpu, 2> &anchorinfo_in,
                           const Tensor<gpu, 3> &bb_out,
                           const float clsthreshold,
                           const int originalH, const int originalW) {
  cuda::PostProcessRPNForward(datacls_in, datareg_in, anchorinfo_in, bb_out, clsthreshold, originalH, originalW);
}

} // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(PostProcessRPNParam param) {
  return new PostProcessRPNOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
