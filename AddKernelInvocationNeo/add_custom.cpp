# include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH{8*2048}; // 每次处理的总量
constexpr int32_t USE_CORE_NUM{8}; // 使用的core的数量
constexpr int32_t BLOCK_LENGTH{TOTAL_LENGTH/USE_CORE_NUM}; // 每个core 上每次处理数据的量
constexpr int32_t TILE_NUM = 8; // 在每个core上将数据进一步划分为8个tile
constexpr int32_t BUFFER_NUM = 2; // double buffering，queue中的tensor规模
constexpr int32_t TILE_LENGTH{BLOCK_LENGTH/TILE_NUM/BUFFER_NUM};// 因为使用了double buffering，所以TILE规模减半

class KernelAdd{
    public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(),BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(),BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(),BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM,TILE_LENGTH * sizeof(half));
        pipe.Init(outQueueZ, BUFFER_NUM*TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for(int32_t i{0};i<loopCount;++i)
        {

        }
    }

    private:

        __aicore__ inline void CopyIn(int32_t progress)
        {
            AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

            AscendC::DataCopy(xLocal,xGm[progress*TILE_LENGTH],TILE_LENGTH);
            AscendC::DataCopy(yLocal,yGm[progress*TILE_LENGTH],TILE_LENGTH);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);
        }

        __aicore__ inline void Compute(int32_t progress)
        {
           AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
           AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
           AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
           AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
           outQueueZ.EnQue<half>(zLocal);
           inQueueX.FreeTensor(xLocal);
           inQueueX.FreeTensor(yLocal);
        }

        __aicore__ incline void CopyOut(int32_t progress)
        {
            AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
            AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);

        }
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
        AscendC::GlobalTensor<half> xGm;
        AscendC::GlobalTensor<half> yGm;
        AscendC::GlobalTensor<half> zGm;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x,GM_ADDR y,GM_ADDR z)
{
    KernelAdd op;
    op.Init(x,y,z);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
{
    add_custom<<<blockDim, nullptr, stream>>>(x, y, z);
}
#endif
