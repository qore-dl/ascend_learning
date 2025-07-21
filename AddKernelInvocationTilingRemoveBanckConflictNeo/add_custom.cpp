

# include "add_custom_tiling.h"
# include "kernel_operator.h"

constexpr size_t BUFFER_NUM{2}; // tensor num for each queue
constexpr int64_t BANKGROUP_SIZE{128*32*16}; // 每个Bank 具有 128x32 个字节，共16个Bank Group (16列)，
// 每个bank Group 有3个Bank
constexpr int64_t  ONE_REPEAT_SIZE = 256; // VECTOR Core 每个reoeat 可以从 UB 中加载一个向量的256 bytes

class KernelAdd{
    public:
    __aicore__ inline KernelAdd(){}

    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR z,
        size_t totalLength,size_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = (this->blockLength / tileNum / BUFFER_NUM);
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        uint64_t total_require_byte = this->tileLength*sizeof(half)*BUFFER_NUM;

        pipe.InitBuffer(inQueueX,BUFFER_NUM,(this->tileLength * sizeof(half)+ONE_REPEAT_SIZE));
        uint64_t given_align_byte = (BANKGROUP_SIZE - total_require_byte - BUFFER_NUM*ONE_REPEAT_SIZE);

        if( total_require_byte > given_align_byte){
            given_align_byte = total_require_byte + BUFFER_NUM*ONE_REPEAT_SIZE;
        }

        given_align_byte = (uint64_t)(given_align_byte/BUFFER_NUM);
        // this->tileLength * sizeof(half)
        pipe.InitBuffer(inQueueY,BUFFER_NUM,given_align_byte);
        pipe.InitBuffer(outQueueZ,BUFFER_NUM,this->tileLength * sizeof(half));

    }

    __aicore__ inline void Process()
    {
        size_t loopCount = this->tileNum * BUFFER_NUM;
        for (size_t i{0}; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
    private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,BUFFER_NUM> inQueueX,inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT,BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    size_t blockLength;
    size_t tileNum;
    size_t tileLength;

    __aicore__ inline void CopyIn(size_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);
        AscendC::DataCopy(yLocal,yGm[progress*this->tileLength],this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(size_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(size_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    KernelAdd op;
    op.Init(x,y,z,tiling.totalLength,tiling.tileNum);
    op.Process();
}

// #ifndef ASCENDC_CPU_DEBUG
// void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, AddCustomTilingData tiling)
// {
//     add_custom<<<blockDim, nullptr, stream>>>(x, y, z,tiling);
// }
// #endif
