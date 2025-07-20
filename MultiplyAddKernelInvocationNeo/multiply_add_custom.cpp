

# include "multiply_add_custom_tiling.h"
# include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM{2}; // tensor num for each queue
constexpr uint32_t DATA_BLOCK_SIZE{32}; // byte size of datablock
constexpr uint64_t VECTORIZED_ACCESS_SIZE{256}; //byte size of vectorized tensor access
constexpr uint8_t BLOCK_INTER_STRIDE{VECTORIZED_ACCESS_SIZE/DATA_BLOCK_SIZE};

// // 先定义一个向上取整函数
// uint32_t RoundUp(uint32_t a, uint32_t b)
// { 
//     return (a + b - 1) / b;
// }



class KernelMultiplyAdd{
    public:
    __aicore__ inline KernelMultiplyAdd(){}

    __aicore__ inline uint32_t Compute_Workspace_size(uint32_t element_size,
        uint32_t vectorized_element_scale,uint32_t vector_repeat_time)
    {
        uint32_t block_element_scale{DATA_BLOCK_SIZE/element_size};
        
        uint32_t iter1OutputCount = vector_repeat_time;                                              // 第一轮操作产生的元素个数
        uint32_t iter1AlignEnd = ((iter1OutputCount + block_element_scale-1)/block_element_scale) * block_element_scale; // 第一轮产生的元素个数做向上取整
        // uint32_t finalWorkLocalNeedSize = iter1AlignEnd;  
        // 最终workLocal所需的elements空间大小就是第一轮操作产生元素做向上取整后的结果
        return iter1AlignEnd;
    }

    __aicore__ inline uint32_t Compute_GM_Sync_Workspace(uint32_t lower_limit)
    {
        int64_t core_num = AscendC::GetBlockNum();
        int64_t event_num = core_num;
        int64_t block_id_num = core_num;

        uint64_t gm_sync_byte_scale = (core_num*32*event_num + block_id_num*32 + 32)*2;

        if(static_cast<uint64_t>(lower_limit) > gm_sync_byte_scale){
            gm_sync_byte_scale = static_cast<uint64_t>(lower_limit);
        }

        return gm_sync_byte_scale;
    }

    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR z, GM_ADDR sync_workplace,
        uint32_t totalLength,uint32_t tileNum,uint32_t element_size,uint32_t lower_sync_size)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = (this->blockLength / tileNum / BUFFER_NUM);
        this->element_size = element_size;
        this->vectorized_element_scale = VECTORIZED_ACCESS_SIZE/element_size;

        this->blockIdx = AscendC::GetBlockIdx();

        uint32_t block_element_scale{DATA_BLOCK_SIZE/element_size};
        // Compute_GM_Sync_Workspace(lower_sync_size)

        uint64_t gm_sync_byte_scale = lower_sync_size;

        // if (gm_sync_byte_scale < 256*sizeof(int32_t)){
        //     gm_sync_byte_scale = 256*sizeof(int32_t);
        // }


        this->block_element_scale = block_element_scale;

        

        this->vector_repeat_time = (this->tileLength + this->vectorized_element_scale - 1)/this->vectorized_element_scale;
        
        uint32_t work_local_element_scale = Compute_Workspace_size(this->element_size,
            this->vectorized_element_scale,this->vector_repeat_time);

        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        if(AscendC::GetBlockIdx() < 1){
            zGm.SetGlobalBuffer((__gm__ half *)z, (tileNum*BUFFER_NUM*block_element_scale)*AscendC::GetBlockNum());
        }else{
            zGm.SetGlobalBuffer((__gm__ half *)z + ((tileNum*BUFFER_NUM*block_element_scale) * AscendC::GetBlockIdx()), 
            (tileNum*BUFFER_NUM*block_element_scale));
        }
        
        // /sizeof(int32_t)

        sync_gm.SetGlobalBuffer((__gm__ int32_t *)sync_workplace,(gm_sync_byte_scale));


        pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY,BUFFER_NUM,this->tileLength * sizeof(half));
        pipe.InitBuffer(vecInSync, BUFFER_NUM, 48 * sizeof(int32_t));
        pipe.InitBuffer(inTotalReduceQueueZ,BUFFER_NUM,(tileNum*BUFFER_NUM*block_element_scale) * sizeof(half));

        // pipe.InitBuffer(outQueueZ,BUFFER_NUM,this->tileLength * sizeof(half));
        // AscendC::TQue<AscendC::TPosition::VECCALC,BUFFER_NUM> calcbufCoreRes;
        // AscendC::TQue<AscendC::TPosition::VECOUT,1> outReduceQueueZ;
        // AscendC::TQue<AscendC::TPosition::VECOUT,1> workLocalSpace;

        // pipe.InitBuffer(calcbufCoreRes,BUFFER_NUM,this->tileLength * sizeof(half));
        pipe.InitBuffer(calcbufCoreRes,block_element_scale);
        // pipe.InitBuffer(workLocalSpace,BUFFER_NUM,work_local_element_scale*sizeof(half));
        pipe.InitBuffer(outReduceQueueZ,BUFFER_NUM,block_element_scale*sizeof(half));
    }

    __aicore__ inline void Process()
    {
        size_t loopCount = this->tileNum * BUFFER_NUM;
        // zCoreRes = calcbufCoreRes.Get<half>();

          
        for (size_t i{0}; i < loopCount; i++) {
                CopyIn(i);
                Compute_ma(i);
                CopyOut(i);
        }

        // 同步核之间的状态
        AscendC::LocalTensor<int32_t> sync_workLocal = vecInSync.AllocTensor<int32_t>();
        AscendC::SyncAll(sync_gm, sync_workLocal);
        vecInSync.FreeTensor(sync_workLocal);
        
        // // 将累加后的结果元素进行ReduceSum：
        // Compute_core_reduce();
        // 若只使用了单核或者是
        size_t total_loopCount = loopCount * AscendC::GetBlockNum();
        if(AscendC::GetBlockIdx() < 1)
        {
            //  CopyOut();
            // outReduceQueueZ.DeQue<half>
            // vecInSync
            // AscendC::LocalTensor<half> zTotalReduce = calcbufCoreRes.Get<half>();
            // AscendC::Duplicate<half>(zTotalReduce, static_cast<half>(0), this->block_element_scale);
            // for(int32_t fb{0};fb<total_loopCount;++fb){
            //     CopyInterCoreIn(fb);
            //     Compute_Add(fb,zTotalReduce);
            // }

            // outReduceQueueZ.EnQue<half>(zTotalReduce);
            // CopyOut(0);  
            
            float sum = 0;
            for(int i = 0; i < total_loopCount; ++i) {
                sum += (float)zGm.GetValue(i * this->block_element_scale);
            }
            zGm.SetValue(0,(half)sum);
            // for(int i = 1; i < this->tileNumAll; ++i) {
            //     sumGm.SetValue(i * this->tileLength, (half)0);
            // }
        }
    }

    private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,BUFFER_NUM> inQueueX,inQueueY,vecInSync;
    // AscendC::TQue<AscendC::TPosition::VECOUT,BUFFER_NUM> outQueueZ;
    // AscendC::TQue<AscendC::TPosition::VECCALC,BUFFER_NUM> calcbufCoreRes;
    AscendC::TQue<AscendC::TPosition::VECOUT,BUFFER_NUM> outReduceQueueZ;
    // AscendC::TQue<AscendC::TPosition::VECOUT,BUFFER_NUM> workLocalSpace; 
    AscendC::TQue<AscendC::TPosition::VECIN,BUFFER_NUM> inTotalReduceQueueZ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcbufCoreRes; //输出数据管理对象，TPosition为VECCALC
    
    // AscendC::LocalTensor<half> zFinalCoreRes;


    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    AscendC::GlobalTensor<int32_t> sync_gm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint64_t vectorized_element_scale;
    uint32_t element_size;
    uint8_t vector_repeat_time;
    uint32_t block_element_scale;
    int32_t blockIdx;

    // __aicore__ inline void InitialLocalSum()
    // {
    //     AscendC::LocalTensor<half> zCoreRes = calcbufCoreRes.AllocTensor<half>();
    //     AscendC::Duplicate(zCoreRes, static_cast<half>(0), 
    //         this->vectorized_element_scale, 
    //         this->vector_repeat_time, 1, BLOCK_INTER_STRIDE);
    //     calcbufCoreRes.EnQue(zCoreRes);
    // }

    __aicore__ inline void CopyIn(size_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

        AscendC::DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);
        AscendC::DataCopy(yLocal,yGm[progress*this->tileLength],this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyInterCoreIn(int32_t progress_id)
    {
        AscendC::LocalTensor<half> tk_zlocal = inTotalReduceQueueZ.AllocTensor<half>();
        // AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

        // AscendC::DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);
        // [blk_id*this->block_element_scale]
        AscendC::DataCopy(tk_zlocal,zGm[progress_id*this->block_element_scale],
            this->block_element_scale);

        inTotalReduceQueueZ.EnQue(tk_zlocal);
        // inQueueY.EnQue(yLocal);
    }

    // ,LocalTensor<half> &z_acc
    __aicore__ inline void Compute_ma(size_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> z_acc =  outReduceQueueZ.AllocTensor<half>();
        // zWorkLocal,
        // AscendC::LocalTensor<half> zWorkLocal = workLocalSpace.AllocTensor<half>();
        // AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);

        // this->vectorized_element_scale,
        // this->vector_repeat_time,{1, 1, 1, BLOCK_INTER_STRIDE, 
            // BLOCK_INTER_STRIDE, BLOCK_INTER_STRIDE}

        AscendC::Mul(xLocal,xLocal,yLocal,this->tileLength);
        AscendC::ReduceSum(z_acc, xLocal, yLocal,this->tileLength);
         
        outReduceQueueZ.EnQue<half>(z_acc);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute_Add(int32_t blk_id,AscendC::LocalTensor<half>& zTotalReduce)
    {
        AscendC::LocalTensor<half> bk_zlocal = inTotalReduceQueueZ.DeQue<half>();
        // AscendC::LocalTensor<half> zTotalReduce = outReduceQueueZ.DeQue<half>();
        // AscendC::Add(zLocal, xLocal, yLocal, TOTAL_LENGTH);
        AscendC::Add(zTotalReduce,bk_zlocal,zTotalReduce,this->block_element_scale);
        // outReduceQueueZ.EnQue<half>(zTotalReduce);
    }

    // __aicore__ inline void Compute_core_reduce()
    // {
    //     AscendC::LocalTensor<half> zCoreRes = calcbufCoreRes.DeQue<half>();
    //     for(uint32_t k{0};k<BUFFER_NUM-1;++k){
    //         AscendC::LocalTensor<half> z_acc_tmp = calcbufCoreRes.DeQue<half>();
    //         AscendC::Add(zCoreRes, zCoreRes, z_acc_tmp, this->tileLength);
    //         calcbufCoreRes.FreeTensor(z_acc_tmp);
    //     }

    //     // AscendC::LocalTensor<half> zCoreReduceRes = outReduceQueueZ.AllocTensor<half>();
    //     AscendC::LocalTensor<half> zWorkLocal = workLocalSpace.AllocTensor<half>();

    //     AscendC::ReduceSum<half>(zCoreReduceRes, zCoreReduceRes, zWorkLocal,
    //         this->vectorized_element_scale,
    //         this->vector_repeat_time,BLOCK_INTER_STRIDE);
        
    //     outReduceQueueZ.EnQue<half>(zCoreReduceRes);
    // }


    // size_t progress
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outReduceQueueZ.DeQue<half>();
        zGm.SetValue(progress * this->block_element_scale, zLocal.GetValue(0));
        outReduceQueueZ.FreeTensor(zLocal);
    }
    // __aicore__ inline void CopyOut(size_t progress)
    // {

    //     AscendC::LocalTensor<half> zLocal = outReduceQueueZ.DeQue<half>();
    //     AscendC::DataCopy(zGm[progress*this->block_element_scale], zLocal, this->block_element_scale);
    //     outReduceQueueZ.FreeTensor(zLocal);

        
    // }

    // 在全局同步的条件下，实现最终结果的累加计算
};

extern "C" __global__ __aicore__ void muladd_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR sync_workplace, MulAddCustomTilingData tiling)
{
    KernelMultiplyAdd op;
    op.Init(x,y,z,sync_workplace,
        tiling.totalLength,tiling.tileNum,
        tiling.element_size,tiling.gm_sync_size);
    op.Process();
}

// #ifndef ASCENDC_CPU_DEBUG
// void muladd_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *sync_workplace, 
//     MulAddCustomTilingData tiling)
// {
//     muladd_custom<<<blockDim, nullptr, stream>>>(x, y, z,sync_workplace,tiling);
// }
// #endif
