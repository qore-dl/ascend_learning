#include "multiply_add_custom_tiling.h"
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_muladd_custom.h"
// #include "acl/acl.h"
// #include "aclnnop/aclnn_dot.h"
#else
#include "tikicpulib.h"
#include <iostream>
extern "C" __global__ __aicore__ void muladd_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR sync_workplace, MulAddCustomTilingData tiling);
#endif

// int64_t GetShapeSize(const std::vector<int64_t> &shape)
// {
//     int64_t shapeSize = 1;
//     for (auto i : shape)
//     {
//         shapeSize *= i;
//     }
//     return shapeSize;
// }

// template <typename T>
// int CreateAclTensor( 
//     const std::vector<int64_t> &shape, 
//     void **deviceAddr,aclDataType dataType, aclTensor **tensor)
// {
//     // auto size = GetShapeSize(shape) * sizeof(T);
//     // // 调用aclrtMalloc申请device侧内存
//     // auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
//     // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
//     // // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
//     // ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
//     // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

//     // const std::vector<T> &hostData,
//     // 计算连续tensor的strides
//     std::vector<int64_t> strides(shape.size(), 1);
//     for (int64_t i = shape.size() - 2; i >= 0; i--)
//     {
//         strides[i] = shape[i + 1] * strides[i + 1];
//     }

//     // 调用aclCreateTensor接口创建aclTensor
//     *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
//          strides.data(), 0, aclFormat::ACL_FORMAT_ND,
//          shape.data(), shape.size(), *deviceAddr);

//     return 0;
// }


uint32_t Compute_GM_OUT_REQUIRED_Byte(uint32_t raw_element_num,uint32_t block_element_scale){
    uint32_t out_size = (uint32_t)((raw_element_num + block_element_scale - 1)/block_element_scale)*32;

    return out_size;

}

int32_t main(int argc,char* argv[])
{
    // size_t blockDim = 8;
    // size_t tilingSize = 2 * sizeof(size_t);
    // size_t inputByteSize = 8*2048*sizeof(uint16_t);
    // size_t outputByteSize = 8*2048*sizeof(uint32_t);
    uint32_t blockDim = 48;
    size_t total_len = 48 * 4096;
    size_t tilingSize = 4 * sizeof(uint32_t);
    size_t inputByteSize = total_len * sizeof(uint16_t);
    
    int32_t buffer_num = 2;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
    
    uint32_t sync_byte_size = *reinterpret_cast<MulAddCustomTilingData *>(tiling).gm_sync_size;
    int32_t tileNum = *reinterpret_cast<MulAddCustomTilingData *>(tiling).tileNum;

    // size_t outputByteSize = blockDim * tileNum*buffer_num*(32 / sizeof(uint16_t))*sizeof(uint16_t);
    // size_t outputByteSize = Compute_GM_OUT_REQUIRED_Byte(blockDim*tileNum*buffer_num,16);
    // tileNum*buffer_num*
    size_t outputByteSize = blockDim * (32 / sizeof(uint16_t))*sizeof(uint16_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *sync_workplace = (uint8_t *)AscendC::GmAlloc(sync_workplace);



    ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(muladd_custom, blockDim, x, y, z,sync_workplace,
                *reinterpret_cast<MulAddCustomTilingData *>(tiling)); // use this macro for cpu debug

    WriteFile("./output/output_z.bin", z, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
    AscendC::GmFree((void *)tiling);

#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 7;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    MulAddCustomTilingData *tiling;
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice, *sync_workplace;


    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);

    uint32_t sync_byte_size = tiling->gm_sync_size;
    uint32_t tileNum = tiling->tileNum;

    // * tileNum*buffer_num
    size_t outputByteSize = blockDim * (32 / sizeof(uint16_t))*sizeof(uint16_t);
    // size_t outputByteSize = Compute_GM_OUT_REQUIRED_Byte(blockDim*tileNum*buffer_num,16);
    

    std::cout<<"sync_byte_size "<<sync_byte_size<<std::endl;
    std::cout<<"tileNum "<<tileNum<<std::endl;
    std::cout<<"OutputSize(Byte) "<<outputByteSize<<std::endl;


    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));

    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&sync_workplace, sync_byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(sync_workplace, sync_byte_size, 0, sync_byte_size));


    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);

    std::cout<<"Read Input Data!!!"<<std::endl;

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout<<"Move Data to Device!!!"<<std::endl;

    for (int i = 0; i < 20; ++i) {
        ACLRT_LAUNCH_KERNEL(muladd_custom)(blockDim, stream, xDevice, yDevice, zDevice, sync_workplace, tiling);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        std::cout<<"Finished warm up: "<<i<<std::endl;
    }



    std::cout<<"Finished Warmup!!"<<std::endl;

    int num_repeat = 10000;

    aclrtEvent start, stop;
    float temp_time = 0;
    float time = 0;
    CHECK_ACL(aclrtCreateEvent(&start));
    CHECK_ACL(aclrtCreateEvent(&stop));
    for (int i = 0; i < num_repeat; ++i) {
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtRecordEvent(start, stream));

        ACLRT_LAUNCH_KERNEL(muladd_custom)(blockDim, stream, xDevice, yDevice, zDevice, sync_workplace, tiling);
        
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtRecordEvent(stop, stream));
        CHECK_ACL(aclrtSynchronizeEvent(stop));
        CHECK_ACL(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }

    printf("m: %u, n: %u, k: %lu, %f GFLOPS, %f ms\n", 1, 1, total_len,
           (float)2 * total_len / (time / num_repeat * 1e-3) / 1e9, (time / num_repeat));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    // // 对比实验
    // std::vector<int64_t> selfShape = {(int64_t)total_len};
    // std::vector<int64_t> tensorShape = {(int64_t)total_len};
    // std::vector<int64_t> outShape = {};

    // aclTensor *self = nullptr;
    // aclTensor *tensor = nullptr;
    // aclTensor *out = nullptr;

    // // 创建self aclTensor
    // ret = CreateAclTensor( 
    //     selfShape, 
    //     &xDevice,aclDataType::ACL_FLOAT16, &self);
    
    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    // // 创建tensor aclTensor
    // ret = CreateAclTensor( 
    //     tensorShape, 
    //     &yDevice,aclDataType::ACL_FLOAT16, &tensor);

    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    // // 创建out aclTensor
    // ret = CreateAclTensor( 
    //     outShape, 
    //     &zDevice,aclDataType::ACL_FLOAT16, &out);

    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    // // 3. 调用CANN算子库API，需要修改为具体的API名称
    // uint64_t workspaceSize = 0;
    // aclOpExecutor *executor;

    // int warmup = 100;
    // int num_repeat = 1000;
    // // aclrtEvent start, stop;
    // temp_time = 0;
    // time = 0;

    // // CHECK_ACL(aclrtCreateEvent(&start));
    // // CHECK_ACL(aclrtCreateEvent(&stop));
    // void *workspaceAddr;

    // for (int i = 0; i < 20; i++)
    // {
    //     // 调用aclnnDot第一段接口
    //     ret = aclnnDotGetWorkspaceSize(self, tensor, out, &workspaceSize, &executor);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    //     // 根据第一段接口计算出的workspaceSize申请device内存
    //     void *workspaceAddr = nullptr;
    //     if (workspaceSize > 0)
    //     {
    //         ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    //         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    //     }
    //     // 调用aclnnDot第二段接口
    //     ret = aclnnDot(workspaceAddr, workspaceSize, executor, stream);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDot failed. ERROR: %d\n", ret); return ret);

    //     // 4.（固定写法）同步等待任务执行结束
    //     // ret = aclrtSynchronizeStream(stream);
    //     // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // }

    // for (int i = 0; i < num_repeat; ++i)
    // {
    //     // 调用aclnnDot第一段接口
    //     ret = aclnnDotGetWorkspaceSize(self, tensor, out, &workspaceSize, &executor);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    //     // 根据第一段接口计算出的workspaceSize申请device内存
    //     void *workspaceAddr = nullptr;
    //     if (workspaceSize > 0)
    //     {
    //         ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    //         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    //     }
    //     // 调用aclnnDot第二段接口
    //     CHECK_ACL(aclrtSynchronizeStream(stream));
    //     CHECK_ACL(aclrtRecordEvent(start, stream));
    //     ret = aclnnDot(workspaceAddr, workspaceSize, executor, stream);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    //     ret = aclrtSynchronizeStream(stream);
    //     CHECK_ACL(aclrtRecordEvent(stop, stream));

    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    //     CHECK_ACL(aclrtSynchronizeEvent(stop));
    //     CHECK_ACL(aclrtEventElapsedTime(&temp_time, start, stop));
    //     time += temp_time;
    //     // 4.（固定写法）同步等待任务执行结束
    // }
    // // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    // auto size = GetShapeSize(outShape);
    // std::vector<float> resultData(size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
    //                   size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // LOG_PRINT("result[0] is: %f\n", resultData[0]);

    // printf("The repeat time is : %d\n", num_repeat);
    // printf("Vmul perf is %f GFLOPS, %f ms \n", (float)2 * total_len / (time / num_repeat * 1e-3) / 1e9,(time / num_repeat));

    // // 6. 释放aclTensor，需要根据具体API的接口定义修改
    // aclDestroyTensor(self);
    // aclDestroyTensor(tensor);
    // aclDestroyTensor(out);



    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));

    // if (workspaceSize > 0)
    // {
    //     aclrtFree(workspaceAddr);
    // }

    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    
#endif
    return 0;
}