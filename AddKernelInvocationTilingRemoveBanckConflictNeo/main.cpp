#include "add_custom_tiling.h"
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_add_custom.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling);
#endif

int32_t main(int argc,char* argv[])
{
    uint32_t blockDim = 48;
    size_t total_len = 48 * 4096;
    size_t tilingSize = 2 * sizeof(uint32_t);
    size_t inputByteSize = total_len * sizeof(uint16_t);
    size_t outputByteSize = total_len * sizeof(uint16_t);
    // size_t workByteSize = 48 * sizeof(int32_t);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(add_custom, blockDim, x, y, z,
                *reinterpret_cast<AddCustomTilingData *>(tiling)); // use this macro for cpu debug

    WriteFile("./output/output_z.bin", z, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t device_id = 7;
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    AddCustomTilingData *tiling;
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&tiling),tilingSize));
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);

    CHECK_ACL(aclrtMallocHost((void **)(&xHost),inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost),inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost),outputByteSize));
    
    CHECK_ACL(aclrtMalloc((void **)(&xDevice),inputByteSize,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)(&yDevice),inputByteSize,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)(&zDevice),inputByteSize,ACL_MEM_MALLOC_HUGE_FIRST));

    std::cout<<"Read Input Data!!!"<<std::endl;

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);

    std::cout<<"Move Data to Device!!!"<<std::endl;

    CHECK_ACL(aclrtMemcpy(xDevice,inputByteSize,xHost,inputByteSize,ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice,inputByteSize,yHost,inputByteSize,ACL_MEMCPY_HOST_TO_DEVICE));

    for (int i = 0; i < 20; ++i) {
        ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice, tiling);
        std::cout<<"Finished warm up: "<<i<<std::endl;
    }

    std::cout<<"Finished Warmup!!"<<std::endl;


    int num_repeat = 1000;

    aclrtEvent start, stop;
    float temp_time = 0;
    float time = 0;
    CHECK_ACL(aclrtCreateEvent(&start));
    CHECK_ACL(aclrtCreateEvent(&stop));
    for (int i = 0; i < num_repeat; ++i) {
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtRecordEvent(start, stream));

        ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice, tiling);
        
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtRecordEvent(stop, stream));
        CHECK_ACL(aclrtSynchronizeEvent(stop));
        CHECK_ACL(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }

    printf("m: %u, n: %u, k: %lu, %f GFLOPS, %f ms\n", 1, 1, total_len,
        (float) total_len / (time / num_repeat * 1e-3) / 1e9, (time / num_repeat));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
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