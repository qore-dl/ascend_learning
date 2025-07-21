#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dot.h"
#include "data_utils.h"

#define CHECK_ACL(x)                                                                        \
    do                                                                                      \
    {                                                                                       \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE)                                                        \
        {                                                                                   \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor( 
    const std::vector<int64_t> &shape, 
    void **deviceAddr,aclDataType dataType, aclTensor **tensor)
{
    // auto size = GetShapeSize(shape) * sizeof(T);
    // // 调用aclrtMalloc申请device侧内存
    // auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    // ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // const std::vector<T> &hostData,
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
         strides.data(), 0, aclFormat::ACL_FORMAT_ND,
         shape.data(), shape.size(), *deviceAddr);

    return 0;
}


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
    // size_t tilingSize = 4 * sizeof(uint32_t);
    size_t inputByteSize = total_len * sizeof(uint16_t);
    
    int32_t buffer_num = 2;

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    // *zHost;
    std::vector<half> outHostData{0};


    uint8_t *xHost, *yHost, 
    
    uint8_t *xDevice, *yDevice, *zDevice, *sync_workplace;




    // CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));
    // ReadFile("/data/huaqin/MultiplyAddKernelInvocationNeo/input/input_tiling.bin", 
    //     tilingSize, tiling, tilingSize);

    // uint32_t sync_byte_size = tiling->gm_sync_size;
    // uint32_t tileNum = tiling->tileNum;

    // size_t outputByteSize = blockDim * tileNum*buffer_num*(32 / sizeof(uint16_t))*sizeof(uint16_t);
    // size_t outputByteSize = Compute_GM_OUT_REQUIRED_Byte(blockDim*tileNum*buffer_num,16);
    

    std::cout<<"sync_byte_size "<<sync_byte_size<<std::endl;
    std::cout<<"tileNum "<<tileNum<<std::endl;
    std::cout<<"OutputSize(Byte) "<<outputByteSize<<std::endl;


    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    // CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));

    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));




    // CHECK_ACL(aclrtMalloc((void **)&sync_workplace, sync_byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMemset(sync_workplace, sync_byte_size, 0, sync_byte_size));

    ret = aclrtMemcpy(zDevice, sizeof(uint16_t), outHostData.data(), 
        sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);


    ReadFile("/data/huaqin/MultiplyAddKernelInvocationNeo/input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("/data/huaqin/MultiplyAddKernelInvocationNeo/input/input_y.bin", inputByteSize, yHost, inputByteSize);

    std::cout<<"Read Input Data!!!"<<std::endl;

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout<<"Move Data to Device!!!"<<std::endl;

    int num_repeat = 1000;

    aclrtEvent start, stop;
    float temp_time = 0;
    float time = 0;
    CHECK_ACL(aclrtCreateEvent(&start));
    CHECK_ACL(aclrtCreateEvent(&stop));


    // CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // WriteFile("./output/output_z.bin", zHost, outputByteSize);

    // 对比实验
    
    std::vector<int64_t> selfShape = {(int64_t)total_len};
    std::vector<int64_t> tensorShape = {(int64_t)total_len};
    std::vector<int64_t> outShape = {};

    aclTensor *self = nullptr;
    aclTensor *tensor = nullptr;
    aclTensor *out = nullptr;

    // 创建self aclTensor
    ret = CreateAclTensor( 
        selfShape, 
        &xDevice,aclDataType::ACL_FLOAT16, &self);
    
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建tensor aclTensor
    ret = CreateAclTensor( 
        tensorShape, 
        &yDevice,aclDataType::ACL_FLOAT16, &tensor);

    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor( 
        outShape, 
        &zDevice,aclDataType::ACL_FLOAT16, &out);

    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    int warmup = 100;
    int num_repeat = 1000;
    // aclrtEvent start, stop;
    temp_time = 0;
    time = 0;

    // CHECK_ACL(aclrtCreateEvent(&start));
    // CHECK_ACL(aclrtCreateEvent(&stop));
    void *workspaceAddr;

    for (int i = 0; i < 20; i++)
    {
        // 调用aclnnDot第一段接口
        ret = aclnnDotGetWorkspaceSize(self, tensor, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnDot第二段接口
        ret = aclnnDot(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDot failed. ERROR: %d\n", ret); return ret);

        // 4.（固定写法）同步等待任务执行结束
        // ret = aclrtSynchronizeStream(stream);
        // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    }
    std::cout<<"Finished Warmup!!"<<std::endl;

    for (int i = 0; i < num_repeat; ++i)
    {
        // 调用aclnnDot第一段接口
        ret = aclnnDotGetWorkspaceSize(self, tensor, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnDot第二段接口
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        ret = aclnnDot(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSynchronizeStream(stream);
        CHECK_ACL(aclrtRecordEvent(stop, stream));

        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        CHECK_ACL(aclrtSynchronizeEvent(stop));
        CHECK_ACL(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
        // 4.（固定写法）同步等待任务执行结束
    }
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    // CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // WriteFile("./output/output_z.bin", zHost, outputByteSize);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("result[0] is: %f\n", resultData[0]);

    printf("The repeat time is : %d\n", num_repeat);
    printf("Vmul perf is %f GFLOPS, %f ms \n", (float)2 * total_len / (time / num_repeat * 1e-3) / 1e9,(time / num_repeat));

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(tensor);
    aclDestroyTensor(out);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));

    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }

    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    // CHECK_ACL(aclrtFreeHost(zHost));
    // CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    
#endif
    return 0;
}