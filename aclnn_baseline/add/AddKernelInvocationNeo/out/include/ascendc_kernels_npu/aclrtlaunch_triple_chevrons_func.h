
#ifndef HEADER_ACLRTLAUNCH_ADD_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ADD_CUSTOM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_add_custom(uint32_t blockDim, void* stream, void* x, void* y, void* z);

inline uint32_t add_custom(uint32_t blockDim, void* hold, void* stream, void* x, void* y, void* z)
{
    (void)hold;
    return aclrtlaunch_add_custom(blockDim, stream, x, y, z);
}

#endif
