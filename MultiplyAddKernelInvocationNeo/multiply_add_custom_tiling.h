
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include <cstdint>

struct MulAddCustomTilingData
{
    uint32_t totalLength;
    uint32_t tileNum;
    // uint32_t ma_method;
    uint32_t element_size;
    uint32_t gm_sync_size;
};
#endif