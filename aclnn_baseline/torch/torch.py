import torch
import torch_npu

print(f"PyTorch 版本: {torch.__version__}")
# 检查 torch_npu 是否已安装且 NPU 是否可用
if hasattr(torch, 'npu') and torch.npu.is_available():
    print(f"NPU 是否可用: True")
    print(f"当前 NPU 设备: {torch.npu.get_device_name(0)}")
    device_npu = torch.device("npu:0")
else:
    print(f"NPU 是否可用: False")
    device_npu = torch.device("cpu")

print(f"将使用设备: {device_npu}")

def benchmark_vec_add(device, size, dtype, num_runs=100, warm_up_runs=20):
    """
    一个用于测试向量加法性能的函数。

    参数:
    - device: torch.device, 计算设备
    - size: int, 向量的维度
    - dtype: torch.dtype, 数据类型 (如 torch.float32)
    - num_runs: int, 正式测试的运行次数
    - warm_up_runs: int, 预热的运行次数
    """
    # 准备数据
    a = torch.randn(size, device=device, dtype=dtype)
    b = torch.randn(size, device=device, dtype=dtype)

    # 预热
    for _ in range(warm_up_runs):
        _ = a + b

    # 使用 torch.npu.Event 进行精确计时
    starter, ender = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    timings = torch.zeros(num_runs) # 用张量记录时间

    # 正式测试
    for i in range(num_runs):
        starter.record()
        _ = a + b
        ender.record()
        
        # 等待 NPU 完成
        torch.npu.synchronize()
        
        # 记录时间 (毫秒)
        timings[i] = starter.elapsed_time(ender)
        
    avg_time_ms = timings.mean().item()

    # 计算 TFLOPS (每秒万亿次浮点运算)
    # 向量加法 (N) + (N) 的计算量大约是 N
    tflops = (size) / (avg_time_ms / 1000) / 1e12
    
    return avg_time_ms, tflops

def benchmark_vec_mul(device, size, dtype, num_runs=100, warm_up_runs=20):
    """
    一个用于测试向量乘法性能的函数。

    参数:
    - device: torch.device, 计算设备
    - size: int, 向量的维度
    - dtype: torch.dtype, 数据类型 (如 torch.float32)
    - num_runs: int, 正式测试的运行次数
    - warm_up_runs: int, 预热的运行次数
    """
    # 准备数据
    a = torch.randn(size, device=device, dtype=dtype)
    b = torch.randn(size, device=device, dtype=dtype)

    # 预热
    for _ in range(warm_up_runs):
        _ = a * b
    
    # 使用 torch.npu.Event 进行精确计时
    starter, ender = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    timings = torch.zeros(num_runs) # 用张量记录时间

    # 正式测试
    for i in range(num_runs):
        starter.record()
        _ = a * b
        ender.record()
        
        # 等待 NPU 完成
        torch.npu.synchronize()
        
        # 记录时间 (毫秒)
        timings[i] = starter.elapsed_time(ender)
        
    avg_time_ms = timings.mean().item()

    # 计算 TFLOPS (每秒万亿次浮点运算)
    # 向量乘法 (N) * (N) 的计算量大约是 N
    tflops = (size) / (avg_time_ms / 1000) / 1e12
    
    return avg_time_ms, tflops

# --- 测试开始 ---
vector_sizes = [100000]

# 仅当 NPU 可用时运行测试
if device_npu.type == 'npu':
    for vector_size in vector_sizes:
        print("="*50)
        print(f"向量维度 (Scale): {vector_size}")
        print("="*50)

        try:
            # --- 测试向量加法 ---
            print("\n--- 向量加法 (Vector Addition) ---")
            # 测试 Float32
            add_time_fp32, add_tflops_fp32 = benchmark_vec_add(device_npu, vector_size, torch.float32)
            print(f"NPU (FP32): 平均耗时 = {add_time_fp32:.6f} ms, 性能 = {add_tflops_fp32:.4f} TFLOPS")

            # 测试 Float16
            add_time_fp16, add_tflops_fp16 = benchmark_vec_add(device_npu, vector_size, torch.float16)
            print(f"NPU (FP16): 平均耗时 = {add_time_fp16:.6f} ms, 性能 = {add_tflops_fp16:.4f} TFLOPS")

            # 测试 BFloat16
            add_time_bf16, add_tflops_bf16 = benchmark_vec_add(device_npu, vector_size, torch.bfloat16)
            print(f"NPU (BF16): 平均耗时 = {add_time_bf16:.6f} ms, 性能 = {add_tflops_bf16:.4f} TFLOPS")

            # --- 测试向量乘法 ---
            print("\n--- 向量乘法 (Vector Multiplication) ---")
            # 测试 Float32
            mul_time_fp32, mul_tflops_fp32 = benchmark_vec_mul(device_npu, vector_size, torch.float32)
            print(f"NPU (FP32): 平均耗时 = {mul_time_fp32:.6f} ms, 性能 = {mul_tflops_fp32:.4f} TFLOPS")

            # 测试 Float16
            mul_time_fp16, mul_tflops_fp16 = benchmark_vec_mul(device_npu, vector_size, torch.float16)
            print(f"NPU (FP16): 平均耗时 = {mul_time_fp16:.6f} ms, 性能 = {mul_tflops_fp16:.4f} TFLOPS")

            # 测试 BFloat16
            mul_time_bf16, mul_tflops_bf16 = benchmark_vec_mul(device_npu, vector_size, torch.bfloat16)
            print(f"NPU (BF16): 平均耗时 = {mul_time_bf16:.6f} ms, 性能 = {mul_tflops_bf16:.4f} TFLOPS")

        except RuntimeError as e:
            print(f"在设备 {device_npu} 上测试失败: {e}")
else:
    print("\n未检测到可用的 NPU 设备，跳过基准测试。")