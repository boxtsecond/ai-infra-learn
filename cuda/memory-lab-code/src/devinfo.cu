// devinfo —— 换机器时第一步：核对规格表。
//
// 不依赖 CUDA samples 的 deviceQuery（CUDA 12.8 的 extras/ 下已不附带，
// 只有 CUPTI 和 Debugger）。所有数字都由 CUDA Runtime API 现场查询，
// 不硬编码任何规格，与 l0–l5 的「环境」块同源。
//
// 编译: nvcc -O2 -arch=sm_90 src/devinfo.cu -o bin/devinfo   （或 make）
// 运行: CUDA_VISIBLE_DEVICES=0 ./bin/devinfo
//
// 注意「CUDA 可见显存」与 nvidia-smi 的 memory.total 不是一个口径：
// ECC 开启时驱动会预留一部分（nvidia-smi 的 memory.reserved），
// 该部分不计入 cudaMemGetInfo / totalGlobalMem。做 数组/L2 倍数一类的
// 容量估算要用这里的 CUDA 可见值。

#include <cstdarg>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            printf("CUDA error %s at line %d: %s\n", #x, __LINE__,             \
                   cudaGetErrorString(e_));                                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// 计算能力 major -> 架构代号。来源：CUDA C++ Programming Guide 的
// Compute Capabilities 一节；只列本仓库可能遇到的几代。
static const char *archName(int major, int minor) {
    switch (major) {
        case 7: return minor == 0 ? "Volta" : "Turing";
        case 8: return minor == 0 ? "Ampere (A100)" : "Ampere / Ada";
        case 9: return "Hopper";
        case 10: return "Blackwell";
        default: return "未知（查 Programming Guide 确认）";
    }
}

// printf 的 %-Ns 按字节补齐，而 UTF-8 的汉字是 3 字节、2 个显示列，
// 直接用 %-28s 会让中文标签那几行对不齐。这里按显示列宽手工补。
static int displayWidth(const char *s) {
    int w = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if ((*p & 0xC0) == 0x80) continue;  // UTF-8 续接字节，不计
        w += (*p >= 0xE0) ? 2 : 1;          // 3 字节起始按全角 2 列
    }
    return w;
}

static void kv(const char *k, const char *fmt, ...) {
    const int kWidth = 26;
    printf("  %s", k);
    for (int i = displayWidth(k); i < kWidth; ++i) putchar(' ');
    putchar(' ');
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    int drv = 0, rt = 0;
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));

    printf("规格表（CUDA 侧，cudaGetDeviceProperties / cudaMemGetInfo）\n");
    kv("CUDA 可见设备数", "%d  (受 CUDA_VISIBLE_DEVICES 影响)", count);
    kv("CUDA drv / rt", "%d.%d / %d.%d", drv / 1000, (drv % 100) / 10,
       rt / 1000, (rt % 100) / 10);

    for (int d = 0; d < count; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        size_t freeB = 0, totalB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

        printf("\n[device %d] %s\n", d, p.name);
        kv("架构 / 计算能力", "%s / %d.%d", archName(p.major, p.minor),
           p.major, p.minor);
        kv("CUDA 可见显存", "%.1f MiB (%.2f GiB)  free %.1f MiB",
           totalB / 1048576.0, totalB / 1073741824.0, freeB / 1048576.0);
        kv("L2 cache", "%d B (%.1f MiB)", p.l2CacheSize,
           p.l2CacheSize / 1048576.0);
        kv("SM 数量", "%d", p.multiProcessorCount);
        kv("Warp size", "%d", p.warpSize);
        kv("每 block 最大线程数", "%d", p.maxThreadsPerBlock);
        kv("每 block 默认 shared mem", "%zu B (%.0f KiB)", p.sharedMemPerBlock,
           p.sharedMemPerBlock / 1024.0);
        kv("每 block opt-in 上限", "%zu B (%.0f KiB)  需 cudaFuncSetAttribute",
           p.sharedMemPerBlockOptin, p.sharedMemPerBlockOptin / 1024.0);
        kv("每 SM shared mem", "%zu B (%.0f KiB)", p.sharedMemPerMultiprocessor,
           p.sharedMemPerMultiprocessor / 1024.0);

        printf("  ---- 带宽相关（l0–l5 的「环境」块同源）----\n");
        // 显存时钟用 cudaDeviceGetAttribute 查询：cudaDeviceProp::memoryClockRate
        // 在新版 CUDA 中已从结构体移除，用属性 API 才能跨版本编译。
        int memClockKHz = 0, busBits = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, d));
        CUDA_CHECK(cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, d));
        kv("显存时钟", "%d kHz (%.0f MHz)", memClockKHz, memClockKHz / 1000.0);
        kv("显存位宽", "%d bit", busBits);
        kv("理论带宽（算出来的）", "%.1f GB/s  = 时钟 x 2 x 位宽/8",
           2.0 * memClockKHz * 1e3 * (busBits / 8.0) / 1e9);
        kv("ECC", "%s", p.ECCEnabled ? "on" : "off");
        kv("驻留线程上限", "%d",
           p.maxThreadsPerMultiProcessor * p.multiProcessorCount);
    }
    return 0;
}
