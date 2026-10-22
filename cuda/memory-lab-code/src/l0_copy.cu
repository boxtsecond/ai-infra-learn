// l0_copy.cu —— L0 基线：纯拷贝 kernel 的有效带宽
//
// 测量设置：
//   - 单个数组默认 1024 MiB，与 L1~L4 的有效数据量相同，并远大于 L2。
//     数组要大于 L2，否则重复 launch 时数据留在 L2 里，测到的是 L2 带宽。
//   - kernel 写成最朴素的形式：不加 __restrict__、不展开、不向量化。
//     向量化的影响由 l0_copy_vec 单独测。
//   - 用 cudaEvent 计时：kernel launch 是异步的，CPU 计时量到的只是 launch 返回。
//   - 先跑 3 次 warm up 不计时，排除首次 launch 的初始化开销。
//   - block = 256，1 线程 1 元素。
//   - 计时结束后把两个数组全量拷回 host 比对，校验不参与计时。
//
// 编译：nvcc -O3 -arch=sm_90 -lineinfo l0_copy.cu -o l0_copy
// 运行：CUDA_VISIBLE_DEVICES=0 ./l0_copy [单数组MiB，默认1024] [计时次数，默认10]
// 带宽波动范围要用独立进程重复运行 3 次来测，见 README。

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cerrno>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d\n  调用: %s\n  原因: %s\n",    \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err_));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------- 终端表格对齐 ----------
// printf 的 %-16s 按字节数填充，而 UTF-8 中文一字 3 字节仅占 2 显示列，
// 中英混排会错位。以下按显示宽度填充。
static int dispWidth(const char* s) {
    int w = 0;
    for (const unsigned char* p = (const unsigned char*)s; *p; ) {
        if (*p < 0x80)                { w += 1; p += 1; }
        else if ((*p & 0xE0) == 0xC0) { w += 1; p += 2; }
        else if ((*p & 0xF0) == 0xE0) { w += 2; p += 3; }
        else if ((*p & 0xF8) == 0xF0) { w += 2; p += 4; }
        else                          { w += 1; p += 1; }
    }
    return w;
}
static void padL(const char* s, int width) {
    fputs(s, stdout);
    for (int i = dispWidth(s); i < width; ++i) putchar(' ');
}
#define KV(key, ...) do { fputs("  ", stdout); padL(key, 16);                  \
                          printf(__VA_ARGS__); putchar('\n'); } while (0)

// ---------- 命令行参数 ----------
// 只接受 [lo, hi] 范围内的十进制整数，拒绝负号、小数、尾随字符和溢出。
static bool parseArg(const char* text, unsigned long long lo, unsigned long long hi,
                     unsigned long long* out) {
    if (text == nullptr || *text == '\0') return false;
    for (const char* p = text; *p != '\0'; ++p)
        if (*p < '0' || *p > '9') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || v < lo || v > hi) return false;
    *out = v;
    return true;
}

// ---------- 环境 ----------
// 理论带宽 = 显存时钟 x 2 x 位宽 / 8（CUDA C++ Best Practices Guide 9.2.1）。
// 时钟和位宽都在运行时查询，换卡不用改代码。
struct Env { double theoGBs; size_t l2Bytes; };

static Env printEnv(int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int smCount = 0, memClockKHz = 0, busBits = 0, l2Bytes = 0, drv = 0, rt = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount,     cudaDevAttrMultiProcessorCount,  dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate,      dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&busBits,     cudaDevAttrGlobalMemoryBusWidth, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2Bytes,     cudaDevAttrL2CacheSize,          dev));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    const double theoGBs = 2.0 * memClockKHz * 1e3 * (busBits / 8.0) / 1e9;

    printf("环境\n");
    KV("GPU",          "%s", prop.name);
    KV("Compute Cap",  "%d.%d", prop.major, prop.minor);
    KV("SM 数",        "%d", smCount);
    KV("理论带宽",     "%.1f GB/s（%d MHz x 2 x %d bit / 8）", theoGBs, memClockKHz / 1000, busBits);
    KV("L2",           "%.1f MiB", l2Bytes / 1048576.0);
    KV("ECC",          "%s", prop.ECCEnabled ? "on" : "off");
    KV("CUDA drv/rt",  "%d.%d / %d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);
    return { theoGBs, (size_t)l2Bytes };
}

// 样本已排序且非空；偶数个时取中间两个的均值
template <typename T>
static T medianOfSorted(const std::vector<T>& v) {
    const size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? T(0.5) * (v[m - 1] + v[m]) : v[m];
}

// ---------- kernel ----------
// 被测 kernel：每个线程拷 1 个 float
__global__ void copyKernel(const float* a, float* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

// 初始化输入，不计时
__global__ void fillRamp(float* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 1.0f + (float)(i % 1024);
}

int main(int argc, char** argv) {
    unsigned long long mib = 1024, reps = 10;
    if (argc > 3 ||
        (argc > 1 && !parseArg(argv[1], 1, 1048576, &mib)) ||
        (argc > 2 && !parseArg(argv[2], 1, 100000, &reps))) {
        fprintf(stderr, "用法: %s [单数组MiB 1..1048576，默认1024] [计时次数 1..100000，默认10]\n",
                argv[0]);
        return 2;
    }
    const int    block = 256;
    const int    warm  = 3;
    const size_t bytes = (size_t)mib * 1024 * 1024;
    const size_t n     = bytes / sizeof(float);
    const size_t grid  = (n + block - 1) / block;
    const double totalBytes = 2.0 * (double)bytes;          // 读 + 写

    CUDA_CHECK(cudaSetDevice(0));
    const Env env = printEnv(0);

    printf("\n配置\n");
    KV("单数组",       "%llu MiB（%zu 个 float）", mib, n);
    KV("读写字节数",   "%.0f MiB（读 + 写）", totalBytes / 1048576.0);
    KV("数组 / L2",    "%.1fx", (double)bytes / (double)env.l2Bytes);
    KV("block",        "%d", block);
    KV("grid",         "%zu（1 线程 1 元素）", grid);
    KV("warm up",      "%d 次（不计时）", warm);
    KV("计时次数",     "%llu", reps);

    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    fillRamp<<<grid, block>>>(d_a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warm; ++i) {
        copyKernel<<<grid, block>>>(d_a, d_b, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 逐次计时，保留每一次的耗时
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    std::vector<float> ms(reps, 0.0f);
    for (size_t r = 0; r < ms.size(); ++r) {
        CUDA_CHECK(cudaEventRecord(evStart));
        copyKernel<<<grid, block>>>(d_a, d_b, n);
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms[r], evStart, evStop));
    }

    // 全量校验，分块拷回 host（每块 4 MiB），不计时
    const size_t chunk = std::min(n, size_t(1) << 20);
    std::vector<float> hostA(chunk), hostB(chunk);
    size_t mismatch = 0;
    for (size_t off = 0; off < n; off += chunk) {
        const size_t count = std::min(chunk, n - off);
        CUDA_CHECK(cudaMemcpy(hostA.data(), d_a + off, count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostB.data(), d_b + off, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i)
            if (hostA[i] != hostB[i]) ++mismatch;
    }

    printf("\n逐次结果\n");
    for (size_t r = 0; r < ms.size(); ++r) {
        const double bw = totalBytes / (ms[r] * 1e-3) / 1e9;
        printf("  run %3zu  %8.4f ms  %8.1f GB/s  占理论带宽 %5.1f%%\n",
               r + 1, ms[r], bw, bw / env.theoGBs * 100.0);
    }

    std::vector<float> msSorted = ms;
    std::sort(msSorted.begin(), msSorted.end());
    std::vector<double> bwSorted;
    for (float t : ms) bwSorted.push_back(totalBytes / (t * 1e-3) / 1e9);
    std::sort(bwSorted.begin(), bwSorted.end());
    const float  msMed = medianOfSorted(msSorted);
    const double bwMed = medianOfSorted(bwSorted);

    printf("\n统计\n");
    KV("耗时",         "min %.4f / median %.4f / max %.4f ms", msSorted.front(), msMed, msSorted.back());
    KV("有效带宽",     "min %.1f / median %.1f / max %.1f GB/s", bwSorted.front(), bwMed, bwSorted.back());
    KV("占理论带宽",   "%.1f%%（按带宽中位数）", bwMed / env.theoGBs * 100.0);
    KV("进程内波动",   "%.2f%%（耗时 (max - min) / median）",
       (msSorted.back() - msSorted.front()) / msMed * 100.0);
    KV("校验",         "%s（全量 %zu 个元素，不一致 %zu 个）", mismatch == 0 ? "PASS" : "FAIL", n, mismatch);

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return mismatch == 0 ? 0 : 1;
}
