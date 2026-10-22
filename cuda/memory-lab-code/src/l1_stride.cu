// l1_stride.cu —— 跨步访问：相邻线程的地址间距对带宽的影响
//
// kernel：b[i * stride] = a[i * stride]，读写两端都按 stride 跨步。
// 自变量只有 stride（1 2 3 4 8 16 32）。有效元素数、kernel、block 大小、计时方式不变。
// 数组按最大 stride 预留 32 x N 个 float，但每个 stride 都只触碰 N 个元素，
// 所以读写字节数在各个 stride 下相同。
// 校验：把 a、b 中被触碰的元素按 stride 收拢成连续数组，抽样比对头、中、尾各 1024 个。
//
// 显存需求：默认 1024 MiB 时约 66 GiB（两个 32 GiB 数组 + 两个 1 GiB 校验缓冲）。
// 显存不够时调小第一个参数，例如 256 → 约 17 GiB。
//
// 编译：nvcc -O3 -arch=sm_90 -lineinfo l1_stride.cu -o l1_stride
// 运行：CUDA_VISIBLE_DEVICES=0 ./l1_stride [有效数据MiB，默认1024] [计时次数，默认10]

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <algorithm>
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
static void padR(const char* s, int width) {
    for (int i = dispWidth(s); i < width; ++i) putchar(' ');
    fputs(s, stdout);
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

// ---------- 计时 ----------
// cudaEvent 计时；warm up 不计入。返回耗时中位数，偶数个时取中间两个的均值。
template <typename LaunchFn>
static float timeMedianMs(LaunchFn launch, int reps, int warmup) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    std::vector<float> ms(reps, 0.0f);
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(evStart));
        launch();
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms[r], evStart, evStop));
    }
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    std::sort(ms.begin(), ms.end());
    return (reps % 2 == 0) ? 0.5f * (ms[reps / 2 - 1] + ms[reps / 2]) : ms[reps / 2];
}

// ---------- 校验：抽样比对头、中、尾各 1024 个元素，返回不一致个数 ----------
static size_t verifySample(const float* d_expect, const float* d_got, size_t n) {
    const size_t len = std::min(n, size_t(1024));
    const size_t off[3] = {0, std::min(n / 2, n - len), n - len};
    std::vector<float> he(len), hg(len);
    size_t bad = 0;
    for (int s = 0; s < 3; ++s) {
        CUDA_CHECK(cudaMemcpy(he.data(), d_expect + off[s], len * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hg.data(), d_got + off[s], len * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < len; ++i)
            if (he[i] != hg[i]) ++bad;
    }
    return bad;
}

// ---------- kernel ----------
__global__ void fillRamp(float* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 1.0f + (float)(i % 1024);
}
// 被测 kernel
__global__ void copyStride(const float* a, float* b, size_t n, size_t stride) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i * stride] = a[i * stride];
}
// 校验用：把 src 中按 stride 分布的 n 个元素收拢成连续数组
__global__ void gatherStride(const float* src, float* dst, size_t n, size_t stride) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i * stride];
}

int main(int argc, char** argv) {
    unsigned long long mib = 1024, repsArg = 10;
    if (argc > 3 ||
        (argc > 1 && !parseArg(argv[1], 1, 1048576, &mib)) ||
        (argc > 2 && !parseArg(argv[2], 1, 100000, &repsArg))) {
        fprintf(stderr, "用法: %s [有效数据MiB 1..1048576，默认1024] [计时次数 1..100000，默认10]\n",
                argv[0]);
        return 2;
    }
    const int    reps  = (int)repsArg;
    const int    block = 256;
    const int    warm  = 3;
    const size_t strides[] = {1, 2, 3, 4, 8, 16, 32};
    const int    nStride   = sizeof(strides) / sizeof(strides[0]);
    const size_t strideMax = strides[nStride - 1];

    const size_t n           = (size_t)mib * 1024 * 1024 / sizeof(float);
    const size_t allocBytes  = n * strideMax * sizeof(float);
    const size_t grid        = (n + block - 1) / block;
    const double usefulBytes = 2.0 * (double)n * sizeof(float);   // 读 + 写

    CUDA_CHECK(cudaSetDevice(0));
    const Env env = printEnv(0);

    printf("\n配置\n");
    KV("有效元素",     "%zu（%llu MiB）", n, mib);
    KV("读写字节数",   "%.0f MiB（读 + 写）", usefulBytes / 1048576.0);
    KV("数组分配",     "%.0f MiB x 2（按 stride=%zu 预留）", allocBytes / 1048576.0, strideMax);
    KV("block",        "%d", block);
    KV("grid",         "%zu", grid);
    KV("warm up",      "%d 次（不计时）", warm);
    KV("计时次数",     "%d", reps);
    KV("stride",       "1 2 3 4 8 16 32");
    KV("触碰跨度 / L2", "%.1fx（stride=1）", (double)(n * sizeof(float)) / (double)env.l2Bytes);

    float *d_a = nullptr, *d_b = nullptr, *d_chk = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, allocBytes));
    CUDA_CHECK(cudaMalloc(&d_b, allocBytes));
    CUDA_CHECK(cudaMalloc(&d_chk, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, n * sizeof(float)));
    fillRamp<<<(n * strideMax + block - 1) / block, block>>>(d_a, n * strideMax);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    enum { W_ST = 8, W_SPAN = 13, W_MS = 11, W_BW = 13, W_REL = 11, W_PREV = 13, W_OK = 8 };
    char buf[64];
    putchar('\n');
    padR("stride", W_ST);   padR("触碰跨度", W_SPAN);   padR("耗时 ms", W_MS);
    padR("有效带宽", W_BW); padR("相对 s=1", W_REL);    padR("相对上一行", W_PREV);
    padR("校验", W_OK);
    putchar('\n');

    double bw1 = 0.0, bwPrev = 0.0;
    size_t totalBad = 0;
    for (int k = 0; k < nStride; ++k) {
        const size_t s = strides[k];
        CUDA_CHECK(cudaMemset(d_b, 0, allocBytes));
        const float ms = timeMedianMs([=] { copyStride<<<grid, block>>>(d_a, d_b, n, s); },
                                      reps, warm);
        gatherStride<<<grid, block>>>(d_b, d_chk, n, s);
        CUDA_CHECK(cudaGetLastError());
        gatherStride<<<grid, block>>>(d_a, d_ref, n, s);
        CUDA_CHECK(cudaGetLastError());
        const size_t bad = verifySample(d_ref, d_chk, n);
        totalBad += bad;

        const double bw = usefulBytes / (ms * 1e-3) / 1e9;
        if (k == 0) bw1 = bw;

        snprintf(buf, sizeof buf, "%zu", s);                                         padR(buf, W_ST);
        snprintf(buf, sizeof buf, "%.0f MiB", (double)(n * s * sizeof(float)) / 1048576.0); padR(buf, W_SPAN);
        snprintf(buf, sizeof buf, "%.4f", ms);                                       padR(buf, W_MS);
        snprintf(buf, sizeof buf, "%.1f GB/s", bw);                                  padR(buf, W_BW);
        snprintf(buf, sizeof buf, "%.3fx", bw / bw1);                                padR(buf, W_REL);
        if (k == 0) padR("-", W_PREV);
        else { snprintf(buf, sizeof buf, "%+.1f%%", (bw / bwPrev - 1.0) * 100.0);   padR(buf, W_PREV); }
        padR(bad == 0 ? "PASS" : "FAIL", W_OK);
        putchar('\n');
        bwPrev = bw;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_chk));
    CUDA_CHECK(cudaFree(d_ref));
    return totalBad == 0 ? 0 : 1;
}
