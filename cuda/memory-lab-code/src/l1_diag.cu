// l1_diag.cu —— 跨步访问的两组对照
//
// 对照 A：跨步发生在读端、写端还是两端。三个 kernel 在同一进程内顺序测量：
//   RW  b[i*s] = a[i*s]   两端跨步（与 l1_stride 相同）
//   R   b[i]   = a[i*s]   仅读端跨步（与 l3_layout 的 AoS 读法相同）
//   W   b[i*s] = a[i]     仅写端跨步
// 对照 B：stride=1 时，分开索引计算和数组分配大小两个因素：
//   plain      b[i]   = a[i]     分配 = 触碰量
//   mul1       b[i*1] = a[i*1]   分配 = 触碰量；即 copyRW，stride 作为运行时参数传入
//   bigalloc   b[i*1] = a[i*1]   分配 = 触碰量 x 32（与 l1_stride 相同）
//   plain_big  b[i]   = a[i]     分配 = 触碰量 x 32
// 每个变体运行前把输出数组清零，跑完后抽样校验被写入的元素（头、中、尾各 1024 个）。
//
// 显存需求：默认 1024 MiB 时约 68 GiB。显存不够时调小第一个参数。
//
// 编译：nvcc -O3 -arch=sm_90 -lineinfo l1_diag.cu -o l1_diag
// 运行：CUDA_VISIBLE_DEVICES=0 ./l1_diag [有效数据MiB，默认1024] [计时次数，默认10]

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
// 对照 A 的三种跨步组合
__global__ void copyRW(const float* a, float* b, size_t n, size_t s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i * s] = a[i * s];
}
__global__ void copyRonly(const float* a, float* b, size_t n, size_t s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i * s];
}
__global__ void copyWonly(const float* a, float* b, size_t n, size_t s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i * s] = a[i];
}
// 对照 B：索引里没有 stride，和 copyRW(s=1) 对比
__global__ void copyPlain(const float* a, float* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}
// 校验用：把 src 中按 stride 分布的 n 个元素收拢成连续数组
__global__ void gatherStride(const float* src, float* dst, size_t n, size_t s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i * s];
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
    const size_t strides[] = {1, 2, 3, 4, 8};
    const int    nStride   = sizeof(strides) / sizeof(strides[0]);
    const size_t strideMax = 32;                 // 与 l1_stride 的数组分配相同

    const size_t n           = (size_t)mib * 1024 * 1024 / sizeof(float);
    const size_t grid        = (n + block - 1) / block;
    const double usefulBytes = 2.0 * (double)n * sizeof(float);   // 读 + 写
    const size_t smallBytes  = n * sizeof(float);
    const size_t bigBytes    = n * strideMax * sizeof(float);

    CUDA_CHECK(cudaSetDevice(0));
    printEnv(0);

    printf("\n配置\n");
    KV("有效元素",     "%zu（%llu MiB）", n, mib);
    KV("读写字节数",   "%.0f MiB（读 + 写）", usefulBytes / 1048576.0);
    KV("小分配",       "%.0f MiB x 2", (double)smallBytes / 1048576.0);
    KV("大分配",       "%.0f MiB x 2", (double)bigBytes / 1048576.0);
    KV("block",        "%d", block);
    KV("grid",         "%zu", grid);
    KV("warm up",      "%d 次（不计时）", warm);
    KV("计时次数",     "%d", reps);

    float *s_a = nullptr, *s_b = nullptr;        // 小分配：恰好容纳触碰量
    float *b_a = nullptr, *b_b = nullptr;        // 大分配：按 strideMax 预留
    float *d_ref = nullptr, *d_chk = nullptr;    // 校验缓冲
    CUDA_CHECK(cudaMalloc(&s_a, smallBytes));
    CUDA_CHECK(cudaMalloc(&s_b, smallBytes));
    CUDA_CHECK(cudaMalloc(&b_a, bigBytes));
    CUDA_CHECK(cudaMalloc(&b_b, bigBytes));
    CUDA_CHECK(cudaMalloc(&d_ref, smallBytes));
    CUDA_CHECK(cudaMalloc(&d_chk, smallBytes));
    fillRamp<<<grid, block>>>(s_a, n);
    CUDA_CHECK(cudaGetLastError());
    fillRamp<<<(n * strideMax + block - 1) / block, block>>>(b_a, n * strideMax);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto bwOf = [&](float ms) { return usefulBytes / (ms * 1e-3) / 1e9; };
    size_t totalBad = 0;
    char buf[64];

    // ---------- 对照 A ----------
    printf("\n对照 A：读写哪一端跨步（有效带宽 GB/s）\n");
    enum { W_ST = 8, W_V = 12, W_REL = 10, W_OK = 8 };
    padR("stride", W_ST);
    padR("RW 两端", W_V);  padR("R 仅读端", W_V);  padR("W 仅写端", W_V);
    padR("R/RW", W_REL);   padR("W/RW", W_REL);    padR("校验", W_OK);
    putchar('\n');

    double bwRW1 = 0.0, bwR1 = 0.0, bwW1 = 0.0, bwRW3 = 0.0, bwR3 = 0.0, bwW3 = 0.0;
    for (int k = 0; k < nStride; ++k) {
        const size_t s = strides[k];
        size_t bad = 0;

        CUDA_CHECK(cudaMemset(b_b, 0, bigBytes));
        const double bwRW = bwOf(timeMedianMs([=] { copyRW<<<grid, block>>>(b_a, b_b, n, s); }, reps, warm));
        gatherStride<<<grid, block>>>(b_a, d_ref, n, s);
        gatherStride<<<grid, block>>>(b_b, d_chk, n, s);
        CUDA_CHECK(cudaGetLastError());
        bad += verifySample(d_ref, d_chk, n);

        CUDA_CHECK(cudaMemset(b_b, 0, bigBytes));
        const double bwR = bwOf(timeMedianMs([=] { copyRonly<<<grid, block>>>(b_a, b_b, n, s); }, reps, warm));
        gatherStride<<<grid, block>>>(b_a, d_ref, n, s);
        CUDA_CHECK(cudaGetLastError());
        bad += verifySample(d_ref, b_b, n);

        CUDA_CHECK(cudaMemset(b_b, 0, bigBytes));
        const double bwW = bwOf(timeMedianMs([=] { copyWonly<<<grid, block>>>(b_a, b_b, n, s); }, reps, warm));
        gatherStride<<<grid, block>>>(b_b, d_chk, n, s);
        CUDA_CHECK(cudaGetLastError());
        bad += verifySample(b_a, d_chk, n);

        totalBad += bad;
        if (s == 1) { bwRW1 = bwRW; bwR1 = bwR; bwW1 = bwW; }
        if (s == 3) { bwRW3 = bwRW; bwR3 = bwR; bwW3 = bwW; }

        snprintf(buf, sizeof buf, "%zu", s);              padR(buf, W_ST);
        snprintf(buf, sizeof buf, "%.1f", bwRW);          padR(buf, W_V);
        snprintf(buf, sizeof buf, "%.1f", bwR);           padR(buf, W_V);
        snprintf(buf, sizeof buf, "%.1f", bwW);           padR(buf, W_V);
        snprintf(buf, sizeof buf, "%.3fx", bwR / bwRW);   padR(buf, W_REL);
        snprintf(buf, sizeof buf, "%.3fx", bwW / bwRW);   padR(buf, W_REL);
        padR(bad == 0 ? "PASS" : "FAIL", W_OK);
        putchar('\n');
    }
    printf("  stride=3 相对各自 stride=1：RW %.3fx，R %.3fx，W %.3fx\n",
           bwRW3 / bwRW1, bwR3 / bwR1, bwW3 / bwW1);

    // ---------- 对照 B ----------
    printf("\n对照 B：stride=1 时的索引计算与分配大小\n");
    struct Row { const char* name; const char* idx; size_t alloc; double bw; size_t bad; };
    Row rows[4];
    {
        CUDA_CHECK(cudaMemset(s_b, 0, smallBytes));
        const double bw = bwOf(timeMedianMs([=] { copyPlain<<<grid, block>>>(s_a, s_b, n); }, reps, warm));
        rows[0] = {"plain", "a[i]", smallBytes, bw, verifySample(s_a, s_b, n)};
    }
    {
        CUDA_CHECK(cudaMemset(s_b, 0, smallBytes));
        const double bw = bwOf(timeMedianMs([=] { copyRW<<<grid, block>>>(s_a, s_b, n, 1); }, reps, warm));
        rows[1] = {"mul1", "a[i*1]", smallBytes, bw, verifySample(s_a, s_b, n)};
    }
    {
        CUDA_CHECK(cudaMemset(b_b, 0, bigBytes));
        const double bw = bwOf(timeMedianMs([=] { copyRW<<<grid, block>>>(b_a, b_b, n, 1); }, reps, warm));
        rows[3] = {"bigalloc", "a[i*1]", bigBytes, bw, verifySample(b_a, b_b, n)};
    }
    {
        CUDA_CHECK(cudaMemset(b_b, 0, bigBytes));
        const double bw = bwOf(timeMedianMs([=] { copyPlain<<<grid, block>>>(b_a, b_b, n); }, reps, warm));
        rows[2] = {"plain_big", "a[i]", bigBytes, bw, verifySample(b_a, b_b, n)};
    }

    enum { W_VAR = 12, W_IDX = 10, W_ALLOC = 13, W_BW = 14, W_REL2 = 13 };
    padL("变体", W_VAR);    padR("索引", W_IDX);       padR("分配", W_ALLOC);
    padR("有效带宽", W_BW); padR("相对 plain", W_REL2); padR("校验", W_OK);
    putchar('\n');
    for (const auto& r : rows) {
        padL(r.name, W_VAR);
        padR(r.idx, W_IDX);
        snprintf(buf, sizeof buf, "%.0f MiB", (double)r.alloc / 1048576.0); padR(buf, W_ALLOC);
        snprintf(buf, sizeof buf, "%.1f GB/s", r.bw);                       padR(buf, W_BW);
        snprintf(buf, sizeof buf, "%.4fx", r.bw / rows[0].bw);              padR(buf, W_REL2);
        padR(r.bad == 0 ? "PASS" : "FAIL", W_OK);
        putchar('\n');
        totalBad += r.bad;
    }
    printf("  只改索引（mul1 相对 plain）：%+.2f%%\n", (rows[1].bw / rows[0].bw - 1.0) * 100.0);
    printf("  只改分配（plain_big 相对 plain）：%+.2f%%\n", (rows[2].bw / rows[0].bw - 1.0) * 100.0);
    printf("  只改分配（bigalloc 相对 mul1）：%+.2f%%\n", (rows[3].bw / rows[1].bw - 1.0) * 100.0);

    CUDA_CHECK(cudaFree(s_a));   CUDA_CHECK(cudaFree(s_b));
    CUDA_CHECK(cudaFree(b_a));   CUDA_CHECK(cudaFree(b_b));
    CUDA_CHECK(cudaFree(d_ref)); CUDA_CHECK(cudaFree(d_chk));
    return totalBad == 0 ? 0 : 1;
}
