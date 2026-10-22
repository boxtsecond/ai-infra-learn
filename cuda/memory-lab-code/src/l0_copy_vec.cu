// l0_copy_vec.cu —— 每线程访存宽度对拷贝带宽的影响
//
// 三个版本拷贝同样多的数据，只改变每个线程一次读写的字节数：
//   float   4 B    copyScalar
//   float2  8 B    copyVec2
//   float4  16 B   copyVec4
// 单个数组默认 1024 MiB，与 l0_copy 相同。
// block 大小、warm up 次数和计时方式相同。grid 随每线程字节数缩小为 1/2、1/4，
// 因此 block 总数和执行的指令数也随之变化。三个版本在同一进程内顺序运行。
// 校验：每个版本跑完后抽样比对头、中、尾各 1024 个元素。
//
// 编译：nvcc -O3 -arch=sm_90 -lineinfo l0_copy_vec.cu -o l0_copy_vec
// 运行：CUDA_VISIBLE_DEVICES=0 ./l0_copy_vec [单数组MiB，默认1024] [计时次数，默认10]

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
// 三个被测 kernel，除元素类型外代码相同
__global__ void copyScalar(const float* a, float* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}
__global__ void copyVec2(const float2* a, float2* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}
__global__ void copyVec4(const float4* a, float4* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

__global__ void fillRamp(float* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 1.0f + (float)(i % 1024);
}

struct Result {
    const char* name;
    int    bytesPerThread;
    size_t grid;
    float  ms;
    double gbs;
    size_t bad;
};

int main(int argc, char** argv) {
    unsigned long long mib = 1024, repsArg = 10;
    if (argc > 3 ||
        (argc > 1 && !parseArg(argv[1], 1, 1048576, &mib)) ||
        (argc > 2 && !parseArg(argv[2], 1, 100000, &repsArg))) {
        fprintf(stderr, "用法: %s [单数组MiB 1..1048576，默认1024] [计时次数 1..100000，默认10]\n",
                argv[0]);
        return 2;
    }
    const int    reps    = (int)repsArg;
    const int    block   = 256;
    const int    warm    = 3;
    const size_t bytes   = (size_t)mib * 1024 * 1024;
    const size_t nFloats = bytes / sizeof(float);
    const double totalBytes = 2.0 * (double)bytes;          // 读 + 写

    CUDA_CHECK(cudaSetDevice(0));
    const Env env = printEnv(0);

    printf("\n配置\n");
    KV("单数组",       "%llu MiB", mib);
    KV("读写字节数",   "%.0f MiB（读 + 写）", totalBytes / 1048576.0);
    KV("数组 / L2",    "%.1fx", (double)bytes / (double)env.l2Bytes);
    KV("block",        "%d", block);
    KV("每线程字节数", "4 / 8 / 16（float / float2 / float4）");
    KV("warm up",      "%d 次（不计时）", warm);
    KV("计时次数",     "%d", reps);

    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    fillRamp<<<(nFloats + block - 1) / block, block>>>(d_a, nFloats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Result> results;
    auto record = [&](const char* name, int bpt, size_t grid, float ms) {
        results.push_back({name, bpt, grid, ms, totalBytes / (ms * 1e-3) / 1e9,
                           verifySample(d_a, d_b, nFloats)});
    };

    {   // float，每线程 4 B
        const size_t n = nFloats, g = (n + block - 1) / block;
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));
        record("float", 4, g,
               timeMedianMs([=] { copyScalar<<<g, block>>>(d_a, d_b, n); }, reps, warm));
    }
    {   // float2，每线程 8 B
        const size_t n = nFloats / 2, g = (n + block - 1) / block;
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));
        record("float2", 8, g,
               timeMedianMs([=] { copyVec2<<<g, block>>>((const float2*)d_a, (float2*)d_b, n); },
                            reps, warm));
    }
    {   // float4，每线程 16 B
        const size_t n = nFloats / 4, g = (n + block - 1) / block;
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));
        record("float4", 16, g,
               timeMedianMs([=] { copyVec4<<<g, block>>>((const float4*)d_a, (float4*)d_b, n); },
                            reps, warm));
    }

    enum { W_NAME = 10, W_BPT = 13, W_GRID = 11, W_MS = 11, W_BW = 12, W_PCT = 13, W_REL = 12, W_OK = 8 };
    char buf[64];
    putchar('\n');
    padL("版本", W_NAME);        padR("每线程字节", W_BPT);  padR("grid", W_GRID);
    padR("耗时 ms", W_MS);       padR("带宽 GB/s", W_BW);    padR("占理论带宽", W_PCT);
    padR("相对 float", W_REL);   padR("校验", W_OK);
    putchar('\n');

    size_t totalBad = 0;
    for (const auto& r : results) {
        padL(r.name, W_NAME);
        snprintf(buf, sizeof buf, "%d B", r.bytesPerThread);              padR(buf, W_BPT);
        snprintf(buf, sizeof buf, "%zu", r.grid);                         padR(buf, W_GRID);
        snprintf(buf, sizeof buf, "%.4f", r.ms);                          padR(buf, W_MS);
        snprintf(buf, sizeof buf, "%.1f", r.gbs);                         padR(buf, W_BW);
        snprintf(buf, sizeof buf, "%.1f%%", r.gbs / env.theoGBs * 100.0); padR(buf, W_PCT);
        snprintf(buf, sizeof buf, "%.3fx", r.gbs / results[0].gbs);       padR(buf, W_REL);
        padR(r.bad == 0 ? "PASS" : "FAIL", W_OK);
        putchar('\n');
        totalBad += r.bad;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return totalBad == 0 ? 0 : 1;
}
