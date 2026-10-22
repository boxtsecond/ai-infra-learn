// l5_bank.cu —— shared memory bank conflict：访问跨步与冲突路数
//
// shared memory 分 32 个 bank，每个 bank 宽 4 字节，bank 编号 = 4 字节字的下标 % 32。
// 一个 warp 的 32 个 lane 读 vs[(lane * STRIDE) & 1023]，STRIDE 取 1 2 4 8 16 32：
// STRIDE=1 时 32 个 lane 落在 32 个不同的 bank；STRIDE=32 时全部落在 bank 0。
// 每个 STRIDE 编译成一个独立的模板实例 bankProbe<STRIDE>，Nsight Compute 里可按名字区分。
// 「冲突路数」一列由主机端按上面的映射计算：同一 bank 里不同下标的最大个数。
//
// 冲突次数用 Nsight Compute 读计数器，只采每个 STRIDE 的第 4 次 launch（跳过 3 次 warm up）：
//   ncu --clock-control none --kernel-name-base demangled --kernel-id '::regex:bankProbe:4' \
//       --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
//       ./l5_bank 1 20000
// 耗时除了冲突造成的串行访问，还包含循环本身的固定开销，耗时比不等于路数比。
//
// 编译：nvcc -O3 -arch=sm_90 -lineinfo l5_bank.cu -o l5_bank
// 运行：CUDA_VISIBLE_DEVICES=0 ./l5_bank [计时次数，默认10] [每次 launch 的循环次数，默认200000]

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

// 按 bank = 下标 % 32 计算一个 warp 访问的冲突路数。
// 多个 lane 读同一个字会广播，不算冲突，所以只数同一 bank 里不同的下标。
static int conflictWays(int stride) {
    int words[32][32];
    int count[32] = {0};
    int ways = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const int idx = (lane * stride) & 1023, bank = idx % 32;
        bool seen = false;
        for (int k = 0; k < count[bank]; ++k) seen = seen || words[bank][k] == idx;
        if (!seen) words[bank][count[bank]++] = idx;
        ways = std::max(ways, count[bank]);
    }
    return ways;
}

// ---------- kernel ----------
// 下标 idx 在循环内不变。vs 声明为 volatile，每次循环都从 shared memory 重新读，
// 否则编译器会把这次读提到循环外，各个 STRIDE 的耗时就没有区别了。
template <int STRIDE>
__global__ void bankProbe(float* out, int iters) {
    __shared__ float s[1024];
    const int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s[i] = (float)i;
    __syncwarp();

    volatile float* vs = s;
    const int idx = (lane * STRIDE) & 1023;
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) acc += vs[idx];
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    unsigned long long repsArg = 10, itersArg = 200000;
    if (argc > 3 ||
        (argc > 1 && !parseArg(argv[1], 1, 100000, &repsArg)) ||
        (argc > 2 && !parseArg(argv[2], 1, 100000000, &itersArg))) {
        fprintf(stderr, "用法: %s [计时次数 1..100000，默认10] [循环次数 1..100000000，默认200000]\n",
                argv[0]);
        return 2;
    }
    const int reps  = (int)repsArg;
    const int iters = (int)itersArg;
    const int warm  = 3;
    const int grid  = 1;      // 单 block、单 warp，只看一个 warp 内部的 bank 行为
    const int block = 32;

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0;
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));

    printf("环境\n");
    KV("GPU",          "%s", prop.name);
    KV("Compute Cap",  "%d.%d", prop.major, prop.minor);
    KV("shared/block", "%zu KiB", prop.sharedMemPerBlock / 1024);
    KV("CUDA drv/rt",  "%d.%d / %d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    printf("\n配置\n");
    KV("shared 数组",  "float s[1024]");
    KV("访问",         "vs[(lane * STRIDE) & 1023]，bank = 下标 %% 32");
    KV("block",        "%d（单 warp）", block);
    KV("grid",         "%d", grid);
    KV("循环次数",     "%d / launch", iters);
    KV("warm up",      "%d 次（不计时）", warm);
    KV("计时次数",     "%d", reps);

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, grid * sizeof(float)));

    enum { W_ST = 8, W_KN = 18, W_WAY = 10, W_MS = 12, W_REL = 11 };
    char buf[64];
    putchar('\n');
    padR("stride", W_ST);   padL("  kernel", W_KN);   padR("冲突路数", W_WAY);
    padR("耗时 ms", W_MS);  padR("相对 s=1", W_REL);
    putchar('\n');

    float ms1 = 0.0f;
    #define PROBE(S)                                                              \
        {                                                                         \
            const float ms = timeMedianMs([=] {                                   \
                bankProbe<S><<<grid, block>>>(d_out, iters); }, reps, warm);      \
            if (S == 1) ms1 = ms;                                                 \
            snprintf(buf, sizeof buf, "%d", S);              padR(buf, W_ST);     \
            snprintf(buf, sizeof buf, "  bankProbe<%d>", S); padL(buf, W_KN);     \
            snprintf(buf, sizeof buf, "%d", conflictWays(S)); padR(buf, W_WAY);   \
            snprintf(buf, sizeof buf, "%.4f", ms);           padR(buf, W_MS);     \
            snprintf(buf, sizeof buf, "%.3fx", ms / ms1);    padR(buf, W_REL);    \
            putchar('\n');                                                        \
        }
    PROBE(1) PROBE(2) PROBE(4) PROBE(8) PROBE(16) PROBE(32)
    #undef PROBE

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
