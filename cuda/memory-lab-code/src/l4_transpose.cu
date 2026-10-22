// l4_transpose.cu —— 矩阵转置三个版本，以及两个拷贝基线
//
// 转置 out[x][y] = in[y][x]，读和写不可能同时按行连续：
//   v1 朴素        直接按转置索引读写。读按行连续，写的地址间距为矩阵边长
//   v2 shared      先把 32x32 的 tile 读进 shared memory，再按转置后的行写出，
//                  global 读写都连续；从 shared 按列读时有 bank conflict
//   v3 shared+pad  tile 每行补 1 个 float（32x33），错开 bank 映射
// 两个拷贝基线，用来衡量转置离纯拷贝还差多少：
//   基线 A  每线程拷 1 个元素（与 l0_copy 相同）
//   基线 B  block 形状、grid 和每线程元素数都与三个转置版本相同，只是索引不转置
// 三个转置版本的 TILE、BLOCK_ROWS、grid 都相同。
// 校验：拷贝基线抽样比对头、中、尾各 1024 个元素；转置抽样比对 4096 个位置。
//
// 编译：nvcc -O3 -arch=sm_90 -lineinfo l4_transpose.cu -o l4_transpose
// 运行：CUDA_VISIBLE_DEVICES=0 ./l4_transpose [矩阵边长，32 的倍数，默认16384] [计时次数，默认10]

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

// ---------- 校验 ----------
// 拷贝：抽样比对头、中、尾各 1024 个元素，返回不一致个数
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
// 转置：抽样 4096 个位置，检查 out[x * h + y] == in[y * w + x]
static size_t verifyTranspose(const float* d_in, const float* d_out, int w, int h) {
    size_t bad = 0;
    for (int k = 0; k < 4096; ++k) {
        const int x = (int)(((long long)k * 7919) % w);
        const int y = (int)(((long long)k * 6271) % h);
        float a = 0.0f, b = 0.0f;
        CUDA_CHECK(cudaMemcpy(&a, d_in  + (size_t)y * w + x, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&b, d_out + (size_t)x * h + y, sizeof(float), cudaMemcpyDeviceToHost));
        if (a != b) ++bad;
    }
    return bad;
}

// ---------- kernel ----------
// tile 32x32；每个 block 32x8 个线程，每个线程处理 tile 中的 4 行
#define TILE       32
#define BLOCK_ROWS 8

__global__ void fillIndex(float* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (float)(i % 65536);
}

// v1 朴素：读 in 按行连续，写 out 的地址间距为矩阵边长
__global__ void transposeNaive(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (x < w && (y + j) < h)
            out[(size_t)x * h + (y + j)] = in[(size_t)(y + j) * w + x];
}

// v2 shared tile：global 读写都按行连续；从 shared 按列读时 32 个线程落在同一个 bank
__global__ void transposeShared(const float* in, float* out, int w, int h) {
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (x < w && (y + j) < h)
            tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * w + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (x < h && (y + j) < w)
            out[(size_t)(y + j) * h + x] = tile[threadIdx.x][threadIdx.y + j];
}

// v3 shared + padding：每行 33 个 float，同一列的元素分散到 32 个不同的 bank
__global__ void transposeSharedPad(const float* in, float* out, int w, int h) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (x < w && (y + j) < h)
            tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * w + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (x < h && (y + j) < w)
            out[(size_t)(y + j) * h + x] = tile[threadIdx.x][threadIdx.y + j];
}

// 基线 A：每线程拷 1 个元素
__global__ void copyBaseline(const float* in, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// 基线 B：与三个转置版本同样的 block 形状、grid 和每线程元素数，只是索引不转置
__global__ void copyTiled(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (x < w && (y + j) < h) {
            const size_t idx = (size_t)(y + j) * w + x;
            out[idx] = in[idx];
        }
}

int main(int argc, char** argv) {
    unsigned long long sideArg = 16384, repsArg = 10;
    if (argc > 3 ||
        (argc > 1 && (!parseArg(argv[1], TILE, 65536, &sideArg) || sideArg % TILE != 0)) ||
        (argc > 2 && !parseArg(argv[2], 1, 100000, &repsArg))) {
        fprintf(stderr, "用法: %s [矩阵边长，%d 的倍数且 <= 65536，默认16384] [计时次数 1..100000，默认10]\n",
                argv[0], TILE);
        return 2;
    }
    const int side = (int)sideArg;
    const int reps = (int)repsArg;
    const int warm = 3;

    const size_t n           = (size_t)side * side;
    const size_t bytes       = n * sizeof(float);
    const double usefulBytes = 2.0 * (double)bytes;          // 读 + 写
    const dim3   gridT(side / TILE, side / TILE);
    const dim3   blockT(TILE, BLOCK_ROWS);
    const int    blockC = 256;
    const size_t gridC  = (n + blockC - 1) / blockC;

    CUDA_CHECK(cudaSetDevice(0));
    const Env env = printEnv(0);

    printf("\n配置\n");
    KV("矩阵",         "%d x %d float", side, side);
    KV("单矩阵",       "%.0f MiB", (double)bytes / 1048576.0);
    KV("读写字节数",   "%.0f MiB（读 + 写）", usefulBytes / 1048576.0);
    KV("tile",         "%d x %d", TILE, TILE);
    KV("block",        "%d x %d", TILE, BLOCK_ROWS);
    KV("grid",         "%d x %d", gridT.x, gridT.y);
    KV("元素/线程",    "%d", TILE / BLOCK_ROWS);
    KV("warm up",      "%d 次（不计时）", warm);
    KV("计时次数",     "%d", reps);
    KV("单矩阵 / L2",  "%.1fx", (double)bytes / (double)env.l2Bytes);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    fillIndex<<<gridC, blockC>>>(d_in, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    struct Ver { const char* name; int ept; int smemBytes; float ms; size_t bad; };
    Ver vers[5];

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    vers[0] = {"基线A 朴素拷贝", 1, 0,
               timeMedianMs([=] { copyBaseline<<<gridC, blockC>>>(d_in, d_out, n); }, reps, warm), 0};
    vers[0].bad = verifySample(d_in, d_out, n);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    vers[1] = {"基线B 同形拷贝", TILE / BLOCK_ROWS, 0,
               timeMedianMs([=] { copyTiled<<<gridT, blockT>>>(d_in, d_out, side, side); }, reps, warm), 0};
    vers[1].bad = verifySample(d_in, d_out, n);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    vers[2] = {"v1 朴素", TILE / BLOCK_ROWS, 0,
               timeMedianMs([=] { transposeNaive<<<gridT, blockT>>>(d_in, d_out, side, side); }, reps, warm), 0};
    vers[2].bad = verifyTranspose(d_in, d_out, side, side);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    vers[3] = {"v2 shared", TILE / BLOCK_ROWS, (int)(TILE * TILE * sizeof(float)),
               timeMedianMs([=] { transposeShared<<<gridT, blockT>>>(d_in, d_out, side, side); }, reps, warm), 0};
    vers[3].bad = verifyTranspose(d_in, d_out, side, side);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    vers[4] = {"v3 shared+pad", TILE / BLOCK_ROWS, (int)(TILE * (TILE + 1) * sizeof(float)),
               timeMedianMs([=] { transposeSharedPad<<<gridT, blockT>>>(d_in, d_out, side, side); }, reps, warm), 0};
    vers[4].bad = verifyTranspose(d_in, d_out, side, side);

    auto bwOf = [&](float ms) { return usefulBytes / (ms * 1e-3) / 1e9; };

    enum { W_VER = 18, W_EPT = 13, W_SM = 15, W_MS = 11, W_BW = 13, W_REL = 12, W_V1 = 11, W_OK = 8 };
    char buf[64];
    putchar('\n');
    padL("版本", W_VER);     padR("元素/线程", W_EPT);   padR("shared/block", W_SM);
    padR("耗时 ms", W_MS);   padR("有效带宽", W_BW);     padR("相对基线B", W_REL);
    padR("相对 v1", W_V1);   padR("校验", W_OK);
    putchar('\n');

    size_t totalBad = 0;
    for (int i = 0; i < 5; ++i) {
        const double bw = bwOf(vers[i].ms);
        padL(vers[i].name, W_VER);
        snprintf(buf, sizeof buf, "%d", vers[i].ept);                         padR(buf, W_EPT);
        snprintf(buf, sizeof buf, "%d B", vers[i].smemBytes);                 padR(buf, W_SM);
        snprintf(buf, sizeof buf, "%.4f", vers[i].ms);                        padR(buf, W_MS);
        snprintf(buf, sizeof buf, "%.1f GB/s", bw);                           padR(buf, W_BW);
        snprintf(buf, sizeof buf, "%.3fx", bw / bwOf(vers[1].ms));            padR(buf, W_REL);
        if (i < 2) padR("-", W_V1);
        else { snprintf(buf, sizeof buf, "%.3fx", bw / bwOf(vers[2].ms));     padR(buf, W_V1); }
        padR(vers[i].bad == 0 ? "PASS" : "FAIL", W_OK);
        putchar('\n');
        totalBad += vers[i].bad;
    }

    printf("\n摘要\n");
    KV("v1 -> v2",     "加速 %.3fx（耗时 v1 / v2）", vers[2].ms / vers[3].ms);
    KV("v2 -> v3",     "加速 %.3fx（耗时 v2 / v3）", vers[3].ms / vers[4].ms);
    KV("v3 / 基线B",   "带宽 %.1f%%", bwOf(vers[4].ms) / bwOf(vers[1].ms) * 100.0);
    KV("基线B / 基线A", "带宽 %.3fx", bwOf(vers[1].ms) / bwOf(vers[0].ms));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return totalBad == 0 ? 0 : 1;
}
