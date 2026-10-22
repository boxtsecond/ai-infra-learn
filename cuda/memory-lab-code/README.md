# memory-lab：CUDA 访存实验

一组单变量的 GPU 访存实验。每个 `.cu` 自包含、可单独编译，都在 device 0 上运行，
依次打印「环境」「配置」和结果表。所有规格（显存时钟、位宽、L2 大小、SM 数）都在运行时查询，
换卡不用改代码。

## 环境要求

- NVIDIA GPU。默认按 Hopper（Compute Capability 9.0）编译，其他架构见下面的编译命令。
- CUDA Toolkit 12 或更高，`make`。
- 显存：`l1_stride` 和 `l1_diag` 默认参数下需要约 66 GiB 和 68 GiB，显存不够时调小第一个参数，见下表。
- 测量期间 GPU 上不能有其他进程。

## 编译

```bash
make                         # 编译全部程序到 bin/
make ARCH=sm_80              # 换架构，例如 A100
make NVCC=/path/to/nvcc      # 指定 nvcc
make clean
```

单个程序也可以直接编译：`nvcc -O3 -arch=sm_90 -lineinfo src/l0_copy.cu -o bin/l0_copy`。
`-lineinfo` 让 Nsight Compute 能把指标对应到源码行。

## 程序一览

| 程序 | 测什么 | 自变量 | 默认参数 | 默认显存 |
|---|---|---|---|---|
| `devinfo` | 设备规格 | - | - | - |
| `l0_copy` | 纯拷贝 `b[i] = a[i]` 的有效带宽 | 无 | 单数组 1024 MiB，计时 10 次 | 2 GiB |
| `l0_copy_vec` | 每线程一次访存的字节数 | 4 / 8 / 16 B | 1024 MiB，10 次 | 2 GiB |
| `l1_stride` | 读写两端跨步 | stride 1 2 3 4 8 16 32 | 1024 MiB，10 次 | 66 GiB |
| `l1_diag` | 跨步在哪一端；stride=1 时的索引计算与分配大小 | 见源码头注释 | 1024 MiB，10 次 | 68 GiB |
| `l2_offset` | 读端起始地址偏移 | offset 0..32 | 1024 MiB，10 次，最大 offset 32 | 2 GiB |
| `l3_layout` | AoS 与 SoA | 布局 | 1024 MiB，10 次 | 5 GiB |
| `l4_transpose` | 矩阵转置三个版本与两个拷贝基线 | 版本 | 边长 16384，10 次 | 2 GiB |
| `l5_bank` | shared memory bank conflict | stride 1 2 4 8 16 32 | 10 次，循环 200000 次 | < 1 MiB |

参数格式见各源码文件头的「运行」一行，不带参数即用默认值。参数非法时程序打印用法并以退出码 2 结束。

## 复现步骤

```bash
# 1. 核对规格
CUDA_VISIBLE_DEVICES=0 ./bin/devinfo

# 2. 确认 GPU 0 上没有其他进程（每次测量前后都要做）
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 3. L0：同一命令用独立进程跑 3 次
for i in 1 2 3; do CUDA_VISIBLE_DEVICES=0 ./bin/l0_copy; done

# 4. 其余实验
CUDA_VISIBLE_DEVICES=0 ./bin/l0_copy_vec
CUDA_VISIBLE_DEVICES=0 ./bin/l1_stride
CUDA_VISIBLE_DEVICES=0 ./bin/l1_diag
CUDA_VISIBLE_DEVICES=0 ./bin/l2_offset
CUDA_VISIBLE_DEVICES=0 ./bin/l3_layout
CUDA_VISIBLE_DEVICES=0 ./bin/l4_transpose
CUDA_VISIBLE_DEVICES=0 ./bin/l5_bank
```

L0 的三次运行各取带宽中位数 Bᵢ，三个 Bᵢ 的中位数记为 B₀；跨进程波动为
`(max(Bᵢ) - min(Bᵢ)) / median(Bᵢ)`。它是判断后续差异的初始门槛：小于它的差异不作结论。
它只代表 L0 这个 kernel 的噪声，其他实验测得更大波动时用更大的门槛。

## 怎么读输出

- **有效带宽**（effective bandwidth）= (读字节数 + 写字节数) / 耗时（CUDA C++ Best Practices Guide 9.2.2），
  只计算法本身需要的字节，不计跨步或非对齐带来的额外传输。「配置」里的「读写字节数」就是公式里的 Br + Bw。
- **理论带宽**（theoretical bandwidth）= 显存时钟 × 2 × 位宽 / 8（同上 9.2.1），由运行时查询的时钟和位宽算出，
  「占理论带宽」以它为分母。它与产品标称带宽可能略有差别。
- **占理论带宽**和 Nsight Compute 的 DRAM Throughput 不是同一个量：后者按 DRAM 实际做的工作计算，
  算法用不到的字节也算在内。纯拷贝时两者接近，跨步访问时相差很大。
- **单位**：数组大小和读写字节数用 MiB（2²⁰ 字节），带宽用 GB/s（10⁹ 字节/秒），后者与 Best Practices Guide 9.2 的公式一致。
- **耗时**：每组先 warm up 3 次不计时，再计时 N 次取中位数，带宽由中位数耗时换算。
  `l0_copy` 另外逐次打印每一次的耗时和带宽，并给出进程内波动。
- **校验**：`l0_copy` 全量比对；其余程序抽样比对头、中、尾各 1024 个元素，转置抽样 4096 个位置。
  任何一行校验失败时，程序以退出码 1 结束。

### 比较分母

「耗时为对照的 N 倍」用 `T实验 / T本级对照`，「加速 N 倍」用反方向的耗时比。
本级对照与实验固定相同的数据规模、启动形状、每线程工作量和计时方式，只保留实验要改的变量。

| 实验 | 本级对照 |
|---|---|
| `l1_stride` | 本程序的 stride=1 |
| `l1_diag` | 对照 A 中 RW、R、W 各自的 stride=1；对照 B 中的 plain |
| `l2_offset` | 本程序的 offset=0 |
| `l3_layout` | 本程序的 SoA |
| `l4_transpose` | 相对本级对照的带宽比以基线 B（同形拷贝）为分母；v1→v2、v2→v3 的加速用对应耗时之比 |
| `l5_bank` | 冲突路数由访问模式和 Nsight Compute 计数器确定，耗时比只作辅助 |

L0 的 B₀ 是带宽参照和噪声门槛，不充当各级「慢 N 倍」的分母。
`l3_layout` 的 AoS 只有读端跨步，对应 `l1_diag` 对照 A 的 R 列，而不是 `l1_stride` 的两端跨步。
`l4_transpose` 每线程处理 4 个元素、grid 与 L0 不同，所以用同形拷贝基线 B 做比较。

## 用 Nsight Compute 采集

Nsight Compute 默认把 GPU 时钟锁到 base clock，下面的命令都加了 `--clock-control none`，
既不影响同一张卡上的其他用户，也让测量条件与正常运行一致。
每个被测 kernel 都是 3 次 warm up 加 N 次计时，所以「第 4 次 launch」就是第一次计时。

```bash
# 单个 kernel：跳过 3 次 warm up，只采第 4 次
ncu --clock-control none --set full --import-source yes \
    -k copyKernel --launch-skip 3 -c 1 -o L0_copy_full ./bin/l0_copy 512 1

# 同名 kernel 依次运行多次（l1_stride、l2_offset）：第 k 个变体（从 0 数）的计时 launch 是第 4k+4 次
ncu --clock-control none --set full --import-source yes \
    -k copyStride --launch-skip $((4*k+3)) -c 1 -o L1_stride_k${k} ./bin/l1_stride 256 1

# 多个不同的 kernel：各采自己的第 4 次
ncu --clock-control none --set full --import-source yes \
    --kernel-id '::regex:^(copyBaseline|copyTiled|transposeNaive|transposeShared|transposeSharedPad)$:4' \
    -o L4_transpose_full ./bin/l4_transpose 8192 1

# bank conflict 计数：模板实例要按 demangled 名字匹配
ncu --clock-control none --kernel-name-base demangled --kernel-id '::regex:bankProbe:4' \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./bin/l5_bank 1 20000
```

ncu 采集用的数据规模比带宽测量小（例如 512 MiB、256 MiB），以减少 kernel replay 的开销。
sector 数和字节数随规模线性变化，比例关系不变。

只采少量计数器、又要和带宽测量用同样的数据规模时，改用 application replay。
默认的 kernel replay 在第一个 pass 前要保存 kernel 能访问的全部显存，`l1_stride` 默认参数下分配了约 66 GiB，会非常慢；
application replay 改为把整个程序重跑几遍，不做保存和恢复（Profiling Guide 2.2.3 Replay）。

```bash
# 1024 MiB 下每个 stride 的 L1 请求、L1 sector、L1→L2 读请求和 DRAM read sector
ncu --clock-control none --replay-mode application \
    --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_requests_srcunit_tex_op_read.sum,dram__sectors_read.sum \
    -k copyStride -o L1_stride_counters ./bin/l1_stride 1024 1
```

每个变体 4 次 launch（3 次 warm up 加 1 次计时），取第 4、8、12……次。`l2_offset` 换成 `-k copyOffset ./bin/l2_offset 1024 1 32` 即可。

## 参考结果与实测环境

`../memory-lab-reports/` 里有一次完整运行的原始输出（`raw/*.stdout.txt`）和 Nsight Compute 报告，
`INDEX.md` 是索引。那次运行的环境：

- 单张 GPU：CC 9.0，132 SM，CUDA 可见显存 143072 MB，显存 3201 MHz × 6144 bit，
  理论带宽 4916.7 GB/s，L2 60 MiB，ECC on。驱动报出的产品名是 NVIDIA H800，规格与 H200 一致。
- 驱动 550.127.08，CUDA 12.4.131，Nsight Compute 2024.1.1。

那些输出由调整输出格式之前的源码生成：被测 kernel 与现在相同，排版和摘要行不同。
各程序的参数以对应 stdout 的「配置」块为准，部分运行用的计时次数不是现在的默认值。
其中 `l0_copy_vec` 当时的单数组是 2048 MiB。

2026-10-21 用当前源码在同一台机器的 GPU 0 上补测了 `l0_copy` 和 `l0_copy_vec`（默认参数，单数组 1024 MiB，各三个独立进程），
原始输出在 `../memory-lab-reports/raw/2026-10-21-1024MiB/`：

- `l0_copy` 三次的带宽中位数为 2666.7、2667.0、2667.3 GB/s，跨进程波动约 0.02%。
- `l0_copy_vec` 的 float、float2、float4 分别为理论带宽的 54.2%、77.7%、86.3%，三个进程之间相差不到 0.2%。

L0 另有一次 1024 MiB、三个独立进程的补测：B₀ = 2666.7 GB/s，跨进程波动约 0.09%，
三次均全量校验通过，记录见 [L0-baseline-1024MiB-2026-10-15](../artifacts/L0-baseline-1024MiB-2026-10-15.md)。

`l5_bank` 用上面的 ncu 命令（循环 20000 次）测得的冲突计数为 0、20000、60000、140000、300000、620000，
满足「冲突计数 = 循环次数 × (冲突路数 − 1)」，与程序按地址算出的路数一致。
