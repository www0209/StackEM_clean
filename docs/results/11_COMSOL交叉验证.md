# COMSOL 交叉验证

本章说明 `reference_results/comsol/`。核对清单是 `docs/checks/results_11.jsonl`。这一步是可选的，需要一份 COMSOL 许可证；没有的话主流程的其他结果都不受影响。

## 1. 这个实验回答什么问题

项目里所有物理结论都以自己写的有限体积求解器为真值。自己写的求解器给自己写的代理解当裁判，说服力有限。本实验用商业有限元软件 COMSOL Multiphysics 回答两个问题。

1. 一个完全独立的求解器在同样的轨上算出的成核时间，与参考求解器差多少。
2. 用通用有限元软件逐条轨求解要多久。

术语：

- 有限元：把求解区域划分成小单元、在每个单元上用简单函数近似解的数值方法。COMSOL 是这类软件。
- LiveLink for MATLAB：COMSOL 的一个接口，可以在 MATLAB 里用脚本创建模型、求解并取回结果。
- 系数形式偏微分方程：COMSOL 的通用数学接口，用户直接填方程各项的系数。

## 2. 原理与做法

三个脚本在 `comsol/` 目录，逐步操作说明在 [comsol/COMSOL_MODEL_BUILD.md](../../comsol/COMSOL_MODEL_BUILD.md)。

### 2.1 被求解的方程

每条轨是一维区间，按段切成若干子区间。未知量是静水应力，方程是 Korhonen 方程，各项的定义与参考求解器相同：

- 段内温度是两端线性插值加焦耳鼓包。
- 扩散系数 $\kappa = D_0 \exp(-E_a/k_BT)\, B\Omega/(k_BT)$。
- 电迁移驱动力 $G = eZ\rho j/\Omega$，热迁移驱动力 $M = Q/(\Omega T)\, dT/dx$。
- 初始条件是每段的残余应力，两端零通量，段与段之间应力和通量连续。

常数写在 `constants.json`：$D_0$ 7.5e-8 m²/s，$E_a$ 0.86 eV，$B$ 100 GPa，$\Omega$ 1.181e-29 m³，电阻率 3e-8 Ω m，有效电荷数 10，$Q$ 0.09 eV，临界应力 500 MPa。

### 2.2 三步流程

1. `comsol/export_rails.py` 读取 base3 的 E2 输出，把每条轨每一段的长度、电流密度、两端温度、焦耳温升幅值、热特征长度、残余应力写成 `rails.csv`，把参考解和 StackEM 的成核时间写成 `rails_tnuc.csv`。轨按参考解的成核时间从早到晚排序。
2. `comsol/livelink_rails.m` 在 Windows 的 MATLAB 里运行，对每条轨从零开始用 API 建一个模型：
   - 几何：一个多区间的一维线段，端点是各段的累积长度。
   - 物理：系数形式偏微分方程，扩散系数填 $\kappa$，驱动力放在守恒通量源项里，初值是残余应力，两端默认零通量。
   - 网格：最大单元尺寸取最短段长的六十分之一，每段至少 60 个单元。
   - 求解：瞬态，输出时刻是 $10^4$ 到 $10^{9.5}$ 秒之间对数均布的 56 个点，相对容差 1e-6。
   - 后处理：取每个输出时刻全域的最大应力，找到第一次超过临界应力的区间，在 $\sqrt t$ 上线性插值得到成核时间。这与参考求解器的规则相同。
   - 计时：`seconds_build` 是建模，`seconds_solve` 是求解本身，`seconds_total` 还包含取结果和插值。
3. `comsol/compare_rails.py` 只在三种方法都成核的轨上算相对误差，并把每轨平均秒数乘以轨数写成 `comsol_timing.csv`。

## 3. 怎么运行

1. 在 Linux 里导出。前提是 base3 的 E2 已在本机跑完。

```bash
STACKEM_ONLY=comsol.export bash scripts/run_all.sh
# 等价于
python comsol/export_rails.py --e2 ~/stackem_work/outputs/base3/e2 --out ~/stackem_work/outputs/comsol
```

2. 把 `~/stackem_work/outputs/comsol` 整个目录和 `comsol/livelink_rails.m` 拷到 Windows，例如 `C:\stackem_comsol\`。
3. 启动 COMSOL Multiphysics with MATLAB，在 MATLAB 命令窗口里输入：

```matlab
setenv('STACKEM_CR', 'C:\stackem_comsol\comsol');
setenv('STACKEM_CR_MAX', '20');      % 可选：先试 20 条
cd C:\stackem_comsol
livelink_rails
```

   每条轨打印一行成核时间和耗时。试跑没问题后把 `STACKEM_CR_MAX` 设为空字符串再跑全部。结束时写出 `comsol_rails_result.csv`。

4. 把 `comsol_rails_result.csv` 拷回 `~/stackem_work/outputs/comsol/`，再运行：

```bash
STACKEM_ONLY=comsol.compare bash scripts/run_all.sh
# 等价于
python comsol/compare_rails.py --dir ~/stackem_work/outputs/comsol
```

   结果文件不存在时这一步只打印一行提示，不算失败。

参考结果用的是 COMSOL 6.2 和 MATLAB R2024a。COMSOL 一侧没有保存日志。耗时只能从结果文件读出：246 条轨的 `seconds_total` 之和是 333 秒，其中建模 74 秒，求解 250 秒。导出和比较两步各只要几秒。

## 4. 输出文件

`reference_results/comsol/` 共 6 个文件。

| 文件 | 内容 |
|---|---|
| `rails.csv` | 每段一行：`rail` 轨号，`k` 段号，`L_m` 长度，`j_A_per_m2` 电流密度，`T_L_K`、`T_R_K` 两端温度，`T_m_K` 焦耳温升幅值，`Gamma_m` 热特征长度，`sigma_T_Pa` 残余应力。共 9840 行，即 246 条轨每条 40 段 |
| `rails_tnuc.csv` | 每条轨一行：轨号、芯片、方向、序号、段数、参考解成核时间 `t_nuc_fdm_s`、StackEM 成核时间 `t_nuc_stackem_s`。不成核写作 `inf` |
| `constants.json` | 电迁移常数和一句方程说明 |
| `comsol_rails_result.csv` | COMSOL 的结果，每条轨一行：轨号、段数、`t_nuc_comsol_s`、`seconds_build`、`seconds_solve`、`seconds_total` |
| `comsol_rails_compare.json` | 三组误差统计，成核判断一致率，每轨秒数，计时轨数，并行进程数的假设 |
| `comsol_timing.csv` | 外推的计时表：轨数、单进程秒数、32 进程秒数 |

自己运行比较脚本时还会画一张 StackEM 对 COMSOL 的对角图 `comsol_rails_parity`，参考结果里没有收录这张图，所以本章没有图。

## 5. 结果

### 5.1 三方误差

`comsol_rails_compare.json`。误差是成核时间的相对差，只统计三种方法都成核的轨。

| 比较 | 轨数 | 中位数 % | p90 % | 最大 % |
|---|---|---|---|---|
| COMSOL 对 参考求解器 | 128 | 0.084 | 0.275 | 0.525 |
| StackEM 对 COMSOL | 128 | 0.185 | 0.848 | 1.887 |
| StackEM 对 参考求解器 | 128 | 0.153 | 0.903 | 2.344 |

COMSOL 与参考求解器对“是否成核”的判断一致率是 1.00，即 246 条轨全部一致：128 条成核，118 条不成核。

最早失效的那条轨，D2 的 V14，三种方法的成核时间分别是：参考求解器 2.389 年，COMSOL 2.394 年，StackEM 2.405 年。

### 5.2 COMSOL 的耗时

| 量 | 值 |
|---|---|
| 计时的轨数 | 246 |
| 每条轨的平均秒数 | 1.354 |
| 每条轨的中位秒数 | 1.353 |
| 每段的平均秒数 | 0.0339 |
| 单条轨最快 | 1.139 秒 |
| 单条轨最慢 | 1.632 秒 |
| 建模占总时间的比例 | 22.3 % |
| 求解占总时间的比例 | 75.0 % |

### 5.3 外推的计时表

`comsol_timing.csv` 的全部内容。第二列等于轨数乘以每轨平均秒数，第三列是第二列除以 32。

| 轨数 | 单进程 秒 | 32 进程 秒 | 轨数乘平均每轨秒数，重算 |
|---|---|---|---|
| 82 | 111.1 | 3.5 | 111.1 |
| 246 | 333.2 | 10.4 | 333.2 |
| 1000 | 1354.3 | 42.3 | 1354.3 |
| 5000 | 6771.7 | 211.6 | 6771.7 |
| 20000 | 27086.7 | 846.5 | 27086.7 |

两万条轨一行换算成更直观的单位：单进程 7.52 小时，32 进程 14.1 分钟。与 E4 里正式引擎两万条轨的 23.4 秒相比，是 1155 倍和 36.1 倍。

## 6. 怎么理解

1. 参考求解器得到了独立软件的确认。两个求解器的离散方法、时间积分、实现都不同，在 128 条会成核的轨上成核时间的中位差是 0.08 %，最大 0.53 %，对哪些轨不成核的判断完全一致。项目里把参考求解器当真值是站得住的。
2. StackEM 相对 COMSOL 的误差与相对参考求解器的误差是同一量级，中位数都在 0.15 % 到 0.2 %。代理解的误差远大于两个求解器之间的差，所以用哪个当真值不影响精度结论。
3. 通用有限元软件逐条轨求解每条要一秒多，其中约两成花在建模上。这个开销来自通用软件的建模和求解框架，不来自问题本身的难度：同样的 246 条轨，自己写的有限体积求解器单进程只要约 6 秒。

## 7. 论文用了哪些，哪些是论文之外的

| 论文中的说法 | 本章对应 |
|---|---|
| 在 base3 的全部 246 条轨上与 COMSOL 6.2 对比，每段 60 个单元 | 2.2 节 |
| 两个求解器对每条不朽轨的判断一致 | 5.1 节 |
| 成核时间中位相差 0.08 %，最大 0.53 % | 5.1 节第一行 |
| COMSOL 在同样的轨上每条 1.35 秒 | 5.2 节 |
| 两万条轨单进程 7.5 小时，32 个独立进程 14 分钟 | 5.3 节末行 |
| StackEM 快 1160 倍和 36 倍 | 5.3 节。按文件算是 1155 倍和 36.1 倍 |

论文之外的内容：StackEM 对 COMSOL 的误差，p90，每段秒数，建模与求解的分项时间。

## 8. 注意事项

1. 计时表是外推。实测的只有 base3 的 246 条轨，每条 40 段。两万条轨的 7.5 小时是 1.35 秒乘以 20000，没有实际跑过。
2. 32 进程一列是除以 32 的理想值。它假设 32 个 COMSOL 进程互不影响、各自需要的许可证和内存都够，没有实测。参考机器是 16 核 32 线程。
3. 1160 倍这个数按文件重算是 1155 倍。
4. COMSOL 的计时含每条轨重新建模的时间。熟练用户可以复用模型、只换参数，建模那两成多的时间可以省掉大部分。即使全部省掉，结论的量级不变。
5. `constants.json` 里 `note` 字段把端点通量写成 `kappa (sigma_x + G + M) = 0`，符号与实际求解的方程 $\kappa(\sigma_x - G - M)$ 相反。这是参考结果文件里一句说明文字的笔误，不影响任何数值。现在的 `comsol/export_rails.py` 写出的说明已经改正，所以重新导出的 `constants.json` 与参考文件在这一句上不同。
6. 实际求解时 COMSOL 的因变量名是默认的 `u`，守恒通量源项的符号为正，见 `comsol/COMSOL_MODEL_BUILD.md`。
7. COMSOL 一侧没有保存运行日志，MATLAB 窗口的输出也没有保存。耗时只来自 `comsol_rails_result.csv`。
8. 比较只做了成核时间。整个应力场 $\sigma(x,t)$ 没有与 COMSOL 逐点比较。
9. 只在 base3 的 `full` 变体上做了这项核对。
