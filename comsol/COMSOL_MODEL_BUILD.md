# COMSOL 交叉验证：逐轨求解与比较

这一步用商业有限元软件 COMSOL 独立求解 base3 堆叠的全部 246 条供电轨，目的有两个：用第三方求解器检验本仓库的参考求解器，以及测出 COMSOL 的单轨耗时供速度图使用。主流程不依赖 COMSOL，没有许可证时跳过这一步即可，E0 里的解析级数检查仍然有效。

本文件夹只保留实际产生了发布结果的那一套脚本。

| 文件 | 在哪里运行 | 作用 |
|---|---|---|
| `export_rails.py` | Linux，StackEM 环境 | 把 E2 求解过的每条轨导出成 COMSOL 能读的表 |
| `livelink_rails.m` | Windows，COMSOL with MATLAB | 逐轨建模求解，记录成核时间和耗时 |
| `compare_rails.py` | Linux，StackEM 环境 | 三方比较，输出误差统计和计时外推表 |

## 环境

发布结果使用 COMSOL Multiphysics 6.2 加 LiveLink for MATLAB。不需要事先在图形界面里建任何模型，脚本通过 API 为每条轨从头建立一维模型。

## 第一步：导出轨

前提是 base3 的 E2 已经跑完，输出目录里有 `e2_full_truth_rails.json`、`e2_full_truth.json`、`e2_full_pred.json`。

```bash
source ~/stackem_work/venv/bin/activate
cd <仓库根目录>
python comsol/export_rails.py --e2 ~/stackem_work/outputs/base3/e2 --out ~/stackem_work/outputs/comsol
```

也可以用 `STACKEM_ONLY=comsol.export bash scripts/run_all.sh`，效果相同。输出三个文件：

| 文件 | 内容 |
|---|---|
| `rails.csv` | 每段一行：轨号、段号、长度、电流密度、两端温度、焦耳温升幅值、热特征长度、残余应力 |
| `rails_tnuc.csv` | 每条轨一行：参考求解器与 StackEM 给出的成核时间 |
| `constants.json` | 电迁移常数和方程的文字说明 |

## 第二步：在 COMSOL 里求解

把导出目录整个拷到 Windows 能访问的位置，例如 `C:\stackem_comsol\comsol`，把 `livelink_rails.m` 放在 `C:\stackem_comsol`。启动 COMSOL Multiphysics with MATLAB，在 MATLAB 命令窗口输入：

```matlab
setenv('STACKEM_CR', 'C:\stackem_comsol\comsol');
% 先试 20 条轨时再加一行：setenv('STACKEM_CR_MAX', '20');
cd C:\stackem_comsol
livelink_rails
```

`STACKEM_CR` 指向导出目录，`STACKEM_CR_MAX` 限制求解的轨数。结果写到同一目录的 `comsol_rails_result.csv`，每条轨一行，含成核时间和建模、求解、合计三个耗时。建议把 MATLAB 窗口的输出另存一份，发布结果当时没有保存这份输出。

### 模型的定义

脚本对每条轨建立的模型如下，手工在图形界面里复现时请按这张表填写。

| 项目 | 设置 |
|---|---|
| 几何 | 一维，每段一个区间，区间端点就是结点 |
| 物理接口 | 系数形式偏微分方程，因变量是应力，单位 Pa |
| 因变量名 | `u`。COMSOL 6.2 上脚本里改名的那一行不生效，因变量保持默认名，初值和取最大值都要写 `u` |
| 系数 | `c = kap`，`da = 1`，`a = f = ea = al = be = 0` |
| 守恒通量源项 | `ga = kap*(G+M)`，符号为正 |
| 边界 | 两端零通量，即默认设置 |
| 初值 | 每段 `u = sT`，即该段的残余应力 |
| 网格 | 最大单元尺寸取最短段长度的六十分之一 |
| 求解 | 瞬态，输出时刻 `10^range(4,0.1,9.5)` 秒共 56 个，相对容差 1e-6 |

每段的变量定义，其中 `xi` 是段内归一化坐标，`lam = L/Gamma`：

```text
T    = T_L + (T_R - T_L)*xi + T_m*(1 - cosh((xi - 0.5)*lam)/cosh(0.5*lam))
dTdx = (T_R - T_L)/L - T_m/Gamma * sinh((xi - 0.5)*lam)/cosh(0.5*lam)
kap  = D0*exp(-Ea/(kB*T)) * B*Omega/(kB*T)
G    = e*Z_eff*rho*j/Omega
M    = Q/(Omega*T) * dTdx
```

### 符号约定，最容易出错的地方

代码求解的方程是

```text
d sigma/dt = d/dx [ kap * ( d sigma/dx - G - M ) ]，Z_eff = +10
```

COMSOL 系数形式里的通量是 `-c*grad(u) + ga`。取 `c = kap` 后，要得到上式就必须令 `ga = +kap*(G+M)`。写成负号会得到方向相反的驱动力。论文正文采用另一种等价写法：括号内为加号，有效电荷数为负 10。两种写法是同一个物理，但不能混用，在 COMSOL 里请一律按本节和 `constants.json` 的写法。

成核时间的取法与参考求解器相同：取整条轨上应力最大值第一次达到临界应力的时刻，在相邻两个输出时刻之间按时间的平方根线性插值。

## 第三步：比较

把 `comsol_rails_result.csv` 拷回 Linux 的导出目录，然后：

```bash
python comsol/compare_rails.py --dir ~/stackem_work/outputs/comsol
```

也可以用 `STACKEM_ONLY=comsol.compare bash scripts/run_all.sh`。输出：

| 文件 | 内容 |
|---|---|
| `comsol_rails_compare.json` | COMSOL 对参考求解器、StackEM 对 COMSOL、StackEM 对参考求解器的成核时间相对差的中位数、90 分位和最大值，以及 COMSOL 的单轨耗时 |
| `comsol_timing.csv` | 轨数与秒数。只有 246 条轨是实测，其余规模是单轨平均耗时乘以轨数的外推，第三列是假设 32 个 COMSOL 进程并行的乐观下界 |
| `comsol_rails_parity.pdf/.png` | StackEM 对 COMSOL 的散点图 |

之后重画速度图，COMSOL 曲线会自动叠加：

```bash
STACKEM_ONLY=figs bash scripts/run_all.sh
```

## 发布结果

`reference_results/comsol/` 里是这一步的发布结果：246 条轨全部求解，其中 128 条在参考求解器和 COMSOL 里都成核。数值见该目录下的 `comsol_rails_compare.json`。

## 局限

- COMSOL 一侧没有保存求解日志，无法事后核对求解器实际采用的时间积分设置，只能信任结果文件。
- 计时是单次测量，并且除 246 条轨以外都是外推。
- 结果是在 COMSOL 6.2 上得到的。其他版本上属性名可能不同，因变量改名那一行是否生效也可能不同，遇到报错时先检查因变量名。
