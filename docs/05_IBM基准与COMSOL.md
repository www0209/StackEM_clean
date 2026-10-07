# 05 IBM 基准与 COMSOL

本文讲两件相互独立的事。第一部分是 IBM 电源网基准，在 Linux 里完成，建议都做。第二部分是用商业软件 COMSOL 做的交叉验证，需要 Windows 上的 COMSOL 许可证，是可选的，没有 COMSOL 的读者看第 2.1 节即可。

前提：已经按 [02_安装.md](02_安装.md) 装好环境。

文中的数字取自 `reference_results/ibmpg/`、`reference_results/comsol/` 和 `reference_results/logs/`，核对清单是 `docs/checks/05_IBM基准与COMSOL.jsonl`。

## 第一部分：IBM 电源网基准

### 1.1 这是什么

IBM 电源网基准是一组公开的电源网网表，共六个，名字是 ibmpg1 到 ibmpg6。网表是一个文本文件，用 SPICE 格式列出电源网里的每一个电阻、电压源和电流源。它们来自真实的设计，规模从三万个节点到一百六十多万个节点，常用来检验电源网分析工具。

前面的三个堆叠算例用的是规则的合成格子。这一组实验要回答的是：换成真实电源网里长短不一、段数不等的轨以后，StackEM 的精度和速度如何。

对每个网表，程序依次做这些事：

1. 读入网表，解整个电源网的直流方程，得到每个电阻上的电流，并与官方给出的直流解核对。
2. 在每一层金属上把首尾相接、方向相同的电阻串成直轨，只保留至少有 3 段的轨。
3. 把 base3 堆叠里顶层芯片 D3 的温度场投影到网表的范围上，给每一段一个温度。
4. 把每条轨的宽度定到使其最大电流密度等于 1e6 A/cm² 的设计上限。
5. 用 Blech 判据筛掉永远不会成核的轨。
6. 对剩下的轨，分别用参考求解器和 StackEM 求成核时间，统计误差和耗时。

第 3 步需要说明：网表本身不带温度信息，这里的温度是借用的，所以这组实验检验的是方法在真实拓扑上的精度和速度，并不代表这些设计的真实寿命。

### 1.2 第一步：下载

```bash
cd ~/StackEM_clean
bash scripts/download_ibmpg.sh
```

脚本从基准的公开网页下载 12 个压缩文件，即六个网表和六个官方直流解，共约 76 MB，用 `bunzip2` 解压，解压后约 610 MB，放在 `~/stackem_work/ibmpg`。最后用 MD5 校验和逐个核对，屏幕上应当出现 12 行 `OK`：

```text
ibmpg1.solution: OK
ibmpg1.spice: OK
...
ibmpg6.spice: OK
benchmarks ready in /home/<你的用户名>/stackem_work/ibmpg
```

可能遇到的情况：

| 现象 | 处理 |
|---|---|
| 下载中途断开 | 重新运行同一条命令。已经解压好的文件会跳过，没下完的文件会续传 |
| 某一行显示 `FAILED` | 这个文件损坏了。删掉它和对应的 `.bz2` 再运行脚本，例如 `rm ~/stackem_work/ibmpg/ibmpg3.spice ~/stackem_work/ibmpg/ibmpg3.spice.bz2` |
| 网站连不上 | 从其他途径取得这 12 个文件，放进 `~/stackem_work/ibmpg`，文件名保持 `ibmpgN.spice` 和 `ibmpgN.solution`。校验和列在 `scripts/download_ibmpg.sh` 里 |
| 想放到别的目录 | `bash scripts/download_ibmpg.sh <目录>`，之后运行时带上 `STACKEM_IBMPG_DIR=<目录>` |

### 1.3 第二步：运行

IBM 基准需要学生网络的权重在输出根目录里。分两种情况。

已经跑过路线 A 或路线 B：权重已经就位，直接运行。

```bash
source ~/stackem_work/venv/bin/activate
cd ~/StackEM_clean
bash scripts/run_ibmpg.sh
```

还没有跑过任何路线，只想做 IBM 基准：用发布权重的脚本，它会先把权重复制过去。

```bash
source ~/stackem_work/venv/bin/activate
cd ~/StackEM_clean
STACKEM_ONLY=ibmpg bash scripts/run_with_released_weights.sh
```

两种写法都会依次处理 `~/stackem_work/ibmpg` 里找到的每一个 `ibmpgN.spice`。如果下载在路线 A 之前就做了，IBM 基准已经作为路线 A 的一部分跑过，不需要再单独运行。

只跑其中几个网表时，把网表的完整路径写在 `run_ibmpg.sh` 后面，用空格隔开：

```bash
bash scripts/run_ibmpg.sh ~/stackem_work/ibmpg/ibmpg1.spice ~/stackem_work/ibmpg/ibmpg2.spice
```

运行前可以先确认显卡可用，这决定了闭合在哪里运行：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

显示 `True` 时闭合在显卡上运行，`False` 时在处理器上运行。两种情况的精度数字相同，只有耗时不同。

### 1.4 屏幕上应当看到什么

以 ibmpg1 为例，参考机的输出主要是下面这些行。

```text
parsed 30636 nodes, 30027 resistors; DC solve 0.1s; Vmin 0.0000 V
coordinate unit: auto -> 1e-06 m (median collinear segment 96 units = 96 um; die 20.8 x 21.0 mm)
temperature field of D3: 342.0-359.9 K projected on the netlist
rails sized to j_max = 1e+06 A/cm^2: W median 11.48 um
DC solve vs official solution: 30635 nodes, max |dV| = 0.006 mV, mean 0.001 mV
1942 straight rails extracted (>= 3 segments); by net: M0: 420, M1: 427, M2: 754, M3: 341
immortal by Blech screen: 0 / 1942 (0.4s)
reference solver: 3.0s on 32 workers, ...
closure (torch/cuda, tabulated): 2.6s for 1942 rails
```

有三行值得逐个网表检查。

1. `coordinate unit` 一行。网表没有写明坐标的单位，程序根据芯片跨度和段长自动判断，并打印它推出的芯片尺寸。这个尺寸必须合理，也就是几毫米到几十毫米。参考结果里各网表的判断如下。

   | 网表 | 判断出的坐标单位 | 推出的芯片尺寸 |
   |---|---|---|
   | ibmpg1 | 1 微米 | 20.8 乘 21.0 mm |
   | ibmpg2 | 1 微米 | 8.1 乘 8.2 mm |
   | ibmpg3 | 1 纳米 | 30.6 乘 30.7 mm |
   | ibmpg4 | 1 微米 | 13.7 乘 13.8 mm |
   | ibmpg5 | 1 微米 | 21.3 乘 21.3 mm |
   | ibmpg6 | 1 纳米 | 10.2 乘 10.2 mm |

2. `DC solve vs official solution` 一行。它是程序自己的直流解与官方解的最大电压差，应当在千分之几毫伏的量级。
3. `closure` 一行。括号里是 `torch/cuda` 还是 `numpy/cpu`，说明闭合实际在显卡还是处理器上运行。

之后程序还会打印按段数、段长等分组的误差统计和误差最大的十条轨，供分析用。

### 1.5 耗时

下表是参考机上每个网表整个步骤的墙钟时间，以及其中两个求解器各自的秒数。参考求解器用 32 个线程，StackEM 的闭合用一块显卡。

| 网表 | 节点数 | 整个步骤 | 参考求解器 | StackEM 闭合 |
|---|---|---|---|---|
| ibmpg1 | 30636 | 9 秒 | 3.0 秒 | 2.6 秒 |
| ibmpg2 | 127236 | 16 秒 | 4.0 秒 | 5.5 秒 |
| ibmpg3 | 851582 | 148 秒 | 69.0 秒 | 43.2 秒 |
| ibmpg4 | 953581 | 108 秒 | 37.2 秒 | 33.3 秒 |
| ibmpg5 | 1079308 | 163 秒 | 98.1 秒 | 34.7 秒 |
| ibmpg6 | 1670492 | 193 秒 | 97.0 秒 | 51.8 秒 |

六个网表合计约 11 分钟。整个步骤的时间除了两个求解器，还包括读网表、解直流方程、跑一次 HotSpot 和提取直轨。没有显卡时闭合慢得多：参考机上 ibmpg1 的闭合用处理器跑是 51.4 秒，用显卡是 2.6 秒。

这些耗时的出处：ibmpg1 取自 `final_4_extras.log`，ibmpg2 到 ibmpg6 取自 `ibmpg_full.log`。`ibmpg_full.log` 如实保留了当时的全部输出，其中同一个网表出现不止一次，与 `reference_results/ibmpg/` 对应的是每个网表最后一次出现的那一段。ibmpg1 在这份日志开头的那一段是只用处理器的闭合，也就是上面 51.4 秒的来源。

### 1.6 结果在哪里，怎么读

每个网表一个文件夹 `~/stackem_work/outputs/ibmpg/ibmpgN/`，里面有：

| 文件 | 内容 |
|---|---|
| `ibmpg_summary.json` | 摘要：节点数、轨数、被筛掉的轨数、误差、排序指标、耗时 |
| `ibmpg_parity.pdf` 和 `.png` | StackEM 对参考求解器的成核时间对照图，读法与 README 里的对照图相同 |
| `ibmpg_rails.npz` | 逐轨的数组：两个求解器的成核时间、段数、段长、电流密度、温度等 |
| `hotspot/` | 这次运行的 HotSpot 输入和输出 |

把摘要打印出来看：

```bash
python -c "import json; d=json.load(open('$HOME/stackem_work/outputs/ibmpg/ibmpg1/ibmpg_summary.json')); [print(k, d[k]) for k in d]"
```

摘要里各字段的含义：

| 字段 | 含义 |
|---|---|
| `n_nodes`、`n_resistors` | 网表的节点数和电阻数 |
| `n_rails` | 提取出的直轨数 |
| `n_immortal` | 被 Blech 判据判为永不成核的轨数 |
| `unit_m` | 判断出的坐标单位，以米计 |
| `n_mortal_truth`、`n_mortal_pred` | 参考求解器和 StackEM 各自判定会成核的轨数 |
| `mortality_agreement` | 两者对“会不会成核”判断一致的比例 |
| `tnuc_rel_median`、`tnuc_rel_p90` | 成核时间相对误差的中位数和 90 分位 |
| `top10_hit` | 真值里最早成核的十条轨被 StackEM 也排进前十的比例 |
| `kendall_top50` | 最早成核的五十条轨上两种排序的 Kendall 相关 |
| `earliest_years`、`n_mortal_10yr` | 真值的最早成核时间和十年内成核的轨数 |
| `seconds_truth`、`seconds_closure`、`closure_backend` | 两个求解器的秒数，以及闭合用的后端 |

参考结果：

| 网表 | 直轨数 | 筛掉的轨数 | 误差中位数 | 误差 90 分位 | 前十命中率 | 前五十的 Kendall 相关 | 成核判断一致的比例 |
|---|---|---|---|---|---|---|---|
| ibmpg1 | 1942 | 0 | 0.35 % | 1.07 % | 100 % | 0.982 | 100 % |
| ibmpg2 | 1177 | 2 | 0.55 % | 1.63 % | 100 % | 0.979 | 100 % |
| ibmpg3 | 11135 | 456 | 0.30 % | 1.04 % | 100 % | 0.997 | 99.87 % |
| ibmpg4 | 13348 | 8245 | 1.38 % | 2.42 % | 100 % | 0.995 | 99.71 % |
| ibmpg5 | 4123 | 0 | 0.72 % | 1.50 % | 100 % | 0.984 | 99.83 % |
| ibmpg6 | 20732 | 2755 | 1.04 % | 2.50 % | 100 % | 0.984 | 99.82 % |

最后一列低于 100 % 表示有少数轨两个求解器一个判为成核、一个判为不成核，各自的数目见摘要里的 `n_mortal_truth` 和 `n_mortal_pred`。

### 1.7 与参考结果比对

```bash
python tools/check_results.py
```

六个 `ibmpg/ibmpgN/ibmpg_summary.json` 应当都是 `PASS`。节点数、轨数、筛掉的轨数、真值的成核轨数必须完全相等，误差指标允许小幅偏差，耗时和闭合后端不比较。

如果 `unit_m` 与上面的表不同，这个文件会显示 `DIFF`，并且其余数字都会跟着不同。这时请看日志里 `coordinate unit` 一行推出的芯片尺寸是否合理。需要手动指定单位时，用第 04 号文档 6.6 节的手动命令，把 `--unit auto` 换成具体的值，例如 `--unit 1e-9`。

## 第二部分：COMSOL 交叉验证

### 2.1 这是什么，没有 COMSOL 怎么办

本项目所有的真值都来自仓库里自带的参考求解器。为了确认这个参考求解器本身没有错，我们用商业有限元软件 COMSOL Multiphysics 把 base3 的全部 246 条轨独立地再解了一遍，同时测出 COMSOL 每条轨要多少时间。

参考结果在 `reference_results/comsol/`，摘要如下。246 条轨里有 128 条在参考求解器和 COMSOL 里都成核，误差在这 128 条上统计。

| 比较 | 成核时间相对差的中位数 | 90 分位 | 最大值 |
|---|---|---|---|
| COMSOL 对参考求解器 | 0.084 % | 0.28 % | 0.53 % |
| StackEM 对 COMSOL | 0.18 % | 0.85 % | 1.89 % |
| StackEM 对参考求解器 | 0.15 % | 0.90 % | 2.34 % |

COMSOL 与参考求解器对每条轨是否成核的判断完全一致。COMSOL 每条轨的耗时中位数是 1.35 秒，246 条轨合计 333 秒。

没有 COMSOL 许可证的读者可以跳过整个第二部分，主流程的任何一步都不依赖它。E0 里对解析级数的检查仍然独立地验证了参考求解器。如果想在没有 COMSOL 的情况下重现比较这一步，可以直接使用仓库里发布的 COMSOL 结果文件，做法见第 2.7 节。

### 2.2 需要什么

| 需要 | 说明 |
|---|---|
| COMSOL Multiphysics 6.2 | 参考结果用的版本。只用到基本模块里的系数形式偏微分方程接口 |
| LiveLink for MATLAB | COMSOL 的一个附加产品，让 MATLAB 脚本能驱动 COMSOL |
| MATLAB | 参考机上是 R2024a |
| base3 的 E2 结果 | 即 `~/stackem_work/outputs/base3/e2/`，由路线 A 或路线 B 产生 |

COMSOL 和 MATLAB 都装在 Windows 一侧。不需要事先在 COMSOL 的图形界面里建任何模型，脚本会为每条轨从头建立一个一维模型。

整个过程分四步：在 Linux 里导出轨，把文件拷到 Windows，在 COMSOL 里求解，把结果拷回 Linux 做比较。

### 2.3 第一步：导出轨

如果已经跑完路线 A 或路线 B，导出已经作为 `comsol.export` 步骤做过了，`~/stackem_work/outputs/comsol/` 里已经有文件，可以直接进入下一步。需要单独重做时：

```bash
source ~/stackem_work/venv/bin/activate
cd ~/StackEM_clean
STACKEM_ONLY=comsol.export bash scripts/run_all.sh
```

检查导出的三个文件：

```bash
ls -l ~/stackem_work/outputs/comsol/
```

| 文件 | 内容 |
|---|---|
| `rails.csv` | 每段一行：轨号、段号、长度、电流密度、两端温度、焦耳温升幅值、热特征长度、残余应力 |
| `rails_tnuc.csv` | 每条轨一行：参考求解器和 StackEM 给出的成核时间 |
| `constants.json` | 电迁移常数和方程的文字说明 |

如果这一步报错说找不到 `e2_full_truth_rails.json`，说明 base3 的 E2 还没有跑，先执行 `STACKEM_ONLY=e2 bash scripts/run_with_released_weights.sh`。

### 2.4 第二步：把文件拷到 Windows

在 WSL 的终端里，Windows 的 C 盘就是 `/mnt/c`。下面三条命令在 C 盘建一个文件夹，把导出目录和求解脚本拷进去。

```bash
mkdir -p /mnt/c/stackem_comsol
cp -r ~/stackem_work/outputs/comsol /mnt/c/stackem_comsol/
cp ~/StackEM_clean/comsol/livelink_rails.m /mnt/c/stackem_comsol/
```

做完以后，Windows 里应当有 `C:\stackem_comsol\livelink_rails.m` 和 `C:\stackem_comsol\comsol\rails.csv` 等文件，可以用资源管理器确认。

用原生 Ubuntu 的读者需要用 U 盘或网络把 `~/stackem_work/outputs/comsol` 文件夹和 `comsol/livelink_rails.m` 拷到装有 COMSOL 的 Windows 电脑上，放成同样的结构。

### 2.5 第三步：在 COMSOL 里求解

1. 在 Windows 的开始菜单里找到 COMSOL 的程序组，点击“COMSOL Multiphysics 6.2 with MATLAB”。注意要点带有“with MATLAB”字样的那一项，不要点普通的 COMSOL 图标。
2. 它会先弹出一个黑色的 COMSOL 服务器窗口，第一次使用时可能要求设置用户名和密码，随意设一个即可。随后 MATLAB 的主窗口会自动打开。两个窗口都不要关。
3. 在 MATLAB 主窗口下方的命令行窗口里，提示符是 `>>`。先试算 20 条轨，确认一切正常。依次输入下面四行，每行回车。

   ```matlab
   setenv('STACKEM_CR', 'C:\stackem_comsol\comsol');
   setenv('STACKEM_CR_MAX', '20');
   cd C:\stackem_comsol
   livelink_rails
   ```

   第一行告诉脚本导出目录在哪里，第二行限制只解前 20 条轨，第三行进入脚本所在的文件夹，第四行运行脚本。
4. 每解完一条轨，窗口里打印一行：

   ```text
   rail    0:  40 segments  t_nuc 2.394 yr  build 0.29s solve 1.24s
   ```

   各项依次是轨号、段数、COMSOL 算出的成核时间、建模秒数、求解秒数。全部结束时打印 `wrote C:\stackem_comsol\comsol\comsol_rails_result.csv`。
5. 试算正常以后，取消轨数限制，解全部 246 条轨。

   ```matlab
   setenv('STACKEM_CR_MAX', '');
   livelink_rails
   ```

   参考机上 246 条轨合计 333 秒，不到 6 分钟。新的结果文件会覆盖试算的那一份。
6. 建议用鼠标选中命令行窗口里的全部输出，复制到一个文本文件里保存。参考结果当时没有保存这份输出，这是第 2.9 节列出的局限之一。

脚本为每条轨建立的模型、各项系数的填法，以及最容易出错的符号约定，写在 [COMSOL_MODEL_BUILD.md](../comsol/COMSOL_MODEL_BUILD.md) 里。只是照做的读者不需要读那份文件，想在图形界面里手工复现或者想改脚本的读者一定要读。

可能遇到的问题：

| 现象 | 处理 |
|---|---|
| 报错信息里提到找不到 `com.comsol.model` 或 `ModelUtil` | MATLAB 不是通过“COMSOL with MATLAB”启动的，没有连上 COMSOL。关掉 MATLAB，按第 1 步重新启动 |
| 报错信息里提到打不开 `rails.csv` | `STACKEM_CR` 指向的文件夹不对。在 MATLAB 里输入 `dir(getenv('STACKEM_CR'))` 看看里面有没有 `rails.csv` |
| 报某个属性名不存在 | 你的 COMSOL 版本与 6.2 的属性名不同。脚本里每一行旁边都注明了它的含义，按说明在新版本里找到对应的属性名 |
| 成核时间与预期相差极大，或全部不成核 | 多半是因变量名的问题。在 6.2 上因变量保持默认名 `u`，其他版本可能不同，详见上面那份说明的“模型的定义”一节 |
| 开始菜单里没有“with MATLAB”这一项 | 没有安装 LiveLink for MATLAB，或者安装 COMSOL 时没有指定 MATLAB 的位置。需要重新运行 COMSOL 的安装程序补装 |

### 2.6 第四步：拷回结果并比较

回到 WSL 的终端。

1. 把结果文件拷回 Linux 的导出目录。

   ```bash
   cp /mnt/c/stackem_comsol/comsol/comsol_rails_result.csv ~/stackem_work/outputs/comsol/
   ```

2. 运行比较。

   ```bash
   source ~/stackem_work/venv/bin/activate
   cd ~/StackEM_clean
   STACKEM_ONLY=comsol.compare bash scripts/run_all.sh
   ```

   屏幕上会打印三组误差统计和计时外推，并在 `~/stackem_work/outputs/comsol/` 里写出三样东西。

   | 文件 | 内容 |
   |---|---|
   | `comsol_rails_compare.json` | 三方两两比较的成核时间相对差的中位数、90 分位和最大值，COMSOL 的单轨耗时 |
   | `comsol_timing.csv` | 轨数与秒数的表，供速度图使用 |
   | `comsol_rails_parity.pdf` 和 `.png` | StackEM 对 COMSOL 的对照图 |

3. 重画速度图。这一次图上会多出一条 COMSOL 的曲线。

   ```bash
   STACKEM_ONLY=figs bash scripts/run_all.sh
   ```

   新图在 `~/stackem_work/outputs/base3/e4/fig_e4_paper_timing.png`。`reference_results/base3/e4/` 里的那张同名图没有 COMSOL 曲线，README 里展示的就是那一张。

4. 与参考结果比对。

   ```bash
   python tools/check_results.py
   ```

   `comsol/comsol_rails_compare.json` 这一行不再是 `not run`。需要注意：你自己用 COMSOL 解出来的成核时间不会与参考机的逐位相同，尤其是 COMSOL 版本不同的时候，而比对工具对 COMSOL 一类的数字用的是千分之一的相对容差，所以这一行显示 `DIFF` 并不意外。判断是否成功应当看量级：COMSOL 对参考求解器的差的中位数应当在千分之一上下，成核判断应当完全一致。

### 2.7 没有 COMSOL 时重现比较这一步

仓库里发布了参考机上 COMSOL 的原始结果文件。把它拷到你的导出目录，就可以在没有 COMSOL 的情况下运行比较和重画速度图。

```bash
cp ~/StackEM_clean/reference_results/comsol/comsol_rails_result.csv ~/stackem_work/outputs/comsol/
cd ~/StackEM_clean
STACKEM_ONLY=comsol.compare bash scripts/run_all.sh
STACKEM_ONLY=figs bash scripts/run_all.sh
python tools/check_results.py
```

前提是 `~/stackem_work/outputs/comsol/` 里已经有导出的 `rails_tnuc.csv`，也就是 `comsol.export` 已经跑过。这样得到的 `comsol_rails_compare.json` 应当通过比对。请清楚这样做的含义：COMSOL 的数字是发布的，并非你自己解出来的，你重现的只是比较和画图。

### 2.8 计时表怎么读

`comsol_timing.csv` 有三列。参考结果如下。

| 轨数 | 顺序求解的秒数 | 32 个 COMSOL 进程并行的乐观下界 |
|---|---|---|
| 82 | 111.1 | 3.5 |
| 246 | 333.2 | 10.4 |
| 1000 | 1354.3 | 42.3 |
| 5000 | 6771.7 | 211.6 |
| 20000 | 27086.7 | 846.5 |

只有 246 条轨这一行是实测。其余各行是用单轨平均耗时 1.354 秒乘以轨数外推出来的。第三列是第二列除以 32，它假设有 32 个 COMSOL 进程同时运行且互不干扰，实际上没有这样跑过。

### 2.9 局限

1. COMSOL 一侧没有保存求解日志，无法事后核对求解器实际采用的时间积分设置，只能信任结果文件。
2. 计时是单次测量，除 246 条轨以外都是外推。
3. 结果是在 COMSOL 6.2 上得到的，其他版本没有试过。
4. COMSOL 里求解的是与参考求解器相同的一维方程和相同的输入，所以这项验证检查的是数值求解是否正确，不检查物理模型本身是否合适。

遇到运行问题请看 [09_常见问题.md](09_常见问题.md)。
