# 随机性实现：natal 引擎、GPU 卷积复刻阻碍与保留卷积的随机方法

> 用途：给未读过 natal 源码的读者解释「我们在建模什么、natal 的随机迁移怎么算、为什么 GPU 卷积不能精确复刻、以及保留卷积的随机实现方法（M1–M4）」。
> 本文件只做说明，不改变任何实现。公式中的符号均在首次出现处定义；推导尽量连续、少跳跃。
> 定位：本文档将扩展为**方法指导**，供未做过数学机制探索、中途接手的 agent 理解并执行探索计划。

---

## 0. 背景：我们在模拟什么

### 0.1 生物学对象

我们模拟一个**空间结构化的、年龄结构的蚊虫种群**，并研究**基因驱动（gene drive）**在其中的传播。把一片栖息地划分成许多个相邻的小区（下面叫 **deme**，即“种群格点”），每个 deme 内记录不同**年龄**、**性别**、**基因型**的个体数量。

- **网格**：六边形（hex）网格。每个 deme 是一个格点，格点之间可以迁移（扩散）。
- **一 tick = 一周**：每个时间步内，所有 deme 各自完成一轮生命周期，然后个体在 deme 之间迁移。
- **年龄结构**（本项目的蚊虫模型）：2 个幼期 + 5 个雄成虫期 + 8 个雌成虫期。
- **基因型**（homing 驱动，一个基因座）：论文用 5 种 $\{dd, dw, ww, rw, dr\}$，其中 $ww$ 是野生型，$dd$ 是驱动纯合子。natal 引擎里也可用 3 种（只有 $WT$、$Drive$ 两个等位基因）或 6 种（再加抗性等位 $Res$）。
- **一个 tick 内的顺序**：natal 引擎固定为 **R-S-A → migration**：
  1. **R（繁殖，reproduction）**：成虫产生新生个体；
  2. **S（密度调节 + 存活，survival）**：幼期受密度依赖调节，再乘存活率；
  3. **A（老化，aging）**：年龄类整体前移；
  4. **migration（迁移）**：个体在本 tick 末尾扩散到邻近 deme。

### 0.2 确定性 vs 随机

- **确定性（deterministic）**：所有数量取期望值，没有随机抽样。结果是唯一的一条轨迹。
- **随机（stochastic）**：每个 tick 的繁殖、存活、迁移都按概率分布**抽样**，结果带随机涨落（demographic stochasticity）。数量可以是连续的（f64），不一定取整。

我们研究的目标现象 **chasing**（驱动与野生型在空间上长期共存、前沿来回移动、不必然一方彻底固定）本质上依赖随机性：小种群的随机灭绝/再定殖会制造“避难所”。因此需要**随机**空间模型。

### 0.3 为什么关心迁移与 GPU

- 迁移使用一个**高斯核**：每个 deme 的个体按距离权重扩散到周围 deme。核越大，扩散范围越广。
- 论文/实际场景常用**很大的核**（几十个格点宽）。若把核展开成稀疏矩阵（CSR），内存和构建成本会爆炸。
- 为此 natal 提供了 **route A**：把迁移写成**归一化卷积**，用 **cuFFT（GPU 快速傅里叶变换）** 逐平面计算，复杂度从 $O(\mathrm{nnz})$（非零边数）降到 $O(n\log n)$（$n$ = 格点数）。
- 问题：route A 目前只实现了**确定性**迁移。若要在 GPU 上做**随机**迁移，就需要回答：**卷积能不能承载随机性？** 本文说明：**不能精确承载 natal 原有的逐边随机迁移**，并给出保留卷积的若干替代方案（M1–M4，见 §4）。

---

## 1. 记号与设定（先统一符号）

我们只关注**一个平面**。所谓“平面”，是固定 `(性别, 年龄, 基因型)` 之后得到的一张空间计数图。迁移对每个平面独立进行，因此所有公式对单个平面成立。

| 符号 | 含义 |
|---|---|
| `n` | deme 总数（$n = \text{rows}\times\text{cols}$） |
| `s, d` | 下标：`s` 是**源** deme（个体从这里出发），`d` 是**目的地** deme |
| `f_s` | 源 deme `s` 在该平面上的个体数（非负实数） |
| `ρ` | **迁移率**，$0\le\rho\le 1$；本 tick 中每个源有比例 $\rho$ 的个体参与迁移 |
| `u` | 位移（源到目的地的偏移） |
| `k(u)` | 高斯**核权重**（未归一化），只依赖位移 `u`（在 hex 度量下按距离衰减） |
| `K'` | 去中心核：$K'(u)=k(u)$（$u\ne 0$），$K'(0)=0$。中心排除意味着“留在原地”不通过迁移核，而由 $1-\rho$ 保留 |
| `valid(s)` | 从源 `s` 出发、落点仍在网格内的所有位移集合 |
| `Z_s` | 源 `s` 的**有效权重和**：$Z_s=\sum_{u\in\mathrm{valid}(s)}K'(u)$ |
| `w_{s,d}` | 源 `s` 到目的地 `d` 的**归一化权重**：$w_{s,d}=K'(d-s)/Z_s$，若 $d-s$ 无效则 $w_{s,d}=0$ |
| `N_s` | 源 `s` 参与迁移前的计数（$=f_s$，后文混用） |
| `O_s` | 源 `s` 抽到的**出境量**（有多少个体本 tick 离开 `s`） |
| `R_s` | 源 `s` 的**保留量**：$R_s=N_s-O_s$ |
| `X_{s,d}` | 源 `s` 实际分配到目的地 `d` 的个体数 |
| `Y_d` | 目的地 `d` 本 tick 迁移后的计数 |
| `G_{s,d}` | 源 `s` 为目的地 `d` 抽取的 Gamma 随机量（中间量） |
| `S_s` | 源 `s` 的 Gamma 总和：$S_s=\sum_d G_{s,d}$ |

由定义，每行权重归一：

$$
\sum_d w_{s,d}=1,\qquad w_{s,s}=0\quad(\text{center excluded})
$$

`w_{s,d}` 是**确定性的**（只依赖几何与核），但 $O_s$、$G_{s,d}$ 是**随机的**。

> 说明：$Z_s$ 处理了“边界”。边界源的有效位移变少，$Z_s$ 变小；用 $Z_s$ 归一化后，每行仍然 $\sum_d w_{s,d}=1$，即“边界源仍把全部出境量分给域内有效目的地”。这就是“丢界 + 重归一化”。

---

## 2. natal 引擎的随机迁移（CPU CSR 路径）

natal 的随机迁移按**源**逐个处理，每个源使用自己的随机数流（RNG），因此结果可复现且与线程调度无关。下面按一个源 `s` 的完整流程走一遍。

### 2.1 第一步：抽“出境量” $O_s$（连续二项分布 = Beta）

离散二项分布 $\mathrm{Binomial}(N,\rho)$ 描述“$N$ 个个体各自以概率 $\rho$ 离开”的总人数：取值 $0..N$，均值 $N\rho$，方差 $N\rho(1-\rho)$。

natal 使用它的**连续版本**（为了让人口保持实数、避免取整）：

$$
\begin{aligned}
O_s &= N_s\cdot B_s,\qquad B_s\sim \mathrm{Beta}(\alpha_s,\beta_s)\\
\alpha_s &= \rho\,(N_s-1)\\
\beta_s &= (1-\rho)\,(N_s-1)
\end{aligned}
$$

其中 $\mathrm{Beta}(\alpha,\beta)$ 是定义在 $[0,1]$ 上的分布，均值和方差为

$$
\mathbb{E}[B]=\frac{\alpha}{\alpha+\beta},\qquad
\mathrm{Var}[B]=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.
$$

**为什么这样取 $\alpha_s,\beta_s$？** 我们希望 $O_s$ 的前两阶矩与 $\mathrm{Binomial}(N_s,\rho)$ 一致。

1. 先看均值：

$$
\alpha_s+\beta_s=\rho(N_s-1)+(1-\rho)(N_s-1)=N_s-1,
$$

$$
\mathbb{E}[B_s]=\frac{\alpha_s}{\alpha_s+\beta_s}=\frac{\rho(N_s-1)}{N_s-1}=\rho,
$$

于是 $\mathbb{E}[O_s]=N_s\rho$，与二项均值一致。

2. 再看方差：

$$
\begin{aligned}
\mathrm{Var}[B_s]
&=\frac{\alpha_s\beta_s}{(\alpha_s+\beta_s)^2(\alpha_s+\beta_s+1)}\\
&=\frac{\rho(1-\rho)(N_s-1)^2}{(N_s-1)^2\cdot N_s}
=\frac{\rho(1-\rho)}{N_s}.
\end{aligned}
$$

于是

$$
\mathrm{Var}[O_s]=N_s^2\,\mathrm{Var}[B_s]
=N_s^2\cdot\frac{\rho(1-\rho)}{N_s}
=N_s\rho(1-\rho),
$$

与二项方差一致。

**$N_s-1$ 是怎么来的？** 关键在 Beta 方差分母里的 $(\alpha+\beta+1)$。要让 $\mathrm{Var}[O_s]=N_s\rho(1-\rho)$，必须让 $\alpha+\beta+1=N_s$，即 $\alpha+\beta=N_s-1$。所以 $N_s-1$ 不是随意选的，而是**方差匹配唯一确定**的。

**Beta 怎么抽？** 用两个独立 Gamma 的比值：

$$
G_a\sim \mathrm{Gamma}(\alpha_s,1),\qquad G_b\sim \mathrm{Gamma}(\beta_s,1),
$$

$$
B_s=\frac{G_a}{G_a+G_b}.
$$

natal 的 `continuous_binomial` 就是这样实现的。这里 $\mathrm{Gamma}(\text{shape},1)$ 的均值 = `shape`、方差 = `shape`。

**边界情形**：当 $N_s\le 1$ 时 $\alpha_s,\beta_s\le 0$，Beta/Gamma 退化，代码直接取均值 $O_s=\rho N_s$。

得到 $O_s$ 后，保留量

$$
R_s=N_s-O_s,
$$

其均值为 $N_s(1-\rho)$，方差同样为 $N_s\rho(1-\rho)$。

### 2.2 第二步：把 $O_s$ 分配到目的地（连续多项分布 = Dirichlet）

离散多项分布 $\mathrm{Multinomial}(O,p)$：把 $O$ 个个体按概率向量 $p=(p_1,\dots,p_k)$ 分到 $k$ 个类别，满足 $\sum_j X_j=O$，且

$$
\begin{aligned}
\mathbb{E}[X_j]&=O\,p_j,\\
\mathrm{Var}[X_j]&=O\,p_j(1-p_j),\\
\mathrm{Cov}[X_i,X_j]&=-O\,p_ip_j\qquad(i\ne j).
\end{aligned}
$$

natal 的连续版本是：对源 `s` 的每个有效目的地 `d`，独立抽 Gamma，再**按总和归一化**：

$$
\begin{aligned}
&\text{for each destination } d\ (w_{s,d}>0):\\
&\alpha_{s,d}=w_{s,d}\,(O_s-1),\\
&G_{s,d}\sim \mathrm{Gamma}(\alpha_{s,d},1),\\
&S_s=\sum_d G_{s,d},\\
&X_{s,d}=O_s\,\frac{G_{s,d}}{S_s}.
\end{aligned}
$$

若 $O_s\le 1$（此时 $\alpha_{s,d}\le 0$ 非法），代码退化为确定性均值 $X_{s,d}=O_s w_{s,d}$。

**为什么抽 Gamma 再归一化？** 因为“独立 Gamma 归一化”正是 Dirichlet 分布的标准构造：

$$
\text{if } G_j\sim\mathrm{Gamma}(\alpha_j,1)\ \text{independent, then }
\left(\frac{G_1}{S},\dots,\frac{G_k}{S}\right)\sim\mathrm{Dirichlet}(\alpha_1,\dots,\alpha_k),\quad
S=\sum_j G_j.
$$

Dirichlet 的均值和方差/协方差为（记 $A=\sum_j\alpha_j$）：

$$
\begin{aligned}
\mathbb{E}[D_j]&=\frac{\alpha_j}{A},\\
\mathrm{Var}[D_j]&=\frac{\alpha_j(A-\alpha_j)}{A^2(A+1)},\\
\mathrm{Cov}[D_i,D_j]&=-\frac{\alpha_i\alpha_j}{A^2(A+1)},\\
\sum_j D_j&=1.
\end{aligned}
$$

我们取 $A=O_s-1$、$\alpha_{s,d}=w_{s,d}(O_s-1)$，于是 $D_{s,d}=G_{s,d}/S_s\sim\mathrm{Dirichlet}(\alpha_{s,\cdot})$，且 $X_{s,d}=O_s D_{s,d}$。代入验证：

1. 均值：

$$
\mathbb{E}[X_{s,d}]=O_s\frac{\alpha_{s,d}}{A}
=O_s\frac{w_{s,d}(O_s-1)}{O_s-1}
=O_s w_{s,d},
$$

与多项均值一致。

2. 方差：

$$
\begin{aligned}
\mathrm{Var}[X_{s,d}]
&=O_s^2\,\mathrm{Var}[D_{s,d}]\\
&=O_s^2\,\frac{\alpha_{s,d}(A-\alpha_{s,d})}{A^2(A+1)}\\
&=O_s^2\,\frac{w(O_s-1)\,(1-w)(O_s-1)}{(O_s-1)^2\cdot O_s}\\
&=O_s\,w(1-w)\qquad(w=w_{s,d}),
\end{aligned}
$$

与多项方差一致。

3. 协方差：同理 $\mathrm{Cov}[X_{s,d},X_{s,d'}]=-O_s w_{s,d}w_{s,d'}$。

4. **守恒**：

$$
\sum_d X_{s,d}=O_s\sum_d D_{s,d}=O_s\cdot 1=O_s.
$$

**$O_s-1$ 又是怎么来的？** 与上一步同理：Dirichlet 方差的分母有 $(A+1)$，要精确匹配多项方差 $O_s w(1-w)$，必须 $A+1=O_s$，即 $A=O_s-1$。这是方差匹配唯一确定的。

**离散版对照**：若不用连续采样，natal 用“逐个目的地的条件二项 + 最后一个目的地取剩余”的离散多项，整数、总和严格等于 $O_s$；连续版则用 Gamma 归一化，实数、总和也严格等于 $O_s$。

### 2.3 第三步：合成目的地计数

源 `s` 的保留量 $R_s$ 留在 `s`；$X_{s,d}$ 被加到目的地 `d`。把所有源加起来：

$$
Y_d=R_d+\sum_{s\ne d}X_{s,d}=(N_d-O_d)+\sum_s X_{s,d}
\qquad(\text{since } w_{d,d}=0).
$$

注意 $X_{d,d}=0$（中心排除），所以“源 `d` 自己的外流”不会回到 `d`；$Y_d$ 由“`d` 的保留量”和“其他源发到 `d` 的量”组成。

### 2.4 均值、方差、协方差、守恒（为后文对照做准备）

**守恒**：

$$
\begin{aligned}
\sum_d Y_d
&=\sum_d (N_d-O_d)+\sum_d\sum_s X_{s,d}\\
&=\sum_d N_d-\sum_d O_d+\sum_s\left(\sum_d X_{s,d}\right)\\
&=\sum_d N_d-\sum_s O_s+\sum_s O_s\\
&=\sum_d N_d.
\end{aligned}
$$

即**总人口严格守恒**（在浮点误差内）。此外，对每个源也有 $\sum_d X_{s,d}=O_s$，即**逐源守恒**。

**均值**：因为 $\mathbb{E}[O_s]=\rho N_s$、$\mathbb{E}[X_{s,d}]=\rho N_s w_{s,d}$，

$$
\mathbb{E}[Y_d]=(1-\rho)N_d+\mu_d,\qquad
\mu_d:=\rho\sum_s N_s w_{s,d}.
$$

$\mu_d$ 就是“所有源按期望发到 `d` 的量”。

**方差**（$Y_d$ 由独立源组成；$R_d$ 与 $s\ne d$ 的 $X_{s,d}$ 独立）：

$$
\begin{aligned}
\mathrm{Var}[Y_d]
&=\mathrm{Var}[R_d]+\sum_{s\ne d}\mathrm{Var}[X_{s,d}]\\
&=N_d\rho(1-\rho)+\sum_s N_s\rho w_{s,d}(1-\rho w_{s,d})\\
&=N_d\rho(1-\rho)+\mu_d-\rho^2\sum_s N_s w_{s,d}^2.
\end{aligned}
$$

三项来源：

- $N_d\rho(1-\rho)$：`d` 自身的保留量有 Beta/二项方差；
- $\mu_d$：`d` 收到的“泊松样”方差；
- $-\rho^2\sum_s N_s w_{s,d}^2$：**同一源分给多个目的地是负相关的**（多项分布比独立泊松更“集中”）。

**协方差**（$d\ne d'$，结论如下）：

$$
\begin{aligned}
\mathrm{Cov}[Y_d,Y_{d'}]
=&-N_d\rho(1-\rho)\,w_{d,d'}\\
&-N_{d'}\rho(1-\rho)\,w_{d',d}\\
&-\rho^2\sum_{s\ne d,d'}N_s w_{s,d}w_{s,d'}.
\end{aligned}
$$

第一、二项来自“`d`（或 `d'`）的保留与外流共享同一个 $O_d$（或 $O_{d'}$）”；第三项来自“同一源发往 `d` 与 `d'` 的量负相关”。这说明**不同目的地之间不是独立的**。这个**密集的、逐源块状的协方差结构**是后文的关键。

### 2.5 随机数与确定性

- 每个源 `s` 有自己持久的 RNG 流；源之间独立，源内按固定顺序消耗随机数。
- 因此固定种子时结果逐位可复现，且与线程数/调度无关。
- 连续采样的阈值：$N_s\le 1$、$O_s\le 1$ 时退化为确定性均值（小计数时“没有噪声”）。

---

## 3. GPU 卷积（route A）如何工作，以及它卡在哪

### 3.1 route A 算的是确定性归一化卷积

route A 对每个平面 $f$ 计算：

$$
\begin{aligned}
g_s&=\rho\,f_s/Z_s,\\
c_d&=\sum_s K'(d-s)\,g_s=\rho\sum_s f_s w_{s,d},\\
Y_d^{\mathrm{det}}&=(1-\rho)f_d+c_d.
\end{aligned}
$$

把 $c_d$ 与 §2.4 的 $\mu_d$ 对比：

$$
\mu_d=\rho\sum_s f_s w_{s,d}=c_d.
$$

所以 **route A 精确给出 $\mathbb{E}[Y_d]$**。写成矩阵：

$$
Y^{\mathrm{det}}=Af,\qquad
A_{d,s}=(1-\rho)\delta_{d,s}+\rho w_{s,d}\quad(w_{d,d}=0).
$$

$A$ 是**固定的、平移不变的线性算子**（核只依赖 $d-s$）。

### 3.2 并行实现（为什么快）

对一个平面，GPU 上分四步（`fft_migrate.rs`）：

1. **`scatter_g`（逐格并行）**：把 $f_s$ 除以 $Z_s$、乘 $\rho$，零填充到 $(\text{rows}+k-1)\times(\text{cols}+k-1)$。每个格点一个线程。
2. **cuFFT `r2c`**：实数平面 → 频域（cuFFT 内部并行）。
3. **`complex_mul`（逐频点并行）**：频域逐点乘核的频谱（并折叠 $1/(f_rf_c)$）。
4. **cuFFT `c2r`**：频域 → 实数域。
5. **`combine`（逐格并行）**：$out_d=(1-\rho)f_d+\text{conv}_d$。

多个平面（$\text{sex}\times\text{age}\times\text{genotype}$）由 host 循环依次提交到同一 CUDA stream。复杂度从 CSR 的 $O(\mathrm{nnz})=O(nk^2)$（$k$ = 核宽）降到 $O(n\log n)$；核越大、域越大，优势越明显。

### 3.3 复刻阻碍：卷积“装不下”逐源随机归一化

CPU 的随机分配是：

$$
X_{s,d}=O_s\,\frac{G_{s,d}}{S_s},\qquad S_s=\sum_{d'}G_{s,d'}.
$$

这里是**逐源**的：每个源先抽 $O_s$，再对它那一行的 $G_{s,d}$ 求和得到 $S_s$，然后归一化。

而卷积给的是

$$
c_d=\sum_s K'(d-s)\,\frac{\rho f_s}{Z_s},
$$

即**对所有源求和**。逐源归一化因子 $S_s$（随机、每次抽样才实现）和“哪个源贡献了 $c_d$ 的哪一部分”**都没有被保留**。因此：

**阻碍 1：源身份被聚合掉了。**
卷积只输出“所有源的加权和”，没有逐源分解；而 $X_{s,d}$ 需要知道源 `s` 那一行的 $G$ 与 $S_s$。

**阻碍 2：$S_s$ 是随机、逐源、不可预计算的。**
对比边界掩膜：$Z_s$ 是**确定性**的、只依赖源位置、可以事先算好（$Z=K'\circledast m$，$m$ 为域指示），所以 route A 可以“先 $f/Z$ 再卷积”。而 $S_s$ 依赖每次抽到的 $G$，无法预计算，也无法写成“逐点缩放 + 固定卷积”。

**阻碍 3：这不是平移不变的算子。**
卷积核只依赖位移 $d-s$；随机权重 $G_{s,d}/S_s$ 依赖源 `s` 的整行（通过 $S_s$），**不是 $d-s$ 的函数**。一次固定卷积无法表达它。

**阻碍 4：随机自由度是 $O(\mathrm{nnz})$。**
每个源有约 $k$ 个独立 Dirichlet 自由度，全体源共 $O(nk)=O(\mathrm{nnz})$；FFT 是 $O(n\log n)$ 的确定性线性变换，无法在 $O(n\log n)$ 内生成这么多独立随机量。

> 补充：**同步不是根本障碍**。逐源归一化只需要“行内归约”（每个源独立），GPU 用 segmented reduction/scatter 完全可以做；现有 GPU 随机 CSR 路径（`migrate_tick_stochastic`）就是这么做的，只是受 `MAX_CSR_ROW=32` 的 scratch 行宽限制（`size=7` 核的行宽是 48，因此被拒）。同步/并行能力不是问题，**表示方式**才是。

### 3.5 朴素加噪为什么不行

一个自然的近似是“卷积均值 + 每个目的地独立加 Gamma 噪声”：$Y_d=(1-\rho)f_d+\mu_d+\varepsilon_d$，$\varepsilon_d\sim\mathrm{Gamma}(\text{shape}=\mu_d)$。它的问题：

1. **漏掉源端保留方差**：CPU 的保留量 $N_d-O_d$ 有方差 $N_d\rho(1-\rho)$；而 $(1-\rho)f_d$ 是确定性的。实测（21×21、200 次重复）方差比只有 **0.70**（GPU 方差低 30%）。
2. **破坏守恒**：独立目的地噪声的和一般不等于出境总量；CPU 总人口严格守恒（std = 0），该近似总人口 std ≈ 55（总量约 6862）。
3. **协方差结构错误**：CPU 的协方差是逐源块状的负相关（§2.4），独立噪声给出的是对角协方差。

所以“卷积 + 独立目的地噪声”既不是好的近似，也不是良定义的模型。

---

## 4. 建议方向：保留卷积的随机实现方法

### 4.0 目标与关键观察

目标：在保留 FFT/卷积的前提下，尽量接近原设计的 Beta（源端）+ Dirichlet（目的地）。

关键观察：Dirichlet 对最终空间场的主要影响体现在**二阶矩**（方差/协方差）；二阶矩是卷积可表示的，只有逐边的精确分布不可卷积化。因此可以在 FFT 框架内做到“二阶矩等价”，或用局部重采样近似非高斯离散。

基线 **M0（上一版）**：源端抽 $O_s$（Beta），再按固定权重 $w_{s,d}$ **确定性铺开**。它保留源端方差、均值、总量守恒，但丢掉行内多项离散（§2.4 中的 $-\rho^2\sum_s N_s w_{s,d}w_{s,d'}$ 与方差里的 $-\rho^2\sum_s N_s w_{s,d}^2$）。下面 M1–M4 都保留卷积，区别在于补回哪一部分随机性。

### 4.1 M1 — 协方差匹配的 Gaussian 随机场（spectral synthesis）

**思路**：不逐边抽样，直接构造零均值随机场 $\eta_d$，使其协方差等于 CPU 的协方差（在平移不变近似下）。

**目标协方差**：在内部、$N_s\approx N$ 近似均匀时，由 §2.4 可把 $Y$ 的涨落协方差写成只依赖位移 $e=d'-d$ 的核

$$
C(e)=[N\rho(1-\rho)+\mu]\,\delta_{e,0}-2N\rho(1-\rho)\,w(e)-\rho^2N\,(w\star w)(e),
$$

其中 $w(e)=w_{s,s+e}$，$(w\star w)(e)=\sum_u w(u)\,w(u+e)$ 是核的自相关，$\mu=\rho N$（内部列和为 1），$\delta_{e,0}$ 是同点指示。三项依次对应：保留量/泊松方差、保留-外流共享 $O_d$ 的负相关、同一源多项分配的负相关。

**谱密度**：设 $\hat w(\omega)$ 为 $w$ 的频谱，则

$$
S(\omega)=[N\rho(1-\rho)+\mu]-2N\rho(1-\rho)\,\hat w(\omega)-\rho^2N\,|\hat w(\omega)|^2.
$$

$C$ 是合法协方差，故 $S(\omega)\ge 0$（离散化后负值截断到 0）。又因 $\hat w(0)=1$，

$$
S(0)=N\rho(1-\rho)+\rho N-2N\rho(1-\rho)-\rho^2N=0,
$$

即**总量守恒**（均匀模无涨落）。

**合成算法**（一次额外 FFT）：

1. 生成白噪声 $\zeta_d\sim\mathcal N(0,1)$；
2. 频域滤波：$\eta=\mathrm{IFFT}\big(\sqrt{S(\omega)}\,\mathrm{FFT}(\zeta)\big)$，取实部；
3. 合成：$Y_d=(1-\rho)N_d+\mu_d+\eta_d$。

边界/有限域用循环嵌入或零填充；非均匀 $N_d$ 用局部调制 $\eta_d\leftarrow\sqrt{N_d/N}\,\eta_d$（线性噪声近似，需再做零均值修正以保持守恒）。

**性质**：均值、方差、协方差（含 Dirichlet 负相关）在二阶矩意义下匹配；$O(n\log n)$。代价：边际是 Gaussian（非 Gamma/Dirichlet），小计数可能为负（需截断或边际变换）。它是 M1–M4 中**二阶矩最接近**的方案。

### 4.2 M2 — 局部/分块多项（block multinomial）

**思路**：用 FFT 得均值场后，在**局部块**内做多项重采样，补回非高斯的行内竞争。

**算法**：

1. FFT 得 $\mu_d=\rho\sum_s N_s w_{s,d}$；
2. 把域划成大小约等于核半径 $k$ 的块 $\mathcal B_b$；
3. 抽总出境量 $T=\sum_s O_s$（$O_s$ 仍用 Beta），按块期望质量 $M_b=\sum_{d\in\mathcal B_b}\mu_d$ 分配 $T$：$\{T_b\}\sim\mathrm{Multinomial}(T,\{M_b/\sum_{b'}M_{b'}\})$；
4. 每块内按 $\mu_d/M_b$ 做多次项分布，得到目的地计数；
5. 合成 $Y_d=(1-\rho)N_d+\text{count}_d$。

**性质**：非高斯、**局部负相关**（同块目的地竞争）、总量守恒；块大小控制局部性。块不重叠，而 CPU 的 Dirichlet 行以每个源为中心、相互重叠，所以是近似，且块边界可能有伪影。

### 4.3 M3 — 全局多项（global multinomial）

**思路**：最简单地补回负相关——所有目的地共享一个总量。

**算法**：

1. FFT 得 $\mu_d$；
2. 抽 $T=\sum_s O_s$；
3. 全局 $\mathrm{Multinomial}(T,\{\mu_d/\sum_{d'}\mu_{d'}\})$；
4. 合成 $Y_d=(1-\rho)N_d+\text{count}_d$。

**性质**：非高斯、总量守恒、负相关；但负相关是**全局**的（空间尺度过大），与 CPU 的局部行内竞争不符，适合作为基线/下限。

### 4.4 M4 — Gaussian copula + 边际变换

**思路**：在 M1 的基础上修正边际，使其更接近 Gamma/Dirichlet 的单点分布。

**算法**：

1. 用 M1 生成相关 Gaussian 场 $z_d$；
2. 对每个 $d$ 做分位变换 $Y_d=Q_d\big(\Phi(z_d)\big)$，其中 $\Phi$ 是标准正态 CDF，$Q_d$ 是目标边际分位函数（Gamma 近似，或小规模 CPU 模拟的经验分位）；
3. 守恒修正（减均值/按总量缩放）。

**性质**：匹配边际 + 秩相关，非负；仍不是精确联合分布；守恒需额外修正，实现比 M1 复杂。

### 4.5 方法对比与推荐

| 方法 | 保留源端 Beta | 复现 Dirichlet 二阶矩 | 非高斯局部离散 | 总量守恒 | 复杂度 | 与 CPU 的距离 |
|---|---|---|---|---|---|---|
| M0 源端+确定性 | ✓ | ✗ | ✗ | ✓ | $O(n\log n)$ | 较大 |
| **M1 协方差匹配 Gaussian** | ✓ | **✓** | ✗ | ✓（$S(0)=0$） | $O(n\log n)$ | **二阶最近** |
| M2 分块多项 | ✓ | 近似 | ✓（局部） | ✓ | $O(n\log n+n)$ | 局部更真 |
| M3 全局多项 | ✓ | 近似（全局） | ✓ | ✓ | $O(n\log n+n)$ | 尺度不符 |
| M4 copula+边际变换 | ✓ | ✓ | 边缘非高斯 | 需修正 | $O(n\log n)$ | 边际更真 |
| （参考）精确稀疏 scatter | ✓ | ✓（精确） | ✓ | ✓ | $O(\mathrm{nnz})$ | 精确 |

**推荐**：

- **首选 M1**：不改卷积结构，把 CPU 的完整二阶矩（含 Dirichlet 负相关）补回，且 $S(0)=0$ 自动守恒。
- 需要非高斯局部离散时用 **M2**（块大小 ≈ 核半径）；**M4** 用于边际精修；**M3** 作基线。
- 精确对齐 CPU 只能回到稀疏 scatter（参考，不作为大核方案）。

**验证（V3）**：小规模 CPU CSR vs 各方法，比较 per-deme 均值、方差比、空间协方差、总量 std，以及 chasing 统计（覆盖/前沿/共存时长）；重点看 M1 的方差比是否从 0.70 收敛到 ≈1、协方差符号/量级是否匹配。

## 5. 一页总结

1. **CPU 随机迁移**：逐源抽 $O_s\sim\mathrm{Beta}(\rho(N-1),(1-\rho)(N-1))$（连续二项），再逐源抽 $G_{s,d}\sim\mathrm{Gamma}(w_{s,d}(O_s-1))$ 并归一化到 $O_s$（连续多项 / Dirichlet）；逐源守恒、全局守恒，均值/方差/协方差有闭式。
2. **GPU route A**：归一化卷积（$g=\rho f/Z$，$K'\circledast g$）+ cuFFT，逐格 kernel + 逐平面流水线；**精确给出均值**，是固定、平移不变的线性算子。
3. **复刻阻碍**：CPU 的 $X_{s,d}=O_sG_{s,d}/S_s$ 是**逐源、随机、非平移不变**的归一化；卷积把源**求和聚合**、丢掉源身份与随机 $S_s$；随机自由度是 $O(\mathrm{nnz})$，FFT 无法在 $O(n\log n)$ 内生成。**同步不是障碍，表示方式才是**。
4. **建议方向（保留卷积）**：M0 = 源端 Beta + 确定性铺开（基线）；M1 = 协方差匹配 Gaussian（二阶矩最接近，$S(0)=0$ 守恒）；M2 = 分块多项（非高斯局部离散）；M3 = 全局多项（基线）；M4 = copula + 边际变换（边际精修）。首选 M1，按需叠加 M2/M4；精确对齐 CPU 仍需稀疏 scatter。

---

## 附录 A：符号速查

| 符号 | 含义 |
|---|---|
| `n` | deme 数 |
| `s, d` | 源 / 目的地下标 |
| `f_s, N_s` | 源计数 |
| `ρ` | 迁移率 |
| `k(u), K'` | 高斯核 / 去中心核 |
| `valid(s)` | 源的有效位移集合 |
| `Z_s` | 有效权重和（确定性边界归一化） |
| `w_{s,d}` | 归一化权重 $K'(d-s)/Z_s$，$\sum_d w_{s,d}=1$ |
| `O_s` | 出境量（随机） |
| `R_s` | 保留量 $N_s-O_s$ |
| `G_{s,d}, S_s` | 逐目的地 Gamma 及其和 |
| `X_{s,d}` | 实际分配量 |
| `Y_d` | 目的地迁移后计数 |
| `μ_d` | 期望流入 $\rho\sum_s N_s w_{s,d}$ |
| `ξ_s` | 源端涨落 $O_s-\rho N_s$ |

## 附录 B：natal 代码位置（供核对）

- CPU 随机迁移：`rust/src/kernels/spatial.rs` 的 `migrate_csr_stochastic_rngs`（逐源主循环）、`sample_outbound`（出境）、`distribute_csr_outbound`（逐行分配）。
- 连续分布：`rust/src/kernels/rng.rs` 的 `continuous_binomial`、`continuous_multinomial`、`gamma`。
- GPU 随机 CSR：`rust/src/gpu/executor.rs` 的 `migrate_tick_stochastic`（受 `MAX_CSR_ROW` 限制）。
- GPU route A 卷积：`rust/src/gpu/fft_migrate.rs`（`scatter_g` / `complex_mul` / `combine` / cuFFT 计划 / 逐平面 `migrate_inplace`）。
