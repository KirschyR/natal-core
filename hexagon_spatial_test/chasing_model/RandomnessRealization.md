## 0. 背景

### 0.1 生物学对象

我们模拟一个**空间结构化的、年龄结构的蚊虫种群**，并研究**基因驱动（gene drive）**在其中的传播。把一片栖息地划分成许多个相邻的小区（下面叫 **deme**，即“种群格点”），每个 deme 内记录不同**年龄**、**性别**、**基因型**的个体数量。

- **网格**：六边形（hex）网格。每个 deme 是一个格点，格点之间可以迁移（扩散）。
- **一 tick = 一周**：每个时间步内，所有 deme 各自完成一轮生命周期，然后个体在 deme 之间迁移。
- **年龄结构**（本项目的蚊虫模型）：2 个幼期 + 5 个雄成虫期 + 8 个雌成虫期。
- **基因型**（homing 驱动，一个基因座）：论文用 5 种 $\{dd, dw, ww, rw, dr\}$，其中 $ww$ 是野生型，$dd$ 是驱动纯合子。natal 引擎里也可用 3 种（只有 $WT$、$Drive$ 两个等位基因）或 6 种（再加抗性等位 $Res$）。
- **一个 tick 内的顺序**：natal 引擎固定为 R-S-A → migration：
  1. **R（繁殖，reproduction）**：成虫产生新生个体；
  2. **S（密度调节 + 存活，survival）**：幼期受密度依赖调节，再乘存活率；
  3. **A（老化，aging）**：年龄类整体前移；
  4. **migration（迁移）**：个体在本 tick 末尾扩散到邻近 deme。

### 0.2 为什么关心迁移与 GPU

- 迁移使用一个**高斯核**：每个 deme 的个体按距离权重扩散到周围 deme。核越大，扩散范围越广。
- 论文/实际场景常用**很大的核**（几十个格点宽）。若把核展开成稀疏矩阵（CSR），内存和构建成本会爆炸。
- 为此 natal 提供了 **route A**：把迁移写成**归一化卷积**，用 **cuFFT（GPU 快速傅里叶变换）** 逐平面计算，复杂度从 $O(\mathrm{nnz})$（非零边数）降到 $O(n\log n)$（$n$ = 格点数）。
- 问题：route A 目前只实现了确定性迁移。若要在 GPU 上做随机迁移，就需要回答：**卷积能不能承载随机性？** 本文说明：**不能精确承载 natal 原有的逐边随机迁移**，并给出一个 FFT 原生的替代方案（噪声场卷积）。

---

## 1. 记号与设定

本节集中说明后文用到的符号；每个符号在第一次出现时给出含义，后文沿用。

后文只关注一个**平面**：固定性别、年龄、基因型后得到的一张空间计数图；迁移对每个平面独立进行，因此所有公式对单个平面成立。用 $n$ 表示该平面的 deme 总数（$n=\text{rows}\times\text{cols}$）。用 $s$ 和 $d$ 分别表示**源**和**目的地** deme 的下标；$f_s$ 是源 $s$ 在该平面上的个体数（也记作 $N_s$）。用 $\rho\in[0,1]$ 表示**迁移率**：本 tick 中每个源有比例 $\rho$ 的个体参与迁移。

用 $u=d-s$ 表示从源到目的地的**位移**。$k(u)$ 是未归一化的高斯**核权重**，只依赖 $u$；去中心核 $K'$ 定义为 $K'(u)=k(u)$（$u\ne 0$）、$K'(0)=0$——中心排除意味着“留在原地”不经过迁移核，而由 $1-\rho$ 保留。从源 $s$ 出发、落点仍在网格内的位移集合记作 $\mathrm{valid}(s)$；$Z_s=\sum_{u\in\mathrm{valid}(s)}K'(u)$ 是该源的有效权重和；归一化权重为 $w_{s,d}=K'(d-s)/Z_s$（$d-s\notin\mathrm{valid}(s)$ 时取 0）。

核边长记作 $\kappa$（奇数，例如 7 或 59）；源 $s$ 的有效目的地个数记作 $r_s$（CSR 行宽），内部 $r_s=\kappa^2-1$，边界更少；非零边总数 $\mathrm{nnz}=\sum_s r_s$。

随机量：$O_s$ 是源 $s$ 的**出境量**，$R_s=N_s-O_s$ 是**保留量**；$X_{s,d}$ 是源 $s$ 实际分配到目的地 $d$ 的量；$Y_d$ 是目的地 $d$ 迁移后的计数；$G_{s,d}$ 是逐目的地的 Gamma 中间量，$S_s=\sum_d G_{s,d}$。

由定义，每行权重归一：$\sum_d w_{s,d}=1$、$w_{s,s}=0$（中心排除）。$w_{s,d}$ 是确定性的（只依赖几何与核），而 $O_s$、$G_{s,d}$ 是随机的。

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

## 3. 卷积如何工作

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

这里的“只依赖 $d-s$”指：原始权重是**位移向量** $u=d-s$ 的函数 $K'(u)$——把源和目的地**同时平移同一个向量**，权重不变，这就是平移不变的含义；它不是“任意 $d,s$ 配对都一样”。注意归一化权重 $w_{s,d}=K'(d-s)/Z_s$ 还依赖 $s$（通过边界归一化 $Z_s$），但 $Z_s$ 是确定性的，route A 通过“先逐点除以 $Z_s$，再卷积”精确处理它。

### 3.2 并行实现

对一个平面，GPU 上分五步（`fft_migrate.rs`）：

1. **`scatter_g`（逐格并行）**：把 $f_s$ 除以 $Z_s$、乘 $\rho$，零填充到 $(\text{rows}+\kappa-1)\times(\text{cols}+\kappa-1)$。每个格点一个线程。
2. **cuFFT `r2c`**：实数平面 → 频域（cuFFT 内部并行）。
3. **`complex_mul`（逐频点并行）**：频域逐点乘核的频谱（并折叠 $1/(f_rf_c)$）。
4. **cuFFT `c2r`**：频域 → 实数域。
5. **`combine`（逐格并行）**：$out_d=(1-\rho)f_d+\text{conv}_d$。

多个平面（$\text{sex}\times\text{age}\times\text{genotype}$，共 $P=2AZ+AZ^2$ 个）由 host 循环依次提交到同一 CUDA stream。每个平面的复杂度从 CSR 的 $O(\mathrm{nnz})=O(\sum_s r_s)$ 降到 $O(n\log n)$：这里 $n=\text{rows}\times\text{cols}$ 是该平面的 deme 数，$r_s$ 是源 $s$ 的有效目的地数（CSR 行宽），内部 $r_s\le\kappa^2-1$（$\kappa$ 为核边长）。若近似 $r_s\approx\kappa^2$，CSR 即 $O(n\kappa^2)$；总开销为 $O(P\,n\log n)$。核越大、域越大，FFT 优势越明显。

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
“平移不变”指权重只随**位移** $u=d-s$ 变化：$K'(u)$ 与源的绝对位置无关，把源、目的地同时平移同一向量时权重不变”。随机权重 $G_{s,d}/S_s$ 额外依赖源 `s` 的整行（通过随机归一化 $S_s$），因此**不是 $u=d-s$ 的函数**，无法写成一次固定卷积。（边界归一化 $Z_s$ 也依赖 $s$，但它是确定性的，可先逐点除以 $Z_s$ 再卷积，所以不构成阻碍。）

**阻碍 4：随机自由度是 $O(\mathrm{nnz})$。**
每个源有 $r_s$ 个目的地，对应约 $r_s-1$ 个独立 Dirichlet 比例自由度；全体源共 $\sum_s(r_s-1)=O(\mathrm{nnz})$（$\mathrm{nnz}=\sum_s r_s$）。FFT 是 $O(n\log n)$ 的确定性线性变换，无法在 $O(n\log n)$ 内生成这么多独立随机量。

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

**M1 近似的是什么。** 把 CPU 的随机涨落 $\eta_d=Y_d-\mathbb{E}[Y_d]$ 近似为**零均值 Gaussian 随机场**，并让它的**协方差**等于 CPU 涨落的协方差（在内部、$N_s\approx N$ 的平移不变近似下）。因此：

- **均值精确**：$\mathbb{E}[Y_d]=(1-\rho)N_d+\mu_d$；
- **方差/协方差匹配**：包括 Dirichlet 带来的行内负相关；
- **不匹配**：边际分布（Gaussian 而非 Gamma/Dirichlet）与更高阶矩；边界/非均匀 $N_d$ 用近似处理。

一句话：M1 不逐边抽样，而是直接“合成”一个具有正确二阶矩的相关噪声场。

**为什么可行。** 由 §2.4，$Y$ 的涨落协方差只依赖 $N_s,\rho,w$；Dirichlet 的影响已经编码在协方差里。二阶矩是**卷积可表示**的（只涉及 $w$ 与 $w\star w$），所以能用 FFT 合成；只有逐边的精确分布不可卷积化。

**目标协方差。** 在内部、$N_s\approx N$ 近似均匀时，把涨落协方差写成只依赖位移 $e=d'-d$ 的核：

$$
C(e)=[N\rho(1-\rho)+\mu]\,\delta_{e,0}-2N\rho(1-\rho)\,w(e)-\rho^2N\,(w\star w)(e),
$$

其中 $w(e)=w_{s,s+e}$，$(w\star w)(e)=\sum_u w(u)\,w(u+e)$，$\mu=\rho N$（内部列和为 1），$\delta_{e,0}$ 是同点指示。三项依次是：保留量/泊松方差、保留-外流共享 $O_d$ 的负相关、同一源多项分配的负相关。

**谱密度与守恒。** 设 $\hat w(\omega)$ 为 $w$ 在（补零）网格上的离散傅里叶变换，则 $C$ 的谱密度为

$$
S(\omega)=[N\rho(1-\rho)+\mu]-2N\rho(1-\rho)\,\hat w(\omega)-\rho^2N\,|\hat w(\omega)|^2.
$$

$C$ 是合法协方差，故 $S(\omega)\ge 0$（离散化后负值截断到 0）。又因 $\hat w(0)=\sum_e w(e)=1$，

$$
S(0)=N\rho(1-\rho)+\rho N-2N\rho(1-\rho)-\rho^2N=0,
$$

即均匀模没有涨落 → **总量守恒**。实践上：先在滞后网格上构造 $w(e)$ 与 $(w\star w)(e)$（后者一次 FFT），再对 $C(e)$ 做 FFT 得 $S(\omega)$；或直接用上式。

**合成算法**（每个平面一次额外 FFT）：

1. 生成白噪声 $\zeta_d\sim\mathcal N(0,1)$（与 $\eta$ 同尺寸）；
2. 频域滤波：$\eta=\mathrm{IFFT}\big(\sqrt{S(\omega)}\odot\mathrm{FFT}(\zeta)\big)$，取实部（$\odot$ 为逐点乘）；
3. 合成：$Y_d=(1-\rho)N_d+\mu_d+\eta_d$。

有限域用循环嵌入/零填充避免环绕；非均匀 $N_d$ 用局部调制 $\eta_d\leftarrow\sqrt{N_d/N}\,\eta_d$（线性噪声近似），再做零均值修正以保持守恒。

**性质。** 均值、方差、协方差（含 Dirichlet 负相关）在二阶矩意义下匹配；每平面 $O(n\log n)$（$n$ 为该平面 deme 数）。代价：边际是 Gaussian，小计数可能为负（需截断或 M4 的边际变换）。它是 M1–M4 中**二阶矩最接近** CPU 的方案。

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
3. **复刻阻碍**：CPU 的 $X_{s,d}=O_sG_{s,d}/S_s$ 是**逐源、随机、非平移不变**的归一化；卷积把源**求和聚合**、丢掉源身份与随机 $S_s$；随机自由度是 $O(\mathrm{nnz})$，FFT 无法在 $O(n\log n)$ 内生成。
4. **建议方向（保留卷积）**：M0 = 源端 Beta + 确定性铺开（基线）；M1 = 协方差匹配 Gaussian（二阶矩最接近，$S(0)=0$ 守恒）；M2 = 分块多项（非高斯局部离散）；M3 = 全局多项（基线）；M4 = copula + 边际变换（边际精修）。首选 M1，按需叠加 M2/M4；精确对齐 CPU 仍需稀疏 scatter。

---

## 附录 C：设计依据

> 本附录单独回答一个设计问题：为什么先抽“走多少人”（二项 / Beta），再对目的地做一次 Gamma 归一化（多项 / Dirichlet）？它是 §2 的补充依据，不改变正文结构。

### C.1 个体层面的两层随机决策

- **第一层（出境量）**：每个个体独立决定“走不走”（概率 $\rho$）→ 走的总人数 $O_s$ 服从二项分布。
- **第二层（目的地）**：给定有 $O_s$ 个个体要走，**每个移动个体再独立选择目的地**（概率 $w_{s,d}$）→ 目的地计数服从多项分布。

Gamma 那一步不是额外处理，而是“给定出境量后，目的地选择仍然随机”的连续实现。

### C.2 离散依据：多项分布的链式分解

设一个源有 $N_s$ 个个体，每个个体有 $k+1$ 种结局：

$$
\text{留下：概率 }1-\rho;\qquad
\text{去目的地 }d:\text{概率 }\rho w_{s,d},
$$

且概率之和 $(1-\rho)+\rho\sum_d w_{s,d}=1$。于是联合计数

$$
(R_s,\,X_{s,1},\dots,X_{s,k})
\sim \mathrm{Multinomial}\big(N_s;\;1-\rho,\;\rho w_{s,1},\dots,\rho w_{s,k}\big).
$$

按概率的链式法则拆成两步：

$$
O_s:=N_s-R_s\sim\mathrm{Binomial}(N_s,\rho),
$$

$$
(X_{s,1},\dots,X_{s,k})\mid O_s
\sim\mathrm{Multinomial}\big(O_s;\;w_{s,1},\dots,w_{s,k}\big).
$$

所以“先抽走多少人、再抽去哪”**不是近似，而是离散模型的精确等价写法**。两步对应两种不同的随机性来源；由全方差公式，

$$
\mathrm{Var}[X_{s,d}]
=\underbrace{\mathbb{E}\big[\mathrm{Var}(X_{s,d}\mid O_s)\big]}_{\text{目的地选择}}
+\underbrace{\mathrm{Var}\big(\mathbb{E}[X_{s,d}\mid O_s]\big)}_{\text{出境量}}.
$$

### C.3 连续依据：逐阶段矩匹配

natal 的计数是连续实数（f64），所以用离散分布的连续版本，原则是**逐阶段匹配前两阶矩**。

**第一层**：离散 $\mathrm{Binomial}(N_s,\rho)$ 的均值为 $N_s\rho$、方差为 $N_s\rho(1-\rho)$。取

$$
O_s=N_s B_s,\qquad
B_s\sim\mathrm{Beta}\big(\rho(N_s-1),(1-\rho)(N_s-1)\big),
$$

则 $\mathbb{E}[O_s]=N_s\rho$、$\mathrm{Var}[O_s]=N_s\rho(1-\rho)$，且 $O_s\in[0,N_s]$。参数里的 $N_s-1$ 正是为了让 Beta 方差分母中的 $(\alpha+\beta+1)$ 恰好等于 $N_s$，从而精确匹配二项方差。

**第二层**：离散 $\mathrm{Multinomial}(O_s,w_{s,\cdot})$ 的均值为 $O_s w_{s,d}$、方差为 $O_s w_{s,d}(1-w_{s,d})$、协方差为 $-O_s w_{s,d}w_{s,d'}$。取

$$
G_{s,d}\sim\mathrm{Gamma}\big(w_{s,d}(O_s-1),1\big),\qquad
S_s=\sum_d G_{s,d},\qquad
X_{s,d}=O_s\frac{G_{s,d}}{S_s}.
$$

独立 Gamma 归一化即 **Dirichlet 分布**的标准构造；浓度 $w_{s,d}(O_s-1)$ 使均值/方差/协方差逐项等于多项分布，同时 $\sum_d X_{s,d}=O_s$ 严格成立。参数里的 $O_s-1$ 同样是为了方差匹配。

### C.4 一个最小数值例子

取 $N_s=10$、$\rho=0.5$、两个目的地 $w_{s,1}=0.7$、$w_{s,2}=0.3$。联合多项概率为：留下 $0.5$、去 1 号 $0.35$、去 2 号 $0.15$。

- 第一步：$O_s\sim\mathrm{Binomial}(10,0.5)$，$\mathbb{E}[O_s]=5$，$\mathrm{Var}[O_s]=2.5$。
- 第二步：给定 $O_s$，$(X_{s,1},X_{s,2})\sim\mathrm{Multinomial}(O_s;(0.7,0.3))$。

于是

$$
\mathbb{E}[X_{s,1}]=0.7\cdot5=3.5,\qquad
\mathrm{Var}[X_{s,1}]=10\cdot0.35\cdot0.65=2.275,
$$

与直接对 $(R_s,X_{s,1},X_{s,2})$ 做多项分布的 $Np_1(1-p_1)$ 一致；协方差

$$
\mathrm{Cov}[X_{s,1},X_{s,2}]=-10\cdot0.35\cdot0.15=-0.525
$$

也与两阶段公式 $-\rho^2N_s w_{s,1}w_{s,2}$ 一致。可见两阶段拆分**不改变前两阶矩**，只是把联合抽样拆成两个更自然、更适配 CSR 行结构的步骤。

### C.5 与单阶段 Dirichlet 的关系（设计选择的边界）

若直接对 $(R_s,X_{s,1},\dots,X_{s,k})$ 做**单阶段**连续化，可用

$$
N_s\cdot\mathrm{Dirichlet}\big((N_s-1)(1-\rho),\;(N_s-1)\rho w_{s,1},\dots\big),
$$

它同样匹配多项分布的均值/协方差。可以验证：**两阶段 Beta+Dirichlet 与这个单阶段 Dirichlet 的无条件前两阶矩完全相同**（$\mathbb{E}$、$\mathrm{Var}$、$\mathrm{Cov}$ 都对得上）。

差别在**给定 $O_s$ 之后的条件分布**：

- 两阶段用浓度 $O_s-1$（对应离散链式法则里“用已实现的出境量作为试验数”）；
- 单阶段 Dirichlet 的条件浓度是 $\rho(N_s-1)$（先验移动强度），不是 $O_s-1$。

所以 natal 选两阶段，依据是：

1. **机制上**对应两个独立的个体决策（走不走、去哪）；
2. **数学上**是多项分布的链式分解；
3. **连续化上**逐阶段矩匹配，且无条件前两阶矩与单阶段 Dirichlet 一致；
4. **工程上**把“迁移率”与“核权重”解耦，天然适配 CSR 行、逐源守恒与实数计数。

它不是某个唯一数学极限，但它是**有明确依据、与离散个体模型逐阶段对应**的连续松弛。

***

几个时间指标：

1. **wall_seconds**：该策略所有 R 次重复 × 全部 tick 的墙钟总时间。包含 lifecycle（繁殖/存活/老化）+ migration + 记录/开销。测量范围是整段 replicate 循环。
2. **per_replicate_seconds**：wall_seconds / R，即一次完整模拟（全部 tick）的平均耗时。
3. **migration_seconds**：只统计迁移算子内部的时间（每个 tick 的 migration 调用），跨所有 tick 和重复累加；不含繁殖/存活/老化/记录。wall ≈ lifecycle + migration + 记录开销。

## 附录 D：用 C2ST 检验两种策略是否「可混淆」

### D.0 符号表

| 符号 | 含义 |
|---|---|
| \(A,B\) | 两种待比较的随机迁移策略 |
| \(R\) | 每种策略的独立重复次数 |
| \(q\) | 特征向量维数 |
| \(\phi_i\in\mathbb R^q\) | 第 \(i\) 次重复的特征向量 |
| \(\mathbb R^q\) | \(q\) 维实向量空间 |
| \(\phi_i^A,\phi_j^B\) | 策略 \(A\) 的第 \(i\) 个、\(B\) 的第 \(j\) 个特征向量 |
| \(T\) | 一次模拟的总 tick 数 |
| \(t\) | tick 指标 |
| \(g\) | deme 指标 |
| \(\tau\) | 覆盖阈值 |
| \(\text{drive}_g(t)\) | tick \(t\)、deme \(g\) 的 drive fraction |
| \(\text{coverage}_\tau(t)\) | 阈值 \(\tau\) 下被覆盖的 deme 数 |
| \(\text{front}(t)\) | tick \(t\) 的前沿半径 |
| \(\text{total}(t)\) | tick \(t\) 的非卵总量 |
| \(\overline{\text{drive}}_g\) | deme \(g\) 的 drive fraction 时间均值 |
| \(\mathrm{SD}_t(\text{drive}_g)\) | 其时间标准差 |
| \(T_{50}\) | 首次达到 0.5 的 tick |
| \(X_A,X_B\) | 两组特征样本（各 \(R\) 个） |
| \(X\) | 池化样本 \(X_A\cup X_B\) |
| \(i,j\) | 样本 / 重复的下标 |
| \(y_i\in\{0,1\}\) | 样本 \(i\) 的方法标签（0=A，1=B） |
| \(N_{\mathrm{tot}}\) | 池化样本数 \(N_{\mathrm{tot}}=2R\) |
| \(H_0,H_1\) | 零假设 / 备择假设 |
| \(\mathcal F_{\mathrm{tr}},\mathcal F_{\mathrm{va}}\) | 训练 / 验证折的样本下标集合 |
| \(N_{\mathrm{tr}}\) | 训练样本数 \(N_{\mathrm{tr}}=|\mathcal F_{\mathrm{tr}}|\) |
| \(\#\{\cdot\},\ |\cdot|\) | 集合元素个数 |
| \(\hat\mu_{\mathrm{tr}},\hat\sigma_{\mathrm{tr}}\) | 训练折的均值 / 标准差 |
| \(\tilde\phi\) | 标准化后的特征 |
| \(w,\beta\) | 逻辑回归权重向量 / 截距 |
| \(\cdot^\top\) | 向量 / 矩阵转置 |
| \(\|w\|^2\) | 权重向量的 L2 范数平方 |
| \(\mathcal L\) | 逻辑回归损失函数 |
| \(p_i\) | 模型给样本 \(i\) 的 \(y=1\) 概率 |
| \(\mathbb P(\cdot)\) | 概率 |
| \(\lambda\) | L2 正则强度 |
| \(\hat y_i\) | 预测标签 |
| \(\mathbb 1[\cdot]\) | 指示函数（条件成立为 1，否则 0） |
| \(\mathrm{Acc}\) | 交叉验证准确率 |
| \(\mathrm{AUC}\) | ROC 曲线下面积 |
| \(K\) | 交叉验证折数 |
| \(C\) | 交叉验证重复次数 |
| \(B\) | 置换次数 |
| \(b\) | 置换指标 |
| \(\mathrm{Acc}^{(b)}\) | 第 \(b\) 次置换的准确率 |
| \(p_{\mathrm{perm}}\) | 置换 p 值 |
| \(p_{\min}\) | 给定 \(B\) 的最小置换 p，\(p_{\min}=1/(1+B)\) |
| \(\mathcal S_1,\mathcal S_2\) | 同种 self-split 的两个重复子集 |
| \(\delta\) | 等价边界（容差） |
| \(\mathrm{MMD}^2\) | 平方最大均值差异 |
| \(k(\cdot,\cdot)\) | 核函数 |
| \(\mathcal H\) | 再生核 Hilbert 空间（RKHS） |
| \(\|\cdot\|_{\mathcal H}\) | RKHS 范数 |

### D.1 问题与零假设

策略 \(A,B\) 各做 \(R\) 次独立重复，第 \(i\) 次重复压缩为特征向量 \(\phi_i\)。去掉方法标签后，问两组样本能否区分：

$$
H_0:\ \{\phi_i^A\}_{i=1}^R\ \text{与}\ \{\phi_j^B\}_{j=1}^R\ \text{同分布（可混淆）}
\qquad\text{vs}\qquad
H_1:\ \text{分布不同}.
$$

检验统计量取「分类器预测方法标签」的交叉验证准确率 \(\mathrm{Acc}\)：\(H_0\) 下应接近 \(0.5\)，越大越可分。这就是 **classifier two-sample test (C2ST)**。

### D.2 每次重复的特征向量

取 tick \(t\in\{T/4,T/2,3T/4,T\}\)（\(T\) 为总 tick 数）：

$$
\phi_i=\Big(
\{\text{drive}_g(t)\}_{g,t},\
\{\text{coverage}_\tau(t)\}_{\tau,t},\
\{\text{front}(t)\}_t,\
\{\text{total}(t)\}_t,\
\{\overline{\text{drive}}_g\}_g,\
\{\mathrm{SD}_t(\text{drive}_g)\}_g,\
T_{50}
\Big).
$$

其中 \(g\) 遍历选定 deme、\(\tau\) 遍历覆盖阈值；\(\{\cdot\}_{g,t}\) 表示把对应下标的量拼成向量。各项含义见 D.0。对高维场可先用 PCA/FPCA，或改用 D.7 的核两样本检验。

### D.3 C2ST 的具体步骤

1. **池化与标签**：\(X=X_A\cup X_B\)，\(y_i=0\)（\(A\)）或 \(1\)（\(B\)），共 \(N_{\mathrm{tot}}=2R\) 个样本。
2. **折内标准化**：用训练折 \(\mathcal F_{\mathrm{tr}}\) 估计 \(\hat\mu_{\mathrm{tr}},\hat\sigma_{\mathrm{tr}}\)，得标准化特征
   $$
   \tilde\phi=(\phi-\hat\mu_{\mathrm{tr}})/\hat\sigma_{\mathrm{tr}}.
   $$
3. **分类器**：L2 逻辑回归
   $$
   \mathbb P(y=1\mid\phi)=\frac{1}{1+e^{-(w^\top\tilde\phi+\beta)}},
   $$
   最小化
   $$
   \mathcal L(w,\beta)=-\frac1{N_{\mathrm{tr}}}\sum_{i\in\mathcal F_{\mathrm{tr}}}
   \big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]+\frac{\lambda}{2}\|w\|^2,
   $$
   其中 \(p_i=1/(1+e^{-(w^\top\tilde\phi_i+\beta)})\)，\(N_{\mathrm{tr}}=|\mathcal F_{\mathrm{tr}}|\)，\(\lambda\) 为 L2 强度。
4. **交叉验证**：重复 \(C\) 次分层 \(K\)-折；每折在 \(\mathcal F_{\mathrm{tr}}\) 上拟合、在 \(\mathcal F_{\mathrm{va}}\) 上预测，累计
   $$
   \mathrm{Acc}=\frac1{N_{\mathrm{tot}}}\sum_{i=1}^{N_{\mathrm{tot}}}\mathbb 1[\hat y_i=y_i],\qquad
   \mathrm{AUC}=\mathbb P\big(p_i>p_j\mid y_i=1,\ y_j=0\big),
   $$
   其中 \(\hat y_i\) 为预测标签，\(p_i\) 为预测概率。
5. **报告**：\(\mathrm{Acc}\)、\(\mathrm{AUC}\) 的交叉验证均值；越接近 \(0.5\) 越不可分。

### D.4 置换零分布与 \(p_{\mathrm{perm}}\) 值

在 \(H_0\) 下方法标签可交换。重复 \(B\) 次：随机打乱 \(y\)，重跑 D.3，得零分布 \(\{\mathrm{Acc}^{(b)}\}_{b=1}^B\)，则

$$
p_{\mathrm{perm}}=\frac{1+\#\{b:\mathrm{Acc}^{(b)}\ge \mathrm{Acc}\}}{1+B}.
$$

- \(B=20\) 时 \(p_{\min}=1/21\approx0.048\)；若所有置换都低于观测值，只能报 \(p_{\mathrm{perm}}\le0.048\)。要更细的 \(p_{\mathrm{perm}}\) 需增大 \(B\)（如 \(B=999\Rightarrow p_{\min}=0.001\)）。
- \(p_{\mathrm{perm}}\) 只表示「可检测性」，不是效应量或生物学重要性；\(\mathrm{Acc}/\mathrm{AUC}\) 才表示差异大小。

### D.5 对角线自校准（self-split）

把同一方法的 \(R\) 次重复随机分成两个子集 \(\mathcal S_1,\mathcal S_2\)，再跑一次 C2ST：

$$
\text{self-split}(A)=\mathrm{C2ST}\big(\{\phi_i^A\}_{i\in\mathcal S_1},\ \{\phi_i^A\}_{i\in\mathcal S_2}\big)\approx0.5.
$$

它检验实现本身是否引入偏置（本项目 small \(0.458/0.432/0.538\)，large \(0.426/0.568/0.522\)）。下三角热图的对角线取此值，而不是人为置 \(1\)。

### D.6 判读：可分 vs 可混淆

- **效应量**：\(\mathrm{Acc},\mathrm{AUC}\to0.5\) 表示不可分（可混淆）；越大越可分。
- **等价边界**：要声称「可合并」，需预先给定容差 \(\delta\)（如 \(\mathrm{Acc}\le0.5+\delta\)，\(\delta=0.1\)），并用置信区间/等价检验（TOST，两个单侧检验）确认落在边界内。
- **\(p_{\mathrm{perm}}\) 的边界**：显著只说明统计上能检测到差异；\(R\) 越大，越小的差异也显著。不显著也不证明等价。
- **生物学判断**：最终看下游 chasing 指标（波前速度、覆盖率、共存时长、固定概率）是否在预设生物学容差内，而不能只看 C2ST。

### D.7 无模型补充（可选）

对高维场或不想选特征时，可用核两样本检验：

$$
\mathrm{MMD}^2=\Big\|\frac1R\sum_{i=1}^R k(\phi_i^A,\cdot)-\frac1R\sum_{j=1}^R k(\phi_j^B,\cdot)\Big\|_{\mathcal H}^2,
$$

其中 \(k(\cdot,\cdot)\) 为核函数，\(\mathcal H\) 为再生核 Hilbert 空间（RKHS），\(\|\cdot\|_{\mathcal H}\) 为其范数；置换给 \(p_{\mathrm{perm}}\) 值。与 C2ST 互为补充。

### D.8 本项目参数与结果

- \(R=100\)；特征见 D.2；逻辑回归 \(\lambda=10^{-2}\)；\(K=5\)，\(C=5\)；\(B=20\)；对角线 self-split。
- 结果（\(\mathrm{Acc}/\mathrm{AUC}\)）：**M2 vs CPU** small \(0.581/0.618\)、large \(0.591/0.604\)；**M0 vs CPU** small \(0.773/0.848\)、large \(0.920/0.974\)；\(p_{\mathrm{perm}}=0.048\)（置换下限）。
- 解读：M2 与 CPU 的效应量接近不可分边界；M0 因方差结构偏低而明显可分。是否「生物学可合并」仍取决于下游 chasing 指标的等价检验。
- 代码：`compare_strategies_v3.py` 的 `_feature_matrix`、`_fit_logreg`、`_cv_scores`、`_c2st_pair`、`_c2st_matrices`、`_plot_c2st`。

---

## 附录 E：不做完整 CSR 的直接 stencil 随机迁移

> **决策记录（2026-11）**：**方案 A 已选定**；执行计划见 `DirectStencilMigrationPlan.md`。本附录保留问题分析与 A/B 对比。
>
> **2026-11 更新**：测试与后续 chasing 统一 `rate=0.5`；对照目标是 CPU CSR；原型顺序为 NumPy 语义原型 →
> 独立 GPU 原型（不接产品）→ 最后接入 natal 产品；M0/M1/M2 的 GPU FFT 近似线暂停。
>
> 本附录讨论一个自然的替代：不把 $n$ 个源的迁移核展开成完整 CSR，而是只保存 $\kappa\times\kappa$ 的核模板，对每个源直接作用模板，用 GPU 并行与“重算”换内存。结论先行：**空间可以大幅降低，但时间不会比 CSR 更优；精确随机的真正难点是逐源 Dirichlet 的行耦合与写冲突。**

### E.1 动机

完整 CSR 为每个源保存它的所有有效目的地与权重。设一个平面的 deme 数为 $n$，源 $s$ 的有效目的地数为 $r_s$，则 CSR 的非零边总数是

$$
\mathrm{nnz}=\sum_{s=1}^{n} r_s .
$$

内部 $r_s=\kappa^2-1$，所以量级为 $O(n\kappa^2)$；当 $n$ 和 $\kappa$ 都很大时，CSR（以及随机路径的 forward scratch、reverse CSR）会占用无法承受的内存。直接 stencil 只保存 $\kappa\times\kappa$ 的核（$O(\kappa^2)$），对每个源用同一模板计算目的地，从而避免保存完整 CSR。

### E.2 先修正两个直觉：空间不是 $O(\kappa^2)$，时间也不会比 CSR 更省

**空间**：即使不存 CSR，仍需要保存每个平面的状态（$O(n)$）、每个源的 $O_s$ 与 $S_s$、边界归一化 $Z_s$（各 $O(n)$）、核模板（$O(\kappa^2)$）以及输出（$O(n)$）。所以是

$$
O(n+\kappa^2)\quad\text{每平面},
$$

而不是 $O(\kappa^2)$。若同时驻留 $P=2AZ+AZ^2$ 个平面（$A$ 为年龄数、$Z$ 为基因型数），状态部分还要再乘 $P$。相比 CSR 的 $O(n\kappa^2)$，这仍是数量级的下降。

**时间**：CSR 的每 tick 代价本来就是 $O(\mathrm{nnz})=O(n\kappa^2)$（逐边处理）；直接 stencil 也是 $O(n\kappa^2)$（逐源遍历模板）。所以直接 stencil **不是把时间从 $O(n)$ 提到 $O(n\kappa^2)$**——CSR 的时间已经在那里了；它省掉的是**预计算邻接的内存**，代价是每个 tick 重算索引/权重，常数可能更大。真正把时间降到 $O(n\log n)$ 的是 route A 的 FFT，但它只能确定性。

| 方案 | 内存 | 每 tick 时间 | 随机语义 |
|---|---|---|---|
| CSR 随机 | $O(n\kappa^2)$ | $O(n\kappa^2)$ | 精确 |
| 直接 stencil | $O(n+\kappa^2)$ | $O(n\kappa^2)$ | 可精确 |
| route A FFT | $O(n)$ | $O(n\log n)$ | 仅确定性 |

### E.3 难点：Dirichlet 的行耦合与写冲突

按 §2.2，源 $s$ 的分配是

$$
X_{s,d}=O_s\,\frac{G_{s,d}}{S_s},\qquad S_s=\sum_{d'}G_{s,d'} .
$$

$X_{s,d}$ 依赖该源**整行**的 $S_s$。这带来两个实现问题：

- **按源 scatter（散射）**：每个源把 $X_{s,d}$ 写到目的地 $d$。不同源可能同时写同一个目的地，需要原子加（atomic add）。浮点原子加的顺序不确定，会破坏固定种子的逐位可复现性。
- **按目的地 gather（聚集）**：每个目的地汇总来自不同源的 $X_{s,d}$。但每个 $X_{s,d}$ 需要该源整行的 $S_s$；要么为每条入边重算整行（$O(\kappa^2)$ 每边，总计 $O(\kappa^4)$ 每目的地），要么把整行存下来（又回到 $O(n\kappa^2)$ 内存）。

现有 GPU 随机 CSR 路径用两遍规避原子加：第一遍逐边抽样写入 **forward scratch**（大小 $O(\mathrm{nnz})$），第二遍按 **reverse CSR** 让每个目的地按固定顺序求和（大小也是 $O(\mathrm{nnz})$）。它可复现，但内存仍是 $O(n\kappa^2)$。

### E.4 方案 A：逐边 RNG + 两遍重算

思路：用**逐边可重放的随机数**换掉 forward scratch 与 reverse CSR。

- **Pass 1（逐源）**：抽 $O_s$；对该源的每条边抽 $G_{s,d}$，累加得到 $S_s$。只保存 $O_s$、$S_s$、$Z_s$（各 $O(n)$），**不保存逐边量**。
- **Pass 2（逐目的地 gather）**：对目的地 $d$，枚举模板偏移 $u$，源为 $s=d-u$；若该位移有效，则用**同一个逐边随机数**重新抽 $G_{s,d}$，计算 $O_sG_{s,d}/S_s$，按固定偏移顺序累加。

要保证两遍抽到同一个 $G_{s,d}$，需要**计数器随机数发生器**（counter-based RNG）：它的输出是 `(seed, 源 s, 目的地 d, tick)` 的哈希，给定键就能 $O(1)$ 重放同一次抽样，而不需要顺序消耗随机流。这样：

- 内存：$O(n+\kappa^2)$；无 forward scratch、无 reverse CSR、无原子加；
- 时间：约两遍 Gamma 抽样 + $O(n\kappa^2)$ 求和；求和顺序固定，可复现；
- 边界：gather 时用 $Z_s$（$O(n)$ 或按位置算）和 $K'(u)$（$O(\kappa^2)$）即可，**不需要 reverse CSR**；
- 代价：Gamma 抽样做两遍；Gamma 采样器需要能为每条边独立 seed（不能用会消耗不定长随机流的全局 RNG）。

### E.5 方案 B：分块 stencil + 有界 scratch

另一种做法是把域分成小块（tile），每块连同其边界外的一圈 **halo** 一起处理：块内用共享内存累加，块间交换 halo。这样每块只需要 $O(\mathrm{tile}\cdot\kappa^2)$ 的逐边 scratch，而不是全局 $O(n\kappa^2)$。

- 内存：$O(n+\mathrm{tile}\cdot\kappa^2)$；
- 时间：约一遍抽样 + $O(n\kappa^2)$；
- 代价：halo 与边界处理更复杂；要保证块间累加顺序固定以维持可复现。

### E.6 模板压缩

内部源的模板完全相同；只有靠近边界的源需要不同的有效位移集合与 $Z_s$。因此可以只保存：

- 一个内部模板（$\kappa^2-1$ 个偏移与权重）；
- 少量边界模板（数量约为 $O(R^2)$，$R=(\kappa-1)/2$ 为核半径）。

这避免了每个 tick 重新判断边界，降低常数；随机路径可以复用同一套模板。M4 的 `m4_template_build.py` 已在确定性情形验证过这类模板压缩。

### E.7 与现有实现、FFT 的选型

- **CSR 随机**：精确，内存 $O(n\kappa^2)$，时间 $O(n\kappa^2)$。
- **直接 stencil（方案 A/B）**：精确，内存 $O(n+\kappa^2)$，时间 $O(n\kappa^2)$（常数更大）。
- **route A FFT**：内存 $O(n)$，时间 $O(n\log n)$，但仅确定性；随机只能近似（§4 的 M1–M4）。

选型取决于瓶颈：内存受限且必须精确随机 → 直接 stencil；$\kappa$ 中等且想快 → CSR；可接受近似或只需均值 → FFT。

### E.8 验证计划（小规模）

1. **算子级原型**：固定一个随机场，分别用 CPU CSR 与方案 A 的直接 stencil 做一次迁移；比较均值、方差、协方差、总量，确认逐边 RNG 重放能得到与 CSR 相同的分布（不是逐位相同，因为 RNG 键不同）。
2. **内存/时间阶梯**：在 $n$ 与 $\kappa$ 的若干组合下测 CSR、直接 stencil、FFT 的显存与每 tick 时间。
3. **可复现性**：固定种子，确认方案 A 的两次运行逐位一致（固定求和顺序）。
4. 若产品化，属高风险：走根 `EVALUATE.md`、全量门禁、独立复核。
