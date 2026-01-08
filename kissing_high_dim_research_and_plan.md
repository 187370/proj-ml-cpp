# Kissing Number（球面码 / spherical code）高维数值搜索：调研报告 + 你们当前算法的改进路线图

> 生成时间：2026-01-08（America/Los_Angeles）  
> 面向目标：在维度 **d > 8** 时，突破“从强 seed 出发无法再插点/无法继续优化”的瓶颈，持续改进 **lower bound**（可行构型的点数 N）。

---

## 0. 你们现在做的是什么（问题等价与现状）

Kissing number \(K(d)\) 的一个标准等价表述是：

- 在单位球面 \(S^{d-1}\) 上找最大点集 \(\{x_i\}\)，使得任意两点夹角 \(\theta_{ij}\ge 60^\circ\)；
- 等价约束：\(\langle x_i,x_j\rangle \le 1/2\)；
- 等价写作欧氏距离（都在单位球上）：\(\|x_i-x_j\|^2 = 2-2\langle x_i,x_j\rangle \ge 1\)。

你贴出的“新版本”代码属于 **增量式（逐点加点） + 局部连续优化 +（可选）basin hopping + k-opt repair** 的组合：  
- 先用结构化 seed（根系 E6/E7/E8, Dn 等），再尝试逐个插点（随机候选取“最远点”），每次插点后用 penalty+repulsion 的力学迭代把最小距离推到 1；
- 插点失败后尝试 small-k 的 remove+reinsert（k-opt repair），最后才大量随机重启。

在 **d≤8** 你们能做到 SOTA（很自然：E8 / E7 / E6 / Dn roots 在这些维度就是最强的结构起点）；但 **d>8** 时你们遇到典型瓶颈：  
> **最强 seed 本身非常“刚性/紧密”，插入第 N+1 个点需要“全局重排”，而你们的插点与局部优化都太局部、候选太随机，难以跨越能量壁垒。**

---

## 1. 高信噪“算法谱系”调研（专注：下界/构造/搜索）

这一节把 **高维能继续推进 N 的方法** 分成 5 类，并明确每类“能带来什么”以及和你们代码的对接点。

### 1.1 结构化构造：格 / 码 / root systems / laminated lattices（最稳定的 baseline）

**关键词：最短向量（minimal vectors）→ kissing configuration。**

- **Conway–Sloane 的 laminated lattices \(\Lambda_n\)**：从 \(\Lambda_{n-1}\) 逐层堆叠得到 \(\Lambda_n\)，很多维度的“最好已知下界”直接来自其 minimal vectors（或其截面/投影）。  
  典型效果：你们 d>8 的“laminated seed”如果不是严格的 \(\Lambda_n\) 构造，只做随机高度叠层，往往**达不到真实 laminated lattice 的强度**。

- **Sloane 的 Spherical codes / packings 数据库**：里面存了大量“最好已知/很强”的球面码坐标或结构描述，是做 seed、做对比表、做复现的第一入口。

- **更高维的非平凡改进**：Cohn–Li（2024）在 17–21 维给出新的更大下界，它的核心信息是：
  > “不要只做 Leech lattice 的截面/投影；可以从‘近似格结构’出发做离散改造，再进行几何微调”，从而获得更多点。

**对你们的启示：**  
- **（必须做）用真实“强 seed”替换随机叠层 seed**：把 Sloane / Conway–Sloane 的标准构造作为初始化，否则算法会在 d>8 一开始就输在起跑线上。  
- **（可选做）在强结构周围做‘离散改造’而不是纯连续微调**：这正是 Cohn–Li 的方向。

---

### 1.2 连续能量最小化 / 平衡条件：从“带排斥势的粒子”到“硬球约束”

经典经验是把“最小距离最大化/硬约束”转成可优化的能量形式：

- Hardin–Sloane–Smith 的 “electrons on the sphere” 系列工作与网站：把点当电荷，在球面上做能量最小化，能产出大量高质量的数值构型，并沉淀了很多“工程细节”（分阶段势函数、重启、局部优化策略）。

- Wang（2008）用能量最小化来找好 spherical codes，并与 Sloane 库做对接比较（强调“势函数与分阶段策略”的重要性）。

- Huang–Pardalos–Shen（2001）的 **Point Balance Algorithm (PBA)**：从 bilevel 观点定义“balanced spherical code”，提出一种把配置推向平衡态的迭代算法（把你们的“力学迭代”做得更“对准平衡条件”）。

- 各类“packing / spherical cap”优化还发展了**Iterated Dynamic Neighborhood Search** 等现代局部搜索框架（2020s 的工程论文多，里面很多“如何跳出局部盆地”的套路可直接借鉴）。

**对你们的启示：**  
- 你们当前优化器是“固定步长的力迭代 + 归一化”，在高维/大 N 下很容易卡住；更强的做法是：
  - 使用 **Riemannian（球面流形）梯度/拟牛顿（L-BFGS）**；
  - 或使用 **多阶段势函数（先软后硬）** + **退火** + **大邻域扰动**（而不只是小 jitter）。

---

### 1.3 全局优化框架：MLSL / VNS / basin-hopping / LNS（把局部搜索“装进大框架”）

Liberti–Kucherenko 等把 KNP 写成非凸数学规划，结合 **MLSL（多层单链）** 与 **VNS（变邻域搜索）** 在“连续+组合”之间跳转，是“能稳定推进”的经典套路。

你们已经有一个雏形：`basin_hopping_optimize` + `kopt_repair`。但目前：

- basin hopping 的接受准则只看 `cand_min`（min distance 的 best），温度也很低；
- k-opt repair 只移除“最差点”且 k 很小；
- 插点候选采样基本是“全随机球面采样”，高维时命中率极差。

**对你们的启示：**  
- 把你们现有框架升级成 **LNS/Large-Neighborhood Search**：  
  反复做 “删除一批点（不只最差点）→ 重插（用强插点器）→ 强局部优化 → 允许临时变差（退火接受）”。  
- 让“全局框架”的温度/扰动规模跟维度、N 自适应，而不是固定常数。

---

### 1.4 AI/自动发现：AlphaEvolve 与 PackingStar（把搜索空间换成“程序/矩阵/博弈”）

近两年的重要趋势是：对 kissing number 这种 **高维巨大组合空间**，纯手工启发式会变慢，AI 会引入新的表示与搜索方式：

- **AlphaEvolve（DeepMind, 2025）**：通过“进化代码 + 自动评估”的方式，在 11 维把下界推到 593（比此前 592 更高）。  
  这类方法的价值不在于“用 AI 直接给答案”，而是提示你们：
  > 把你们的算法当作一个可变程序，搜索的不只是点，而是“搜索策略/插点策略/扰动策略/参数日程”。

- **PackingStar（Ma et al., 2025, arXiv:2511.13391）**：把问题建模成“Gram 矩阵补全的两人博弈”，在更高维声称超过许多既有记录（尤其 25–31 维）。  
  即使你们不复现 RL，也可以借鉴其核心表示：
  > **把点集用 Gram 矩阵/接触图来表达，从‘矩阵条目’层面做离散/连续混合更新。**

---

### 1.5 上界（LP/SDP）只做“对照与差距评估”

虽然你们主攻下界，但报告里需要放上界作对照：

- Bachoc–Vallentin 的 SDP bound 是现代经典上界框架；
- Machado–Oliveira Filho 等利用对称性改进了 9–23 维的 SDP 上界。

---

## 2. 诊断：你们当前实现为什么在 d>8 “插不动了”

下面专门对你贴的那份代码（增量插点 + optimize_points）做“结构性诊断”。

### 2.1 插点器 `insert_point_best`：高维里随机候选会崩

你们的插点策略是：

1. 在球面上随机采 `candidates` 个点；
2. 选其中与现有点集的最小距离最大者，作为新点。

问题：在高维、且 N 已经很大时，“可插入区域”是许多球帽补集的交集，体积很小。  
**随机抽 200~600 个点几乎不可能落到好的区域**，所以你们看到：
- 插入点通常先天离得不够远；
- 局部优化再怎么推也推不动（因为一开始冲突太多，落入坏盆地）。

> 这是你们高维停滞的第一大根因。

---

### 2.2 优化器 `optimize_points`：步长/方向/力模型在高维不够“数值强”

你们的更新是：

- 对所有 pairs 计算 diff 与 dist；
- 若 dist2 < target_dist2，则加 penalty push；
- 对所有 pairs 加 repulsion；
- 用固定日程 step（且按 1/sqrt(n) 缩放），更新后 normalize。

几个高维问题：

1. **球面流形上的梯度方向**：你们直接在 \(\mathbb{R}^d\) 加力后 normalize，等价于某种“隐式投影”，但在高维/强约束时会变得很慢。  
   更稳的做法：在每个点的切空间更新：  
   \[
   F_\perp = F - (F\cdot x)x,\quad x\leftarrow \mathrm{normalize}(x+\eta F_\perp).
   \]

2. **全对全 O(N²) 的 repulsion 代价巨大**：  
   当你们想做 11D 的 593 或更高维更大 N 时，O(N²·iters) 会成为主要瓶颈，导致不得不缩短 iters/减少重启，从而搜索质量下降。

3. **“只追 min distance 到 1”会把问题做成硬可行性判定**：  
   插点失败时，优化器只是在“把最差那对拉开”，但实际上需要全局重排，靠这种力学迭代很难跨越能量壁垒。

---

### 2.3 k-opt repair 仍然太“局部”：删除最差点 ≠ 唯一有效邻域

你们的 `worst_points_by_min_distance` 永远删“最近邻最差”的点，这在高维经常不够：

- 有些时候需要删的是“挡住插点通道”的点（它们未必是最差点）；
- 需要 **随机化/多样化的邻域**：随机删、基于接触图删、按簇删、按方向删等；
- k 也需要变大（例如 5~20 甚至更多），并配合更强插点器与退火接受。

---

## 3. 能继续推进 d>8 的“硬核改进路线”（强烈建议按优先级做）

下面每一条都写清：**做什么、为什么、怎么接进你们现有代码、预期收益**。

---

## 3.1 先把“种子”做对：真实 laminated lattice / Sloane 数据库导入（优先级 S）

> 你们的 `generate_laminated_seed` 是“随机高度叠层 + 贪心插点”，它不是 Conway–Sloane 的 laminated lattice 构造，强度差异会非常大。

**建议：**

1. **直接导入 Sloane 的 spherical code 坐标**（或从其文档/附录提取）作为 d=9..24 的 seed。  
2. **实现真正的 \(\Lambda_n\) minimal vectors 生成**（如果你们愿意做“算法性构造”而不只读文件）：  
   - 用 Conway–Sloane laminated lattice 的定义/构造流程生成 Gram 矩阵/基，再枚举 minimal vectors；  
   - 或至少把典型维度的已知 lattice seed（如 \(\Lambda_9,\Lambda_{10},...,\Lambda_{24}\)）以文件形式固化到项目中。

**收益：**
- d>8 的 baseline 立刻对齐“人类最好已知结构”；  
- 你们再谈“能不能超越 seed”才有意义（否则算法进步可能被 seed 差掩盖）。

---

## 3.2 换掉插点器：从“随机采样”升级为“maximin 插点优化”（优先级 S）

### 方案 A：Soft-min 目标 + Riemannian 梯度上升（推荐，工程最简单）

对新点 \(x\) 定义：
\[
m(x)=\min_i \|x-p_i\|^2\approx -\frac1\beta \log\sum_i \exp(-\beta\|x-p_i\|^2)
\]
最大化 \(m(x)\)（在球面上），用多次随机初始化 + 梯度上升，取最好的作为插入候选。

**怎么接入：**
- 把 `insert_point_best` 改成：
  - 多起点（例如 50~200 个起点）；
  - 每个起点做 100~300 步切空间梯度上升；
  - 输出最优的 1~k 个候选。

**收益：**
- 候选点一开始就更“接近真正的 Voronoi 最远点”，插点成功率会比随机高一个数量级以上。

### 方案 B：近似 Voronoi / active constraints 生成候选（更强，但工程更重）

最远点往往在一些“紧约束”的交界处（接触图附近）。做法：
- 找到当前配置里“接近 1 的 pair”形成的接触图；
- 对每个点取其 k 近邻，构造“候选法向/交点方向”，生成一批候选再局部精化。

**收益：**
- 让插点专注在“真正有希望的几何位置”，尤其适合从极强 lattice seed 上突破。

---

## 3.3 局部优化器升级：球面流形优化 + 自适应步长/拟牛顿（优先级 A）

把 `optimize_points` 的“力学迭代”升级为更标准的流形优化：

1. **切空间投影**：更新前把 force 投影到切空间 \(F_\perp = F-(F\cdot x)x\)；
2. **自适应学习率**：用 line search 或 Adam/RMSProp（即便简单也比固定 step 稳）；
3. **拟牛顿（L-BFGS）**：对“平滑势函数”（repulsion + soft penalty）非常有效。

同时把 penalty 从 dist2 写成 inner-product 更自然：  
- 约束 \(\langle x_i,x_j\rangle \le 1/2\)；  
- penalty 用 \(\max(0,\langle x_i,x_j\rangle-1/2)^2\)。  
这样梯度更干净，也更接近 Gram 矩阵表示。

**收益：**
- 同样的迭代次数下，收敛速度更快；  
- 对强 seed 的“微扰后重新平衡”更容易。

---

## 3.4 变邻域/大邻域：把 k-opt 做成真正的 LNS（优先级 A）

你们目前的 repair 是：

- 删最差 remove_k 个点；
- 重新插 remove_k+1 个点（贪心随机候选）；
- 再跑局部优化。

建议升级成 LNS：

1. **删除策略多样化**（每次随机选一种）  
   - 删最差点（你们已有）；  
   - 删随机点；  
   - 删“同一簇/同一方向”的点（用 PCA 或随机投影分簇）；  
   - 删接触图中高度数节点（可能是“阻塞点”）。

2. **k 取值扩大**：例如在失败时 k 从 2,3,5,8,13,… 逐渐增大，并允许偶尔非常大的 shake（比如删 20）。

3. **重插用强插点器（3.2）**，而不是纯随机候选。

4. **接受准则退火**：允许临时变差（min distance 降一点或 penalty 上升一点），再逐步收紧。

**收益：**
- 这是从 “N 到 N+1” 最常见且最有效的工程套路；  
- 你们现有框架已具备雏形，改动收益/成本比很高。

---

## 3.5 性能工程：把 O(N²) 做成“近邻主导”（优先级 A）

当 N 上到几百/几千时，O(N²) 直接卡死，会迫使你们减少 iters 和 restarts → 质量下降。  
优化方向：

1. **只对“近邻对”计算 penalty**（你们已经是 dist2 < target_dist2 才加 penalty；关键是找到这些对不要每次全扫）：
   - 周期性构建近邻表（kNN），中间若干步只用近邻对更新；
   - 高维用 ANN（近似近邻）也够用。

2. **repulsion 只对近邻/抽样对**：  
   - 全局 repulsion 可以用抽样近似（每点采样 M 个远点）；
   - 或者直接只保留“短程排斥”（因为真正决定 feasibility 的是短程约束）。

3. **并行化**：force 计算对 pairs 很适合 OpenMP / SIMD。

**收益：**
- 让你们能用更多 restarts、更深的 LNS、更强插点器，而不是被 N² 计算耗尽。

---

## 3.6 借鉴 PackingStar：加入 Gram 矩阵/接触图视角（优先级 B）

PackingStar 的核心做法是从 Gram 矩阵（两两内积）入手做“补全+修正”。你们可做一个轻量版：

- 维护接触图（哪些 pair 接近 \(\langle x_i,x_j\rangle = 1/2\)）；
- 在局部优化时对这些 active constraints 加更强权重（类似 active-set）；
- 插点或 LNS 时优先处理接触图冲突区域。

**收益：**
- 让“跨越局部盆地”的扰动更有结构，而不是纯随机 kick。

---

## 4. 给你们一套“可落地”的新管线（按实现顺序）

下面是一条你们可以直接照着写第二版/第三版的路线图。

### Phase 1：把 baseline 对齐（1~2 天）
- 导入 Sloane 的 d=9..24 seed（文件方式即可）。  
- 修正你们 `default_target_for_dimension`（例如 11D=593 是现成热点；13D 应该不是 593）。

### Phase 2：插点器升级（2~4 天）
- 实现 Soft-min 插点优化（多起点 + 梯度上升）；  
- 替换掉 `insert_point_best` 的纯随机候选。

### Phase 3：LNS 化（2~5 天）
- 将 repair 改为多策略删除 + 可变 k + 退火接受；  
- 每次 LNS 后跑更强局部优化（流形投影 + 自适应步长）。

### Phase 4：性能工程（并行推进）
- 近邻表 + repulsion 抽样；  
- OpenMP 并行 pair forces。

---

## 5. 实验设计与报告写法建议（让结果更“像研究”）

1. **分维度报告**：9~16 用 laminated / known seeds 做基线；17~21 做对照 Cohn–Li；11 维对照 593 的公开新闻。  
2. **报告三类曲线**：
   - “插点成功率 vs 候选生成方式”（随机 vs soft-min 插点）；  
   - “NNS/LNS 强度 vs 成功率”（k、删除策略、温度）；  
   - “耗时分解”（O(N²) 优化前后）。  
3. **输出可复现记录**：
   - seed 来源、随机种子、参数日程；  
   - 最终点集与最小距离验证脚本。

---

## 6. 参考文献（你们报告可以直接引用）

- Henry Cohn: *Kissing numbers*（上下界表与更新入口）  
- Neil Sloane: *Spherical Codes / Packings database*（强 seed 与对照）  
- Conway & Sloane (1982): *Laminated lattices*（\(\Lambda_n\) 构造理论基础）  
- Cohn & Li (2024): *Improved kissing numbers in seventeen through twenty-one dimensions*（非截面新构造）  
- Huang, Pardalos, Shen (2001): *A Point Balance Algorithm for the Spherical Code Problem*  
- Liberti & Kucherenko (2007): *New formulations for the Kissing Number Problem*（MLSL/VNS 全局框架）  
- Wang (2008): *Energy minimization for spherical codes*  
- DeepMind (2025): *AlphaEvolve*（11D=593 新下界，提示“搜索策略”层面的进化）  
- Ma et al. (2025): *Finding Kissing Numbers with Game-theoretic Reinforcement Learning (PackingStar)*（Gram 矩阵博弈视角）

> 注：你们若要进一步“复刻/对齐”某些维度的具体坐标，建议直接以 Sloane 数据库或其引用论文为准，避免从二手表格抄写带来的偏差。

---
