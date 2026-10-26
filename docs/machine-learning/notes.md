# 机器学习讲解

机器学习的考查分两类：一是推导与概念，如偏差-方差、正则化、XGBoost 的二阶展开、AUC、反向传播、注意力机制；二是结合项目的追问，如时间序列如何切分、是否存在标签泄漏、样本外表现为何衰减、因子如何合成。幻方、衍复、世纪前沿、玄信等设有机器学习研究岗，宽德、诚奇的部分笔试版本含机器学习题。公开面经未记录完整题干，配套[题库](problems.md)以经典题与押题为主。

前置知识：[统计与计量](../statistics/notes.md)中的 OLS、最大似然与多重检验；[编程与算法](../programming/notes.md)中的 NumPy。

---

# 第一部分：通用框架

## 1. 监督学习的基本框架

### 1.1 三要素

李航《统计学习方法》把一个统计学习方法拆成三部分：

| 要素 | 含义 | 例子 |
|---|---|---|
| 模型 | 假设空间 $\mathcal F$，即候选函数的集合 | 线性函数 $f(x)=w^\top x+b$；深度不超过 6 的树 |
| 策略 | 用什么准则挑选函数，即损失函数与风险 | 平方损失、交叉熵、hinge 损失 |
| 算法 | 怎么求解最优函数 | 闭式解、梯度下降、贪心分裂 |

### 1.2 损失、风险与经验风险

- **损失函数** $L(y,f(x))$：单个样本上预测的代价。
- **期望风险** $R(f)=\mathbb E_{(x,y)\sim P}[L(y,f(x))]$：真正想最小化的量，但分布 $P$ 未知。
- **经验风险** $\hat R(f)=\frac1n\sum_{i=1}^n L(y_i,f(x_i))$：训练集上的平均损失。
- **结构风险** $\hat R(f)+\lambda J(f)$：经验风险加上模型复杂度惩罚 $J(f)$，即正则化。

常见损失函数：

| 损失 | 公式 | 用于 | 最优预测 |
|---|---|---|---|
| 平方损失 | $(y-f)^2$ | 回归 | 条件均值 $\mathbb E[y\mid x]$ |
| 绝对损失 | $\lvert y-f\rvert$ | 稳健回归 | 条件中位数 |
| Huber | 小残差平方、大残差线性 | 有异常值的回归 | 介于均值与中位数之间 |
| 交叉熵（对数损失） | $-y\log p-(1-y)\log(1-p)$ | 分类 | 条件概率 $P(y=1\mid x)$ |
| hinge | $\max(0,1-yf)$，$y\in\{-1,1\}$ | SVM | $\mathrm{sign}(P(y=1\mid x)-\tfrac12)$ |
| 0-1 损失 | $\mathbf 1[y\ne \hat y]$ | 评价分类 | 贝叶斯分类器 |

> **量化用途**：收益率预测用平方损失时，模型学的是条件均值，会被少数极端收益拉动；所以实践中常先对标签做截面去极值或排名，或改用 Huber 损失。

### 1.3 生成模型与判别模型

| | 生成模型 | 判别模型 |
|---|---|---|
| 学什么 | 联合分布 $P(x,y)$，再用贝叶斯公式得 $P(y\mid x)$ | 直接学 $P(y\mid x)$ 或决策函数 $f(x)$ |
| 例子 | 朴素贝叶斯、高斯判别分析、HMM | 逻辑回归、SVM、树模型、神经网络 |
| 特点 | 样本少时收敛快，可处理缺失 | 样本多时精度通常更高 |

---

## 2. 偏差-方差与正则化

### 2.1 偏差-方差分解

设 $y=f(x)+\varepsilon$，$\mathbb E\varepsilon=0$，$\mathrm{Var}(\varepsilon)=\sigma^2$。$\hat f$ 由随机抽取的训练集得到。对固定的测试点 $x$：

$$\mathbb E\big[(y-\hat f(x))^2\big]=\underbrace{\sigma^2}_{\text{不可约误差}}+\underbrace{\big(f(x)-\mathbb E\hat f(x)\big)^2}_{\text{偏差}^2}+\underbrace{\mathbb E\big[(\hat f(x)-\mathbb E\hat f(x))^2\big]}_{\text{方差}}$$

**推导**：记 $\bar f=\mathbb E\hat f(x)$。

$$y-\hat f=\varepsilon+(f-\bar f)+(\bar f-\hat f)$$

三项两两乘积的期望为 0：$\varepsilon$ 与训练集独立且均值为 0；$f-\bar f$ 是常数，而 $\mathbb E(\bar f-\hat f)=0$。平方取期望即得上式。

| 模型复杂度 | 偏差 | 方差 | 典型表现 |
|---|---|---|---|
| 低（线性、浅树、强正则） | 高 | 低 | 欠拟合：训练误差和测试误差都高 |
| 高（深树、大网络、弱正则） | 低 | 高 | 过拟合：训练误差低，测试误差高 |

> **量化用途**：金融收益的不可约误差 $\sigma^2$ 极大（日收益的可预测部分通常不到总方差的 1%），方差项稍大就会盖过偏差上的改进。因此量化模型普遍**偏向低方差**：强正则、浅树、多模型平均。

### 2.2 正则化

| 方法 | 惩罚项 | 效果 | 贝叶斯解释 |
|---|---|---|---|
| L2（Ridge） | $\lambda\lVert w\rVert_2^2$ | 系数整体收缩，不为零；缓解共线性 | 高斯先验 $w\sim N(0,\tau^2 I)$ 下的 MAP |
| L1（Lasso） | $\lambda\lVert w\rVert_1$ | 部分系数精确为 0，做特征选择 | 拉普拉斯先验下的 MAP |
| Elastic Net | $\lambda_1\lVert w\rVert_1+\lambda_2\lVert w\rVert_2^2$ | 兼有稀疏与成组选择 | — |
| 早停 | 验证集损失不再下降时停止 | 近似 L2 | — |
| Dropout | 训练时随机置零神经元 | 近似模型平均 | — |
| 树模型约束 | 最大深度、叶子数、最小叶子样本、学习率 | 限制单棵树复杂度 | — |

**为什么 L1 会产生稀疏解**：约束形式 $\lVert w\rVert_1\le t$ 的可行域是菱形，顶点在坐标轴上，等高线（椭圆）最先碰到的通常是顶点；L2 的可行域是圆，没有尖角。代数上，正交设计下 Lasso 的解是**软阈值**：

$$\hat w_j=\mathrm{sign}(z_j)\,(\lvert z_j\rvert-\lambda)_+,\qquad z_j=x_j^\top y$$

（目标函数取 $\tfrac12\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$，且 $X^\top X=I$。）而 Ridge 的解是 $\hat w_j=z_j/(1+\lambda)$，只缩不截。

**Ridge 的闭式解**：

$$\hat w=(X^\top X+\lambda I)^{-1}X^\top y$$

$X^\top X$ 接近奇异（共线性）时，加 $\lambda I$ 让矩阵可逆且条件数变小。

> **量化用途**：因子之间高度相关（如多个窗口的动量、多种口径的估值），OLS 系数会剧烈波动甚至符号相反；Ridge 让权重稳定，Lasso 用于从几百个候选因子中筛选。注意 L1/L2 惩罚对尺度敏感，**必须先标准化特征**。

---

## 3. 交叉验证与时间序列验证

### 3.1 标准方法

| 方法 | 做法 | 适用 |
|---|---|---|
| 留出法 | 一次性划分训练/验证/测试 | 数据量大 |
| K 折交叉验证 | 分 K 份，轮流用 1 份验证、其余训练，取平均 | 样本独立同分布 |
| 留一法 | K = n | 样本很少 |
| 自助法（bootstrap） | 有放回抽 n 个，约 $1-1/e\approx63.2\%$ 的样本被抽到，其余作袋外样本 | 估计方差、随机森林 OOB |

**超参数要在验证集上选，测试集只能用一次。** 在测试集上反复调参，测试集就变成了验证集。

### 3.2 为什么金融数据不能直接用随机 K 折

随机 K 折假设样本独立同分布，金融数据有三点不满足：

1. **时间依赖**：收益有波动聚集，特征（如 20 日动量）有很强的自相关。随机打乱后，验证集中某天的"邻居"大多在训练集里，模型只需记住邻近样本就能得到虚高分数。
2. **标签重叠**：标签是未来 $h$ 日收益时，第 $t$ 天的标签覆盖 $[t+1,t+h]$，与第 $t+1,\dots,t+h-1$ 天的标签共享部分收益。相邻样本的标签高度相关，训练集与验证集之间存在信息泄漏。
3. **非平稳**：用未来数据训练、过去数据验证，等于让模型提前知道了未来的市场状态。

### 3.3 Walk-forward（滚动 / 扩展窗口）

```
扩展窗口：  [训练 ......][验证]
            [训练 ...........][验证]
            [训练 ................][验证]
滚动窗口：  [训练 ....][验证]
                 [训练 ....][验证]
                      [训练 ....][验证]
```

| | 扩展窗口 | 滚动窗口 |
|---|---|---|
| 训练集 | 从起点到当前 | 最近固定长度 |
| 优点 | 样本多、估计稳定 | 适应市场变化 |
| 缺点 | 旧数据权重过大 | 样本少、方差大 |

walk-forward 最贴近实盘（每次只用过去的数据），缺点是每个时点只得到一条路径，结果方差大，早期数据从未作为验证集。

### 3.4 Purged K-Fold 与 Embargo

López de Prado 在 AFML 第 7 章提出的方法：仍然做 K 折（每折是一段连续时间），但做两处修正：

- **Purge（清除）**：从训练集中删除标签区间与验证集标签区间**有重叠**的样本。设样本 $i$ 的标签区间为 $[t_{i,0},t_{i,1}]$，验证集覆盖 $[T_0,T_1]$，则只要 $[t_{i,0},t_{i,1}]\cap[T_0,T_1]\ne\varnothing$，就删除样本 $i$。
- **Embargo（禁止期）**：验证集结束后再删除一小段训练样本（AFML 建议约 $0.01T$），因为特征自相关会把验证集的信息"带"到紧随其后的样本中。验证集之前的样本已被 purge 处理，因此只需在验证集之后设 embargo。

```
时间 →   ............|== 验证集 ==|..................
训练集： ██████████░░             ░░░░████████████████
                  ↑ purge            ↑ embargo
```

| 问题 | 解决方法 |
|---|---|
| 标签区间重叠 | Purge |
| 特征序列相关 | Embargo |
| 只有一条回测路径 | CPCV（组合清除交叉验证，AFML 第 12 章）：把数据分成 N 组，每次取 k 组作验证，可拼出多条回测路径 |

### 3.5 标签泄漏与前视偏差的常见形式

| 形式 | 例子 | 修正 |
|---|---|---|
| 特征用到未来 | 财报按报告期而非公告日对齐；用当日收盘价算信号并在当日成交 | point-in-time 数据；信号用 $t$ 日收盘前信息，$t+1$ 日成交 |
| 预处理泄漏 | 用全样本的均值方差做标准化；全样本 PCA；全样本去极值阈值 | 预处理参数只在训练集上估计；截面标准化只用当天数据 |
| 特征选择泄漏 | 先在全样本上挑出与标签相关性最高的特征，再做交叉验证 | 特征选择放进每一折内部 |
| 超参数泄漏 | 在测试期上调参 | 三段划分，测试集只用一次 |
| 标签重叠 | 未来 20 日收益作标签，随机切分 | Purge + Embargo |
| 幸存者偏差 | 只用当前存续股票 | 使用历史全部股票和历史成分股 |
| 目标编码泄漏 | 用包含自身标签的类别均值编码 | 折外（out-of-fold）编码 |

> **面试回答模板**："我用的是滚动窗口：训练 3 年、验证 6 个月、测试 6 个月，每季度向前滚动；标签是未来 5 日收益，所以训练集末尾和验证集之间留了 5 个交易日的间隔（purge），验证集后再留 5 天（embargo）；标准化只用当天截面；超参数只在验证集上选。"

---

# 第二部分：经典模型

## 4. 线性模型与逻辑回归

### 4.1 线性回归

| 方法 | 目标 | 解 |
|---|---|---|
| OLS | $\min\lVert y-Xw\rVert^2$ | $\hat w=(X^\top X)^{-1}X^\top y$ |
| Ridge | $+\lambda\lVert w\rVert_2^2$ | $(X^\top X+\lambda I)^{-1}X^\top y$ |
| Lasso | $+\lambda\lVert w\rVert_1$ | 无闭式解，用坐标下降（每次对一个系数做软阈值） |
| WLS | $\min\sum_i \omega_i(y_i-x_i^\top w)^2$ | $(X^\top\Omega X)^{-1}X^\top\Omega y$ |

OLS 的统计性质（无偏性、Gauss-Markov、t 检验）见[统计与计量](../statistics/notes.md)。这里强调 ML 视角：OLS 是平方损失下的经验风险最小化，也是高斯噪声下的极大似然估计。

### 4.2 逻辑回归

**模型**：$P(y=1\mid x)=\sigma(w^\top x)=\dfrac{1}{1+e^{-w^\top x}}$，即对数几率 $\log\frac{p}{1-p}=w^\top x$ 是线性的。

**损失**：极大似然等价于最小化交叉熵

$$\ell(w)=-\sum_{i=1}^n\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big],\qquad p_i=\sigma(w^\top x_i)$$

**梯度**：利用 $\sigma'(z)=\sigma(z)(1-\sigma(z))$，

$$\frac{\partial\ell}{\partial w}=\sum_i (p_i-y_i)\,x_i=X^\top(p-y)$$

**Hessian**：$H=X^\top SX$，$S=\mathrm{diag}(p_i(1-p_i))$，半正定，所以损失是凸的，没有局部极小。牛顿法的迭代就是**迭代重加权最小二乘（IRLS）**。

| 问题 | 回答 |
|---|---|
| 为什么不用平方损失 | 平方损失套 sigmoid 后非凸；交叉熵对应伯努利似然，梯度形式简洁 |
| 数据线性可分会怎样 | 似然可以无限逼近 1，$\lVert w\rVert\to\infty$，不收敛；需要加正则 |
| 多分类 | softmax 回归：$p_k=e^{z_k}/\sum_j e^{z_j}$，梯度同样是 $p-y$（$y$ 为 one-hot） |
| 系数的含义 | $x_j$ 增加 1 单位，对数几率增加 $w_j$，几率乘以 $e^{w_j}$ |

> **量化用途**：预测"明日收益是否跑赢中位数"这类二分类标签，逻辑回归是可解释的基线；其输出概率也可直接作为打分。

---

## 5. 决策树与随机森林

### 5.1 决策树

决策树把特征空间划分成若干矩形区域，每个区域给一个常数预测。构建方法是**贪心的递归二分**：每一步遍历所有特征和切分点，选使不纯度下降最多的分裂。

| 算法 | 分裂准则 | 说明 |
|---|---|---|
| ID3 | 信息增益 $g(D,A)=H(D)-H(D\mid A)$ | 偏向取值多的特征 |
| C4.5 | 信息增益比 $g(D,A)/H_A(D)$ | 修正 ID3 的偏向 |
| CART 分类树 | 基尼指数 $\mathrm{Gini}(D)=1-\sum_k p_k^2$ | 二叉树，sklearn 默认 |
| CART 回归树 | 平方误差 $\sum(y_i-\bar y_L)^2+\sum(y_i-\bar y_R)^2$ | 叶子取均值 |

熵 $H(D)=-\sum_k p_k\log_2 p_k$。例：10 个样本中 6 正 4 负，$H=-0.6\log_2 0.6-0.4\log_2 0.4\approx0.971$，$\mathrm{Gini}=1-0.36-0.16=0.48$。

**剪枝**：预剪枝（最大深度、最小叶子样本、最小增益）与后剪枝（CART 的代价复杂度剪枝 $C_\alpha(T)=C(T)+\alpha\lvert T\rvert$）。

| 优点 | 缺点 |
|---|---|
| 可解释；自动捕捉交互项和非线性；对特征单调变换不变；不需要标准化 | 单棵树方差大（数据小变动导致结构大变）；预测是阶梯函数；外推能力差 |

### 5.2 Bagging 与随机森林

**Bagging**：对训练集做 $B$ 次自助采样，各训练一个模型，回归取平均、分类投票。

若 $B$ 个模型的预测方差均为 $\sigma^2$，两两相关系数为 $\rho$，则平均后的方差为

$$\mathrm{Var}\Big(\frac1B\sum_b \hat f_b\Big)=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$

$B\to\infty$ 时第二项消失，剩下 $\rho\sigma^2$。所以**降低模型之间的相关性**是关键。

**随机森林** 在 Bagging 基础上，每次分裂只从随机抽取的 $m$ 个特征中选（分类常取 $m=\sqrt p$，回归常取 $p/3$），进一步降低树之间的相关性 $\rho$。

| 概念 | 说明 |
|---|---|
| 袋外误差（OOB） | 每个样本约有 36.8% 的树没用到它，用这些树预测它，得到近似交叉验证误差 |
| 特征重要性（MDI） | 每个特征带来的不纯度下降之和；偏向取值多、连续的特征 |
| 置换重要性（MDA） | 打乱某特征后验证集误差上升多少；更可靠但更慢 |
| 增加树的数量 | 不会过拟合（只是方差趋于 $\rho\sigma^2$），但边际收益递减 |

> **量化用途**：AFML 第 6 章指出，金融样本标签重叠时，自助采样会抽到大量相互重叠的样本，袋外误差偏乐观；建议降低 `max_samples` 或按样本唯一性（uniqueness）加权采样。

---

## 6. GBDT、XGBoost、LightGBM

### 6.1 Boosting 与 GBDT

**Bagging 降方差，Boosting 降偏差。** Boosting 串行地训练弱学习器，每个新模型修正前面模型的错误。

**前向分步加法模型**：$F_m(x)=F_{m-1}(x)+\eta\,h_m(x)$，$\eta$ 为学习率（shrinkage）。

**GBDT（Friedman 2001）**：第 $m$ 步用一棵回归树拟合损失的**负梯度**（伪残差）

$$r_{im}=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}$$

| 损失 | 负梯度 |
|---|---|
| 平方损失 $\tfrac12(y-F)^2$ | $y-F$（就是残差） |
| 绝对损失 | $\mathrm{sign}(y-F)$ |
| 对数损失（$p=\sigma(F)$） | $y-p$ |

拟合出树结构后，每个叶子的输出再按原损失单独优化（平方损失下就是叶内残差均值）。

**AdaBoost** 是指数损失下的前向分步算法，通过提高错分样本的权重迭代（李航第 8 章）。

### 6.2 XGBoost：二阶展开（必须会推）

**目标函数**（第 $t$ 轮，新加的树记为 $f_t$）：

$$\mathcal L^{(t)}=\sum_{i=1}^n L\big(y_i,\hat y_i^{(t-1)}+f_t(x_i)\big)+\Omega(f_t),\qquad \Omega(f)=\gamma T+\tfrac12\lambda\sum_{j=1}^T w_j^2$$

$T$ 是叶子数，$w_j$ 是叶子 $j$ 的输出值。

**第 1 步：二阶泰勒展开**。记 $g_i=\partial_{\hat y}L(y_i,\hat y_i^{(t-1)})$，$h_i=\partial^2_{\hat y}L(y_i,\hat y_i^{(t-1)})$：

$$\mathcal L^{(t)}\approx\sum_i\Big[L(y_i,\hat y_i^{(t-1)})+g_if_t(x_i)+\tfrac12h_if_t^2(x_i)\Big]+\Omega(f_t)$$

第一项与 $f_t$ 无关，去掉。

**第 2 步：按叶子归并**。记 $I_j=\{i:x_i\text{ 落在叶子 }j\}$，$G_j=\sum_{i\in I_j}g_i$，$H_j=\sum_{i\in I_j}h_i$。由于同一叶子内 $f_t(x_i)=w_j$：

$$\tilde{\mathcal L}^{(t)}=\sum_{j=1}^T\Big[G_jw_j+\tfrac12(H_j+\lambda)w_j^2\Big]+\gamma T$$

**第 3 步：对每个 $w_j$ 求最小值**（关于 $w_j$ 的一元二次函数）：

$$w_j^*=-\frac{G_j}{H_j+\lambda},\qquad \tilde{\mathcal L}^*=-\frac12\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T$$

$\tilde{\mathcal L}^*$ 可以当作树结构的打分，越小越好。

**第 4 步：分裂增益**。把一个叶子分成左右两个，目标函数的下降量为

$$\mathrm{Gain}=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$$

$\mathrm{Gain}\le0$ 时不分裂，所以 $\gamma$ 起到预剪枝作用。

**小例子**：4 个样本，$x=1,2,3,4$，$y=0,0,1,1$，对数损失，初始预测概率 $0.5$。对数损失关于 logit 的 $g_i=p_i-y_i$，$h_i=p_i(1-p_i)$，所以 $g=(0.5,0.5,-0.5,-0.5)$，$h_i=0.25$。取 $\lambda=1,\gamma=0$，在 $x=2.5$ 处分裂：$G_L=1,H_L=0.5$，$G_R=-1,H_R=0.5$。

- 叶子权重：$w_L=-1/1.5=-0.667$，$w_R=0.667$；
- 增益：$\frac12\left[\frac{1}{1.5}+\frac1{1.5}-\frac{0}{2}\right]=0.667$。

 用 xgboost 3.4（`max_depth=1, learning_rate=1, reg_lambda=1, base_score=0.5`）验证，叶子值为 $\mp0.6667$；导出的 `gain=1.333`，是上式**不含系数 $\tfrac12$** 的值，读模型时注意。

**为什么用二阶**：

| 理由 | 说明 |
|---|---|
| 收敛更快 | 相当于在函数空间做牛顿步，而 GBDT 是梯度步 |
| 统一框架 | 任意二阶可导损失只需提供 $g_i,h_i$，树的构建代码不变 |
| $h_i$ 作为样本权重 | $\tilde{\mathcal L}$ 可写成 $\sum_i\tfrac12h_i(f_t(x_i)+g_i/h_i)^2$，即以 $h_i$ 为权重拟合 $-g_i/h_i$；`min_child_weight` 限制的就是叶内 $\sum h_i$ |

**XGBoost 的其他设计**：

| 设计 | 作用 |
|---|---|
| 正则项 $\gamma T+\tfrac12\lambda\lVert w\rVert^2$（及 L1 项 $\alpha$） | 控制叶子数和叶子值大小 |
| 学习率 $\eta$ | 每棵树乘以 $\eta$，给后续树留空间 |
| 行采样 `subsample`、列采样 `colsample_bytree` 等 | 降低方差，借鉴随机森林 |
| 稀疏感知分裂 | 缺失值分别尝试放到左、右，选增益大的方向作为默认方向 |
| 加权分位数草图 | 以 $h_i$ 为权重找候选切分点 |
| 列块结构、缓存优化、并行 | 特征预排序存成块，在特征维度上并行找切分点 |

### 6.3 LightGBM：三项优化

LightGBM（Ke et al., NeurIPS 2017）针对"样本多、特征多时找切分点太慢"做了以下改进：

| 技术 | 做法 | 效果 |
|---|---|---|
| **直方图算法** | 把连续特征离散成最多 `max_bin`（默认 255）个桶，只在桶边界上找切分点；父节点直方图减去一个子节点直方图得到另一个子节点（**直方图做差**） | 找切分点的代价从 $O(\#\text{数据})$ 降到 $O(\#\text{桶})$，内存用 uint8 存桶号；离散化本身也有正则作用 |
| **GOSS**（单边梯度采样） | 保留梯度绝对值最大的 $a\times100\%$ 样本，从其余样本中随机抽 $b\times100\%$，并把抽到的小梯度样本权重乘以 $\frac{1-a}{b}$ | 梯度小的样本已经拟合得较好，少用它们；乘权重保持增益估计近似无偏 |
| **EFB**（互斥特征捆绑） | 很少同时非零的稀疏特征（如 one-hot）捆绑成一个特征，用偏移量区分原特征的取值；捆绑问题等价于图着色，用贪心近似 | 减少有效特征数 |
| **Leaf-wise 生长** | 每次在所有叶子中选增益最大的一个分裂（XGBoost 默认 level-wise，逐层分裂） | 同样叶子数下损失更低，但容易长出很深的树、过拟合，需配合 `num_leaves` 和 `max_depth` |
| 类别特征原生支持 | 按每个类别的 $G/H$ 排序后找最优划分 | 不必 one-hot |

> 在 LightGBM 4.x 中，GOSS 通过 `data_sample_strategy="goss"` 开启（旧写法 `boosting="goss"` 已弃用）。

### 6.4 常用超参数与调参顺序

| 类别 | XGBoost | LightGBM | 作用 |
|---|---|---|---|
| 树的复杂度 | `max_depth` | `num_leaves`、`max_depth` | 越大越容易过拟合 |
| 叶子约束 | `min_child_weight` | `min_child_samples`、`min_sum_hessian_in_leaf` | 防止叶子过小 |
| 正则 | `reg_lambda`、`reg_alpha`、`gamma` | `lambda_l2`、`lambda_l1`、`min_gain_to_split` | 收缩叶子值，提高分裂门槛 |
| 采样 | `subsample`、`colsample_bytree` | `bagging_fraction`（需配 `bagging_freq`）、`feature_fraction` | 降方差 |
| 迭代 | `n_estimators`、`learning_rate` | 同左 | 学习率越小需要的树越多，配合早停 |

调参顺序：先固定较大学习率（0.05–0.1）用早停确定树的数量；再调树的复杂度和叶子约束；再调采样和正则；最后降学习率、增加树。

### 6.5 特征重要性

| 方法 | 含义 | 问题 |
|---|---|---|
| split（次数） | 特征被用于分裂的次数 | 偏向连续、取值多的特征 |
| gain（增益） | 特征带来的总增益 | 相关特征之间会分摊重要性 |
| 置换重要性 | 打乱特征后验证误差的上升 | 相关特征打乱一个，另一个可替代，重要性被低估 |
| SHAP | 基于 Shapley 值的逐样本贡献，满足可加性 | 计算量大；仍受特征相关影响 |

> **量化用途**：GBDT 是量化因子合成的主力模型。微软 Qlib 的基准中，LightGBM + Alpha158 是常用的基线。与神经网络相比，GBDT 对特征尺度不敏感、对噪声较稳健、训练快，便于每月或每周滚动重训。

---

## 7. SVM、kNN、聚类、PCA

### 7.1 支持向量机

**硬间隔**：线性可分时，求间隔最大的超平面

$$\min_{w,b}\ \tfrac12\lVert w\rVert^2\quad\text{s.t. } y_i(w^\top x_i+b)\ge1$$

几何间隔为 $1/\lVert w\rVert$，所以最大化间隔等价于最小化 $\lVert w\rVert$。

**软间隔**：允许违反约束，等价于最小化 hinge 损失加 L2 正则

$$\min_{w,b}\ \tfrac12\lVert w\rVert^2+C\sum_i\max\big(0,1-y_i(w^\top x_i+b)\big)$$

$C$ 越大，对误分类惩罚越重，越容易过拟合。

**对偶问题**：

$$\max_\alpha\ \sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,x_i^\top x_j\quad\text{s.t. }0\le\alpha_i\le C,\ \sum_i\alpha_iy_i=0$$

$w=\sum_i\alpha_iy_ix_i$。只有 $\alpha_i>0$ 的样本（**支持向量**）影响解。对偶问题中样本只以内积 $x_i^\top x_j$ 出现，替换成核函数 $K(x_i,x_j)$ 即得非线性 SVM（**核技巧**）。常用核：线性、多项式、RBF $K=\exp(-\gamma\lVert x-x'\rVert^2)$。

| 优点 | 缺点 |
|---|---|
| 凸优化，全局最优；高维小样本效果好；核方法表达力强 | 样本量大时训练慢（核矩阵 $O(n^2)$）；输出不是概率；对特征尺度敏感 |

### 7.2 kNN

预测时找训练集中距离最近的 $k$ 个样本，分类投票、回归取平均。

- $k$ 小：方差大、易过拟合；$k$ 大：偏差大。$k=1$ 时训练误差为 0。
- 需要标准化；高维时距离失去区分度（**维数灾难**）。
- 训练 $O(1)$，预测 $O(nd)$；可用 KD 树、球树加速（低维有效）。

> **量化用途**：可用于"相似历史行情"检索（类比法），但高维时效果很差，一般先降维。

### 7.3 聚类

**K-means**：最小化簇内平方和 $\sum_k\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$。Lloyd 算法交替执行"分配到最近中心"和"更新中心为簇均值"，每一步目标函数不增，因此收敛，但只到局部最优。

| 问题 | 方法 |
|---|---|
| 初值敏感 | k-means++：按与已选中心距离的平方成比例抽取下一个中心；多次初始化取最好 |
| 选 $k$ | 肘部法、轮廓系数、Gap 统计量 |
| 簇形状 | 只适合凸的、大小相近的簇；非凸用 DBSCAN、谱聚类 |
| 软分配 | 高斯混合模型（GMM），用 EM 算法估计；K-means 是 GMM 在协方差 $\sigma^2I$、$\sigma\to0$ 时的极限 |

**层次聚类**：自底向上合并，链接方式有 single、complete、average、Ward。

> **量化用途**：按收益相关性对股票聚类，构造数据驱动的行业分类；López de Prado 的 HRP（层次风险平价）先对资产做层次聚类再分配权重。

### 7.4 PCA

**目标**：找方差最大的正交方向。对中心化数据 $X$（$n\times p$），协方差 $S=\frac1{n-1}X^\top X$，第一主成分方向是

$$v_1=\arg\max_{\lVert v\rVert=1}v^\top Sv$$

由拉格朗日乘子法得 $Sv=\lambda v$，即 $v_1$ 是 $S$ 最大特征值对应的特征向量，投影方差等于该特征值。

| 要点 | 说明 |
|---|---|
| 计算 | 对 $X$ 做 SVD：$X=U\Sigma V^\top$，$V$ 的列是主成分方向，$\lambda_k=\sigma_k^2/(n-1)$ |
| 方差解释比例 | $\lambda_k/\sum_j\lambda_j$ |
| 必须中心化；量纲不同时要标准化 | 否则主成分被大尺度变量主导 |
| 等价表述 | 最大化投影方差 ⇔ 最小化重构误差 |
| 局限 | 只捕捉线性结构；主成分不一定与预测目标相关（无监督） |

> **量化用途**：对股票收益矩阵做 PCA，第一主成分近似市场因子（各股载荷大多同号），之后几个对应行业或风格。统计风险模型、因子去相关（对称正交化）、协方差矩阵降噪都用到 PCA。

---

## 8. 评价指标

### 8.1 分类指标

**混淆矩阵**：

| | 预测正 | 预测负 |
|---|---|---|
| 实际正 | TP | FN |
| 实际负 | FP | TN |

| 指标 | 公式 | 关注 |
|---|---|---|
| 准确率 | $(TP+TN)/N$ | 类别不平衡时失真 |
| 精确率 Precision | $TP/(TP+FP)$ | 预测为正的里面有多少是真的 |
| 召回率 Recall（TPR） | $TP/(TP+FN)$ | 真正的正例找回了多少 |
| 假正率 FPR | $FP/(FP+TN)$ | ROC 横轴 |
| F1 | $\dfrac{2PR}{P+R}$ | 精确率与召回率的调和平均 |

### 8.2 ROC 与 AUC

ROC 曲线：阈值从高到低变化，横轴 FPR、纵轴 TPR。AUC 是曲线下面积。

**概率解释（必会）**：AUC 等于随机取一个正样本和一个负样本，正样本得分高于负样本的概率（得分相同计 1/2）：

$$\mathrm{AUC}=\frac{1}{n_+n_-}\sum_{i\in+}\sum_{j\in-}\Big(\mathbf 1[s_i>s_j]+\tfrac12\mathbf 1[s_i=s_j]\Big)$$

等价于 Mann-Whitney U 统计量除以 $n_+n_-$，可以用秩计算：

$$\mathrm{AUC}=\frac{\sum_{i\in+}\mathrm{rank}_i-\frac{n_+(n_++1)}{2}}{n_+n_-}$$

**例**：得分 $(0.9,0.8,0.7,0.6,0.55,0.4,0.3)$，标签 $(1,1,0,1,0,0,1)$。4 个正样本分别高于 3、3、2、0 个负样本，$\mathrm{AUC}=8/12=0.667$。 与 `sklearn.metrics.roc_auc_score` 一致。

| 性质 | 说明 |
|---|---|
| 只依赖排序 | 得分做任意单调变换，AUC 不变；不反映概率是否校准 |
| 对类别比例不敏感 | TPR、FPR 分别在正、负类内部计算 |
| 不平衡时的补充 | 正例很少时 ROC 看起来很好，PR 曲线（精确率-召回率）更能反映问题 |
| 随机猜测 | AUC = 0.5；AUC < 0.5 说明信号方向反了 |

### 8.3 交叉熵与对数损失

二分类对数损失 $-\frac1n\sum[y\log p+(1-y)\log(1-p)]$。多分类交叉熵 $-\frac1n\sum_i\log p_{i,y_i}$。

- 对过度自信的错误惩罚极重：真实为 1 却预测 $p=0.01$，损失 $-\ln0.01=4.6$。
- 恒预测先验概率 $\pi$ 的对数损失是 $H(\pi)$（以 $e$ 为底），模型必须低于它才有价值。
- 交叉熵 = 真实分布的熵 + KL 散度，所以最小化交叉熵等于最小化 KL 散度。

### 8.4 回归与量化指标

| 指标 | 说明 |
|---|---|
| MSE、MAE | 对尺度敏感，受极端收益影响大 |
| $R^2$ | 样本外 $R^2_{oos}=1-\sum(y-\hat y)^2/\sum(y-\bar y_{train})^2$ 可以为负 |
| IC / RankIC | 每期截面上预测与实际收益的 Pearson / Spearman 相关，再取时间平均 |
| ICIR | IC 均值 / IC 标准差 |
| 分层收益、多空 Sharpe、换手 | 最终评价标准 |

> **量化用途**：收益预测的 MSE 或 $R^2$ 往往很难看（日频 $R^2$ 在 1% 以下很常见），但只要截面排序正确，就能赚钱。所以量化中更看重 RankIC 和组合层面的指标，模型选择时也常用验证集 IC 做早停。

---

## 9. 特征工程、类别不平衡、过拟合排查

### 9.1 特征工程

| 操作 | 方法 | 量化中的注意点 |
|---|---|---|
| 缺失值 | 均值/中位数填充、加缺失指示变量、树模型原生处理 | 缺失本身可能有信息（如未披露）；填充统计量只用当天截面或训练集 |
| 异常值 | 截尾（winsorize）、MAD 去极值、rank 变换 | 截面上处理 |
| 标准化 | z-score、min-max、rank 到 $[0,1]$ 或高斯化 | **截面标准化**同时去掉了时间上的水平漂移，缓解非平稳 |
| 类别特征 | one-hot、目标编码（折外）、嵌入 | 行业哑变量 |
| 构造 | 比率、差分、滚动统计、交互项 | 所有滚动窗口只能向后看 |
| 降维与去相关 | PCA、正交化、聚类后取代表 | 避免多个高度相关的因子重复计权 |
| 中性化 | 对行业、市值回归取残差 | 让模型学到的不只是风格暴露 |

### 9.2 类别不平衡

| 方法 | 做法 | 注意 |
|---|---|---|
| 评价指标 | 用 PR-AUC、F1、召回率，不用准确率 | — |
| 重采样 | 欠采样多数类、过采样少数类、SMOTE | 只在训练集上做；时间序列中 SMOTE 插值会混合不同时期的样本，慎用 |
| 类别权重 | 损失中给少数类更大权重（`class_weight="balanced"`、`scale_pos_weight`） | 会改变输出概率，需要重新校准 |
| 阈值调整 | 不改模型，按业务需求选阈值 | 最简单 |
| 改造标签 | 回归或排序代替分类 | 量化中常用 |

> **量化用途**：用三重障碍法（AFML 第 3 章）打标签时，"触及止盈/止损/时间到期"三类往往不平衡；股票截面上做"涨停预测"之类任务时正例极少。

### 9.3 过拟合排查清单

| 现象 | 可能原因 | 检查方法 |
|---|---|---|
| 训练好、验证差 | 模型太复杂 | 学习曲线；加正则看验证误差变化 |
| 验证好、测试差 | 在验证集上调参过度 | 统计尝试过的配置数量；换一段测试期 |
| 样本外极好（如日频 IC 0.2） | **泄漏** | 逐个排查第 3.5 节的泄漏形式；把标签往后错一期，若效果依旧很好，说明特征中有未来信息 |
| 换随机种子结果差异大 | 方差大 | 多种子平均，报告均值与标准差 |
| 少数特征重要性极高 | 可能是泄漏特征或 ID 类特征 | 删除后对比 |
| 只在某一段时间有效 | 机制依赖市场状态 | 分年、分市场状态评估 |

---

# 第三部分：深度学习

## 10. 神经网络与反向传播

### 10.1 多层感知机

一层全连接：$z=Wa_{\text{prev}}+b$，$a=\phi(z)$。没有非线性激活时，多层网络仍然是线性模型。

| 激活函数 | 公式 | 导数 | 特点 |
|---|---|---|---|
| Sigmoid | $\frac1{1+e^{-z}}$ | $\sigma(1-\sigma)\le\frac14$ | 输出 $(0,1)$；饱和区梯度消失；非零中心 |
| Tanh | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $1-\tanh^2\le1$ | 零中心；仍会饱和 |
| ReLU | $\max(0,z)$ | $\mathbf 1[z>0]$ | 计算快，正区间不饱和；可能出现"死神经元" |
| Leaky ReLU | $\max(\alpha z,z)$ | $\alpha$ 或 1 | 缓解死神经元 |
| GELU | $z\,\Phi(z)$ | — | Transformer 常用，平滑 |

### 10.2 反向传播

反向传播就是**链式法则 + 动态规划**：从输出层往回逐层计算损失对每层输入的梯度，复用已算出的中间结果。

以两层网络、softmax + 交叉熵为例（批大小 $n$，行向量写法）：

$$Z_1=XW_1+b_1,\quad A_1=\mathrm{ReLU}(Z_1),\quad Z_2=A_1W_2+b_2,\quad P=\mathrm{softmax}(Z_2),\quad L=-\tfrac1n\sum_i\log P_{i,y_i}$$

反向：

| 步骤 | 梯度 |
|---|---|
| 输出层 | $\dfrac{\partial L}{\partial Z_2}=\dfrac1n(P-Y)$（$Y$ 为 one-hot），这是 softmax 与交叉熵组合后的简洁结果 |
| 第二层参数 | $\dfrac{\partial L}{\partial W_2}=A_1^\top\dfrac{\partial L}{\partial Z_2}$，$\dfrac{\partial L}{\partial b_2}=\sum_i\dfrac{\partial L}{\partial Z_2}$ |
| 传到隐藏层 | $\dfrac{\partial L}{\partial A_1}=\dfrac{\partial L}{\partial Z_2}W_2^\top$，$\dfrac{\partial L}{\partial Z_1}=\dfrac{\partial L}{\partial A_1}\odot\mathbf 1[Z_1>0]$ |
| 第一层参数 | $\dfrac{\partial L}{\partial W_1}=X^\top\dfrac{\partial L}{\partial Z_1}$ |

**维度检查法**：$\partial L/\partial W$ 的形状必须与 $W$ 相同，据此可以确定矩阵乘法的顺序和转置位置。

**梯度检验**：用中心差分 $\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$ 近似每个参数的梯度，与反向传播结果比较相对误差（双精度下应在 $10^{-7}$ 量级以下）。

### 10.3 梯度消失与爆炸

$L$ 层网络的梯度含连乘 $\prod_l W_l^\top\,\mathrm{diag}(\phi'(z_l))$。若每层的放大倍数小于 1，梯度指数衰减；大于 1，指数爆炸。

| 对策 | 原理 |
|---|---|
| ReLU 类激活 | 正区间导数为 1 |
| 合适的初始化 | Xavier：$\mathrm{Var}(W)=\frac{2}{n_{in}+n_{out}}$（tanh）；He：$\mathrm{Var}(W)=\frac{2}{n_{in}}$（ReLU），使各层激活方差大致不变 |
| 残差连接 | $a_{l+1}=a_l+F(a_l)$，梯度可以沿恒等支路直接传回 |
| 归一化（BN/LN） | 稳定每层输入分布 |
| 梯度裁剪 | 梯度范数超过阈值时等比缩小，主要防爆炸（RNN 常用） |
| LSTM/GRU 门控 | 见第 12 节 |

---

## 11. 优化器与归一化

### 11.1 优化器

| 优化器 | 更新规则 | 特点 |
|---|---|---|
| SGD | $\theta\leftarrow\theta-\eta g$ | 噪声有助于跳出尖锐极小值；对学习率敏感 |
| Momentum | $v\leftarrow\beta v+g$，$\theta\leftarrow\theta-\eta v$ | 累积历史方向，减少震荡，加速穿过平坦区域 |
| Nesterov | 在"预测位置" $\theta-\eta\beta v$ 处求梯度 | 动量的改进 |
| AdaGrad | $s\leftarrow s+g^2$，$\theta\leftarrow\theta-\eta g/(\sqrt s+\epsilon)$ | 每个参数自适应学习率；$s$ 只增不减，后期学习率过小 |
| RMSProp | $s\leftarrow\rho s+(1-\rho)g^2$ | 用指数平均代替累加 |
| Adam | 见下 | 动量 + RMSProp + 偏差校正，默认首选 |
| AdamW | Adam + 解耦的权重衰减 | Transformer 训练的标准做法 |

**Adam**（Kingma & Ba 2015），默认 $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad \theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

**偏差校正的原因**：$m_0=v_0=0$，初期 $m_t$、$v_t$ 偏向 0。例如 $t=1$ 时 $m_1=0.1g_1$，除以 $1-0.9=0.1$ 后恢复为 $g_1$。校正后第一步的步长约为 $\eta\cdot\mathrm{sign}(g_1)$，与梯度大小无关。

**L2 正则与权重衰减的区别**：在 SGD 中，损失加 $\frac\lambda2\lVert\theta\rVert^2$ 与每步把参数乘以 $(1-\eta\lambda)$ 等价；在 Adam 中，L2 的梯度 $\lambda\theta$ 也会被 $\sqrt{\hat v}$ 缩放，梯度大的参数衰减反而弱。AdamW（Loshchilov & Hutter 2019）把衰减项从梯度中拿出来，直接作用于参数。

**学习率调度**：warmup（前若干步线性升高，Adam 早期方差估计不准）、余弦退火、按验证损失衰减。

### 11.2 Batch Normalization 与 Layer Normalization

两者都是"减均值、除标准差，再乘可学习的 $\gamma$、加 $\beta$"，区别在**沿哪个维度统计**：

$$\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\hat x+\beta$$

| | BatchNorm（Ioffe & Szegedy 2015） | LayerNorm（Ba et al. 2016） |
|---|---|---|
| 统计维度 | 同一特征、跨样本（batch 维） | 同一样本、跨特征 |
| 训练与推理 | 训练用当前 batch 统计量，推理用训练时的滑动平均 | 两者一致 |
| 依赖 batch 大小 | 是，batch 小时估计不准 | 否 |
| 适用 | CNN | RNN、Transformer、变长序列 |

**为什么有效**：最初的解释是减少"内部协变量偏移"；后来的研究（Santurkar et al. 2018）认为主要原因是让损失曲面更平滑，允许更大的学习率。

> **量化用途**：股票截面数据中，一个 batch 若取同一天的全部股票，BN 的统计量就是当天截面的均值方差，相当于自动做了截面标准化；若 batch 混合多天，则会混入时间上的信息。设计时要清楚自己在哪个维度归一化。

### 11.3 Dropout

训练时每个神经元以概率 $p$ 置零。**inverted dropout**：保留的神经元除以 $1-p$，使期望不变，推理时不做任何处理。

- 解释：每次训练一个随机子网络，推理时近似对指数多个子网络取平均（集成）。
- 防止神经元之间的共适应。
- 与 BN 同用时要注意顺序和方差偏移问题。

---

## 12. RNN、LSTM、Transformer

### 12.1 RNN

$$h_t=\tanh(W_hh_{t-1}+W_xx_t+b)$$

参数在时间上共享。沿时间反向传播（BPTT）时梯度含 $\prod_t W_h^\top\mathrm{diag}(1-h_t^2)$，长序列下梯度消失或爆炸，难以学习长期依赖。

### 12.2 LSTM

LSTM（Hochreiter & Schmidhuber 1997）引入细胞状态 $c_t$ 和三个门（$\sigma$ 为 sigmoid，$[\cdot,\cdot]$ 为拼接）：

| 组件 | 公式 | 作用 |
|---|---|---|
| 遗忘门 | $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$ | 保留多少旧记忆 |
| 输入门 | $i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$ | 写入多少新信息 |
| 候选记忆 | $\tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c)$ | 新信息内容 |
| 细胞状态 | $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ | **加法更新** |
| 输出门 | $o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$ | 输出多少 |
| 隐状态 | $h_t=o_t\odot\tanh(c_t)$ | — |

**为什么缓解梯度消失**：$\partial c_t/\partial c_{t-1}=f_t$（沿细胞状态的直接路径），是逐元素乘遗忘门而不是乘权重矩阵；遗忘门接近 1 时梯度几乎无损地传回。实践中常把遗忘门偏置初始化为 1。

参数量：4 组门，每组 $(d_h+d_x)\times d_h+d_h$，共 $4\big[(d_h+d_x)d_h+d_h\big]$。GRU 把遗忘门和输入门合并为更新门，只有 3 组参数。

### 12.3 注意力与 Transformer

**缩放点积注意力**（Vaswani et al. 2017）：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $Q=XW_Q$，$K=XW_K$，$V=XW_V$。每个位置的输出是所有位置 $V$ 的加权平均，权重由查询与键的相似度决定。
- **为什么除以 $\sqrt{d_k}$**：若 $q,k$ 各分量独立、均值 0、方差 1，则 $q^\top k=\sum_{i=1}^{d_k}q_ik_i$ 的方差为 $d_k$。不缩放时 $d_k$ 越大，点积越大，softmax 越接近 one-hot，梯度越小。
- **掩码**：解码器中把未来位置的分数置为 $-\infty$（因果掩码），保证位置 $t$ 只看到 $\le t$ 的信息。在金融时间序列上，这是防止前视偏差的必要条件。

**多头注意力**：把 $d_{model}$ 拆成 $h$ 个头，每个头 $d_k=d_{model}/h$，分别做注意力后拼接再线性变换。总计算量与单头相近，但能在不同子空间学习不同的关系。参数量为 $4d_{model}^2$（$W_Q,W_K,W_V,W_O$，不计偏置）。

**Transformer 块**：多头自注意力 → 残差 + LayerNorm → 前馈网络（两层 MLP，中间维度通常为 $4d_{model}$）→ 残差 + LayerNorm。现代实现多把 LN 放在子层之前（Pre-LN），训练更稳定。

**位置编码**：注意力本身对输入顺序是置换等变的，需要显式加入位置信息。原论文用正弦编码 $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$，$PE_{(pos,2i+1)}=\cos(\cdot)$；也可学习位置嵌入，或用相对位置编码（如 RoPE）。

| | RNN / LSTM | Transformer |
|---|---|---|
| 长程依赖 | 信息需逐步传递，路径长度 $O(n)$ | 任意两位置直接相连，路径长度 $O(1)$ |
| 并行 | 时间上串行 | 序列内并行 |
| 复杂度 | $O(nd^2)$ | $O(n^2d)$，长序列内存开销大 |
| 数据需求 | 较小 | 大，归纳偏置弱 |

> **量化用途**：
> - 时间维度：对单只股票的历史序列做注意力（如分钟级量价序列），注意因果掩码。
> - 截面维度：把同一天的股票当作"序列"做注意力，学习股票间的关系；此时不需要位置编码，且天然满足置换等变。
> - 金融数据样本有效量小、噪声大，Transformer 常常不比 GBDT 或简单的 MLP 好，需要强正则和大量集成。

---

# 第四部分：量化应用

## 13. 机器学习在量化中的应用与陷阱

### 13.1 金融数据为什么难

| 特点 | 表现 | 后果 |
|---|---|---|
| **低信噪比** | 单因子日频 IC 0.03–0.05 已有价值；单变量线性预测下 $R^2=\mathrm{IC}^2$，IC 0.05 对应 $R^2=0.25\%$ | 模型很容易拟合噪声；验证分数的波动可能大于模型之间的差距 |
| **非平稳** | 风格切换、监管变化、市场参与者结构变化 | 历史规律会失效；需要滚动训练，越近的数据越重要 |
| **对抗性 / 拥挤** | 信号被越来越多的资金交易后被抹平 | 样本外衰减；收益与容量此消彼长 |
| **样本有效量小** | 日频 10 年约 2500 个交易日；截面样本之间高度相关 | 看起来有几百万行，独立信息远少于行数 |
| **重尾与异常值** | 涨跌停、停复牌、财务造假 | 平方损失被极端值主导 |

Gu, Kelly & Xiu（2020，RFS）用美股 1957–2016 年数据比较多种 ML 模型预测个股月收益，最好的神经网络样本外月度 $R^2$ 也只有约 0.4%，但据此构造的多空组合 Sharpe 显著高于线性模型。**$R^2$ 小不代表没有经济价值。**

### 13.2 因子合成：从线性到 ML

| 方法 | 做法 | 优缺点 |
|---|---|---|
| 等权 | 标准化后直接相加 | 最稳健，无需估计 |
| IC / ICIR 加权 | 按历史 IC 或 ICIR 赋权 | 简单；忽略因子间相关 |
| 最大化 ICIR | $w\propto\Sigma_{IC}^{-1}\overline{IC}$ | 考虑相关性；协方差估计误差大时需收缩 |
| 线性回归 / Ridge | 截面上以因子预测下期收益 | 可解释；只能捕捉线性关系 |
| GBDT | 以全部因子为特征，滚动训练 | 捕捉非线性与交互；当前主流 |
| 神经网络 | MLP、时序模型、截面注意力、GNN | 端到端，可直接用原始量价；调参与算力成本高 |

**ML 合成的标准流程**：

1. **标签**：未来 $h$ 日收益，通常做截面处理（去极值、标准化、排名、对行业市值中性化后的超额收益）。
2. **特征**：截面标准化后的因子值；可加入行业、市值作为条件变量。
3. **切分**：滚动窗口，训练与验证之间留出至少 $h$ 天（purge）。
4. **损失与早停**：MSE 或排序损失；早停指标常用验证集的平均 RankIC 而不是 MSE。
5. **集成**：多随机种子、多训练窗口、多模型（GBDT + NN）平均，降低方差。
6. **评估**：IC 序列、分层收益、多空和多头超额、换手、扣费后收益、风格暴露。

### 13.3 量化特有的陷阱

| 陷阱 | 说明 | 对策 |
|---|---|---|
| 标签与特征时间错位 | 用 $t$ 日收盘价算特征，标签用 $t$ 日收盘到 $t+1$ 日收盘；但实际 $t$ 日收盘才拿到信号，最早 $t+1$ 日开盘成交 | 标签改为 $t+1$ 日开盘（或 VWAP）到 $t+h+1$ 日 |
| 不可交易样本 | 涨停买不到、跌停卖不出、停牌 | 训练时可保留，回测和评估时剔除或按实际成交模拟 |
| 预测了风格而非 alpha | 模型主要靠市值、波动率等风格得分 | 标签和特征做中性化；检查预测值的风格暴露 |
| 回测过拟合 | 尝试了大量模型和参数，只报告最好的 | 记录尝试次数；用 Deflated Sharpe Ratio（Bailey & López de Prado 2014）、CPCV |
| 换手过高 | 预测值逐日波动，扣费后收益消失 | 对预测做时间平滑；组合优化中加换手约束 |
| 数据窥探 | 特征的构造思路来自对全样本的观察 | 保留一段从未看过的最终测试期 |

### 13.4 "样本外为什么变差" 的回答框架

1. **过拟合**：参数多、尝试次数多、验证期太短。→ 看学习曲线，统计尝试次数，做多重检验修正。
2. **泄漏**：特征或预处理用了未来信息。→ 逐项排查第 3.5 节表格；把标签错后一期做"安慰剂"检验。
3. **市场变化**：风格切换、因子拥挤、监管变化。→ 分时间段看 IC，看因子与拥挤度指标的关系。
4. **实现差异**：成交价、成本、可交易性在回测与实盘中不一致。→ 用实盘成交数据校准成本模型。
5. **统计噪声**：样本外期太短，差异本身不显著。→ 计算 IC 均值的 t 值和置信区间。

### 13.5 其他应用方向

| 方向 | 说明 |
|---|---|
| 自然语言处理 / 大模型 | 新闻、研报、公告的情绪与事件抽取；用大模型辅助因子挖掘和代码编写 |
| 图神经网络 | 以供应链、同行业、基金共同持股等关系建图，做信息传播 |
| 强化学习 | 交易执行（拆单）、做市；难点是模拟环境与真实市场的差距、奖励稀疏 |
| 遗传规划 | 自动搜索因子表达式（如 gplearn），需严格控制过拟合 |
| 生成模型 | 生成合成行情做压力测试，数据增强 |

---

## 14. 面试高频问题速查

| 问题 | 要点 | 详见 |
|---|---|---|
| 偏差-方差分解 | $\sigma^2+\text{Bias}^2+\text{Var}$，会推交叉项为 0 | §2.1 |
| L1 为什么稀疏 | 菱形顶点；软阈值 $(\lvert z\rvert-\lambda)_+$ | §2.2 |
| 时间序列为什么不能随机 K 折 | 自相关、标签重叠、非平稳 | §3.2 |
| Purge 和 Embargo 的区别 | Purge 处理标签区间重叠；Embargo 处理验证集之后的序列相关 | §3.4 |
| 逻辑回归梯度 | $X^\top(p-y)$，损失为凸 | §4.2 |
| 随机森林为什么要随机选特征 | 降低树间相关 $\rho$，方差 $\rho\sigma^2+\frac{1-\rho}B\sigma^2$ | §5.2 |
| Bagging 与 Boosting | 并行降方差 / 串行降偏差 | §5.2、§6.1 |
| XGBoost 叶子权重与增益 | $w^*=-G/(H+\lambda)$；Gain 公式 | §6.2 |
| XGBoost 相对 GBDT 的改进 | 二阶展开、正则项、缺失值默认方向、列采样、并行 | §6.2 |
| LightGBM 为什么快 | 直方图（含做差）、GOSS、EFB、leaf-wise | §6.3 |
| GOSS 小梯度样本权重 | $(1-a)/b$ | §6.3 |
| SVM 对偶与核技巧 | 样本只以内积出现；支持向量 $\alpha_i>0$ | §7.1 |
| PCA 与 SVD 的关系 | $V$ 为主成分方向，$\lambda=\sigma^2/(n-1)$ | §7.4 |
| AUC 的概率含义 | 正样本得分高于负样本的概率；只看排序 | §8.2 |
| 不平衡时用什么指标 | PR-AUC、F1、召回率 | §8.2、§9.2 |
| 过拟合怎么排查 | 学习曲线、泄漏检查、多种子、分时间段 | §9.3 |
| softmax + 交叉熵的梯度 | $P-Y$ | §10.2 |
| Adam 偏差校正 | 初值为 0 导致早期估计偏小 | §11.1 |
| Adam 与 AdamW | L2 梯度被自适应缩放 / 解耦衰减 | §11.1 |
| BN 与 LN | 跨样本 / 跨特征统计；训练推理是否一致 | §11.2 |
| Dropout 推理时怎么处理 | inverted dropout 训练时除以 $1-p$，推理不变 | §11.3 |
| LSTM 为什么缓解梯度消失 | 细胞状态加法更新，$\partial c_t/\partial c_{t-1}=f_t$ | §12.2 |
| 注意力为什么除以 $\sqrt{d_k}$ | 点积方差为 $d_k$，防止 softmax 饱和 | §12.3 |
| Transformer 复杂度 | $O(n^2d)$ | §12.3 |
| 金融 ML 为什么难 | 低信噪比、非平稳、对抗性、有效样本少 | §13.1 |
| IC 与 $R^2$ | 单变量线性预测下 $R^2=\mathrm{IC}^2$ | §13.1 |
| 样本外为什么变差 | 过拟合、泄漏、市场变化、实现差异、噪声 | §13.4 |

---

## 15. 参考来源

**书籍**
- 李航，《统计学习方法》（第 2 版），清华大学出版社，2019：第 1 章（三要素、损失与风险）、第 3 章（kNN）、第 5 章（决策树、ID3/C4.5/CART）、第 6 章（逻辑斯谛回归）、第 7 章（SVM、对偶、核函数）、第 8 章（AdaBoost、提升树）、第 9 章（EM 与 GMM）、第 14 章（聚类）、第 16 章（PCA）
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*（ISLR，第 2 版，2021）：第 2.2 节（偏差-方差）、第 5 章（交叉验证、自助法）、第 6 章（Ridge、Lasso）、第 8 章（树、Bagging、随机森林、Boosting）、第 9 章（SVM）、第 10 章（深度学习）、第 12 章（PCA、聚类）
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*：第 10 章（Boosting）、第 15 章（随机森林方差公式）
- Marcos López de Prado, *Advances in Financial Machine Learning*（AFML），Wiley，2018：第 3 章（三重障碍法）、第 4 章（样本唯一性与权重）、第 6 章（集成方法在金融中的问题）、第 7 章（Purged K-Fold、Embargo）、第 8 章（特征重要性 MDI/MDA/SFI）、第 12 章（CPCV）
- Goodfellow, Bengio & Courville, *Deep Learning*：第 6 章（反向传播）、第 7 章（正则化、Dropout）、第 8 章（优化器、BN）、第 10 章（RNN、LSTM）

**论文**
- Friedman (2001), Greedy Function Approximation: A Gradient Boosting Machine, *Annals of Statistics*
- Chen & Guestrin (2016), XGBoost: A Scalable Tree Boosting System, *KDD*
- Ke et al. (2017), LightGBM: A Highly Efficient Gradient Boosting Decision Tree, *NeurIPS*
- Kingma & Ba (2015), Adam；Loshchilov & Hutter (2019), Decoupled Weight Decay Regularization（AdamW）
- Ioffe & Szegedy (2015), Batch Normalization；Ba, Kiros & Hinton (2016), Layer Normalization；Santurkar et al. (2018), How Does Batch Normalization Help Optimization?
- Srivastava et al. (2014), Dropout, *JMLR*；Hochreiter & Schmidhuber (1997), Long Short-Term Memory
- Vaswani et al. (2017), Attention Is All You Need, *NeurIPS*
- Gu, Kelly & Xiu (2020), Empirical Asset Pricing via Machine Learning, *Review of Financial Studies*
- Bailey & López de Prado (2014), The Deflated Sharpe Ratio, *Journal of Portfolio Management*

**GitHub 与文档**
- [SoYuCry/awesome-quant-interview](https://github.com/SoYuCry/awesome-quant-interview)：Q41（偏差-方差）、Q42（L1/L2/ElasticNet）、Q43（时间序列交叉验证）、Q44（XGBoost/LightGBM 为何流行）、Q45（特征工程）、Q46（RNN/LSTM/Transformer）、Q51（标签不平衡）、Q52（信息泄漏）、Q53（Attention 在选股中的应用）、Q54（集成学习）、Q55（大模型在量化中的应用）
- [microsoft/qlib](https://github.com/microsoft/qlib)：Alpha158 特征与 LightGBM 等基准模型
- [XGBoost 文档 · Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)（二阶展开推导）；[LightGBM 文档 · Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)（`data_sample_strategy`、`num_leaves`、`max_bin` 等）

**国内面经**
- 宽德笔试含可选机器学习题（[牛客](https://www.nowcoder.com/discuss/382695299372896256)、[牛客](https://www.nowcoder.com/discuss/393797475918774272)）；诚奇笔试含机器学习题（[知乎](https://www.zhihu.com/question/62031149)、[CSDN](https://blog.csdn.net/gitblog_09797/article/details/142943590)）；幻方设深度学习方向（[知乎](https://www.zhihu.com/question/313149375/answer/610309132)）。以上面经均未记录题干。
