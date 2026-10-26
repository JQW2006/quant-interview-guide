# 机器学习题库

配套讲解见[讲解](notes.md)。代码均为 Python 3，深度学习题用 NumPy 实现。

| 标签 | 含义 |
|---|---|
| `经典·书名` | 出自面试书或教材。李航 = 《统计学习方法》；ISLR = *An Introduction to Statistical Learning*；ESL = *The Elements of Statistical Learning*；AFML = *Advances in Financial Machine Learning*；DL = Goodfellow 等 *Deep Learning*；AQI = awesome-quant-interview 题号；论文以“作者 年份”标注；其后数字为章节号 |
| `押题` | 依据往年考点整理的高频题型 |
| `基础` `中等` `困难` | 难度 |

---

## A. 基础概念：偏差-方差、正则化、验证

> 对应讲解第 1–3.1 节。

### A1. 偏差-方差分解 `中等` `经典·ISLR 2.2` `经典·AQI Q41`

设 $y=f(x)+\varepsilon$，$\mathbb E\varepsilon=0$，$\mathrm{Var}(\varepsilon)=\sigma^2$，$\hat f$ 由随机训练集得到，且与测试点的噪声独立。推导测试点 $x$ 处的期望平方误差分解，并说明增加训练样本量、增加模型复杂度分别影响哪一项。

<details>
<summary>答案与解析</summary>

记 $\bar f=\mathbb E\hat f(x)$，则 $y-\hat f=\varepsilon+(f-\bar f)+(\bar f-\hat f)$。平方后取期望：

- $\mathbb E[\varepsilon(f-\bar f)]=0$，$\mathbb E[\varepsilon(\bar f-\hat f)]=0$（$\varepsilon$ 均值为 0 且与 $\hat f$ 独立）；
- $\mathbb E[(f-\bar f)(\bar f-\hat f)]=(f-\bar f)\,\mathbb E[\bar f-\hat f]=0$。

所以

$$\mathbb E(y-\hat f)^2=\sigma^2+(f-\bar f)^2+\mathbb E(\hat f-\bar f)^2=\text{不可约误差}+\text{偏差}^2+\text{方差}$$

| 操作 | 偏差 | 方差 |
|---|---|---|
| 增加样本量 | 基本不变（由模型类决定） | 下降 |
| 增加模型复杂度 | 下降 | 上升 |
| 加强正则化 | 上升 | 下降 |
| Bagging | 基本不变 | 下降 |

**追问**：分类问题的 0-1 损失没有这样干净的加法分解；对数损失可以类似地分解为"噪声 + 偏差 + 方差"的推广形式，但面试中一般只要求平方损失。

</details>

### A2. 用模拟验证偏差-方差权衡 `中等` `押题`

真实函数 $f(x)=\sin(2\pi x)$，在 $x\in[0,1]$ 上取 30 个固定点，噪声 $\sigma=0.3$。用 1、3、9 阶多项式拟合，各重复 500 次，估计偏差平方与方差。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def bias_variance(degree, n_rep=500, n=30, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    f = np.sin(2 * np.pi * x)
    x_test = np.linspace(0.02, 0.98, 50)
    f_test = np.sin(2 * np.pi * x_test)
    preds = np.empty((n_rep, len(x_test)))
    for r in range(n_rep):
        y = f + sigma * rng.standard_normal(n)
        coef = np.polyfit(x, y, degree)
        preds[r] = np.polyval(coef, x_test)
    bias2 = np.mean((preds.mean(axis=0) - f_test) ** 2)
    var = np.mean(preds.var(axis=0))
    return bias2, var

for d in (1, 3, 9):
    b, v = bias_variance(d)
    print(f"degree={d}: bias^2={b:.4f}, var={v:.4f}, sum={b + v:.4f}")
```

输出：

```
degree=1: bias^2=0.1887, var=0.0058, sum=0.1945
degree=3: bias^2=0.0045, var=0.0105, sum=0.0150
degree=9: bias^2=0.0000, var=0.0267, sum=0.0267
```

 阶数升高，偏差单调下降、方差单调上升；总误差在 3 阶附近最小。测试点取在 $[0.02,0.98]$ 内，避开端点处高阶多项式的剧烈震荡。

</details>

### A3. L1 为什么产生稀疏解 `中等` `经典·ESL 3.4` `经典·AQI Q42`

在正交设计 $X^\top X=I$ 下，分别求 $\tfrac12\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$ 与 $\tfrac12\lVert y-Xw\rVert^2+\tfrac\lambda2\lVert w\rVert_2^2$ 的解，并从几何上解释 L1 的稀疏性。

<details>
<summary>答案与解析</summary>

记 $z=X^\top y$（即 OLS 解）。正交设计下 $\tfrac12\lVert y-Xw\rVert^2=\tfrac12\lVert w-z\rVert^2+\text{常数}$，问题对每个坐标解耦。

**Lasso**：$\min_{w_j}\tfrac12(w_j-z_j)^2+\lambda\lvert w_j\rvert$。$w_j>0$ 时导数为 $w_j-z_j+\lambda=0$，得 $w_j=z_j-\lambda$，需 $z_j>\lambda$；$w_j<0$ 同理；$\lvert z_j\rvert\le\lambda$ 时，0 处次梯度 $[-\lambda,\lambda]$ 包含 $z_j$，最优解为 0。合起来就是软阈值：

$$\hat w_j=\mathrm{sign}(z_j)(\lvert z_j\rvert-\lambda)_+$$

**Ridge**：$\min\tfrac12(w_j-z_j)^2+\tfrac\lambda2w_j^2$，得 $\hat w_j=\dfrac{z_j}{1+\lambda}$，按比例收缩，永远不为 0（除非 $z_j=0$）。

**几何解释**：约束形式 $\lVert w\rVert_1\le t$ 的可行域是菱形（高维是多面体），顶点和棱位于坐标轴或坐标平面上；损失函数的椭圆等高线从外向内扩张时，最先接触的通常是顶点，此时部分坐标为 0。L2 的可行域是球，边界光滑，切点一般不在坐标轴上。

**Elastic Net**：L1 在一组高度相关的特征中往往只随机选一个；加入 L2 后相关特征的系数趋于相近（成组效应）。

</details>

### A4. Ridge 闭式解 `基础` `经典·ISLR 6.2`

写出 Ridge 回归的闭式解，用 NumPy 实现，并与 `sklearn.linear_model.Ridge` 对比。说明为什么 $\lambda>0$ 时矩阵一定可逆。

<details>
<summary>答案与解析</summary>

目标 $\lVert y-Xw\rVert^2+\lambda\lVert w\rVert^2$，求导 $-2X^\top(y-Xw)+2\lambda w=0$，得

$$\hat w=(X^\top X+\lambda I)^{-1}X^\top y$$

$X^\top X$ 半正定，特征值 $\ge0$；加 $\lambda I$ 后特征值 $\ge\lambda>0$，所以正定可逆，条件数从 $\mu_{\max}/\mu_{\min}$ 降为 $(\mu_{\max}+\lambda)/(\mu_{\min}+\lambda)$。

```python
import numpy as np
from sklearn.linear_model import Ridge

def ridge_closed_form(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)

rng = np.random.default_rng(1)
X = rng.standard_normal((200, 5))
y = X @ np.array([1.0, -2.0, 0.0, 0.5, 3.0]) + rng.standard_normal(200)

w_np = ridge_closed_form(X, y, lam=10.0)
w_sk = Ridge(alpha=10.0, fit_intercept=False).fit(X, y).coef_
print(np.round(w_np, 4))
print(np.abs(w_np - w_sk).max())
```

 两者最大差异在 $10^{-12}$ 量级。

**注意**：
- 用 `np.linalg.solve` 而不是先求逆，数值更稳定、更快。
- 有截距时，截距一般**不加惩罚**：先对 $X$、$y$ 中心化，求出 $w$ 后再算截距 $\bar y-\bar x^\top w$（sklearn 的 `fit_intercept=True` 就是这样做的）。

</details>

### A5. 手写 Lasso 坐标下降 `困难` `经典·ESL 3.8`

sklearn 的 Lasso 目标是 $\frac1{2n}\lVert y-Xw\rVert^2+\alpha\lVert w\rVert_1$。用坐标下降实现，并与 sklearn 的结果对比。

<details>
<summary>答案与解析</summary>

**思路**：固定其他坐标，只优化 $w_j$。记偏残差 $r^{(j)}=y-Xw+x_jw_j$，则子问题为

$$\min_{w_j}\ \frac1{2n}\lVert r^{(j)}-x_jw_j\rVert^2+\alpha\lvert w_j\rvert$$

解为 $w_j=\dfrac{S\big(\tfrac1n x_j^\top r^{(j)},\ \alpha\big)}{\tfrac1n x_j^\top x_j}$，$S(z,\alpha)=\mathrm{sign}(z)(\lvert z\rvert-\alpha)_+$。

```python
import numpy as np
from sklearn.linear_model import Lasso

def soft_threshold(z, a):
    return np.sign(z) * np.maximum(np.abs(z) - a, 0.0)

def lasso_cd(X, y, alpha, n_iter=1000, tol=1e-10):
    n, p = X.shape
    w = np.zeros(p)
    r = y.copy()                        # r = y - Xw
    col_sq = (X ** 2).sum(axis=0) / n
    for _ in range(n_iter):
        max_change = 0.0
        for j in range(p):
            old = w[j]
            rho = X[:, j] @ r / n + col_sq[j] * old   # = x_j^T r^(j) / n
            w[j] = soft_threshold(rho, alpha) / col_sq[j]
            if w[j] != old:
                r -= X[:, j] * (w[j] - old)          # 增量更新残差
                max_change = max(max_change, abs(w[j] - old))
        if max_change < tol:
            break
    return w

rng = np.random.default_rng(2)
X = rng.standard_normal((300, 8))
w_true = np.array([3.0, -2.0, 0, 0, 1.5, 0, 0, 0])
y = X @ w_true + 0.5 * rng.standard_normal(300)

w_cd = lasso_cd(X, y, alpha=0.1)
w_sk = Lasso(alpha=0.1, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y).coef_
print(np.round(w_cd, 4))
print(np.abs(w_cd - w_sk).max())
```

 与 sklearn 的差异小于 $10^{-8}$；5 个真实系数为 0 的特征中，估计值全部精确为 0。

**复杂度**：每轮 $O(np)$。残差增量更新是关键，否则每个坐标都要重算 $Xw$，每轮变成 $O(np^2)$。

</details>

### A6. 正则化的贝叶斯解释 `中等` `经典·ESL 3.4`

设 $y\mid X,w\sim N(Xw,\sigma^2I)$。证明：先验 $w\sim N(0,\tau^2I)$ 下的最大后验估计等价于 Ridge，并求 $\lambda$；拉普拉斯先验对应什么？

<details>
<summary>答案与解析</summary>

$$\log p(w\mid y)=\log p(y\mid w)+\log p(w)+C=-\frac{1}{2\sigma^2}\lVert y-Xw\rVert^2-\frac{1}{2\tau^2}\lVert w\rVert^2+C$$

最大化后验等价于最小化 $\lVert y-Xw\rVert^2+\dfrac{\sigma^2}{\tau^2}\lVert w\rVert^2$，即 Ridge，$\lambda=\sigma^2/\tau^2$。

- 先验越集中（$\tau$ 小）或噪声越大（$\sigma$ 大），$\lambda$ 越大。
- 拉普拉斯先验 $p(w_j)\propto e^{-\lvert w_j\rvert/b}$ 对应 Lasso，$\lambda=2\sigma^2/b$（目标写成 $\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$ 时）。拉普拉斯分布在 0 处有尖峰，所以 MAP 估计会精确取 0。
- 注意 Lasso 只是 MAP（后验众数），后验均值并不稀疏。

> **量化用途**：$\lambda=\sigma^2/\tau^2$ 给出直觉：收益数据噪声 $\sigma$ 极大，而真实因子载荷 $\tau$ 很小，所以合理的正则强度远大于其他领域的经验值。

</details>

### A7. 为什么正则化前要标准化 `基础` `经典·ISLR 6.2`

一个因子用"元"做单位，另一个用"万元"。不做标准化直接跑 Ridge，会发生什么？树模型需要标准化吗？

<details>
<summary>答案与解析</summary>

- 惩罚项 $\lambda\sum w_j^2$ 对所有系数一视同仁，但系数大小取决于特征单位。用"万元"计的特征系数是用"元"计时的 $10^4$ 倍，被惩罚得重得多；用"元"计的特征系数很小，几乎不受惩罚。结果取决于单位的选择，这是不合理的。
- OLS 对单位不敏感（系数按比例缩放，预测不变），Ridge、Lasso、kNN、SVM、PCA、神经网络都敏感。
- 树模型只依赖特征的排序（切分点），对单调变换不变，**不需要**标准化。
- 量化中更常用截面 rank 或截面 z-score，同时解决尺度问题和时间上的非平稳。

</details>

### A8. 训练集、验证集、测试集 `基础` `经典·ISLR 5.1`

为什么需要三段划分？如果在测试集上反复调参，报告的测试误差有什么问题？

<details>
<summary>答案与解析</summary>

| 数据 | 用途 |
|---|---|
| 训练集 | 拟合参数 |
| 验证集 | 选超参数、选模型、早停 |
| 测试集 | 只用一次，估计最终泛化误差 |

在测试集上调参，相当于把测试集当作验证集用。选出的是"在这一段测试数据上碰巧最好"的配置，报告的测试误差是对泛化误差的**乐观有偏估计**：即使所有配置都没有预测能力，从中挑最好的一个，其测试分数也会显著好于随机（见 L7）。

量化中的对应做法：保留最近一段时间作为最终样本外，研究阶段不看；所有尝试过的配置记录在案，用于多重检验修正。

</details>

### A9. 自助采样中约 63.2% 的样本被抽到 `基础` `经典·ISLR 5.2`

从 $n$ 个样本中有放回地抽 $n$ 次，某个样本至少被抽到一次的概率是多少？$n\to\infty$ 时的极限？用模拟验证。

<details>
<summary>答案与解析</summary>

某个样本一次没被抽中的概率是 $1-\frac1n$，$n$ 次都没被抽中的概率是 $(1-\frac1n)^n\to e^{-1}\approx0.368$。所以至少被抽中一次的概率 $\to1-e^{-1}\approx0.632$。

```python
import numpy as np

rng = np.random.default_rng(0)
n = 10_000
fracs = [len(np.unique(rng.integers(0, n, n))) / n for _ in range(200)]
print(np.mean(fracs), 1 - (1 - 1 / n) ** n, 1 - np.exp(-1))
```

 模拟均值约 0.6321，与理论值 $1-(1-1/n)^n=0.63214$ 一致。

**应用**：随机森林中每棵树没用到的约 36.8% 样本称为袋外样本（OOB），可以用来估计泛化误差。

</details>

---

## B. 时间序列验证与标签泄漏

> 对应讲解第 3.2–3.5 节。这一章是量化面试和互联网 ML 面试差别最大的地方。

### B1. 为什么金融数据不能用随机 K 折 `基础` `经典·AQI Q43` `经典·AFML 7.2`

面试官问："你做收益预测时用 5 折交叉验证调参，有什么问题？"

<details>
<summary>答案与解析</summary>

随机 K 折假设样本独立同分布，金融数据有三点不满足：

| 问题 | 机制 | 后果 |
|---|---|---|
| 序列相关 | 特征（滚动均值、动量）和收益波动都有自相关，随机打乱后验证样本的相邻样本在训练集中 | 模型只需"记住邻居"即可得高分 |
| 标签重叠 | 标签为未来 $h$ 日收益时，相邻样本的标签共享 $h-1$ 天收益 | 训练集直接包含验证标签的一部分 |
| 非平稳 | 用未来训练、过去验证 | 模型提前知道了未来的市场状态 |

**正确做法**：walk-forward（滚动或扩展窗口）；或 Purged K-Fold + Embargo；需要多条路径时用 CPCV。截面模型还要保证同一天的样本全部在同一折（按日期分组切分）。

</details>

### B2. 随机 K 折在自相关数据上的虚高分数 `中等` `经典·AFML 7.2`

构造一个高度自相关的目标序列，唯一的特征是时间下标。比较随机 5 折和按时间顺序 5 折的 $R^2$。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
n = 1000
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.98 * y[t - 1] + rng.standard_normal()   # AR(1)，接近随机游走
X = np.arange(n).reshape(-1, 1).astype(float)          # 特征只有时间戳

model = KNeighborsRegressor(n_neighbors=5)
r2_random = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
r2_time = cross_val_score(model, X, y, cv=TimeSeriesSplit(5)).mean()
print(f"随机 K 折 R^2 = {r2_random:.3f}")
print(f"时间顺序切分 R^2 = {r2_time:.3f}")
```

 随机 K 折 $R^2\approx0.95$，按时间切分 $R^2$ 为负（约 $-0.62$）。

**解释**：时间戳本身对未来没有任何预测力，但随机打乱后，每个验证点两侧的邻居都在训练集里，kNN 做的是插值而不是预测。真实的因子模型中，滚动窗口特征、波动聚集起着同样的作用，只是虚高程度没这么夸张，因此更隐蔽。

</details>

### B3. 实现 walk-forward 切分 `中等` `押题`

实现生成器 `walk_forward(n, train_len, test_len, gap=0, expanding=False)`，按时间顺序产出 `(train_idx, test_idx)`；`gap` 为训练集末尾与测试集开头之间丢弃的样本数。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def walk_forward(n, train_len, test_len, gap=0, expanding=False):
    start = 0
    while True:
        train_end = start + train_len             # 训练集 [train_start, train_end)
        test_start = train_end + gap
        test_end = test_start + test_len
        if test_end > n:
            break
        train_start = 0 if expanding else start
        yield np.arange(train_start, train_end), np.arange(test_start, test_end)
        start += test_len                          # 每次向前滚动一个测试窗口

for tr, te in walk_forward(20, train_len=8, test_len=4, gap=2):
    print(f"train {tr[0]}-{tr[-1]}  test {te[0]}-{te[-1]}")
```

输出：

```
train 0-7  test 10-13
train 4-11  test 14-17
```

 用随机参数测试：训练集与测试集不相交、训练集全部早于测试集、间隔恰为 `gap`、各测试窗口首尾相接不重叠。

**gap 取多少**：至少等于标签的前瞻长度 $h$（如未来 5 日收益取 5），这就是 walk-forward 下的 purge。

</details>

### B4. 实现 Purged K-Fold + Embargo `困难` `经典·AFML 7.4`

样本 $i$ 在时刻 `t0[i]` 观测特征，标签区间为 `[t0[i], t1[i]]`（如 `t1 = t0 + h`）。实现 K 折切分：每折是一段连续时间；训练集中删除标签区间与测试集重叠的样本（purge），并删除测试集结束后 `embargo` 个时间单位内开始的样本。

<details>
<summary>答案与解析</summary>

**思路**：测试集覆盖的时间为 $[T_0,T_1]$，$T_0$ 是测试集第一个样本的 `t0`，$T_1$ 是测试集所有样本 `t1` 的最大值。训练样本 $i$ 保留的条件是：

- 在测试集之前且标签不进入测试期：`t1[i] < T0`；或
- 在测试集之后且越过 embargo：`t0[i] > T1 + embargo`。

```python
import numpy as np

def purged_kfold(t0, t1, n_splits=5, embargo=0):
    t0, t1 = np.asarray(t0), np.asarray(t1)
    n = len(t0)
    for test_idx in np.array_split(np.arange(n), n_splits):   # 要求样本按 t0 排序
        T0 = t0[test_idx[0]]
        T1 = t1[test_idx].max()
        keep = (t1 < T0) | (t0 > T1 + embargo)
        keep[test_idx] = False
        yield np.flatnonzero(keep), test_idx

n, h = 100, 5
t0 = np.arange(n)
t1 = t0 + h                       # 标签：未来 5 期收益
for tr, te in purged_kfold(t0, t1, n_splits=5, embargo=3):
    before = tr[tr < te[0]]
    after = tr[tr > te[-1]]
    print(f"test {te[0]:>2}-{te[-1]:>2} | 训练集前段末尾 {before.max() if len(before) else '-':>2}"
          f" | 后段起点 {after.min() if len(after) else '-':>2} | 训练样本数 {len(tr)}")
```

输出：

```
test  0-19 | 训练集前段末尾  - | 后段起点 28 | 训练样本数 72
test 20-39 | 训练集前段末尾 14 | 后段起点 48 | 训练样本数 67
test 40-59 | 训练集前段末尾 34 | 后段起点 68 | 训练样本数 67
test 60-79 | 训练集前段末尾 54 | 后段起点 88 | 训练样本数 67
test 80-99 | 训练集前段末尾 74 | 后段起点  - | 训练样本数 75
```

 对每一折逐个样本检查：保留的训练样本标签区间与 $[T_0,T_1]$ 都不相交，且没有样本落在 $(T_1,T_1+\text{embargo}]$ 内。

**读输出**：测试集 20–39 的标签延伸到 44。前段：样本 15–19 的标签区间 $[15,20]$ 至 $[19,24]$ 进入测试期，被 purge，前段到 14 为止。后段：样本 40–44 的特征期在测试标签区间内被 purge，再加 3 期 embargo（45–47），后段从 48 开始。

**与 sklearn 的区别**：`TimeSeriesSplit` 只用过去训练，有 `gap` 参数但没有 embargo；Purged K-Fold 允许用测试集之后的数据训练，样本利用率更高，但前提是市场规律在时间上大致对称，实盘前仍需 walk-forward 复核。

</details>

### B5. Purge 和 Embargo 的区别 `中等` `经典·AFML 7.4`

两者都是删除训练样本，为什么需要两种？为什么 embargo 只设在测试集之后？

<details>
<summary>答案与解析</summary>

| | Purge | Embargo |
|---|---|---|
| 针对 | **标签区间重叠**：训练样本的标签时间与测试样本的标签时间有交集 | **特征序列相关**：测试期结束后不久的样本，特征（如 20 日均线）仍包含测试期的数据 |
| 删除范围 | 由标签区间精确决定 | 经验设定，AFML 建议约 $0.01T$，或取特征的最大回看窗口 |
| 位置 | 测试集两侧 | 测试集之后 |

**为什么 embargo 只在之后**：测试集之前的训练样本，其特征只用到更早的数据，不包含测试期信息；它们的标签可能进入测试期，这已由 purge 处理。测试集之后的训练样本，特征的回看窗口会覆盖测试期，从而把测试期的价格信息带入训练集，而 purge 只看标签区间，不处理这种情况。

</details>

### B6. 全样本标准化造成的泄漏 `中等` `押题` `经典·AQI Q52`

对单只股票的时间序列特征做 z-score，`(x - x.mean()) / x.std()` 有什么问题？写出不含未来信息的版本，并用"改动未来数据，过去的值是否改变"来检验。

<details>
<summary>答案与解析</summary>

全样本均值和标准差用到了未来数据。例如价格长期上涨时，早期的标准化值普遍为负，等于告诉模型"以后会涨"。

```python
import numpy as np
import pandas as pd

def zscore_full(x):
    return (x - x.mean()) / x.std()

def zscore_expanding(x, min_periods=20):
    m = x.expanding(min_periods=min_periods).mean()     # 只用 t 及以前
    s = x.expanding(min_periods=min_periods).std()
    return (x - m) / s

rng = np.random.default_rng(0)
x = pd.Series(np.cumsum(rng.standard_normal(300)))
x2 = x.copy()
x2.iloc[200:] += 50                                     # 只改动 t >= 200 的数据

for f in (zscore_full, zscore_expanding):
    changed = not np.allclose(f(x).iloc[:200], f(x2).iloc[:200], equal_nan=True)
    print(f.__name__, "过去的值是否改变:", changed)
```

 `zscore_full` 输出 `True`（有泄漏），`zscore_expanding` 输出 `False`。

**其他写法**：滚动窗口 `x.rolling(250).mean()`；截面模型中更常用**截面标准化**（每天只用当天所有股票的数据），天然不含未来信息。这一检验方法对任何特征函数都适用，可以写进特征库的单元测试。

</details>

### B7. 特征选择泄漏 `中等` `经典·ESL 7.10.2`

100 个样本、标签随机、5000 个纯噪声特征。方案一：先在全部数据上选出与标签相关性最高的 50 个特征，再做 5 折交叉验证；方案二：把特征选择放到每一折的训练集内部。比较两者的 AUC。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 5000))
y = rng.integers(0, 2, 100)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# 方案一：错误，先在全样本上选特征
X_sel = SelectKBest(f_classif, k=50).fit_transform(X, y)
auc_wrong = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=cv, scoring="roc_auc").mean()

# 方案二：正确，选择放在每折内部
pipe = make_pipeline(SelectKBest(f_classif, k=50), LogisticRegression(max_iter=1000))
auc_right = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
print(f"全样本先选特征 AUC = {auc_wrong:.3f}")
print(f"折内选特征   AUC = {auc_right:.3f}")
```

 方案一 AUC 约 0.97；方案二约 0.41，即在 0.5 附近随机波动（样本只有 100 个，波动较大）。数据中没有任何信号。

**结论**：凡是用到标签的步骤（特征选择、目标编码、超参数搜索、因子筛选），都必须放进交叉验证的每一折内部。量化中的对应场景：先在全样本上算出 IC 最高的 50 个因子，再回测这 50 个因子的合成，结果必然虚高。

</details>

### B8. 判断是否存在泄漏 `中等` `押题` `经典·AQI Q52`

判断以下做法是否泄漏未来信息，并给出修正：

1. 用 $t$ 日收盘价计算的反转因子，标签为 $t$ 日收盘到 $t+1$ 日收盘的收益；
2. 财务因子 ROE 按报告期（如 3 月 31 日）对齐；
3. 每天对全市场股票做截面 rank 标准化；
4. 用 2015–2024 年全部数据做 PCA，取前 10 个主成分作为特征，再做 walk-forward；
5. 股票池使用当前的沪深 300 成分股，回测 2015 年至今；
6. 用每只股票的历史平均收益做目标编码，计算时包含了当期。

<details>
<summary>答案与解析</summary>

| 序号 | 是否泄漏 | 说明与修正 |
|---|---|---|
| 1 | 取决于成交假设 | 若假设 $t$ 日收盘价成交，需要在收盘前拿到信号，实际不可行（集合竞价期间无法计算最终收盘价）。修正：标签改为 $t+1$ 日开盘或 VWAP 到 $t+2$ 日同一价格；或用 14:50 的价格计算信号 |
| 2 | 泄漏 | 一季报最晚 4 月 30 日才披露。修正：按公告日对齐，使用 point-in-time 数据 |
| 3 | 不泄漏 | 只用当天截面信息 |
| 4 | 泄漏 | 主成分方向由全样本估计。修正：每个训练窗口内重新估计 PCA，再应用到测试期 |
| 5 | 泄漏（幸存者偏差） | 当前成分股是"后来表现好"的股票。修正：使用每个历史时点的成分股 |
| 6 | 泄漏 | 编码值包含当期标签。修正：只用 $t-1$ 及以前，或用折外编码 |

</details>

---

## C. 线性模型与逻辑回归

> 对应讲解第 4 节。

### C1. OLS 闭式解与几何意义 `基础` `经典·ESL 3.2`

推导 OLS 闭式解，说明 $\hat y$ 的几何意义，以及残差与什么正交。

<details>
<summary>答案与解析</summary>

$\nabla_w\lVert y-Xw\rVert^2=-2X^\top(y-Xw)=0$，得正规方程 $X^\top X\hat w=X^\top y$，$X$ 列满秩时 $\hat w=(X^\top X)^{-1}X^\top y$。

**几何意义**：$\hat y=X\hat w=Hy$，$H=X(X^\top X)^{-1}X^\top$ 是到 $X$ 列空间的**正交投影矩阵**（$H^2=H$，$H^\top=H$）。正规方程 $X^\top(y-\hat y)=0$ 说明残差与 $X$ 的每一列正交；含截距时残差之和为 0。

**数值求解**：实际中用 QR 分解（`np.linalg.lstsq`）或 Cholesky，而不是显式求逆；$X$ 病态时用 SVD。

统计性质（无偏、Gauss-Markov、标准误）见[统计与计量讲解](../statistics/notes.md)第 4 节。

</details>

### C2. 逻辑回归的梯度与凸性 `中等` `经典·李航 6.1`

写出逻辑回归的负对数似然，推导梯度和 Hessian，证明它是凸函数。

<details>
<summary>答案与解析</summary>

$p_i=\sigma(z_i)$，$z_i=w^\top x_i$，负对数似然

$$\ell(w)=-\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]$$

利用 $\sigma'(z)=\sigma(z)(1-\sigma(z))$：

$$\frac{\partial\ell}{\partial z_i}=-\frac{y_i}{p_i}p_i(1-p_i)+\frac{1-y_i}{1-p_i}p_i(1-p_i)=p_i-y_i$$

所以 $\nabla\ell=X^\top(p-y)$。再求一次导：$\partial p_i/\partial w=p_i(1-p_i)x_i$，

$$\nabla^2\ell=\sum_ip_i(1-p_i)x_ix_i^\top=X^\top SX,\qquad S=\mathrm{diag}\big(p_i(1-p_i)\big)$$

对任意 $v$，$v^\top X^\top SXv=\sum_ip_i(1-p_i)(x_i^\top v)^2\ge0$，Hessian 半正定，$\ell$ 是凸函数，任何局部极小都是全局极小。

**比较**：若用平方损失 $\sum(y_i-p_i)^2$，对 $w$ 不是凸的，并且在 $p_i$ 接近 0 或 1 时梯度含因子 $p_i(1-p_i)$，几乎为 0，学习停滞。

</details>

### C3. 手写逻辑回归（牛顿法） `中等` `经典·ESL 4.4`

用牛顿法（IRLS）实现逻辑回归，与 sklearn 的无正则结果对比。

<details>
<summary>答案与解析</summary>

牛顿迭代 $w\leftarrow w-H^{-1}\nabla\ell=w+(X^\top SX)^{-1}X^\top(y-p)$，一般 5–10 步收敛。

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_newton(X, y, n_iter=50, tol=1e-12):
    X1 = np.hstack([np.ones((len(X), 1)), X])      # 加截距列
    w = np.zeros(X1.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X1 @ w)
        grad = X1.T @ (p - y)
        H = X1.T @ (X1 * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < tol:
            break
    return w

rng = np.random.default_rng(3)
X = rng.standard_normal((2000, 3))
logit = 0.5 + X @ np.array([1.0, -1.5, 0.3])
y = (rng.random(2000) < sigmoid(logit)).astype(float)

w_newton = logistic_newton(X, y)
sk = LogisticRegression(C=1e10, tol=1e-12, max_iter=10000).fit(X, y)
w_sk = np.r_[sk.intercept_, sk.coef_.ravel()]
print(np.round(w_newton, 4))
print(np.abs(w_newton - w_sk).max())
```

 与 sklearn（`C=1e10`，近似无正则）的系数差异小于 $10^{-5}$，估计值接近真实参数 $(0.5,1,-1.5,0.3)$。

**梯度下降与牛顿法**：梯度下降每步 $O(np)$，需要很多步；牛顿法每步 $O(np^2+p^3)$，二次收敛。特征维度小时用牛顿法，维度大时用 L-BFGS（sklearn 默认）或 SGD。

</details>

### C4. 线性可分时逻辑回归不收敛 `中等` `经典·ESL 4.4`

数据完全线性可分时，无正则的逻辑回归会怎样？用实验说明正则强度 $C$ 对系数范数的影响。

<details>
<summary>答案与解析</summary>

线性可分时存在 $w$ 使所有 $y_i=1$ 的样本 $w^\top x_i>0$，$y_i=0$ 的样本 $w^\top x_i<0$。把 $w$ 放大为 $cw$，$c\to\infty$ 时所有 $p_i$ 趋于正确标签，似然趋于 1、损失趋于 0 但取不到，没有有限的最优解，$\lVert w\rVert\to\infty$。

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
X = np.r_[rng.normal(-2, 0.5, (50, 2)), rng.normal(2, 0.5, (50, 2))]
y = np.r_[np.zeros(50), np.ones(50)]

for C in (0.1, 1, 100, 10000):
    w = LogisticRegression(C=C, max_iter=100000).fit(X, y).coef_
    print(f"C={C:>6}: ||w|| = {np.linalg.norm(w):.2f}")
```

 $C$ 越大（正则越弱），$\lVert w\rVert$ 单调增大（1.15、1.89、3.57、4.26），大致按 $\log C$ 的速度增长，没有上界。

**处理**：加 L2 正则；或认识到此时概率输出不可信（全部接近 0 或 1）。高维稀疏数据（特征数大于样本数）几乎总是线性可分的，必须正则化。

</details>

### C5. 数值稳定的 softmax 与交叉熵 `基础` `经典·DL 4.1`

直接计算 `np.exp(z) / np.exp(z).sum()` 在 $z=(1000,1001,1002)$ 时会出什么问题？写出数值稳定的 softmax 和 log-softmax。

<details>
<summary>答案与解析</summary>

$e^{1000}$ 溢出为 `inf`，得到 `inf/inf = nan`。softmax 对所有分量加同一常数不变，所以先减去最大值：

$$\mathrm{softmax}(z)_k=\frac{e^{z_k-m}}{\sum_je^{z_j-m}},\qquad \log\mathrm{softmax}(z)_k=z_k-m-\log\sum_je^{z_j-m},\quad m=\max_jz_j$$

```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def log_softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))

def cross_entropy(logits, y):
    """logits: (n, K)，y: (n,) 整数标签"""
    return -log_softmax(logits)[np.arange(len(y)), y].mean()

z = np.array([[1000.0, 1001.0, 1002.0]])
print(softmax(z))
print(cross_entropy(z, np.array([2])))
```

 输出 `[[0.0900 0.2447 0.6652]]` 和 $0.4076$（$=-\log0.6652$），无溢出；与 `scipy.special.softmax`、`log_softmax` 一致。

**为什么交叉熵要用 log-softmax 而不是 `log(softmax)`**：概率极小时 softmax 下溢为 0，`log(0) = -inf`；log-softmax 直接在对数域计算，避免下溢。深度学习框架中的 `CrossEntropyLoss` 都接收 logits 而不是概率，原因就在这里。

</details>

### C6. 共线性对 OLS 与 Ridge 的影响 `中等` `经典·ISLR 6.2`

$x_2=x_1+0.05\,\eta$（相关系数约 0.999），$y=x_1+x_2+\varepsilon$。重复 500 次，比较 OLS 和 Ridge 系数的均值与标准差。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

rng = np.random.default_rng(0)
coefs = {"OLS": [], "Ridge": []}
for _ in range(500):
    x1 = rng.standard_normal(100)
    x2 = x1 + 0.05 * rng.standard_normal(100)
    X = np.c_[x1, x2]
    y = x1 + x2 + rng.standard_normal(100)
    coefs["OLS"].append(LinearRegression().fit(X, y).coef_)
    coefs["Ridge"].append(Ridge(alpha=10).fit(X, y).coef_)

for k, v in coefs.items():
    v = np.array(v)
    print(f"{k:5s} 均值 {np.round(v.mean(0), 3)}  标准差 {np.round(v.std(0), 3)}  "
          f"系数和的标准差 {v.sum(1).std():.3f}")
```

 OLS 两个系数均值约为 $(1,1)$（无偏），但标准差约 2.1，经常一正一负；Ridge 均值约为 $(0.95,0.95)$（有偏），标准差约 0.05。两种方法的系数和都很稳定（标准差约 0.1）。

**解释**：数据只能确定 $w_1+w_2$，无法区分 $w_1$ 与 $w_2$。$X^\top X$ 在 $(1,-1)$ 方向的特征值很小，OLS 在这个方向上的方差被放大；Ridge 在该方向上收缩最多。

> **量化用途**：5 日、10 日、20 日反转因子高度相关，OLS 合成会得到一正一负的大权重，样本外很不稳定。可以用 Ridge、对称正交化，或只保留一个代表。

</details>

### C7. 逻辑回归系数的解释 `基础` `经典·ISLR 4.3`

逻辑回归中某特征系数为 0.693。该特征增加 1 单位，几率（odds）和概率分别怎么变？

<details>
<summary>答案与解析</summary>

$\log\frac{p}{1-p}=w^\top x$，特征增加 1 单位，对数几率增加 0.693，几率乘以 $e^{0.693}\approx2$，即**几率翻倍**。

概率的变化不是常数，取决于原来的概率：原来 $p=0.5$（几率 1）时变成几率 2，$p=2/3$；原来 $p=0.1$（几率 1/9）时变成几率 2/9，$p=2/11\approx0.18$。概率对特征的边际效应为 $w\,p(1-p)$，在 $p=0.5$ 时最大。

</details>

### C8. 能否用线性回归做分类 `基础` `经典·ISLR 4.2`

把标签编码为 0/1 后直接用 OLS 拟合，再以 0.5 为阈值分类，有什么问题？

<details>
<summary>答案与解析</summary>

| 问题 | 说明 |
|---|---|
| 预测值超出 $[0,1]$ | 不能解释为概率 |
| 对远离边界的点敏感 | 一个"非常正确"的样本（特征值很大）也会产生大残差，拉动决策边界 |
| 多分类编码 | 用 1、2、3 编码隐含了类别之间的顺序和等间距 |
| 误差分布 | 残差只能取两个值，同方差和正态假设不成立 |

二分类时 OLS 的 $\hat y$ 实际上是 $P(y=1\mid x)$ 的线性近似（线性概率模型），在计量经济学中有时用来解释边际效应；做预测和概率估计时应使用逻辑回归。

</details>

---

## D. 决策树与随机森林

> 对应讲解第 5 节。

### D1. 计算信息增益与基尼指数 `基础` `经典·李航 5.2`

贷款申请数据共 15 个样本，9 个批准、6 个拒绝。按"有自己的房子"划分：有房 6 人全部批准；无房 9 人中 3 人批准。按"有工作"划分：有工作 5 人全部批准；无工作 10 人中 4 人批准。分别计算两个特征的信息增益和划分后的基尼指数，应该先按哪个特征分？

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def entropy(counts):
    p = np.asarray(counts, float) / sum(counts)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()

def gini(counts):
    p = np.asarray(counts, float) / sum(counts)
    return 1 - (p ** 2).sum()

def split_scores(parent, children):
    n = sum(parent)
    cond_h = sum(sum(c) / n * entropy(c) for c in children)
    cond_g = sum(sum(c) / n * gini(c) for c in children)
    return entropy(parent) - cond_h, cond_g

parent = [9, 6]
print("H(D) =", round(entropy(parent), 3), " Gini(D) =", round(gini(parent), 3))
for name, children in {"有房": [[6, 0], [3, 6]], "有工作": [[5, 0], [4, 6]]}.items():
    ig, g = split_scores(parent, children)
    print(f"{name}: 信息增益 = {ig:.3f}, 划分后基尼 = {g:.3f}")
```

输出：

```
H(D) = 0.971  Gini(D) = 0.48
有房: 信息增益 = 0.420, 划分后基尼 = 0.267
有工作: 信息增益 = 0.324, 划分后基尼 = 0.320
```

 两个准则都选"有房"：信息增益更大，划分后基尼指数更小。这与《统计学习方法》第 5 章例题的结论一致（书中四个特征的信息增益分别为 0.083、0.324、0.420、0.363）。

</details>

### D2. 回归树的最优单次分裂 `中等` `经典·ESL 9.2`

给定一维特征 $x$ 和目标 $y$，找使左右两侧平方误差之和最小的切分点。要求 $O(n\log n)$，并与 `DecisionTreeRegressor(max_depth=1)` 的结果对比。

<details>
<summary>答案与解析</summary>

**思路**：排序后，切分点只需考虑相邻不同取值的中点。一侧的平方误差 $\mathrm{SSE}=\sum y_i^2-(\sum y_i)^2/m$，用前缀和在 $O(1)$ 内算出每个切分点的左右 SSE。排序 $O(n\log n)$，扫描 $O(n)$。

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def best_split(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    cs, cs2 = np.cumsum(ys), np.cumsum(ys ** 2)
    total, total2 = cs[-1], cs2[-1]
    best_sse, best_thr = np.inf, None
    for i in range(1, n):                        # 左边取前 i 个
        if xs[i] == xs[i - 1]:
            continue
        sl, sl2 = cs[i - 1], cs2[i - 1]
        sse = (sl2 - sl ** 2 / i) + ((total2 - sl2) - (total - sl) ** 2 / (n - i))
        if sse < best_sse:
            best_sse, best_thr = sse, (xs[i] + xs[i - 1]) / 2
    return best_thr, best_sse

rng = np.random.default_rng(0)
x = rng.random(500)
y = np.where(x > 0.37, 2.0, 0.0) + 0.5 * rng.standard_normal(500)

thr, sse = best_split(x, y)
tree = DecisionTreeRegressor(max_depth=1).fit(x.reshape(-1, 1), y)
print(round(thr, 4), round(tree.tree_.threshold[0], 4))
```

 切分点与 sklearn 完全一致（0.3751），也与 $O(n^2)$ 暴力解在随机数据上一致。

**推广**：多特征时对每个特征做一次，总复杂度 $O(pn\log n)$；预排序可以在整棵树的构建中复用（XGBoost 的列块）。直方图算法把排序替换成分桶，复杂度降为 $O(pn)$ 建直方图加 $O(p\cdot\#\text{桶})$ 找切分点。

</details>

### D3. ID3、C4.5、CART 的区别 `基础` `经典·李航 5.3`

列表比较三种决策树算法。

<details>
<summary>答案与解析</summary>

| | ID3 | C4.5 | CART |
|---|---|---|---|
| 准则 | 信息增益 | 信息增益比 | 基尼指数（分类）、平方误差（回归） |
| 树结构 | 多叉 | 多叉 | 二叉 |
| 连续特征 | 不支持 | 支持（二分） | 支持 |
| 缺失值 | 不支持 | 支持（按比例分配） | 支持（代理分裂） |
| 回归 | 否 | 否 | 是 |
| 剪枝 | 无 | 悲观剪枝 | 代价复杂度剪枝 |

**信息增益的偏向**：取值多的特征（极端情况是样本 ID）能把数据切得很纯，信息增益很大但毫无泛化能力。信息增益比除以特征本身的熵 $H_A(D)$，对取值多的特征施加惩罚。sklearn 实现的是优化版 CART，`criterion` 可选 `gini`、`entropy`、`log_loss`。

</details>

### D4. 随机森林的方差公式 `中等` `经典·ESL 15.2`

$B$ 个同分布的预测，方差均为 $\sigma^2$，两两相关系数为 $\rho$。求平均值的方差，并据此解释随机森林为什么要在每次分裂时随机选特征。

<details>
<summary>答案与解析</summary>

$$\mathrm{Var}\Big(\frac1B\sum_b\hat f_b\Big)=\frac1{B^2}\Big[B\sigma^2+B(B-1)\rho\sigma^2\Big]=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$

- $B\to\infty$ 时，方差下降到 $\rho\sigma^2$，再多的树也无法消除。
- Bagging 的树都在相似的数据上训练，强特征总是在顶部被选中，树与树高度相关，$\rho$ 大。
- 随机森林每次分裂只从 $m$ 个随机特征中选，强特征有时不可用，树的结构更加多样，$\rho$ 下降；代价是单棵树的 $\sigma^2$ 略有上升（偏差也略升）。$m$ 是在两者之间的权衡。

**数值例子**：$\rho=0.5$ 时，无论多少棵树，方差至少是单棵树的一半；$\rho=0.1$、$B=100$ 时，方差为 $0.1+0.009=0.109$ 倍。

> **量化用途**：同一思想解释了多因子、多策略组合的分散化：$n$ 个两两相关 $\rho$ 的策略等权组合，方差下限是 $\rho\sigma^2$。

</details>

### D5. 袋外误差（OOB） `基础` `经典·ISLR 8.2`

什么是 OOB 误差？为什么它近似交叉验证误差？在金融数据上为什么会偏乐观？

<details>
<summary>答案与解析</summary>

- 每棵树用自助样本训练，约 36.8% 的样本没有被这棵树用到（A9）。对样本 $i$，只用没见过它的那些树（约 $B/3$ 棵）预测，再与真实值比较，汇总得到 OOB 误差。
- 每个样本的预测都来自没有用它训练的模型，类似留一法交叉验证，不需要额外切分数据。
- **金融数据中偏乐观**：标签重叠时，样本 $i$ 虽然不在某棵树的训练集中，但与它标签重叠的相邻样本 $i\pm1,\dots$ 很可能在，等价于 B1 中的随机 K 折泄漏。AFML 第 6 章建议设置较小的 `max_samples`（接近样本平均唯一性），并用 Purged K-Fold 而不是 OOB 评估。

</details>

### D6. Bagging 与 Boosting `基础` `经典·ISLR 8.2` `经典·AQI Q54`

从训练方式、降低的误差、基学习器、过拟合倾向几个方面比较 Bagging 和 Boosting。

<details>
<summary>答案与解析</summary>

| | Bagging（随机森林） | Boosting（GBDT、XGBoost） |
|---|---|---|
| 训练方式 | 并行，各模型独立 | 串行，每个模型修正前面的残差 |
| 主要降低 | 方差 | 偏差 |
| 基学习器 | 低偏差高方差（深树） | 高偏差低方差（浅树） |
| 增加模型数量 | 不会过拟合，方差趋于 $\rho\sigma^2$ | 会过拟合，需早停 |
| 对噪声标签 | 较稳健 | 敏感（会努力拟合难样本） |
| 调参难度 | 低 | 高 |

其他集成方法：**Stacking**（用一个元模型组合多个基模型的折外预测）、**Blending**（用留出集代替折外）、简单平均。量化中常用的是多种子、多窗口、多模型的简单平均或按验证 IC 加权。

</details>

### D7. 树模型不能外推 `基础` `押题`

$x\in[0,10]$，$y=2x+$ 噪声。分别用线性回归和随机森林拟合，预测 $x=20$。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, (500, 1))
y = 2 * X.ravel() + rng.standard_normal(500)

lr = LinearRegression().fit(X, y)
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y)
print("线性回归 f(20) =", round(lr.predict([[20.0]])[0], 2))
print("随机森林 f(20) =", round(rf.predict([[20.0]])[0], 2))
```

 线性回归约 40.0，随机森林约 20.8（训练集中 $y$ 的最大值附近）。

**解释**：树的预测是叶子内训练样本的均值，所以不会超出训练标签的范围；特征超出训练范围时，落入最边缘的叶子。

> **量化用途**：
> - 市场进入历史上未见过的状态（如 2024 年初的极端流动性冲击）时，树模型只能给出边缘叶子的值，不会"按趋势外推"。这有时是优点（预测有界），有时是缺点。
> - 截面 rank 标准化后的特征始终在 $[0,1]$ 内，外推问题大幅缓解，这也是量化中偏好 rank 特征的原因之一。

</details>

### D8. MDI 特征重要性的偏差 `困难` `经典·AFML 8.3`

构造一个二值的有效特征和一个连续的纯噪声特征，比较随机森林的 MDI（不纯度下降）重要性和测试集上的置换重要性。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 2000
signal = rng.integers(0, 2, n)                       # 二值有效特征
y = np.where(rng.random(n) < 0.7, signal, 1 - signal)  # 70% 与特征一致
noise = rng.random(n)                                # 连续噪声特征
X = np.c_[signal, noise]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=0)
print("MDI 重要性   [signal, noise] =", np.round(rf.feature_importances_, 3))
print("置换重要性   [signal, noise] =", np.round(perm.importances_mean, 3))
```

 MDI 认为噪声特征**更重要**（0.83 对 0.17）；测试集上的置换重要性则正确地给出有效特征约 0.09、噪声特征约 0（$-0.01$）。

**原因**：MDI 在训练集上累计不纯度下降。树完全生长时，连续噪声特征有大量切分点可用来拟合训练集中的噪声，累计下降量很大；二值特征只能切一次。

**对策**：用验证集上的置换重要性（AFML 称为 MDA）；或用 SHAP；或限制树的深度、叶子最小样本数。AFML 还指出，相关特征会互相"替代"，置换一个时另一个能补上，所以应先对特征聚类，再按簇计算重要性。

</details>

---

## E. GBDT、XGBoost、LightGBM

> 对应讲解第 6 节。E2、E3 为常考推导。

### E1. 手写 GBDT（平方损失） `中等` `经典·ESL 10.10`

用 `DecisionTreeRegressor` 作为基学习器实现平方损失的 GBDT，与 `GradientBoostingRegressor` 对比预测值。

<details>
<summary>答案与解析</summary>

**算法**：$F_0=\bar y$；第 $m$ 轮计算负梯度 $r_i=y_i-F_{m-1}(x_i)$，用回归树拟合 $r$，更新 $F_m=F_{m-1}+\eta h_m$。平方损失下叶子的最优输出就是叶内残差均值，正好是回归树的默认输出，不需要再单独优化叶子值。

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

class SimpleGBDT:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators, self.lr, self.max_depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        self.f0 = y.mean()
        self.trees = []
        F = np.full(len(y), self.f0)
        for _ in range(self.n_estimators):
            residual = y - F                       # 平方损失的负梯度
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            F += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, (500, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * rng.standard_normal(500)

ours = SimpleGBDT().fit(X, y).predict(X)
sk = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3).fit(X, y).predict(X)
print(np.abs(ours - sk).max())
```

 与 sklearn 的预测最大差异在 $10^{-15}$ 量级。（sklearn 1.9 起 `GradientBoostingRegressor` 的 `criterion` 参数已弃用、不再起作用，内部树统一使用平方误差准则；更早的版本默认 `friedman_mse`，切分点可能略有不同。）

**对数损失时**：负梯度为 $y-p$，但叶子值不再是残差均值，需要做一步牛顿更新 $\sum r_i/\sum p_i(1-p_i)$，这正是 XGBoost 用二阶信息的思路。

</details>

### E2. 推导 XGBoost 的叶子权重和分裂增益 `困难` `经典·Chen & Guestrin 2016`

写出 XGBoost 第 $t$ 轮的目标函数，用二阶泰勒展开推导最优叶子权重 $w_j^*$ 和分裂增益公式，并说明 $\lambda$、$\gamma$ 的作用。

<details>
<summary>答案与解析</summary>

**目标函数**：

$$\mathcal L^{(t)}=\sum_iL\big(y_i,\hat y_i^{(t-1)}+f_t(x_i)\big)+\gamma T+\tfrac12\lambda\sum_{j=1}^Tw_j^2$$

**二阶展开**（$g_i,h_i$ 为损失对 $\hat y^{(t-1)}$ 的一、二阶导）：

$$\mathcal L^{(t)}\approx\text{const}+\sum_i\Big[g_if_t(x_i)+\tfrac12h_if_t(x_i)^2\Big]+\gamma T+\tfrac12\lambda\sum_jw_j^2$$

**按叶子归并**（叶子 $j$ 中 $f_t(x_i)=w_j$，$G_j=\sum_{i\in I_j}g_i$，$H_j=\sum_{i\in I_j}h_i$）：

$$\tilde{\mathcal L}=\sum_{j=1}^T\Big[G_jw_j+\tfrac12(H_j+\lambda)w_j^2\Big]+\gamma T$$

**最优权重**：对 $w_j$ 求导置零，

$$w_j^*=-\frac{G_j}{H_j+\lambda},\qquad\tilde{\mathcal L}^*=-\frac12\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T$$

**分裂增益**：分裂前该叶子贡献 $-\frac12\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}+\gamma$，分裂后贡献 $-\frac12\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}\Big]+2\gamma$，相减：

$$\mathrm{Gain}=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$$

| 参数 | 作用 |
|---|---|
| $\lambda$ | 出现在分母中：收缩叶子权重；样本少（$H$ 小）的叶子收缩更多 |
| $\gamma$ | 每多一个叶子的固定代价；增益不超过 $\gamma$ 就不分裂 |
| $h_i$ | 样本权重；平方损失下 $h_i=1$，$H_j$ 就是叶内样本数 |

**平方损失检验**：$L=\frac12(y-\hat y)^2$，$g_i=\hat y_i-y_i$，$h_i=1$，$\lambda=0$ 时 $w_j^*=\frac1{n_j}\sum(y_i-\hat y_i)$，就是残差均值，与 GBDT 一致。

</details>

### E3. XGBoost 分裂数值题 `中等` `押题`

6 个样本 $x=1,\dots,6$，$y=(0,1,0,0,1,1)$，对数损失，初始概率 0.5，$\lambda=1$，$\gamma=0$。求第一棵深度为 1 的树的最佳切分点、增益和两个叶子的权重，并用 xgboost 验证。

<details>
<summary>答案与解析</summary>

$g_i=p_i-y_i=(0.5,-0.5,0.5,0.5,-0.5,-0.5)$，$h_i=p_i(1-p_i)=0.25$。总 $G=0$，$H=1.5$。

| 切分（左侧样本数） | $G_L$ | $H_L$ | Gain |
|---|---|---|---|
| 1 | 0.5 | 0.25 | $\frac12\big[\frac{0.25}{1.25}+\frac{0.25}{2.25}\big]=0.156$ |
| 2 | 0 | 0.5 | 0 |
| 3 | 0.5 | 0.75 | $\frac12\big[\frac{0.25}{1.75}+\frac{0.25}{1.75}\big]=0.143$ |
| 4 | 1.0 | 1.0 | $\frac12\big[\frac{1}{2}+\frac{1}{1.5}\big]=0.583$ |
| 5 | 0.5 | 1.25 | 0.156 |

最佳切分在 $x<4.5$：$w_L=-\frac{1}{1+1}=-0.5$，$w_R=-\frac{-1}{0.5+1}=0.667$。

```python
import numpy as np
import xgboost as xgb

def xgb_best_split(x, g, h, lam=1.0, gamma=0.0):
    order = np.argsort(x)
    x, g, h = x[order], g[order], h[order]
    G, H = g.sum(), h.sum()
    GL = HL = 0.0
    best = (-np.inf, None, None, None)
    for i in range(len(x) - 1):
        GL += g[i]; HL += h[i]
        if x[i] == x[i + 1]:
            continue
        GR, HR = G - GL, H - HL
        gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam)) - gamma
        if gain > best[0]:
            best = (gain, (x[i] + x[i + 1]) / 2, -GL / (HL + lam), -GR / (HR + lam))
    return best

x = np.arange(1, 7, dtype=float)
y = np.array([0, 1, 0, 0, 1, 1])
p = np.full(6, 0.5)
gain, thr, wl, wr = xgb_best_split(x, p - y, p * (1 - p))
print(f"threshold={thr}, gain={gain:.4f}, w_left={wl:.4f}, w_right={wr:.4f}")

model = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=1.0,
                          gamma=0.0, min_child_weight=0, base_score=0.5, tree_method="exact")
model.fit(x.reshape(-1, 1), y)
print(model.get_booster().get_dump(with_stats=True)[0])
```

输出：

```
threshold=4.5, gain=0.5833, w_left=-0.5000, w_right=0.6667
0:[f0<4.5] yes=1,no=2,missing=1,gain=1.16666663,cover=1.5
	1:leaf=-0.5,cover=1
	2:leaf=0.666666687,cover=0.5
```

 切分点、叶子值一致；xgboost 导出的 `gain` 是 $1.1667=2\times0.5833$，**不含系数 $\frac12$**；`cover` 是叶内 $\sum h_i$。

</details>

### E4. 常见损失的 $g$ 和 $h$ `基础` `经典·Chen & Guestrin 2016`

分别写出平方损失、对数损失（对 logit 求导）的一阶和二阶导。若自定义损失的二阶导可能为负或为 0，会有什么问题？

<details>
<summary>答案与解析</summary>

| 损失 | $g_i$ | $h_i$ |
|---|---|---|
| 平方损失 $\frac12(y-\hat y)^2$ | $\hat y-y$ | 1 |
| 对数损失，$\hat y$ 为 logit，$p=\sigma(\hat y)$ | $p-y$ | $p(1-p)$ |
| Huber（$\lvert r\rvert\le\delta$ 区间） | $\hat y-y$ | 1 |
| Huber（$\lvert r\rvert>\delta$ 区间） | $\delta\,\mathrm{sign}(\hat y-y)$ | 0 |

对数损失推导：$L=-y\log p-(1-y)\log(1-p)$，$\frac{\partial L}{\partial\hat y}=\frac{\partial L}{\partial p}\cdot p(1-p)=p-y$，再求导得 $p(1-p)$。

**$h_i\le0$ 的问题**：叶子权重 $-G/(H+\lambda)$ 的分母可能接近 0 或为负，二次近似不再有最小值，权重爆炸或方向错误。处理：给 $h$ 设下限（如 $\max(h,10^{-6})$）、增大 $\lambda$、用平滑近似（如 Pseudo-Huber 代替 Huber）。

**自定义目标**：xgboost 和 lightgbm 都允许传入返回 `(grad, hess)` 的函数，量化中可用来实现加权 MSE、排序损失等。

</details>

### E5. 正则化参数的作用 `中等` `押题`

解释 XGBoost 中 `reg_lambda`、`gamma`、`min_child_weight`、`max_depth`、`subsample`、`colsample_bytree`、`learning_rate` 的作用。在日频选股这种低信噪比任务中，一般如何设置？

<details>
<summary>答案与解析</summary>

| 参数 | 作用 | 低信噪比任务的倾向 |
|---|---|---|
| `reg_lambda` | 叶子权重的 L2 惩罚，$w^*=-G/(H+\lambda)$ | 调大 |
| `gamma` | 分裂的最小增益 | 视增益量级调大 |
| `min_child_weight` | 叶内 $\sum h_i$ 下限；平方损失下就是最小叶子样本数 | 调大，如数百至数千 |
| `max_depth` | 树的深度，控制交互阶数 | 较浅，4–8 |
| `subsample` | 每棵树的行采样比例 | 0.5–0.8 |
| `colsample_bytree` | 每棵树的列采样比例 | 0.5–0.8 |
| `learning_rate` | 每棵树的缩放 | 较小（0.01–0.05），配合早停 |

**原因**：信号很弱时，每棵树都容易拟合噪声，需要让每一步都"迈小步、看大样本"。截面选股的单日样本数在 5000 左右，叶子样本数太小时，叶子值几乎完全由噪声决定。

</details>

### E6. XGBoost 如何处理缺失值 `中等` `经典·Chen & Guestrin 2016`

解释 XGBoost 的稀疏感知分裂，并构造一个缺失值本身带有信息的例子验证。

<details>
<summary>答案与解析</summary>

**方法**：对每个候选切分，只用该特征非缺失的样本计算 $G_L,H_L$；把缺失样本整体分别放到左边和右边，各算一次增益，取较大者，并把这个方向记为该节点的**默认方向**。预测时缺失值走默认方向。只需遍历非缺失样本，对稀疏数据效率很高。

```python
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
x = rng.random(1000)
y = (x > 0.5).astype(int)
x[:200] = np.nan                    # 缺失的 200 个样本
y[:200] = 1                         # 缺失样本全部为正类

model = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, base_score=0.5)
model.fit(x.reshape(-1, 1), y)
dump = model.get_booster().get_dump()[0]
print(dump)
```

输出：

```
0:[f0<0.49997282] yes=1,no=2,missing=2
	1:leaf=-1.97938144
	2:leaf=1.98064518
```

 默认方向 `missing=2` 指向输出为正的右叶子，即模型学到了"缺失 → 正类"。

> **量化用途**：财务数据缺失（未披露、新上市）往往本身有信息。直接交给 XGBoost 或 LightGBM 处理，通常比填充均值更好；但要确认缺失模式在训练期和实盘期一致，例如数据商回补历史数据会改变缺失模式。

</details>

### E7. 直方图算法与直方图做差 `中等` `经典·Ke et al. 2017`

实现基于直方图的切分点搜索：把特征按分位数分成最多 `max_bin` 个桶，累积每个桶的 $G$、$H$ 后扫描。验证"父节点直方图减去左子节点直方图等于右子节点直方图"。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def build_bins(x, max_bin=16):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, max_bin + 1)[1:-1]))
    return np.searchsorted(edges, x, side="right"), edges   # 桶号 0..len(edges)

def histogram(bins, g, h, n_bins):
    return np.bincount(bins, g, n_bins), np.bincount(bins, h, n_bins)

def best_split_hist(Gh, Hh, lam=1.0):
    G, H = Gh.sum(), Hh.sum()
    GL, HL = np.cumsum(Gh)[:-1], np.cumsum(Hh)[:-1]      # 在第 k 个桶之后切
    gain = 0.5 * (GL**2 / (HL + lam) + (G - GL)**2 / (H - HL + lam) - G**2 / (H + lam))
    k = int(np.argmax(gain))
    return k, gain[k]

rng = np.random.default_rng(0)
n = 100_000
x = rng.standard_normal(n)
y = (x > 0.3).astype(float)
p = np.full(n, y.mean())
g, h = p - y, p * (1 - p)

bins, edges = build_bins(x, max_bin=16)
n_bins = len(edges) + 1
Gh, Hh = histogram(bins, g, h, n_bins)
k, gain = best_split_hist(Gh, Hh)
print(f"切在第 {k} 个桶之后, 阈值约 {edges[k]:.3f}, gain = {gain:.2f}")

# 直方图做差：父节点 = 全部样本；左子节点 = x < 0
left = x < 0
GhL, HhL = histogram(bins[left], g[left], h[left], n_bins)
GhR, HhR = histogram(bins[~left], g[~left], h[~left], n_bins)
print(np.allclose(Gh - GhL, GhR), np.allclose(Hh - HhL, HhR))
```

 最佳阈值落在离 0.3 最近的分位数边界（16 个桶时边界间距较大，约 0.32）；做差得到的直方图与直接统计的完全一致（`True True`）。另外对分桶后的数据用暴力搜索验证，最佳切分与直方图扫描一致。

| | 预排序（exact） | 直方图 |
|---|---|---|
| 找切分点 | $O(n)$ 每特征 | $O(\#\text{桶})$ 每特征 |
| 建统计量 | 排序 $O(n\log n)$（可复用） | 一次遍历 $O(n)$ |
| 内存 | 排序索引 + 原始值 | 每个值 1 字节桶号 |
| 子节点 | 两个都要统计 | 统计样本少的一个，另一个做差 |

**做差的意义**：每次分裂只需对较小的子节点遍历样本建直方图，较大的子节点用父节点减得到，至少节省一半计算。

</details>

### E8. GOSS 的权重为什么是 $(1-a)/b$ `困难` `经典·Ke et al. 2017`

GOSS 保留梯度绝对值最大的 $a$ 比例样本，从剩余样本中随机抽 $b$ 比例（相对总样本数），并给抽到的小梯度样本乘以 $(1-a)/b$。证明这样估计的 $\sum g_i$ 是无偏的，并用模拟比较它与"均匀随机抽同样数量样本"的估计方差。

<details>
<summary>答案与解析</summary>

**无偏性**：设总样本 $n$，大梯度集合 $A$（$an$ 个）全部保留。小梯度集合 $A^c$ 有 $(1-a)n$ 个，从中无放回抽 $bn$ 个，每个样本被抽中的概率为 $\frac{bn}{(1-a)n}=\frac{b}{1-a}$。因此

$$\mathbb E\Big[\sum_{i\in A}g_i+\frac{1-a}{b}\sum_{i\in B}g_i\Big]=\sum_{i\in A}g_i+\frac{1-a}{b}\cdot\frac{b}{1-a}\sum_{i\in A^c}g_i=\sum_ig_i$$

这就是 Horvitz-Thompson 估计：权重等于入样概率的倒数。

```python
import numpy as np

def goss_estimate(g, mask, a, b, rng):
    n = len(g)
    order = np.argsort(-np.abs(g))
    top = order[: int(a * n)]
    rest = order[int(a * n):]
    sampled = rng.choice(rest, int(b * n), replace=False)
    w = np.zeros(n)
    w[top] = 1.0
    w[sampled] = (1 - a) / b
    return (w * g * mask).sum()

def uniform_estimate(g, mask, frac, rng):
    n = len(g)
    idx = rng.choice(n, int(frac * n), replace=False)
    return (g[idx] * mask[idx]).sum() / frac

rng = np.random.default_rng(0)
n = 20_000
g = rng.standard_t(df=3, size=n)                     # 厚尾梯度
mask = rng.random(n) < 0.5                           # 某个候选左子节点
true = (g * mask).sum()

a, b = 0.2, 0.1
est_goss = np.array([goss_estimate(g, mask, a, b, rng) for _ in range(2000)])
est_unif = np.array([uniform_estimate(g, mask, a + b, rng) for _ in range(2000)])
print(f"真实值 {true:.1f}")
print(f"GOSS    均值 {est_goss.mean():.1f}  标准差 {est_goss.std():.1f}")
print(f"均匀采样 均值 {est_unif.mean():.1f}  标准差 {est_unif.std():.1f}")
```

 真实值 $-68.7$；GOSS 均值 $-68.0$、标准差 184，均匀采样均值 $-72.1$、标准差 257。两者的均值与真实值的差都在模拟误差（标准差 $/\sqrt{2000}$）以内，即无偏；GOSS 的标准差约为均匀采样的 70%。原因是方差主要来自大梯度样本，GOSS 把它们全部保留，只对小梯度样本抽样。

**注意**：
- 论文中证明的是方差增益估计的误差界，这里用 $\sum g$ 演示无偏性的核心思想。
- LightGBM 4.x 用 `data_sample_strategy="goss"`，默认 `top_rate=0.2`、`other_rate=0.1`。
- 金融数据中大梯度样本往往是极端收益，GOSS 会更关注它们；若极端收益主要是噪声，这可能有害，需要实验确认。

</details>

### E9. EFB 互斥特征捆绑 `中等` `经典·Ke et al. 2017`

两个特征几乎不会同时非零（如 one-hot 编码的不同类别），LightGBM 如何把它们合并成一个特征而不丢失信息？写出捆绑与还原的代码。

<details>
<summary>答案与解析</summary>

**做法**：
1. **找捆绑**：把特征看作图的顶点，两个特征同时非零的样本数作为边的权重（冲突）；按度排序，贪心地把特征放入冲突不超过阈值的捆绑中。这等价于图着色问题（NP 难），所以用贪心近似。
2. **合并**：给后加入的特征的取值加上偏移量，使不同特征的取值落在不相交的区间。

```python
import numpy as np

def bundle(f1, f2, f1_max):
    """f1 取值 0..f1_max，f2 取值 0..；二者不同时非零"""
    assert not np.any((f1 > 0) & (f2 > 0))
    return np.where(f2 > 0, f2 + f1_max, f1)

def unbundle(b, f1_max):
    f1 = np.where(b <= f1_max, b, 0)
    f2 = np.where(b > f1_max, b - f1_max, 0)
    return f1, f2

rng = np.random.default_rng(0)
n = 1000
which = rng.integers(0, 3, n)                    # 0: 都为 0；1: 仅 f1 非零；2: 仅 f2 非零
f1 = np.where(which == 1, rng.integers(1, 4, n), 0)
f2 = np.where(which == 2, rng.integers(1, 3, n), 0)
b = bundle(f1, f2, f1_max=3)
r1, r2 = unbundle(b, f1_max=3)
print(np.unique(b), np.array_equal(r1, f1) and np.array_equal(r2, f2))
```

 捆绑后取值为 0–5（0 表示两者都为 0，1–3 来自 f1，4–5 来自 f2），可以无损还原。

**效果**：特征数从 $p$ 降为捆绑数，建直方图的代价相应下降。对 one-hot 特征很多的数据（如股票 ID、行业），提速明显。

</details>

### E10. Level-wise 与 Leaf-wise `基础` `经典·AQI Q44`

比较 XGBoost 默认的 level-wise 生长和 LightGBM 的 leaf-wise 生长。`num_leaves` 与 `max_depth` 是什么关系？XGBoost 和 LightGBM 为什么在量化比赛中流行？

<details>
<summary>答案与解析</summary>

| | Level-wise（逐层） | Leaf-wise（逐叶） |
|---|---|---|
| 做法 | 同一层的所有叶子都分裂 | 每次只分裂全局增益最大的叶子 |
| 同样叶子数下的训练损失 | 较高 | 较低 |
| 过拟合风险 | 较低，树是平衡的 | 较高，可能长出很深的分支 |
| 主要参数 | `max_depth` | `num_leaves`（再用 `max_depth` 兜底） |

深度为 $d$ 的二叉树最多 $2^d$ 个叶子，所以应设 `num_leaves` $<2^{\text{max\_depth}}$；例如 `max_depth=7` 时完全二叉树有 128 个叶子，`num_leaves` 取 70–80 更稳妥。XGBoost 也可通过 `grow_policy="lossguide"` 使用逐叶生长。

**在量化中流行的原因**：
- 表格数据上效果好，能自动处理非线性与交互；
- 对特征尺度不敏感、能处理缺失值、对异常值较稳健；
- 训练快，便于滚动重训和大量实验；
- 正则化手段丰富，可解释性工具成熟（特征重要性、SHAP）。

</details>

### E11. LightGBM 早停实战 `中等` `押题`

用 LightGBM 在有噪声的非线性数据上训练，按时间顺序切出验证集，用早停确定树的数量，比较早停模型和训练满 2000 棵树的模型在测试集上的表现。

<details>
<summary>答案与解析</summary>

```python
import lightgbm as lgb
import numpy as np

rng = np.random.default_rng(0)
n = 6000
X = rng.standard_normal((n, 10))
signal = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
y = 0.3 * signal + rng.standard_normal(n)              # 低信噪比
tr, va, te = slice(0, 3000), slice(3000, 4500), slice(4500, n)   # 按顺序切分

params = dict(n_estimators=2000, learning_rate=0.05, num_leaves=15,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, random_state=0)
es = lgb.LGBMRegressor(**params).fit(
    X[tr], y[tr], eval_X=X[va], eval_y=y[va],          # 4.7 起推荐写法；旧版用 eval_set=[(X, y)]
    callbacks=[lgb.early_stopping(100, verbose=False)])
full = lgb.LGBMRegressor(**params).fit(X[tr], y[tr])

def corr(a, b):
    return np.corrcoef(a, b)[0, 1]

print("最佳迭代次数:", es.best_iteration_)
print(f"早停模型   测试集相关系数 {corr(es.predict(X[te]), y[te]):.3f}")
print(f"2000 棵树 测试集相关系数 {corr(full.predict(X[te]), y[te]):.3f}")
```

 最佳迭代次数为几十轮（此例为 71），早停模型的测试集相关系数（0.177）明显高于训练满 2000 棵树的模型（0.111）。

**要点**：
- 早停用的验证集不能和测试集重合，否则测试结果偏乐观。
- 指标可以自定义，量化中常用验证集的截面 RankIC 作为早停指标。
- 滚动训练时，每个窗口的最佳轮数会波动，可以取近几个窗口的中位数作为固定值，避免每次都要切验证集。

</details>

---

## F. SVM、kNN、聚类、PCA

> 对应讲解第 7 节。

### F1. SVM 的硬间隔与支持向量 `中等` `经典·李航 7.1`

推导线性可分 SVM 的优化问题，并验证：对线性可分数据，`SVC(kernel="linear", C` 很大`)` 的支持向量满足 $y_i(w^\top x_i+b)=1$，间隔宽度为 $2/\lVert w\rVert$。

<details>
<summary>答案与解析</summary>

**推导**：点 $x_i$ 到超平面 $w^\top x+b=0$ 的几何距离为 $y_i(w^\top x_i+b)/\lVert w\rVert$。$(w,b)$ 同时放缩不改变超平面，所以可以规定最近点满足 $y_i(w^\top x_i+b)=1$（函数间隔为 1），此时几何间隔为 $1/\lVert w\rVert$。最大化间隔等价于

$$\min_{w,b}\ \tfrac12\lVert w\rVert^2\quad\text{s.t. }y_i(w^\top x_i+b)\ge1$$

KKT 互补松弛条件 $\alpha_i[y_i(w^\top x_i+b)-1]=0$：只有约束取等号的点（支持向量）才可能 $\alpha_i>0$。

```python
import numpy as np
from sklearn.svm import SVC

rng = np.random.default_rng(0)
X = np.r_[rng.normal([-2, -2], 0.6, (40, 2)), rng.normal([2, 2], 0.6, (40, 2))]
y = np.r_[-np.ones(40), np.ones(40)]

svm = SVC(kernel="linear", C=1e6).fit(X, y)
w, b = svm.coef_.ravel(), svm.intercept_[0]
margins = y[svm.support_] * (X[svm.support_] @ w + b)
print("支持向量个数:", len(svm.support_))
print("支持向量的函数间隔:", np.round(margins, 4))
print("所有样本的最小函数间隔:", round((y * (X @ w + b)).min(), 4))
print("间隔宽度 2/||w|| =", round(2 / np.linalg.norm(w), 4))
```

 支持向量只有 2 个，函数间隔均为 1.0000，所有样本的最小函数间隔也是 1；间隔宽度约 2.78。只用支持向量重新训练，得到的 $(w,b)$ 不变。

</details>

### F2. 软间隔 SVM 与 hinge 损失 `中等` `经典·李航 7.2`

证明软间隔 SVM 等价于"hinge 损失 + L2 正则"，并说明 $C$ 的作用。比较 hinge 损失与逻辑回归的对数损失。

<details>
<summary>答案与解析</summary>

软间隔问题：

$$\min\ \tfrac12\lVert w\rVert^2+C\sum_i\xi_i\quad\text{s.t. }y_i(w^\top x_i+b)\ge1-\xi_i,\ \xi_i\ge0$$

对固定的 $(w,b)$，最优的 $\xi_i=\max(0,1-y_if(x_i))$。代回得

$$\min\ \sum_i\max\big(0,1-y_if(x_i)\big)+\frac{1}{2C}\lVert w\rVert^2$$

即 hinge 损失 + L2 正则，$\lambda=\frac1{2C}$。$C$ 越大正则越弱，越倾向于把每个点都分对，越容易过拟合。

| 损失（$m=yf(x)$） | 公式 | 特点 |
|---|---|---|
| hinge | $\max(0,1-m)$ | $m\ge1$ 时损失为 0，解是稀疏的（只依赖支持向量） |
| 对数损失 | $\log(1+e^{-m})$ | 处处为正，所有样本都有贡献；输出概率 |
| 0-1 | $\mathbf 1[m<0]$ | 两者都是它的凸上界（对数损失除以 $\ln2$ 后） |

</details>

### F3. 核技巧 `基础` `经典·李航 7.3`

验证二维输入下多项式核 $K(x,z)=(x^\top z+1)^2$ 等于某个 6 维特征映射的内积，并写出该映射。

<details>
<summary>答案与解析</summary>

展开 $(x_1z_1+x_2z_2+1)^2=x_1^2z_1^2+x_2^2z_2^2+2x_1x_2z_1z_2+2x_1z_1+2x_2z_2+1$，所以

$$\phi(x)=\big(1,\ \sqrt2x_1,\ \sqrt2x_2,\ x_1^2,\ x_2^2,\ \sqrt2x_1x_2\big)$$

```python
import numpy as np

def poly_kernel(x, z):
    return (x @ z + 1) ** 2

def phi(x):
    x1, x2 = x
    s = np.sqrt(2)
    return np.array([1, s * x1, s * x2, x1 ** 2, x2 ** 2, s * x1 * x2])

rng = np.random.default_rng(0)
for _ in range(3):
    x, z = rng.standard_normal(2), rng.standard_normal(2)
    print(round(poly_kernel(x, z), 10), round(phi(x) @ phi(z), 10))
```

 两者相等。

**意义**：$d$ 维输入的 $p$ 次多项式核对应 $\binom{d+p}{p}$ 维特征空间，直接计算核只需 $O(d)$。RBF 核对应无穷维特征空间。核方法的代价是需要 $n\times n$ 的核矩阵，样本量大时不可行。

</details>

### F4. 维数灾难 `中等` `经典·ESL 2.5`

在 $[0,1]^d$ 中均匀撒 1000 个点，计算一个随机查询点到最近点和最远点的距离之比 $(d_{\max}-d_{\min})/d_{\min}$，观察 $d=2,10,100,1000$ 时的变化。这对 kNN 意味着什么？

<details>
<summary>答案与解析</summary>

```python
import numpy as np

rng = np.random.default_rng(0)
for d in (2, 10, 100, 1000):
    ratios = []
    for _ in range(20):
        X = rng.random((1000, d))
        q = rng.random(d)
        dist = np.linalg.norm(X - q, axis=1)
        ratios.append((dist.max() - dist.min()) / dist.min())
    print(f"d={d:>4}: (dmax-dmin)/dmin = {np.mean(ratios):.3f}")
```

 比值随维度单调下降：$d=2$ 时约 120（最近点比最远点近得多），$d=10$ 时约 3，$d=100$ 时约 0.44，$d=1000$ 时约 0.12，所有点到查询点的距离几乎相同。

**对 kNN 的影响**：高维下"最近邻"与"随机点"差不多近，邻居失去意义。另一种表述：要在单位超立方体中覆盖 10% 的数据，$d=10$ 时需要边长 $0.1^{1/10}\approx0.79$ 的邻域，已经不"局部"了。对策：降维（PCA）、特征选择、度量学习。

</details>

### F5. 手写 K-means++ `中等` `经典·李航 14.3`

实现 K-means（k-means++ 初始化 + Lloyd 迭代），验证目标函数单调不增，并与 sklearn 的结果比较。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.cluster import KMeans

def kmeans_pp_init(X, k, rng):
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])   # 按距离平方加权抽样
    return np.array(centers)

def kmeans(X, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(X, k, rng)
    history = []
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        labels = d2.argmin(1)
        history.append(d2[np.arange(len(X)), labels].sum())
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, labels, history

rng = np.random.default_rng(1)
centers_true = np.array([[0, 0], [5, 5], [0, 6], [6, 0]])
X = np.vstack([c + rng.standard_normal((150, 2)) for c in centers_true])

best = min((kmeans(X, 4, seed=s) for s in range(10)), key=lambda r: r[2][-1])
C, labels, hist = best
sk = KMeans(4, n_init=10, random_state=0).fit(X)
print("目标函数单调不增:", all(a >= b - 1e-9 for a, b in zip(hist, hist[1:])))
print(f"手写 SSE = {hist[-1]:.2f}, sklearn inertia = {sk.inertia_:.2f}")
```

 目标函数单调不增；10 次初始化取最好时，SSE 与 sklearn 的 `inertia_` 相同。

**为什么单调**：分配步骤固定中心、把每个点分给最近的中心，目标不增；更新步骤固定分配，簇均值是使簇内平方和最小的点，目标不增。可能的分配方式有限，所以有限步内收敛，但只保证局部最优。

</details>

### F6. 用 SVD 实现 PCA `基础` `经典·李航 16.2`

用 SVD 实现 PCA，输出主成分方向和方差解释比例，与 `sklearn.decomposition.PCA` 对比。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.decomposition import PCA

def pca_svd(X, k):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)                 # 各主成分方差 = 协方差矩阵特征值
    return Vt[:k], var[:k] / var.sum(), Xc @ Vt[:k].T

rng = np.random.default_rng(0)
A = rng.standard_normal((5, 5))
X = rng.standard_normal((500, 5)) @ A           # 相关的 5 维数据

comps, ratio, scores = pca_svd(X, 3)
sk = PCA(3).fit(X)
print(np.round(ratio, 4))
print(np.allclose(ratio, sk.explained_variance_ratio_),
      np.allclose(np.abs(comps), np.abs(sk.components_)))   # 方向只差符号
```

 方差解释比例与 sklearn 一致；主成分方向一致（特征向量的符号不唯一，比较绝对值）。

**为什么用 SVD 而不是对协方差矩阵做特征分解**：$X^\top X$ 的条件数是 $X$ 的平方，直接对 $X$ 做 SVD 数值上更稳定；$p\gg n$ 时也可以只算前 $k$ 个奇异向量（截断 SVD、随机化 SVD）。

</details>

### F7. 股票收益的第一主成分是市场因子 `中等` `押题`

用单因子模型 $r_{it}=\beta_if_t+\varepsilon_{it}$ 生成 200 只股票 500 天的收益，做 PCA，检验第一主成分得分与市场因子 $f_t$ 的相关性，以及载荷的符号。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

rng = np.random.default_rng(0)
T, N = 500, 200
f = 0.01 * rng.standard_normal(T)                    # 市场因子，日波动 1%
beta = rng.uniform(0.5, 1.5, N)
R = np.outer(f, beta) + 0.02 * rng.standard_normal((T, N))   # 特质波动 2%

Rc = R - R.mean(0)
U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
pc1_score = Rc @ Vt[0]
loading = Vt[0] * np.sign(Vt[0].sum())               # 统一符号
print("PC1 方差解释比例:", round(S[0] ** 2 / (S ** 2).sum(), 3))
print("PC1 得分与市场因子的相关系数:", round(abs(np.corrcoef(pc1_score, f)[0, 1]), 4))
print("载荷全部同号:", bool(np.all(loading > 0)))
print("载荷与 beta 的相关系数:", round(np.corrcoef(loading, beta)[0, 1], 3))
```

 PC1 解释约 22% 的方差，得分与市场因子相关系数约 0.99，载荷全部同号且与 $\beta$ 高度相关（约 0.95）。即使单只股票的特质波动是市场波动的 2 倍，200 只股票平均后市场成分依然清晰。

> **量化用途**：真实 A 股数据中，第一主成分通常解释 20%–40% 的收益方差（随市场状态变化，系统性下跌时更高），后续主成分对应行业和风格（如大小盘）。统计风险模型就是用前若干个主成分作为风险因子。

</details>

### F8. GMM、EM 与 K-means `中等` `经典·李航 9.3`

简述 EM 算法在高斯混合模型中的 E 步和 M 步，并说明 K-means 是 GMM 的什么特例。

<details>
<summary>答案与解析</summary>

**GMM**：$p(x)=\sum_k\pi_kN(x\mid\mu_k,\Sigma_k)$。隐变量 $z$ 表示样本来自哪个分量。

| 步骤 | 计算 |
|---|---|
| E 步 | 责任 $\gamma_{ik}=\dfrac{\pi_kN(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_jN(x_i\mid\mu_j,\Sigma_j)}$ |
| M 步 | $N_k=\sum_i\gamma_{ik}$，$\pi_k=N_k/n$，$\mu_k=\frac1{N_k}\sum_i\gamma_{ik}x_i$，$\Sigma_k=\frac1{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$ |

EM 每次迭代不降低对数似然（E 步构造下界并使其在当前点取等，M 步最大化下界），收敛到局部极大。

**K-means 是特例**：令所有 $\Sigma_k=\sigma^2I$、$\pi_k$ 相等，并让 $\sigma\to0$，责任 $\gamma_{ik}$ 变成"离哪个中心最近就是 1，否则为 0"的硬分配，M 步变成取簇均值，正是 Lloyd 算法。

> **量化用途**：用 GMM 或隐马尔可夫模型对市场收益与波动率做状态划分（牛市、熊市、震荡），再按状态调整策略。注意状态划分是事后拟合的，实时识别会有滞后。

</details>

---

## G. 评价指标

> 对应讲解第 8 节。

### G1. 手算 AUC `基础` `经典·ISLR 4.4`

7 个样本的得分为 $(0.9,0.8,0.7,0.6,0.55,0.4,0.3)$，标签为 $(1,1,0,1,0,0,1)$。计算 AUC，并说明其概率含义。

<details>
<summary>答案与解析</summary>

正样本 4 个（0.9、0.8、0.6、0.3），负样本 3 个（0.7、0.55、0.4），共 12 个正负对。

| 正样本得分 | 高于几个负样本 |
|---|---|
| 0.9 | 3 |
| 0.8 | 3 |
| 0.6 | 2 |
| 0.3 | 0 |

$\mathrm{AUC}=8/12=0.667$：随机取一个正样本和一个负样本，正样本得分更高的概率是 2/3。

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_pairs(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0) + 0.5 * (diff == 0)).mean()

s = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3]
y = [1, 1, 0, 1, 0, 0, 1]
print(auc_pairs(y, s), roc_auc_score(y, s))
```

 两者均为 0.6667。

</details>

### G2. 用排序在 $O(n\log n)$ 内计算 AUC `中等` `押题`

G1 的两两比较是 $O(n_+n_-)$。用秩实现 $O(n\log n)$ 的 AUC，正确处理得分相同的情况，并与 sklearn 随机对拍。

<details>
<summary>答案与解析</summary>

**公式**：把全部得分升序排列，相同得分取平均秩。正样本秩之和减去正样本内部的 $\frac{n_+(n_++1)}2$，剩下的就是"正样本高于负样本"的对数（并列算半对）：

$$\mathrm{AUC}=\frac{\sum_{i\in+}\mathrm{rank}_i-\frac{n_+(n_++1)}{2}}{n_+n_-}$$

```python
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def auc_rank(y, s):
    y = np.asarray(y)
    r = rankdata(s)                       # 并列取平均秩
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

rng = np.random.default_rng(0)
max_err = 0
for _ in range(500):
    n = rng.integers(5, 200)
    y = rng.integers(0, 2, n)
    if y.min() == y.max():
        continue
    s = rng.integers(0, 10, n)            # 大量并列
    max_err = max(max_err, abs(auc_rank(y, s) - roc_auc_score(y, s)))
print(max_err)
```

 500 组含大量并列的随机数据上与 sklearn 的最大差异小于 $10^{-12}$。

**追问**：这就是 Mann-Whitney U 检验，$U=\mathrm{AUC}\cdot n_+n_-$。因此"AUC 是否显著大于 0.5"可以直接用 Mann-Whitney U 检验。

</details>

### G3. AUC 的不变性与类别不平衡 `中等` `押题`

(1) 对得分做单调变换（如取指数），AUC 会变吗？(2) 正负样本比例从 1:1 变为 1:20，同一模型的 AUC 和固定阈值下的精确率分别怎么变？用模拟说明。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score

rng = np.random.default_rng(0)
def sample(n_pos, n_neg):
    s = np.r_[rng.normal(1, 1, n_pos), rng.normal(0, 1, n_neg)]   # 同一个打分模型
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    return y, s

y, s = sample(5000, 5000)
print("AUC 原始 / 取指数:", round(roc_auc_score(y, s), 4), round(roc_auc_score(y, np.exp(s)), 4))
for n_neg in (5000, 100000):
    y, s = sample(5000, n_neg)
    print(f"正:负 = 1:{n_neg // 5000:<2} AUC={roc_auc_score(y, s):.3f}  "
          f"精确率(阈值0.5)={precision_score(y, s > 0.5):.3f}  PR-AUC={average_precision_score(y, s):.3f}")
```

 (1) 单调变换后 AUC 完全不变（只依赖排序）。(2) 负样本增加 20 倍后，AUC 基本不变（约 0.76），而精确率从约 0.69 降到约 0.10，PR-AUC 从约 0.76 降到约 0.17。

**结论**：
- ROC 的 TPR、FPR 分别在正类、负类内部计算，与类别比例无关；精确率的分母同时包含正负样本，随比例变化。
- 正样本稀少、关心"预测为正的有多少是真的"时，应看 PR 曲线。
- AUC 不反映概率是否校准（见 G7）。

</details>

### G4. 对数损失与常数基准 `基础` `押题`

正样本比例为 30%。(1) 恒预测 $p=0.3$ 的对数损失是多少？(2) 一个模型的对数损失为 0.62，它有价值吗？(3) 真实为 1、预测 $p=0.01$ 的单样本损失是多少？

<details>
<summary>答案与解析</summary>

(1) $-(0.3\ln0.3+0.7\ln0.7)=0.6109$，即伯努利分布的熵 $H(0.3)$。这是不用任何特征能达到的最好对数损失（恒预测其他常数只会更差）。

(2) 0.62 > 0.6109，比常数预测还差，没有价值（可能过拟合或概率未校准）。

(3) $-\ln0.01=4.605$，一个过度自信的错误相当于约 7.5 个基准样本的损失。

```python
import numpy as np
from sklearn.metrics import log_loss

y = np.r_[np.ones(300), np.zeros(700)]
print(round(log_loss(y, np.full(1000, 0.3)), 4))
for c in (0.2, 0.25, 0.35, 0.4):
    print(c, round(log_loss(y, np.full(1000, c)), 4))
print(round(-np.log(0.01), 3))
```

 常数 0.3 的对数损失为 0.6109，其他常数均更大；$-\ln0.01=4.605$。

</details>

### G5. 精确率、召回率与准确率陷阱 `基础` `经典·ISLR 4.4`

1000 个样本中 50 个正例。模型预测 40 个为正，其中 30 个正确。求准确率、精确率、召回率、F1。全部预测为负的准确率是多少？

<details>
<summary>答案与解析</summary>

$TP=30$，$FP=10$，$FN=20$，$TN=940$。

| 指标 | 计算 | 值 |
|---|---|---|
| 准确率 | $(30+940)/1000$ | 0.970 |
| 精确率 | $30/40$ | 0.750 |
| 召回率 | $30/50$ | 0.600 |
| F1 | $2\times0.75\times0.6/1.35$ | 0.667 |
| 全部预测为负的准确率 | $950/1000$ | 0.950 |

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

y = np.r_[np.ones(50), np.zeros(950)]
pred = np.r_[np.ones(30), np.zeros(20), np.ones(10), np.zeros(940)]
print(accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred), round(f1_score(y, pred), 4))
print(accuracy_score(y, np.zeros(1000)))
```

 输出 0.97、0.75、0.6、0.6667 和 0.95。准确率只比"什么都不做"高 2 个百分点，几乎不反映模型质量。

</details>

### G6. 交叉熵、熵与 KL 散度 `基础` `经典·DL 3.13`

证明 $H(p,q)=H(p)+D_{KL}(p\Vert q)$，并说明为什么最小化交叉熵等价于极大似然。

<details>
<summary>答案与解析</summary>

$$H(p,q)=-\sum_xp(x)\log q(x)=-\sum_xp(x)\log p(x)+\sum_xp(x)\log\frac{p(x)}{q(x)}=H(p)+D_{KL}(p\Vert q)$$

$H(p)$ 由数据决定，与模型 $q$ 无关，所以最小化交叉熵等价于最小化 KL 散度。$D_{KL}\ge0$（Jensen 不等式），当且仅当 $q=p$ 时为 0。

取 $p$ 为经验分布 $\hat p=\frac1n\sum_i\delta_{y_i}$，则 $H(\hat p,q_\theta)=-\frac1n\sum_i\log q_\theta(y_i)$，正是平均负对数似然。

KL 散度不对称：$D_{KL}(p\Vert q)$ 惩罚 $p$ 大而 $q$ 小的地方（$q$ 要覆盖 $p$ 的全部支撑），$D_{KL}(q\Vert p)$ 则倾向于让 $q$ 集中在 $p$ 的一个峰上。

</details>

### G7. AUC 高不代表概率准确 `中等` `押题`

把一个已校准模型的 logit 乘以 3，AUC、对数损失、Brier 分数分别怎么变？如何校准概率？

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

rng = np.random.default_rng(0)
z = rng.normal(0, 1.5, 20000)                       # 真实 logit
p = 1 / (1 + np.exp(-z))
y = (rng.random(20000) < p).astype(int)             # 按真实概率生成标签

for k in (1, 3):
    q = 1 / (1 + np.exp(-k * z))
    print(f"logit x{k}: AUC={roc_auc_score(y, q):.4f}  logloss={log_loss(y, q):.4f}  Brier={brier_score_loss(y, q):.4f}")
```

 AUC 完全相同（0.8109），对数损失从 0.529 升到 0.747，Brier 分数从 0.178 升到 0.206。

**校准方法**：
- **Platt 缩放**：在验证集上以模型输出的 logit 为特征拟合一个一维逻辑回归 $\sigma(as+b)$；
- **保序回归**（isotonic）：拟合单调阶梯函数，更灵活，需要更多数据；
- sklearn 的 `CalibratedClassifierCV`；用可靠性图（reliability diagram）检查。

> **量化用途**：如果只用预测值做截面排序选股，校准无关紧要；如果用预测概率决定仓位大小（如按 Kelly 公式），校准就很重要。

</details>

### G8. IC、$R^2$ 与样本外 $R^2$ `中等` `押题`

(1) 证明一元线性回归（含截距）的样本内 $R^2$ 等于预测值与真实值相关系数的平方。(2) 样本外 $R^2$ 为什么可能为负？(3) 某模型日频截面 IC 为 0.05，相当于多大的 $R^2$？

<details>
<summary>答案与解析</summary>

(1) 含截距时 $\hat y$ 是 $y$ 在 $\mathrm{span}\{1,x\}$ 上的正交投影，残差与 $\hat y-\bar y$ 正交，$\mathrm{Cov}(y,\hat y)=\mathrm{Var}(\hat y)$，于是

$$\mathrm{Corr}(y,\hat y)^2=\frac{\mathrm{Var}(\hat y)^2}{\mathrm{Var}(y)\mathrm{Var}(\hat y)}=\frac{\mathrm{Var}(\hat y)}{\mathrm{Var}(y)}=R^2$$

(2) 样本外 $R^2_{oos}=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y_{train})^2}$，分母是"用训练集均值预测"的误差。若模型预测比常数预测还差（尺度过大、方向错误、过拟合），$R^2_{oos}<0$。**相关系数为正的预测，若尺度过大，$R^2_{oos}$ 仍可为负。**

(3) 若预测与收益是线性关系且尺度最优，$R^2=\mathrm{IC}^2=0.0025$，即 0.25%。

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
x = rng.standard_normal(5000)
y = 0.05 * x + rng.standard_normal(5000)              # IC 约 0.05

yhat = LinearRegression().fit(x[:, None], y).predict(x[:, None])
print(round(r2_score(y, yhat), 5), round(np.corrcoef(y, yhat)[0, 1] ** 2, 5))
print("预测尺度放大 10 倍的 R^2:", round(r2_score(y, 10 * yhat), 4),
      " 相关系数:", round(np.corrcoef(y, 10 * yhat)[0, 1], 4))
```

 样本内 $R^2$ 与相关系数平方相等（此例为 0.0041）；把预测放大 10 倍后，相关系数不变，$R^2$ 变为约 $-0.35$。

> **量化用途**：选股只用预测值的排序，尺度无关，所以 IC 比 $R^2$ 更适合作为评价指标。

</details>

---

## H. 特征工程、类别不平衡、过拟合排查

> 对应讲解第 9 节。

### H1. 截面预处理流水线 `基础` `押题`

长表 `df` 含 `date`、`code`、`factor` 三列。按日期分组实现：(1) MAD 去极值（中位数 ± 5 × 1.4826 × MAD）；(2) z-score 标准化；(3) rank 高斯化（排名映射为标准正态分位数）。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
import pandas as pd
from scipy.stats import norm

def mad_clip(s, k=5.0):
    med = s.median()
    mad = (s - med).abs().median()
    bound = k * 1.4826 * mad
    return s.clip(med - bound, med + bound)

def zscore(s):
    return (s - s.mean()) / s.std()

def rank_gauss(s):
    r = s.rank(method="average")
    return pd.Series(norm.ppf((r - 0.5) / s.count()), index=s.index)

rng = np.random.default_rng(0)
dates = pd.date_range("2024-01-01", periods=5, freq="B")
df = pd.DataFrame({
    "date": np.repeat(dates, 300),
    "code": np.tile(np.arange(300), 5),
    "factor": rng.standard_t(df=2, size=1500),       # 厚尾
})

g = df.groupby("date")["factor"]
df["clipped"] = g.transform(mad_clip)
df["z"] = df.groupby("date")["clipped"].transform(zscore)
df["rg"] = g.transform(rank_gauss)
print(df.groupby("date")[["z", "rg"]].agg(["mean", "std"]).round(3))
print("原始最大值:", round(df["factor"].abs().max(), 1), " 去极值后:", round(df["clipped"].abs().max(), 1))
```

 每个日期的 `z` 均值为 0、标准差为 1；`rg` 均值为 0、标准差约 1；厚尾的原始最大值（约 40）被截到约 7。

**要点**：
- `transform` 返回与原表等长的结果，可直接赋值为新列；`apply` 返回的形状不固定。
- 1.4826 使正态分布下 MAD × 1.4826 等于标准差。
- rank 高斯化对异常值最稳健，但丢失了因子值的距离信息。

</details>

### H2. 目标编码的泄漏与折外编码 `中等` `押题`

5000 个样本、一个有 1000 个取值的类别特征，标签完全随机。比较"用全样本类别均值编码"和"5 折折外编码"后，编码值与标签的 AUC。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

rng = np.random.default_rng(0)
n = 5000
df = pd.DataFrame({"cat": rng.integers(0, 1000, n), "y": rng.integers(0, 2, n)})

# 错误：编码值包含样本自身的标签
naive = df.groupby("cat")["y"].transform("mean")

# 正确：每个样本的编码只用其他折的数据
oof = np.full(n, df["y"].mean())
for tr, va in KFold(5, shuffle=True, random_state=0).split(df):
    means = df.iloc[tr].groupby("cat")["y"].mean()
    oof[va] = df["cat"].iloc[va].map(means).fillna(df["y"].iloc[tr].mean()).to_numpy()

print("全样本编码 AUC:", round(roc_auc_score(df["y"], naive), 3))
print("折外编码 AUC:  ", round(roc_auc_score(df["y"], oof), 3))
```

 全样本编码的 AUC 约 0.75（每个类别平均只有 5 个样本，自身标签占比很大），折外编码约 0.5。

**改进**：对样本少的类别向全局均值收缩，$\frac{n_c\bar y_c+m\bar y}{n_c+m}$；时间序列中只用过去数据计算均值（expanding）。

</details>

### H3. 类别不平衡的处理 `基础` `经典·AQI Q51`

正负样本比例为 1:50。列出可用的处理方法，并说明在金融时间序列中使用时的注意点。

<details>
<summary>答案与解析</summary>

| 层面 | 方法 | 金融数据中的注意点 |
|---|---|---|
| 评价 | PR-AUC、F1、召回率、按阈值的收益 | 最终看策略收益而不是分类指标 |
| 数据 | 欠采样、过采样、SMOTE | 只在训练集内做；SMOTE 在不同时间的样本之间插值，会制造不存在的样本；欠采样会丢掉大量正常市场状态 |
| 损失 | 类别权重、Focal Loss | 改变了输出概率的含义，需要重新校准（H4） |
| 决策 | 调整阈值 | 最简单，不改变模型 |
| 问题重构 | 改成回归或排序；改成异常检测 | 量化中常把"大涨"分类改为预测收益的排序 |

</details>

### H4. 欠采样后的概率修正 `困难` `押题`

对负样本按比例 $\beta$ 欠采样（每个负样本以概率 $\beta$ 保留）后训练逻辑回归，预测概率会偏高。推导修正公式，并用模拟验证截距的变化。

<details>
<summary>答案与解析</summary>

欠采样后，正样本全部保留，负样本保留比例 $\beta$，所以对任意 $x$，

$$\frac{p_s(x)}{1-p_s(x)}=\frac{P(y=1\mid x)}{\beta P(y=0\mid x)}=\frac{1}{\beta}\cdot\frac{p(x)}{1-p(x)}$$

即 $\mathrm{logit}\,p_s=\mathrm{logit}\,p-\log\beta$。逻辑回归中只有截距改变：$b_s=b-\log\beta$，斜率不变。修正：$\mathrm{logit}\,p=\mathrm{logit}\,p_s+\log\beta$，等价于

$$p=\frac{\beta p_s}{\beta p_s+1-p_s}$$

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 200_000
x = rng.standard_normal((n, 1))
p = 1 / (1 + np.exp(-(-4 + 1.0 * x[:, 0])))           # 正样本约 3%
y = (rng.random(n) < p).astype(int)

beta = 0.05
keep = (y == 1) | (rng.random(n) < beta)
full = LogisticRegression(C=1e6).fit(x, y)
sub = LogisticRegression(C=1e6).fit(x[keep], y[keep])
print("截距差 b_s - b =", round(sub.intercept_[0] - full.intercept_[0], 3), " 理论 -log(beta) =", round(-np.log(beta), 3))
print("斜率:", round(full.coef_[0, 0], 3), round(sub.coef_[0, 0], 3))

ps = sub.predict_proba(x)[:, 1]
corrected = beta * ps / (beta * ps + 1 - ps)
print("平均预测概率 修正前/修正后/真实:", round(ps.mean(), 4), round(corrected.mean(), 4), round(y.mean(), 4))
```

 截距差约 3.0，与理论值 $-\log0.05=2.996$ 一致；斜率不变；修正后的平均概率与真实正样本比例一致。

</details>

### H5. 过拟合排查清单 `中等` `押题` `经典·AQI Q39`

面试官："你的模型回测年化 80%，实盘第一个月就亏钱。你会怎么排查？"

<details>
<summary>答案与解析</summary>

按"先排除错误，再排除过拟合，最后考虑市场变化"的顺序：

| 步骤 | 检查内容 | 方法 |
|---|---|---|
| 1. 泄漏 | 特征是否用到未来；预处理、特征选择是否在全样本上做；财务数据是否按公告日 | B6 的"改动未来数据"测试；B8 的清单；逐个删除重要特征看效果 |
| 2. 回测实现 | 成交价假设、涨跌停与停牌、成本、复权方式 | 用实盘成交记录复算同期回测，对比每日持仓和收益 |
| 3. 过拟合 | 尝试过多少种配置；验证期多长；参数是否在平坦区域 | 统计尝试次数并计算 Deflated Sharpe；参数小幅扰动后看收益是否稳定 |
| 4. 统计噪声 | 实盘期太短 | 一个月约 20 个交易日，Sharpe 2 的策略单月亏损的概率并不小 |
| 5. 市场变化 | 风格切换、拥挤、监管 | 分析亏损来源于哪些风格暴露；对比同类产品同期表现 |

最后一条要说明：如果只有一个月，第 4 步可能就是答案；回测年化 80% 本身就是需要怀疑的信号。

</details>

### H6. 基于相关性的特征去重 `基础` `押题`

有 $p$ 个因子，要求保留的因子两两相关系数绝对值不超过 0.7，并优先保留 IC 绝对值高的因子。实现贪心算法。

<details>
<summary>答案与解析</summary>

**思路**：按 $\lvert\mathrm{IC}\rvert$ 从高到低遍历，若当前因子与已保留的所有因子的相关系数绝对值都不超过阈值，则保留。

```python
import numpy as np

def greedy_dedupe(corr, ic, thr=0.7):
    order = np.argsort(-np.abs(ic))
    kept = []
    for j in order:
        if all(abs(corr[j, k]) <= thr for k in kept):
            kept.append(int(j))
    return sorted(kept)

rng = np.random.default_rng(0)
base = rng.standard_normal((1000, 3))
F = np.c_[base[:, 0], base[:, 0] + 0.3 * rng.standard_normal(1000),   # 0、1 高度相关
          base[:, 1], base[:, 1] + 0.3 * rng.standard_normal(1000),   # 2、3 高度相关
          base[:, 2]]
corr = np.corrcoef(F, rowvar=False)
ic = np.array([0.03, 0.05, 0.04, 0.02, 0.01])
print(greedy_dedupe(corr, ic))
```

 输出 `[1, 2, 4]`：每组相关因子中保留 IC 较高的一个。

**其他方法**：对相关矩阵做层次聚类，每簇取代表或取簇内平均；对称正交化（$F(F^\top F)^{-1/2}$ 型变换，保持与原因子最接近）；PCA；在模型中加 L1。

</details>

### H7. 缺失值处理 `基础` `押题`

量化数据中常见的缺失有哪些？分别如何处理？

<details>
<summary>答案与解析</summary>

| 缺失来源 | 例子 | 处理 |
|---|---|---|
| 停牌 | 当天无价格、无成交 | 量价因子置缺失，不要前向填充收益（会制造零收益）；停牌股不可交易，回测时剔除 |
| 新股 | 回看窗口不足（如 250 日波动率） | 设最小样本数；新股前若干日单独处理 |
| 财务未披露 | 季报尚未发布 | 用上一期 point-in-time 值；缺失本身可作为特征 |
| 数据商缺漏 | 字段为空 | 截面中位数或行业中位数填充，加缺失指示变量 |
| 分析师覆盖不足 | 小盘股无一致预期 | 缺失与市值相关，填充时注意引入市值暴露 |

**原则**：填充所用的统计量只能来自当天截面或过去；树模型可直接处理缺失（E6）；线性模型和神经网络需要填充，同时加缺失指示变量。

</details>

---

## I. 神经网络与反向传播

> 对应讲解第 10 节。本章代码只用 NumPy（不依赖深度学习框架），I4、I5 共用同一组函数。

### I1. 为什么需要非线性激活 `基础` `经典·DL 6.1`

去掉所有激活函数后，一个 10 层的全连接网络等价于什么模型？

<details>
<summary>答案与解析</summary>

$W_{10}(\cdots(W_2(W_1x+b_1)+b_2)\cdots)+b_{10}=\tilde Wx+\tilde b$，其中 $\tilde W=W_{10}\cdots W_1$。多层线性变换的复合仍是线性变换，所以等价于一个线性模型（回归时就是线性回归，接 softmax 时就是多项逻辑回归），无法表示 XOR 这样的非线性关系。

有非线性激活时，单隐藏层网络在隐藏单元足够多时就能以任意精度逼近紧集上的连续函数（万能逼近定理）；但深层网络用更少的参数就能表示同样复杂的函数，这是"深"的价值。

</details>

### I2. Sigmoid 导致梯度消失 `基础` `经典·DL 6.3`

证明 sigmoid 的导数最大为 $1/4$。若一个 10 层网络每层都用 sigmoid，且权重约为 1，传回第一层的梯度最多缩小多少倍？

<details>
<summary>答案与解析</summary>

$\sigma'(z)=\sigma(z)(1-\sigma(z))$，令 $s=\sigma(z)\in(0,1)$，$s(1-s)\le\frac14$，在 $s=\frac12$（$z=0$）时取等。

链式法则下每层贡献一个因子 $w\,\sigma'(z)$，权重约为 1 时每层最多 $\frac14$，10 层后最多为 $(1/4)^{10}\approx9.5\times10^{-7}$，即梯度至少缩小约一百万倍；若神经元处于饱和区（$\lvert z\rvert$ 大），$\sigma'$ 更接近 0，衰减更严重。

```python
import numpy as np
z = np.linspace(-10, 10, 100001)
s = 1 / (1 + np.exp(-z))
print((s * (1 - s)).max(), 0.25 ** 10)
```

 数值最大导数为 0.25，$0.25^{10}=9.54\times10^{-7}$。

**对比**：ReLU 正区间导数为 1；tanh 导数最大为 1，但同样会饱和。

</details>

### I3. Softmax + 交叉熵的梯度 `中等` `经典·DL 6.2`

$p=\mathrm{softmax}(z)$，$L=-\sum_ky_k\log p_k$，$y$ 为 one-hot。证明 $\partial L/\partial z=p-y$。

<details>
<summary>答案与解析</summary>

softmax 的雅可比：$\dfrac{\partial p_k}{\partial z_j}=p_k(\delta_{kj}-p_j)$。

$$\frac{\partial L}{\partial z_j}=-\sum_ky_k\frac{1}{p_k}\cdot p_k(\delta_{kj}-p_j)=-\sum_ky_k(\delta_{kj}-p_j)=-y_j+p_j\sum_ky_k=p_j-y_j$$

最后一步用了 $\sum_ky_k=1$。

**意义**：
- 梯度有界（每个分量在 $[-1,1]$ 内），不会因 softmax 饱和而消失，这是分类任务用交叉熵而不用平方损失的原因之一；
- 二分类 sigmoid + 对数损失是 $K=2$ 的特例，梯度为 $p-y$（与 C2 一致）；
- 实现时把 softmax 与交叉熵合在一起计算（C5），数值更稳定，梯度也更简单。

</details>

### I4. 用 NumPy 实现两层网络并做梯度检验 `困难` `经典·DL 6.5`

实现两层全连接网络（ReLU 隐藏层、softmax 输出、交叉熵损失）的前向和反向传播，并用中心差分做梯度检验。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def init_params(d_in, d_hidden, d_out, rng):
    return {
        "W1": rng.standard_normal((d_in, d_hidden)) * np.sqrt(2 / d_in),   # He 初始化
        "b1": np.zeros(d_hidden),
        "W2": rng.standard_normal((d_hidden, d_out)) * np.sqrt(2 / d_hidden),
        "b2": np.zeros(d_out),
    }

def forward_backward(params, X, y, reg=0.0):
    n = len(X)
    # 前向
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0)
    Z2 = A1 @ params["W2"] + params["b2"]
    Z2s = Z2 - Z2.max(axis=1, keepdims=True)
    P = np.exp(Z2s) / np.exp(Z2s).sum(axis=1, keepdims=True)
    loss = -np.log(P[np.arange(n), y]).mean()
    loss += 0.5 * reg * ((params["W1"] ** 2).sum() + (params["W2"] ** 2).sum())
    # 反向
    dZ2 = P.copy()
    dZ2[np.arange(n), y] -= 1
    dZ2 /= n                                   # dL/dZ2 = (P - Y) / n
    grads = {"W2": A1.T @ dZ2 + reg * params["W2"], "b2": dZ2.sum(axis=0)}
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)                       # ReLU 的导数
    grads["W1"] = X.T @ dZ1 + reg * params["W1"]
    grads["b1"] = dZ1.sum(axis=0)
    return loss, grads, P

def grad_check(params, X, y, reg=1e-3, eps=1e-6):
    _, grads, _ = forward_backward(params, X, y, reg)
    worst = 0.0
    for k, W in params.items():
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = W[i]
            W[i] = old + eps; lp, _, _ = forward_backward(params, X, y, reg)
            W[i] = old - eps; lm, _, _ = forward_backward(params, X, y, reg)
            W[i] = old
            num[i] = (lp - lm) / (2 * eps)
        rel = np.abs(num - grads[k]).max() / max(1e-12, np.abs(num).max() + np.abs(grads[k]).max())
        worst = max(worst, rel)
        print(f"{k}: 相对误差 {rel:.2e}")
    return worst

rng = np.random.default_rng(0)
X_small = rng.standard_normal((20, 5))
y_small = rng.integers(0, 3, 20)
params_small = init_params(5, 8, 3, rng)
worst = grad_check(params_small, X_small, y_small)
```

 四组参数的相对误差都在 $10^{-9}$ 量级以下（双精度、$\epsilon=10^{-6}$）。

**易错点**：
- `dZ2` 要除以 $n$（损失取了平均），偏置梯度要沿 batch 维求和；
- 正则项只加在权重上，梯度检验时也要包含它；
- ReLU 在 0 处不可导，梯度检验时若有 $Z_1$ 恰好接近 0 可能出现较大误差，属于正常现象（"kink"）。

</details>

### I5. 训练网络拟合螺旋数据 `中等` `押题`

用 I4 的网络和小批量梯度下降，拟合 3 类螺旋数据（线性模型无法分开），要求训练准确率超过 95%。

<details>
<summary>答案与解析</summary>

```python
def make_spiral(n_per_class=200, K=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.0, 1, n_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, n_per_class) + rng.standard_normal(n_per_class) * 0.2
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, k))
    return np.vstack(X), np.concatenate(y)

def train(X, y, d_hidden=100, lr=1.0, reg=1e-3, epochs=3000, batch=200, seed=0):
    rng = np.random.default_rng(seed)
    params = init_params(X.shape[1], d_hidden, y.max() + 1, rng)
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            b = idx[s:s + batch]
            loss, grads, _ = forward_backward(params, X[b], y[b], reg)
            for k in params:
                params[k] -= lr * grads[k]
    return params

X_sp, y_sp = make_spiral()
params_sp = train(X_sp, y_sp)
_, _, P = forward_backward(params_sp, X_sp, y_sp)
acc_mlp = (P.argmax(1) == y_sp).mean()

from sklearn.linear_model import LogisticRegression
acc_lin = LogisticRegression(max_iter=1000).fit(X_sp, y_sp).score(X_sp, y_sp)
print(f"两层网络训练准确率 {acc_mlp:.3f}，逻辑回归 {acc_lin:.3f}")
```

 两层网络训练准确率 0.992，逻辑回归 0.540（线性边界对异或结构无能为力）。

**说明**：这里只看训练准确率，用来验证反向传播和优化是否正确实现；实际应用要在验证集上评估。

</details>

### I6. 初始化与激活方差 `中等` `经典·He et al. 2015`

20 层 ReLU 全连接网络（每层 512 个单元），分别用标准差 0.01、Xavier（$\sqrt{1/n_{in}}$）和 He（$\sqrt{2/n_{in}}$）初始化权重，观察各层激活的标准差。推导 He 初始化的系数。

<details>
<summary>答案与解析</summary>

**推导**：$z=\sum_{i=1}^{n}w_ia_i$，$w_i$ 独立、均值 0、方差 $\mathrm{Var}(w)$，与 $a_i$ 独立，则 $\mathrm{Var}(z)=n\,\mathrm{Var}(w)\,\mathbb E[a^2]$。若上一层 $z$ 对称、均值 0，ReLU 后 $\mathbb E[a^2]=\frac12\mathrm{Var}(z_{\text{prev}})$。要求方差逐层不变：$n\,\mathrm{Var}(w)\cdot\frac12=1$，得 $\mathrm{Var}(w)=2/n$。Xavier 针对 tanh 等在 0 附近近似线性的激活，取 $\mathrm{Var}(w)=1/n$（或 $2/(n_{in}+n_{out})$），用于 ReLU 时每层方差减半。

```python
import numpy as np

def activation_stds(scale_fn, n_layers=20, width=512, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((1000, width))
    stds = []
    for _ in range(n_layers):
        W = rng.standard_normal((width, width)) * scale_fn(width)
        a = np.maximum(a @ W, 0)
        stds.append(a.std())
    return stds

for name, fn in [("0.01", lambda n: 0.01), ("Xavier", lambda n: np.sqrt(1 / n)), ("He", lambda n: np.sqrt(2 / n))]:
    s = activation_stds(fn)
    print(f"{name:>6}: 第 1 层 {s[0]:.3e}  第 10 层 {s[9]:.3e}  第 20 层 {s[19]:.3e}")
```

 标准差 0.01：激活每层缩小约 $0.01\times\sqrt{512/2}\approx0.16$ 倍，20 层后约 $8.8\times10^{-17}$；Xavier：每层约缩小 $1/\sqrt2$，20 层后约 $7.1\times10^{-4}$；He：各层基本稳定在 0.73 左右。

</details>

### I7. 残差连接为什么有效 `中等` `经典·He et al. 2016`

残差块 $a_{l+1}=a_l+F(a_l)$。从梯度传播的角度解释它为什么能训练上百层的网络。

<details>
<summary>答案与解析</summary>

$$\frac{\partial a_L}{\partial a_l}=\prod_{k=l}^{L-1}\Big(I+\frac{\partial F_k}{\partial a_k}\Big)$$

展开后包含恒等项 $I$，因此 $\dfrac{\partial\mathcal L}{\partial a_l}=\dfrac{\partial\mathcal L}{\partial a_L}\Big(I+\cdots\Big)$：即使所有 $\partial F/\partial a$ 都很小，梯度仍能通过恒等路径无衰减地传回浅层。没有残差时是 $\prod_k\frac{\partial F_k}{\partial a_k}$，容易指数衰减。

**另两种理解**：
- 学习残差 $F(a)=H(a)-a$ 比直接学习 $H(a)$ 容易：若最优映射接近恒等，只需把 $F$ 推向 0；
- 残差网络可以看作许多不同深度路径的集成。

Transformer 中每个子层都用残差连接，并配合 LayerNorm。

</details>

### I8. 参数量计算 `基础` `押题`

输入 784 维，隐藏层 256 个单元，输出 10 类的全连接网络，有多少参数？若改为两个隐藏层（256、128）呢？

<details>
<summary>答案与解析</summary>

- 一个隐藏层：$784\times256+256+256\times10+10=200704+256+2560+10=203530$。
- 两个隐藏层：$784\times256+256+256\times128+128+128\times10+10=200960+32896+1290=235146$。

```python
def mlp_params(sizes):
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

print(mlp_params([784, 256, 10]), mlp_params([784, 256, 128, 10]))
```

 203530 和 235146。

**经验**：全连接层的参数主要集中在输入层（维度最高的那一层）。

</details>

---

## J. 优化器与归一化

> 对应讲解第 11 节。

### J1. 比较 SGD、Momentum 与 Adam `中等` `经典·DL 8.3`

在病态二次函数 $f(x,y)=\tfrac12(x^2+100y^2)$ 上，从 $(10,1)$ 出发，比较 SGD、Momentum、Adam 迭代 200 步后的函数值，并解释差异。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def grad(p):
    return np.array([p[0], 100 * p[1]])

def f(p):
    return 0.5 * (p[0] ** 2 + 100 * p[1] ** 2)

def sgd(p, lr=0.015, steps=200):
    for _ in range(steps):
        p = p - lr * grad(p)
    return p

def momentum(p, lr=0.015, beta=0.9, steps=200):
    v = np.zeros_like(p)
    for _ in range(steps):
        v = beta * v + grad(p)
        p = p - lr * v
    return p

def adam(p, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, steps=200):
    m = np.zeros_like(p); v = np.zeros_like(p)
    for t in range(1, steps + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        p = p - lr * mh / (np.sqrt(vh) + eps)
    return p

p0 = np.array([10.0, 1.0])
for name, opt in [("SGD", sgd), ("Momentum", momentum), ("Adam", adam)]:
    print(f"{name:>8}: f = {f(opt(p0.copy())):.3e}")
```

 200 步后：SGD 约 $1.2\times10^{-1}$，Momentum 约 $4.8\times10^{-8}$，Adam 约 $1.2\times10^{-2}$。病态问题上动量收敛最快；Adam 的逐坐标缩放消除了条件数的影响，但固定学习率下末段会在最优点附近振荡。

**解释**：
- 条件数为 100。SGD 的学习率受陡峭方向限制（$\eta<2/100$），在平缓方向 $x$ 上每步只缩小 $1-0.015=0.985$ 倍，200 步后 $x$ 仍约为 0.5。
- Momentum 在平缓方向上累积速度，等效步长放大约 $1/(1-\beta)=10$ 倍，同时在陡峭方向上的震荡相互抵消。
- Adam 按每个坐标的梯度大小归一化，两个方向的步长接近，相当于对角预条件；但学习率固定时会在最优点附近以约 $\eta$ 的幅度波动，需要配合学习率衰减。

</details>

### J2. Adam 的偏差校正 `中等` `经典·Kingma & Ba 2015`

$\beta_1=0.9$，$\beta_2=0.999$，$\eta=0.001$。第一步梯度 $g_1=0.5$。分别计算有、无偏差校正时第一步的更新量。

<details>
<summary>答案与解析</summary>

$m_1=0.1\times0.5=0.05$，$v_1=0.001\times0.25=0.00025$。

| | $m$ | $v$ | 更新量 $\eta\,m/\sqrt v$ |
|---|---|---|---|
| 无校正 | 0.05 | 0.00025 | $0.001\times0.05/0.01581=0.00316$ |
| 有校正 | $0.05/0.1=0.5$ | $0.00025/0.001=0.25$ | $0.001\times0.5/0.5=0.001$ |

```python
import numpy as np
b1, b2, lr, g = 0.9, 0.999, 1e-3, 0.5
m, v = (1 - b1) * g, (1 - b2) * g ** 2
print(lr * m / np.sqrt(v), lr * (m / (1 - b1)) / np.sqrt(v / (1 - b2)))
```

 无校正 0.00316，有校正 0.001。

**解释**：无校正时第一步更新量为 $\eta\frac{1-\beta_1}{\sqrt{1-\beta_2}}\mathrm{sign}(g)=3.16\eta$，与梯度大小无关但被放大约 3 倍；更一般地，早期 $m$ 和 $v$ 都偏向初值 0，且偏差程度不同。校正后第一步恰为 $\eta\,\mathrm{sign}(g_1)$，Adam 的步长大致由 $\eta$ 控制，这让学习率更容易设定。

</details>

### J3. Adam 中的 L2 正则与权重衰减 `困难` `经典·Loshchilov & Hutter 2019`

在 SGD 中，L2 正则与权重衰减等价；在 Adam 中为什么不等价？AdamW 做了什么修改？

<details>
<summary>答案与解析</summary>

**SGD**：损失加 $\frac\lambda2\lVert\theta\rVert^2$，更新 $\theta\leftarrow\theta-\eta(g+\lambda\theta)=(1-\eta\lambda)\theta-\eta g$，等价于每步把参数乘以 $(1-\eta\lambda)$ 再做梯度下降。

**Adam + L2**：把 $\lambda\theta$ 加进梯度后，它也进入 $m$ 和 $v$，被 $1/\sqrt{\hat v}$ 缩放：

$$\theta\leftarrow\theta-\eta\frac{\hat m(g+\lambda\theta)}{\sqrt{\hat v(g+\lambda\theta)}+\epsilon}$$

历史梯度大的参数，$\sqrt{\hat v}$ 大，正则效果被削弱；梯度小的参数反而衰减更强。正则强度因参数而异，与"惩罚所有权重"的初衷不符。

**AdamW**：梯度中不含正则项，衰减直接作用于参数：

$$\theta\leftarrow\theta-\eta\Big(\frac{\hat m}{\sqrt{\hat v}+\epsilon}+\lambda\theta\Big)$$

实验中 AdamW 的泛化更好，且最优学习率与最优衰减系数更解耦，便于调参。PyTorch 中 `torch.optim.Adam(weight_decay=...)` 是 L2 形式，`torch.optim.AdamW` 是解耦形式。

</details>

### J4. 实现 BatchNorm 的前向与反向 `困难` `经典·Ioffe & Szegedy 2015`

实现训练模式下 BatchNorm 的前向与反向传播（输入 $n\times d$），并做梯度检验。

<details>
<summary>答案与解析</summary>

前向：$\mu=\frac1n\sum x_i$，$\sigma^2=\frac1n\sum(x_i-\mu)^2$，$\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$，$y=\gamma\hat x+\beta$。

反向（记 $\delta=\partial L/\partial y$）：$\partial L/\partial\gamma=\sum_i\delta_i\hat x_i$，$\partial L/\partial\beta=\sum_i\delta_i$，以及

$$\frac{\partial L}{\partial x}=\frac{\gamma}{n\sqrt{\sigma^2+\epsilon}}\Big(n\,\delta-\sum_i\delta_i-\hat x\sum_i\delta_i\hat x_i\Big)$$

推导要点：$\mu$ 和 $\sigma^2$ 都依赖每个 $x_i$，梯度有三条路径（直接、经过 $\mu$、经过 $\sigma^2$），合并后得到上式。

```python
import numpy as np

def bn_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(0)
    var = x.var(0)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, var, gamma, eps)

def bn_backward(dy, cache):
    xhat, var, gamma, eps = cache
    n = dy.shape[0]
    dgamma = (dy * xhat).sum(0)
    dbeta = dy.sum(0)
    dx = gamma / (n * np.sqrt(var + eps)) * (n * dy - dy.sum(0) - xhat * (dy * xhat).sum(0))
    return dx, dgamma, dbeta

rng = np.random.default_rng(0)
x = rng.normal(3, 2, (16, 4))
gamma, beta = rng.standard_normal(4), rng.standard_normal(4)
dy = rng.standard_normal((16, 4))                     # 上游梯度，定义 L = sum(y * dy)

y, cache = bn_forward(x, gamma, beta)
dx, dgamma, dbeta = bn_backward(dy, cache)

def L(x_):
    return (bn_forward(x_, gamma, beta)[0] * dy).sum()

num = np.zeros_like(x)
for i in np.ndindex(x.shape):
    xp, xm = x.copy(), x.copy()
    xp[i] += 1e-6; xm[i] -= 1e-6
    num[i] = (L(xp) - L(xm)) / 2e-6
print("dx 最大误差:", np.abs(num - dx).max())
print("输出每列均值、标准差:", np.round(((y - beta) / gamma).mean(0), 6), np.round(((y - beta) / gamma).std(0), 4))
```

 $dx$ 的数值梯度与解析梯度最大误差在 $10^{-9}$ 量级；$\hat x$ 每列均值为 0、标准差约为 1。

**性质**：$\sum_i\partial L/\partial x_i=0$。BN 的输出对输入整体平移不变，所以梯度在 batch 维上的和为 0。

</details>

### J5. BatchNorm 与 LayerNorm `中等` `经典·Ba et al. 2016`

(1) BN 在推理时如何计算？batch 很小时为什么效果差？(2) LN 沿什么维度归一化？为什么 Transformer 用 LN？用代码说明两者的统计维度。

<details>
<summary>答案与解析</summary>

(1) 推理时一个样本没有 batch 统计量，所以用训练时累积的滑动平均 $\mu_{run}\leftarrow(1-m)\mu_{run}+m\mu_B$（方差同理）。batch 小时 $\mu_B,\sigma_B^2$ 噪声大，训练与推理用的统计量不一致，效果下降。

(2) LN 对每个样本在特征维上归一化，与 batch 无关，训练和推理完全相同；序列长度可变时也不受影响，因此适合 RNN 和 Transformer。

```python
import numpy as np

def batch_norm(x, eps=1e-5):
    return (x - x.mean(0)) / np.sqrt(x.var(0) + eps)          # 每个特征跨样本

def layer_norm(x, eps=1e-5):
    return (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + eps)   # 每个样本跨特征

rng = np.random.default_rng(0)
x = rng.normal(5, 3, (8, 6))
bn, ln = batch_norm(x), layer_norm(x)
print("BN 每列均值:", np.round(bn.mean(0), 6))
print("LN 每行均值:", np.round(ln.mean(1), 6))
print("单个样本做 LN 与整个 batch 中的结果相同:", np.allclose(layer_norm(x[:1]), ln[:1]))
print("单个样本做 BN 与整个 batch 中的结果相同:", np.allclose(batch_norm(x[:2])[:1], bn[:1]))
```

 BN 每列均值为 0，LN 每行均值为 0；LN 的结果与 batch 中其他样本无关，BN 不是。

> **量化用途**：股票截面数据中，把同一天的股票放在一个 batch 里，BN 就相当于截面标准化；用于时间序列模型时，对每个样本在特征维上做 LN 更稳妥。

</details>

### J6. Inverted Dropout `基础` `经典·Srivastava et al. 2014`

实现 inverted dropout，验证训练时输出的期望与推理时相同。为什么不在推理时乘以 $(1-p)$？

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def dropout(x, p, training, rng):
    if not training or p == 0:
        return x
    mask = rng.random(x.shape) >= p
    return x * mask / (1 - p)                  # 训练时放大，推理时原样输出

rng = np.random.default_rng(0)
x = rng.random((1000, 100)) + 1.0
outs = np.mean([dropout(x, 0.3, True, rng) for _ in range(200)], axis=0)
print(round(x.mean(), 4), round(outs.mean(), 4))
print(np.array_equal(dropout(x, 0.3, False, rng), x))
```

 输入均值 1.4996，训练模式下 200 次平均输出的均值 1.4993，二者一致；推理模式直接返回输入（`True`）。

**原因**：原始 dropout 在推理时把权重乘以 $(1-p)$；inverted dropout 把缩放放在训练阶段，推理代码不需要知道 $p$，部署更简单，推理也更快。两者在期望意义上等价。

</details>

### J7. 梯度裁剪 `基础` `经典·DL 10.11`

实现按全局范数的梯度裁剪：所有参数梯度拼成一个向量，范数超过阈值时等比缩小。与逐元素截断有什么区别？

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def clip_by_global_norm(grads, max_norm):
    total = np.sqrt(sum((g ** 2).sum() for g in grads))
    scale = min(1.0, max_norm / (total + 1e-12))
    return [g * scale for g in grads], total

rng = np.random.default_rng(0)
grads = [rng.standard_normal((10, 10)) * 5, rng.standard_normal(10) * 5]
clipped, before = clip_by_global_norm(grads, max_norm=1.0)
after = np.sqrt(sum((g ** 2).sum() for g in clipped))
cos = sum((a * b).sum() for a, b in zip(grads, clipped)) / (before * after)
print(round(before, 2), round(after, 4), round(cos, 6))
```

 裁剪后范数为 1.0，与原梯度的余弦相似度为 1（方向不变）。

**区别**：全局范数裁剪保持梯度方向，只缩小步长；逐元素截断（`clip(g, -c, c)`）会改变方向。RNN 训练中梯度偶尔爆炸，全局范数裁剪是标准做法。

</details>

---

## K. RNN、LSTM、Transformer

> 对应讲解第 12 节。

### K1. RNN 的梯度消失与爆炸 `中等` `经典·DL 10.7`

线性 RNN $h_t=Wh_{t-1}$ 中，$\partial h_T/\partial h_0=W^T$。分别取 $W$ 的谱半径为 0.9 和 1.1，计算 $T=50$ 时 $\lVert W^T\rVert$。

<details>
<summary>答案与解析</summary>

$W=Q\Lambda Q^{-1}$ 时 $W^T=Q\Lambda^TQ^{-1}$，最大特征值的模 $\rho<1$ 时 $\lVert W^T\rVert\sim\rho^T\to0$，$\rho>1$ 时指数爆炸。

```python
import numpy as np

rng = np.random.default_rng(0)
A = rng.standard_normal((32, 32))
A /= np.max(np.abs(np.linalg.eigvals(A)))          # 谱半径归一化为 1
for rho in (0.9, 1.0, 1.1):
    W = rho * A
    print(f"rho={rho}: ||W^50|| = {np.linalg.norm(np.linalg.matrix_power(W, 50), 2):.3e}")
```

 $\rho=0.9$ 时 $8.1\times10^{-3}$（$0.9^{50}=5.2\times10^{-3}$，乘以与 $Q$ 有关的常数），$\rho=1.1$ 时 $184.5$（$1.1^{50}=117$）。

加上 tanh 后，雅可比为 $\mathrm{diag}(1-h_t^2)W$，导数因子 $\le1$ 使消失更严重。对策：梯度裁剪（防爆炸）、LSTM/GRU（防消失）、正交初始化、截断 BPTT。

</details>

### K2. 实现 LSTM 单元 `中等` `经典·Hochreiter & Schmidhuber 1997`

用 NumPy 实现 LSTM 单元的前向计算，在长度为 30 的序列上运行，并验证参数量公式 $4[(d_h+d_x)d_h+d_h]$。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LSTMCell:
    def __init__(self, d_x, d_h, rng):
        self.d_h = d_h
        self.W = rng.standard_normal((d_x + d_h, 4 * d_h)) / np.sqrt(d_x + d_h)
        self.b = np.zeros(4 * d_h)
        self.b[:d_h] = 1.0                       # 遗忘门偏置初始化为 1

    def step(self, x, h, c):
        z = np.concatenate([x, h], axis=-1) @ self.W + self.b
        d = self.d_h
        f, i, o = sigmoid(z[..., :d]), sigmoid(z[..., d:2 * d]), sigmoid(z[..., 2 * d:3 * d])
        g = np.tanh(z[..., 3 * d:])
        c = f * c + i * g                         # 加法更新细胞状态
        h = o * np.tanh(c)
        return h, c

    def n_params(self):
        return self.W.size + self.b.size

rng = np.random.default_rng(0)
d_x, d_h, T, batch = 8, 16, 30, 4
cell = LSTMCell(d_x, d_h, rng)
xs = rng.standard_normal((T, batch, d_x))
h = c = np.zeros((batch, d_h))
for t in range(T):
    h, c = cell.step(xs[t], h, c)
print(h.shape, cell.n_params(), 4 * ((d_h + d_x) * d_h + d_h))
print("h 的取值范围在 (-1, 1) 内:", bool(np.all(np.abs(h) < 1)))
```

 输出形状 `(4, 16)`；参数量 1600，与公式 $4\times[(16+8)\times16+16]=1600$ 一致。

**实现细节**：四个门的权重拼成一个 $(d_x+d_h)\times4d_h$ 矩阵，一次矩阵乘法算完，这也是框架中的标准做法。

</details>

### K3. 实现缩放点积注意力与因果掩码 `中等` `经典·Vaswani et al. 2017` `经典·AQI Q53`

实现 $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$，支持因果掩码。验证：每行注意力权重之和为 1；加因果掩码后，修改未来位置的输入不影响过去位置的输出。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, causal=False):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)          # (..., n, n)
    if causal:
        n = scores.shape[-1]
        scores = np.where(np.tril(np.ones((n, n), bool)), scores, -np.inf)
    w = softmax(scores)
    return w @ V, w

def self_attention(X, Wq, Wk, Wv, causal=False):
    return attention(X @ Wq, X @ Wk, X @ Wv, causal)

rng = np.random.default_rng(0)
n, d = 10, 16
Wq, Wk, Wv = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3))
X = rng.standard_normal((n, d))

out, w = self_attention(X, Wq, Wk, Wv, causal=True)
X2 = X.copy()
X2[6:] += 10 * rng.standard_normal((n - 6, d))              # 修改第 6 个位置之后的输入
out2, _ = self_attention(X2, Wq, Wk, Wv, causal=True)
out3, _ = self_attention(X2, Wq, Wk, Wv, causal=False)
print("每行权重之和为 1:", np.allclose(w.sum(-1), 1))
print("权重矩阵是下三角:", np.allclose(np.triu(w, 1), 0))
print("因果掩码下前 6 个位置输出不变:", np.allclose(out[:6], out2[:6]))
print("无掩码时前 6 个位置输出改变:", not np.allclose(out[:6], out3[:6]))
```

 四项检查全部为 `True`。

> **量化用途**：在单只股票的历史序列上做注意力，必须加因果掩码，否则位置 $t$ 的表示用到了 $t$ 之后的价格，这是一种隐蔽的前视偏差。在同一天的股票截面上做注意力则不需要掩码。

</details>

### K4. 为什么除以 $\sqrt{d_k}$ `基础` `经典·Vaswani et al. 2017`

$q,k\in\mathbb R^{d_k}$ 各分量独立、均值 0、方差 1。求 $q^\top k$ 的方差，并用模拟说明不缩放时 softmax 的饱和程度。

<details>
<summary>答案与解析</summary>

$\mathrm{Var}(q^\top k)=\sum_{i=1}^{d_k}\mathrm{Var}(q_ik_i)=\sum_i\mathbb E[q_i^2]\mathbb E[k_i^2]=d_k$。除以 $\sqrt{d_k}$ 后方差为 1，与维度无关。

```python
import numpy as np

rng = np.random.default_rng(0)
for d_k in (16, 64, 256, 1024):
    q = rng.standard_normal((2000, d_k))
    k = rng.standard_normal((2000, 32, d_k))               # 每个查询对应 32 个键
    s = np.einsum("nd,nmd->nm", q, k)
    def max_weight(scores):
        e = np.exp(scores - scores.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).max(1).mean()
    print(f"d_k={d_k:>4}: Var(q·k)={s.var():7.1f}  最大权重 不缩放 {max_weight(s):.3f}  缩放 {max_weight(s / np.sqrt(d_k)):.3f}")
```

 点积方差约等于 $d_k$；不缩放时，$d_k=1024$ 的平均最大权重约 0.95（接近 one-hot），缩放后各维度下都约 0.16–0.17。

**影响**：softmax 接近 one-hot 时，其雅可比 $\mathrm{diag}(p)-pp^\top$ 接近 0，梯度消失，训练变慢。

</details>

### K5. 多头注意力的参数量与复杂度 `基础` `经典·Vaswani et al. 2017`

$d_{model}=512$，$h=8$ 个头，前馈网络中间维度 2048。求一个 Transformer 编码器块的参数量（含偏置，不含 LayerNorm），以及序列长度 $n$ 时自注意力的时间和空间复杂度。

<details>
<summary>答案与解析</summary>

| 部分 | 参数量 |
|---|---|
| $W_Q,W_K,W_V,W_O$ | $4\times(512\times512+512)=1050624$ |
| 前馈网络 | $512\times2048+2048+2048\times512+512=2099712$ |
| 合计 | 3150336 |

LayerNorm 每层还有 $2\times512$ 个参数，两层共 2048。

```python
d, ff = 512, 2048
attn = 4 * (d * d + d)
ffn = d * ff + ff + ff * d + d
print(attn, ffn, attn + ffn)
```

 1050624、2099712、3150336。

**多头不增加参数**：每个头的维度为 $d_k=d_{model}/h=64$，8 个头的 $W_Q$ 拼起来仍是 $512\times512$。

**复杂度**：计算 $QK^\top$ 需要 $O(n^2d)$ 时间、$O(n^2)$ 内存存放注意力矩阵；投影和前馈网络为 $O(nd^2)$。$n$ 大于 $d$ 时注意力部分占主导，这是长序列的瓶颈（稀疏注意力、线性注意力、FlashAttention 等针对这一点）。

</details>

### K6. 自注意力的置换等变性与位置编码 `中等` `押题`

证明不加位置编码时，自注意力对输入序列的置换是等变的，即 $\mathrm{SA}(PX)=P\,\mathrm{SA}(X)$。这对时间序列和股票截面分别意味着什么？

<details>
<summary>答案与解析</summary>

$Q'=PXW_Q=PQ$，$K'=PK$，$V'=PV$。

$$\mathrm{softmax}\Big(\frac{PQK^\top P^\top}{\sqrt{d_k}}\Big)PV=P\,\mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)P^\top PV=P\,\mathrm{SA}(X)$$

用到了按行 softmax 与行列同时置换可交换，以及 $P^\top P=I$。

```python
perm = rng.permutation(n)
out_a, _ = self_attention(X[perm], Wq, Wk, Wv)
out_b, _ = self_attention(X, Wq, Wk, Wv)
print(np.allclose(out_a, out_b[perm]))
```

 输出 `True`（沿用 K3 中定义的函数和变量）。

**含义**：
- **时间序列**：不加位置编码，模型无法区分"先涨后跌"和"先跌后涨"，必须加入位置信息（正弦编码、可学习编码、相对位置编码）。
- **股票截面**：股票的排列顺序本来就没有意义，置换等变正是需要的性质；加了位置编码反而会让模型依赖股票的排列顺序。

</details>

### K7. RNN、LSTM、Transformer 在量化中的取舍 `中等` `经典·AQI Q46` `经典·AQI Q53`

面试官："你会用 LSTM 还是 Transformer 预测股票收益？为什么不直接用 GBDT？"

<details>
<summary>答案与解析</summary>

| 模型 | 优点 | 缺点 | 适用 |
|---|---|---|---|
| GBDT | 训练快、稳健、对特征尺度不敏感、可解释工具成熟 | 需要手工构造时序特征；不直接建模序列 | 日频因子合成的基线和主力 |
| MLP | 简单，可端到端 | 需要标准化、正则、集成 | 与 GBDT 互补，常做集成 |
| LSTM / GRU | 自然建模序列，参数少 | 串行训练慢；长依赖仍有限 | 分钟级或日级量价序列 |
| Transformer | 长依赖、并行；可做截面注意力 | 数据需求大、容易过拟合、$O(n^2)$ | 大数据量的高频序列；截面关系建模 |
| GNN | 利用股票间的关系（行业、供应链、持仓） | 图的构造有主观性，图也会随时间变化 | 关系信息丰富时 |

**回答思路**：
1. 先建 GBDT 基线（手工因子 + LightGBM），这是比较的标准；
2. 神经网络的价值在于从原始量价序列中自动提取特征，或建模股票间关系；
3. 金融数据信噪比低、有效样本少，复杂模型的优势不一定能体现，需要强正则、早停、多种子集成；
4. 最终看与基线的增量：样本外 IC 的提升、与基线预测的相关性（相关性低时集成收益大）、扣费后收益。

</details>

---

## L. 机器学习在量化中的应用与陷阱

> 对应讲解第 13 节。L4 为完整的因子合成流程。

### L1. 金融机器学习为什么难 `基础` `经典·AFML 1.2`

面试官："为什么在图像识别上效果很好的模型，用来预测股票收益效果很差？"

<details>
<summary>答案与解析</summary>

| 方面 | 图像识别 | 股票收益预测 |
|---|---|---|
| 信噪比 | 标签几乎无噪声 | 可预测部分占收益方差不到 1% |
| 平稳性 | 猫的样子不会变 | 市场规律随风格、监管、参与者变化 |
| 对抗性 | 图片不会因为被识别而改变 | 信号被交易后会被抹平（alpha 衰减） |
| 样本 | 可以无限采集、可做数据增强 | 历史只有一条路径，日频 10 年约 2500 天 |
| 独立性 | 样本近似独立 | 截面相关、时间相关、标签重叠 |
| 评价 | 准确率直接可用 | 需要考虑成本、容量、风险暴露 |

因此金融 ML 的重点不是更复杂的模型，而是：正确的验证（防泄漏）、合理的标签与特征、强正则和集成、对过拟合的严格控制。

</details>

### L2. IC 0.05 意味着什么 `中等` `经典·Grinold & Kahn`

(1) 某因子日频截面 IC 为 0.05，单变量线性预测的 $R^2$ 是多少？(2) 用模拟说明：3000 只股票的截面上，IC 为 0.05 的因子做多前 10%、做空后 10% 的日收益期望。(3) 用主动管理基本定律说明为什么量化追求很多弱信号。

<details>
<summary>答案与解析</summary>

(1) $R^2=\mathrm{IC}^2=0.25\%$。

(2) 设截面收益 $r=\mathrm{IC}\cdot\sigma\cdot z+\sqrt{1-\mathrm{IC}^2}\,\sigma\,\varepsilon$，$z$ 为标准化因子值，截面波动 $\sigma=2\%$。

```python
import numpy as np

rng = np.random.default_rng(0)
N, T, ic, sigma = 3000, 500, 0.05, 0.02
ls, ics = [], []
for _ in range(T):
    z = rng.standard_normal(N)
    r = ic * sigma * z + np.sqrt(1 - ic ** 2) * sigma * rng.standard_normal(N)
    order = np.argsort(z)
    k = N // 10
    ls.append(r[order[-k:]].mean() - r[order[:k]].mean())
    ics.append(np.corrcoef(z, r)[0, 1])
ls = np.array(ls)
print(f"实测 IC 均值 {np.mean(ics):.4f}, R^2 = {np.mean(ics) ** 2:.5f}")
print(f"多空日收益均值 {ls.mean() * 1e4:.1f} bp, 年化 Sharpe {ls.mean() / ls.std() * np.sqrt(250):.1f}")
```

 实测 IC 0.0489，多空日收益 33.8 bp（理论值 $\mathrm{IC}\cdot\sigma\cdot(\mathbb E[z\mid\text{前}10\%]-\mathbb E[z\mid\text{后}10\%])=0.05\times2\%\times2\times1.755\approx35.1$ bp，其中 $\mathbb E[z\mid z>z_{0.9}]=\varphi(1.2816)/0.1=1.755$），年化 Sharpe 约 33。这一 Sharpe 远高于现实，因为模拟假设 IC 每天恒定、没有成本；真实因子的 IC 本身逐日波动，ICIR 才是 Sharpe 的主要决定因素。

**注意**：这里假设 IC 每天恒为 0.05，且截面噪声独立。真实市场中 IC 本身每天波动很大（标准差常在 0.1 以上），多空收益还受风格暴露影响，实际 Sharpe 远低于此，扣除双边成本后更低。这个例子说明的是 $R^2$ 很低的信号在截面上仍有可观的经济价值。

(3) **基本定律** $\mathrm{IR}\approx\mathrm{IC}\times\sqrt{\mathrm{Breadth}}$，Breadth 为每年独立下注的次数。IC 很难提高，但可以通过增加股票数量、提高调仓频率、组合多个低相关信号来增加 Breadth。这就是量化"用大量弱信号在大量股票上分散下注"的理论依据。

</details>

### L3. 最大化 ICIR 的因子权重 `困难` `经典·Grinold & Kahn` `经典·AQI Q33`

$K$ 个因子的 IC 序列均值为 $\mu$（$K$ 维）、协方差为 $\Sigma$。合成因子的 IC 近似为 $w^\top\mathrm{IC}_t$。求使 $\mathrm{ICIR}=\frac{w^\top\mu}{\sqrt{w^\top\Sigma w}}$ 最大的 $w$，并用模拟验证它优于等权和随机权重。

<details>
<summary>答案与解析</summary>

ICIR 对 $w$ 的缩放不变，固定 $w^\top\Sigma w=1$ 最大化 $w^\top\mu$，拉格朗日条件 $\mu=2\lambda\Sigma w$，得

$$w^*\propto\Sigma^{-1}\mu,\qquad\mathrm{ICIR}^*=\sqrt{\mu^\top\Sigma^{-1}\mu}$$

这与最大 Sharpe 组合 $w\propto\Sigma^{-1}\mu$ 形式完全相同。由柯西-施瓦茨不等式，$(w^\top\mu)^2=\big((\Sigma^{1/2}w)^\top(\Sigma^{-1/2}\mu)\big)^2\le(w^\top\Sigma w)(\mu^\top\Sigma^{-1}\mu)$，任何权重的 ICIR 都不超过 $\sqrt{\mu^\top\Sigma^{-1}\mu}$。

```python
import numpy as np

rng = np.random.default_rng(0)
T = 500
mu_true = np.array([0.03, 0.025, 0.02, 0.01])
C = np.array([[1, .7, .2, 0], [.7, 1, .1, 0], [.2, .1, 1, -.3], [0, 0, -.3, 1]])
sd = np.array([0.08, 0.07, 0.06, 0.05])
ic = rng.multivariate_normal(mu_true, C * np.outer(sd, sd), T)   # 模拟的 IC 序列

mu, S = ic.mean(0), np.cov(ic, rowvar=False)
def icir(w):
    s = ic @ w
    return s.mean() / s.std(ddof=1)

w_opt = np.linalg.solve(S, mu)
w_opt /= np.abs(w_opt).sum()
w_eq = np.ones(4) / 4
best_random = max(icir(rng.standard_normal(4)) for _ in range(5000))
print("最优权重:", np.round(w_opt, 3))
print(f"ICIR 最优 {icir(w_opt):.3f} | 理论上界 {np.sqrt(mu @ np.linalg.solve(S, mu)):.3f} | "
      f"等权 {icir(w_eq):.3f} | 5000 组随机权重中最好 {best_random:.3f}")
```

 样本内最优权重的 ICIR 为 0.562，等于理论上界；等权 0.520，5000 组随机权重中最好 0.558。

**实践中的问题**：
- $\Sigma$ 由有限样本估计，$\Sigma^{-1}$ 会放大估计误差，可能给出极端权重甚至负权重。常用 Ledoit-Wolf 收缩、只用对角线（退化为 $w_k\propto\mu_k/\sigma_k^2$）或限制权重非负。
- 这是样本内最优；样本外应在滚动窗口中估计 $\mu$ 和 $\Sigma$，再应用到下一期。
- 合成因子的 IC 等于 $w^\top\mathrm{IC}_t$ 只是近似（严格成立需要因子截面标准化且合成后重新标准化时的方差不变），但这一近似在实践中足够好。

</details>

### L4. LightGBM 因子合成全流程 `困难` `押题`

在模拟的 400 天 × 300 只股票面板上（10 个因子，收益与因子有线性和非线性关系），实现：滚动训练（训练窗口 200 天，每 40 天重训，训练集与预测期之间留 1 天间隔）、样本外预测、计算每日 RankIC，并与 Ridge 基线比较。分别考虑"线性成分为主"和"非线性成分为主"两种情形。

<details>
<summary>答案与解析</summary>

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

T, N, K = 400, 300, 10

def simulate(coef, scale, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, N, K))
    signal = (coef[0] * F[..., 0] + coef[1] * np.tanh(2 * F[..., 1])
              + coef[2] * F[..., 2] * F[..., 3] - coef[3] * (np.abs(F[..., 4]) > 1.5))
    ret = scale * signal + rng.standard_normal((T, N))      # 次日收益，信噪比很低
    label = (pd.DataFrame(ret).rank(axis=1, pct=True) - 0.5).to_numpy()   # 截面 rank 标签
    return F, ret, label

def rolling_predict(F, label, make_model, train_len=200, step=40, gap=1):
    preds = np.full((T, N), np.nan)
    for start in range(train_len + gap, T, step):
        tr = slice(start - gap - train_len, start - gap)   # 标签在 start - gap 之前全部已知
        model = make_model().fit(F[tr].reshape(-1, K), label[tr].reshape(-1))
        te = slice(start, min(start + step, T))
        preds[te] = model.predict(F[te].reshape(-1, K)).reshape(-1, N)
    return preds

def daily_rank_ic(preds, ret):
    return np.array([spearmanr(preds[t], ret[t])[0] for t in range(T) if not np.isnan(preds[t]).any()])

make_lgb = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7,
                                     min_child_samples=1000, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1, random_state=0)
make_ridge = lambda: Ridge(alpha=10.0)

scenarios = {"线性为主": ((0.4, 0.3, 0.3, 0.2), 0.1), "非线性为主": ((0.2, 0.3, 0.5, 0.3), 0.15)}
for sname, (coef, scale) in scenarios.items():
    F, ret, label = simulate(coef, scale)
    for mname, mk in [("LightGBM", make_lgb), ("Ridge", make_ridge)]:
        ic = daily_rank_ic(rolling_predict(F, label, mk), ret)
        print(f"{sname} {mname:>8}: 样本外 {len(ic)} 天, RankIC {ic.mean():.4f}, "
              f"ICIR {ic.mean() / ic.std():.2f}, t 值 {ic.mean() / ic.std() * np.sqrt(len(ic)):.1f}")
```

输出：

```
线性为主 LightGBM: 样本外 199 天, RankIC 0.0316, ICIR 0.57, t 值 8.0
线性为主    Ridge: 样本外 199 天, RankIC 0.0417, ICIR 0.70, t 值 9.9
非线性为主 LightGBM: 样本外 199 天, RankIC 0.0603, ICIR 1.15, t 值 16.3
非线性为主    Ridge: 样本外 199 天, RankIC 0.0428, ICIR 0.73, t 值 10.2
```

 线性成分为主时，Ridge 好于 LightGBM（0.042 对 0.032）；交互项 $F_2F_3$ 和阈值效应 $\lvert F_4\rvert>1.5$ 占比较大时，LightGBM 的 RankIC 明显更高（0.060 对 0.043）。两种情形下 t 值都显著。

**结论与流程要点**：
- **模型选择取决于数据**：信噪比极低、非线性成分弱时，GBDT 拟合的非线性多半是噪声，不一定胜过线性模型。所以必须保留线性基线，比较样本外增量。
- **正则很关键**：这里 `min_child_samples=1000`、`num_leaves=7`。线性为主情形下，把叶子样本数降到 200、叶子数升到 15，LightGBM 的 RankIC 从 0.032 降到 0.025；降到 20、31 时为 0.021。正则越弱，越多噪声被当作非线性拟合进去。
- **标签**：截面 rank 去掉了收益的整体水平和极端值，模型专注于排序。
- **间隔**：标签是次日收益，训练集最后一天的标签在预测期开始前才能确认，所以留 1 天；标签为未来 $h$ 日收益时间隔至少为 $h$。
- **评估**：只用样本外预测计算 RankIC，并报告 t 值；真实研究中还要做分层回测、换手、成本和风格暴露分析。

</details>

### L5. 标签设计与三重障碍法 `中等` `经典·AFML 3.4`

比较几种常见标签：原始未来收益、截面排名、行业市值中性化后的超额收益、三重障碍法。实现三重障碍法：止盈 $+u$、止损 $-l$、最长持有 $h$ 期，先触及哪个就以哪个为标签（时间到期时按收益符号）。

<details>
<summary>答案与解析</summary>

| 标签 | 优点 | 缺点 |
|---|---|---|
| 原始未来 $h$ 日收益 | 直观 | 受市场整体涨跌主导，截面模型学到的是择时噪声；极端值多 |
| 截面排名 / 标准化 | 去掉市场水平和极端值，直接对应选股目标 | 丢失收益幅度信息 |
| 中性化超额收益 | 去掉风格暴露，模型学的是纯 alpha | 依赖风险模型的质量 |
| 三重障碍法 | 考虑路径和止盈止损，贴近交易规则 | 参数多；标签区间长度不一，需要 purge 和样本权重 |

```python
import numpy as np

def triple_barrier(prices, up, down, h):
    """返回每个起点的 (标签, 触及时刻)；标签 1 止盈，-1 止损，时间到期时取收益符号"""
    n = len(prices)
    labels, touch = np.zeros(n - 1, int), np.zeros(n - 1, int)
    for t in range(n - 1):
        end = min(t + h, n - 1)
        path = prices[t + 1:end + 1] / prices[t] - 1
        hit_up = np.flatnonzero(path >= up)
        hit_dn = np.flatnonzero(path <= -down)
        first_up = hit_up[0] if len(hit_up) else np.inf
        first_dn = hit_dn[0] if len(hit_dn) else np.inf
        if first_up == first_dn == np.inf:
            labels[t], touch[t] = np.sign(path[-1]), end
        elif first_up < first_dn:
            labels[t], touch[t] = 1, t + 1 + int(first_up)
        else:
            labels[t], touch[t] = -1, t + 1 + int(first_dn)
    return labels, touch

p = np.array([100, 101, 103, 99, 96, 97, 100, 104, 103], float)
labels, touch = triple_barrier(p, up=0.03, down=0.03, h=4)
print(labels)
print(touch)
```

输出：

```
[ 1 -1 -1 -1  1  1  1 -1]
[2 4 3 4 6 6 7 8]
```

 与逐点手算一致。例如：起点 0（100）在第 2 期到 103，触及 $+3\%$；起点 1（101）第 2 期 $+1.98\%$、第 3 期 $-1.98\%$，第 4 期到 96（$-4.95\%$）触及止损；起点 7（104）只剩一期，到期收益 $103/104-1<0$，标签为 $-1$。

**要点**：
- 触及时刻 `touch` 就是标签区间的终点 `t1`，可直接用于 B4 的 Purged K-Fold；
- AFML 建议止盈止损宽度按近期波动率设定（如 $u=l=2\sigma_t$），而不是固定百分比；
- 标签区间相互重叠，AFML 第 4 章用"平均唯一性"给样本加权，降低重叠样本的影响。

</details>

### L6. 多种子集成 `中等` `押题` `经典·AQI Q54`

同一组 LightGBM 参数，只改 `random_state`（行采样、列采样随之变化），训练 10 个模型并对预测取平均。用模拟数据比较：单个模型测试集相关系数的均值与标准差、10 个模型平均后的相关系数。解释结果。

<details>
<summary>答案与解析</summary>

```python
import lightgbm as lgb
import numpy as np

rng = np.random.default_rng(0)
n, k = 20000, 20
X = rng.standard_normal((n, k))
signal = 0.3 * X[:, 0] + 0.2 * np.tanh(2 * X[:, 1]) + 0.2 * X[:, 2] * X[:, 3]
y = 0.3 * signal + rng.standard_normal(n)            # 低信噪比
tr, te = slice(0, 15000), slice(15000, n)

params = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
preds = np.array([lgb.LGBMRegressor(**params, random_state=s).fit(X[tr], y[tr]).predict(X[te])
                  for s in range(10)])
single = np.array([np.corrcoef(p, y[te])[0, 1] for p in preds])
ens = np.corrcoef(preds.mean(axis=0), y[te])[0, 1]
pair = np.corrcoef(preds)[np.triu_indices(10, 1)].mean()
print(f"单模型相关系数 均值 {single.mean():.4f}, 标准差 {single.std():.4f}, 最好 {single.max():.4f}")
print(f"10 个模型平均  {ens:.4f}；模型之间预测的平均相关系数 {pair:.3f}")
print(f"真实信号与 y 的相关系数（上限） {np.corrcoef(signal[te], y[te])[0, 1]:.4f}")
```

输出：

```
单模型相关系数 均值 0.0555, 标准差 0.0050, 最好 0.0647
10 个模型平均  0.0585；模型之间预测的平均相关系数 0.890
真实信号与 y 的相关系数（上限） 0.0863
```

 集成后的相关系数 0.0585 高于单模型均值 0.0555，约提高半个标准差。单模型最好的 0.0647 是在测试集上事后挑出的，事前无法知道哪个种子最好；种子之间预测相关系数 0.89，所以集成的收益有限。

**解释**：每个模型的预测 = 共同的信号部分 + 随种子变化的拟合噪声。设单个模型预测方差中噪声占比较高、模型之间噪声相关系数为 $\rho$，则 $M$ 个模型平均后噪声方差变为 $\rho\sigma^2+(1-\rho)\sigma^2/M$（与随机森林方差公式相同，见 D4），信号部分不变，所以相关系数上升。

**要点**：
- 收益只来自降低方差，不会降低偏差；种子之间相关性越低，收益越大。
- 比"挑出最好的种子"更可靠：最好种子的样本外表现多半是运气（与 L7 同理）。
- 实践中常同时集成种子、训练窗口长度、标签周期和模型类型（GBDT + 线性 + NN），后几种多样性更大。

</details>

### L7. 多重检验：N 个无效策略中最大的 Sharpe `困难` `经典·Bailey & López de Prado 2014`

研究员回测了 $N=100$ 个**真实 Sharpe 均为 0** 的策略，每个策略有 5 年日频数据，挑出年化 Sharpe 最高的一个。(1) 这个最高 Sharpe 的期望约为多少？(2) 用模拟验证。(3) 如何修正？

<details>
<summary>答案与解析</summary>

**(1) 推导**：真实 Sharpe 为 0 时，年化 Sharpe 估计量近似服从 $N(0,1/T)$，$T$ 为年数（日频 Sharpe 估计的标准误约 $1/\sqrt{n_{\text{days}}}$，年化乘以 $\sqrt{250}$，得 $\sqrt{250/n_{\text{days}}}=1/\sqrt{T}$）。$N$ 个独立标准正态的最大值期望约为

$$
\mathbb E\Big[\max_{i\le N} Z_i\Big]\approx(1-\gamma)\,\Phi^{-1}\Big(1-\frac1N\Big)+\gamma\,\Phi^{-1}\Big(1-\frac1{Ne}\Big)\le\sqrt{2\ln N},
$$

其中 $\gamma\approx0.5772$ 是欧拉常数（Bailey & López de Prado 2014 的 False Strategy Theorem）。$N=100$ 时约为 2.5，所以最高年化 Sharpe 的期望约 $2.5/\sqrt5\approx1.1$。

**(2) 模拟**：

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
N, years, n_sim = 100, 5, 2000
days = 250 * years
best = []
for _ in range(n_sim):
    r = rng.standard_normal((N, days)) * 0.01            # 日收益，真实均值为 0
    sr = r.mean(axis=1) / r.std(axis=1) * np.sqrt(250)
    best.append(sr.max())
g = 0.5772156649
emax = (1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e))
print(f"模拟 最大年化 Sharpe 均值 {np.mean(best):.3f}")
print(f"理论 {emax / np.sqrt(years):.3f}，上界 sqrt(2lnN)/sqrt(T) = {np.sqrt(2 * np.log(N) / years):.3f}")
```

输出：

```
模拟 最大年化 Sharpe 均值 1.130
理论 1.132，上界 sqrt(2lnN)/sqrt(T) = 1.357
```

 模拟值 1.130 与理论值 1.132 一致；$\sqrt{2\ln N}$ 是偏松的上界。也就是说，100 个毫无预测力的策略中，最好的一个 5 年回测年化 Sharpe 约 1.1，看起来已经"可以上线"。

**(3) 修正**：
- **Deflated Sharpe Ratio**：把"最大 Sharpe 的期望"作为零假设下的基准，再考虑偏度、峰度和样本长度，计算真实 Sharpe 大于 0 的概率（Bailey & López de Prado 2014）。
- 记录**全部**尝试过的配置数量 $N$，而不是只报告最好的一个；策略之间相关时，用有效独立次数代替 $N$。
- Bonferroni、Holm、BH（FDR）等多重检验校正；Harvey, Liu & Zhu (2016) 建议新因子的 t 值门槛提高到 3 左右。
- 保留从未用过的样本外区间，只在最后用一次。

</details>

### L8. 样本外为什么变差 `中等` `押题` `经典·AQI Q39`

面试官："你的模型回测很好，实盘或样本外明显变差，你怎么排查？"

<details>
<summary>答案与解析</summary>

按"先排除错误，再看过拟合，最后看环境变化"的顺序回答：

| 步骤 | 检查内容 | 方法 |
|---|---|---|
| 1. 泄漏与前视 | 特征是否用到当时不可得的数据（财报发布日、复权、指数成分、全样本标准化、标签与训练集重叠） | 逐个特征检查时间戳；用 Purged K-Fold + Embargo 重新验证（B4、B6） |
| 2. 实现差异 | 回测与实盘的数据源、复权方式、停牌和涨跌停处理、成交价格假设是否一致 | 用实盘当天的数据重放回测，逐笔对比 |
| 3. 成本与容量 | 冲击成本、手续费、印花税是否低估；规模变大后成交量占比上升 | 按成交量占比加冲击成本，做容量曲线 |
| 4. 过拟合 | 尝试过多少配置；参数是否在"悬崖边" | 统计尝试次数 $N$ 并做 Deflated Sharpe（L7）；参数邻域内看表现是否平滑 |
| 5. 非平稳 | 风格切换、市场结构变化（例如 2024 年初小盘股流动性危机、融券限制） | 分年度、分市场状态看 IC；检查风格暴露（市值、Beta、动量） |
| 6. 拥挤与衰减 | 同类信号被多家机构交易 | 看 IC 随时间的趋势、与公开因子的相关性 |

**结论句式**："先确认是不是错误（泄漏、实现差异、成本），再判断是过拟合还是环境变化；后两者的区别在于：过拟合时样本外从第一天起就差，环境变化时样本外一开始正常、某一时点后变差。"

</details>

### L9. GBDT 还是神经网络 `中等` `押题` `经典·AQI Q44`

截面选股任务中，什么时候用 GBDT，什么时候用神经网络？

<details>
<summary>答案与解析</summary>

| 维度 | GBDT（LightGBM、XGBoost） | 神经网络（MLP、GRU、Transformer） |
|---|---|---|
| 输入 | 表格型特征（几十到几千个因子） | 原始序列（量价、订单簿）、文本、多模态 |
| 预处理 | 对单调变换不敏感，不需要标准化 | 需要标准化、去极值，对异常值敏感 |
| 数据量 | 几十万样本即可 | 需要更多样本，否则过拟合 |
| 调参与稳定性 | 稳定，默认参数可用 | 对初始化、学习率、种子敏感，需要多种子集成 |
| 速度 | CPU 上分钟级 | 通常需要 GPU |
| 可解释性 | 特征重要性、SHAP | 较弱 |
| 优势场景 | 手工因子合成（日频选股的主流基线） | 从原始数据端到端学特征；高频量价；特征之间有序列或结构关系 |

**实践**：表格因子上先用 GBDT 和线性模型建立基线；NN 的价值在于与 GBDT 的预测相关性较低，两者集成通常比单独使用更好。公开研究中，Gu, Kelly & Xiu (2020) 在美股月频数据上比较了多种模型，树模型和浅层神经网络都优于线性模型，较深的网络没有进一步提升。

</details>

### L10. 大语言模型在量化中的应用 `中等` `经典·AQI Q55`

LLM 可以用在量化研究的哪些环节？有哪些特有的风险？

<details>
<summary>答案与解析</summary>

**应用**：

| 环节 | 做法 |
|---|---|
| 文本因子 | 对新闻、公告、研报、电话会议纪要打情绪分或事件标签，作为因子；或取文本嵌入向量作为特征输入下游模型 |
| 信息抽取 | 从公告中抽取结构化字段（业绩预告数值、回购金额、股东减持） |
| 研究辅助 | 代码生成、因子构造想法、文献整理 |
| 时间序列基础模型 | 用预训练的时序模型做特征提取（效果尚有争议） |

**特有风险**：
- **前视偏差**：模型的预训练语料包含回测区间之后的信息。例如用 2024 年训练的模型给 2020 年的新闻打分，模型可能"知道"后来的股价走势。应只在模型训练截止日之后的数据上评估（Glasserman & Lin 2023 专门讨论了这一问题）。
- **不可复现**：闭源模型会更新版本，同一输入的输出可能变化；需要固定版本、温度设为 0 并缓存结果。
- **成本与延迟**：全市场每天数万条文本，调用成本和延迟限制了它在高频场景中的使用。
- **幻觉**：信息抽取的结果需要与原文校验。

**公开研究**：Lopez-Lira & Tang (2023) 用 ChatGPT 对新闻标题打分，发现分数能预测次日收益，且效果集中在小盘股和负面新闻上。

</details>

---

## 参考来源

- 李航，《统计学习方法》（第 2 版），清华大学出版社，2019：第 5 章（决策树）、第 6 章（逻辑回归）、第 7 章（SVM）、第 9 章（EM 与 GMM）、第 14 章（聚类）、第 16 章（PCA）
- James, Witten, Hastie, Tibshirani，*An Introduction to Statistical Learning*（ISLR，第 2 版），2021：第 2、4、5、6、8 章
- Hastie, Tibshirani, Friedman，*The Elements of Statistical Learning*（ESL，第 2 版），2009：第 2–4、7、9、10、15 章
- Marcos López de Prado，*Advances in Financial Machine Learning*（AFML），Wiley，2018：第 1 章、第 3 章（三重障碍法）、第 7 章（Purged K-Fold、Embargo）、第 8 章（特征重要性）
- Goodfellow, Bengio, Courville，*Deep Learning*（DL），MIT Press，2016：第 3、4、6、8、10 章
- Grinold & Kahn，*Active Portfolio Management*（第 2 版），1999：主动管理基本定律、IC 与信息比率
- 论文：Chen & Guestrin (2016) XGBoost；Ke et al. (2017) LightGBM；He et al. (2015) 初始化、(2016) 残差网络；Ioffe & Szegedy (2015) BatchNorm；Ba et al. (2016) LayerNorm；Srivastava et al. (2014) Dropout；Kingma & Ba (2015) Adam；Loshchilov & Hutter (2019) AdamW；Hochreiter & Schmidhuber (1997) LSTM；Vaswani et al. (2017) Transformer；Bailey & López de Prado (2014) *The Deflated Sharpe Ratio*；Harvey, Liu & Zhu (2016) *…and the Cross-Section of Expected Returns*；Gu, Kelly & Xiu (2020) *Empirical Asset Pricing via Machine Learning*；Lopez-Lira & Tang (2023) *Can ChatGPT Forecast Stock Price Movements?*；Glasserman & Lin (2023) *Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis*
- GitHub · [SoYuCry/awesome-quant-interview](https://github.com/SoYuCry/awesome-quant-interview)：Q39、Q41–Q46、Q51–Q55（标签中的 AQI 题号）
- 官方文档：[XGBoost](https://xgboost.readthedocs.io/)（`get_dump` 输出格式）、[LightGBM](https://lightgbm.readthedocs.io/)（GOSS、早停）、[scikit-learn](https://scikit-learn.org/)
- 往年真题：公开面经仅记录宽德、诚奇笔试含机器学习题，未给出题干，因此本题库不含 `往年` 标签。幻方、衍复考过的回归 $R^2$ 题见[统计与计量题库](../statistics/problems.md) D 章。

