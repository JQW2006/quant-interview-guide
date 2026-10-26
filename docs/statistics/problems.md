# 统计与计量题库

配套讲解见[讲解](notes.md)。答案中的“面试回答要点”为口述结构，“解析”为推导过程。

| 标签 | 含义 |
|---|---|
| `往年·公司名` | 公开面经中记录的往年笔试或面试题，题干经过整理 |
| `经典·书名` | 出自面试书或教材。Wooldridge = *Introductory Econometrics*；C&B = Casella & Berger *Statistical Inference*；ISLR = *An Introduction to Statistical Learning*；Tsay = *Analysis of Financial Time Series*；绿皮书 = Zhou *A Practical Guide to Quantitative Finance Interviews*；AFML = López de Prado *Advances in Financial Machine Learning* |
| `押题` | 依据往年考点整理的高频题型 |
| `基础` `中等` `困难` | 难度 |

---

## A. 描述统计与相关

> 对应讲解第 2 节。

### A1. 为什么除以 $n-1$ `基础` `押题（高频）`

证明样本方差 $s^2=\frac1{n-1}\sum(X_i-\bar X)^2$ 是 $\sigma^2$ 的无偏估计。

<details>
<summary>答案与解析</summary>

**解析**：
$$\sum(X_i-\bar X)^2=\sum\big[(X_i-\mu)-(\bar X-\mu)\big]^2=\sum(X_i-\mu)^2-2(\bar X-\mu)\sum(X_i-\mu)+n(\bar X-\mu)^2$$
因为 $\sum(X_i-\mu)=n(\bar X-\mu)$，上式 $=\sum(X_i-\mu)^2-n(\bar X-\mu)^2$。取期望：
$$E\Big[\sum(X_i-\bar X)^2\Big]=n\sigma^2-n\cdot\frac{\sigma^2}{n}=(n-1)\sigma^2$$

**面试回答要点**：$\bar X$ 是使 $\sum(X_i-c)^2$ 最小的 $c$，所以用它代替 $\mu$ 会系统性地低估离散程度；估计均值用掉了 1 个自由度。

</details>

### A2. 相关系数的传递 `中等` `经典·绿皮书`

$\mathrm{Corr}(A,B)=0.8$，$\mathrm{Corr}(B,C)=0.8$。$\mathrm{Corr}(A,C)$ 的取值范围？

<details>
<summary>答案与解析</summary>

**答案**：$[0.28,\ 1]$

**解析**：相关矩阵必须半正定。三阶相关矩阵的行列式
$$1-\rho_{AB}^2-\rho_{BC}^2-\rho_{AC}^2+2\rho_{AB}\rho_{BC}\rho_{AC}\ge0$$
代入得 $\rho_{AC}^2-1.28\rho_{AC}+0.28\le0$，解得 $0.28\le\rho_{AC}\le1$。
一般公式：$\rho_{AC}\in\Big[\rho_1\rho_2-\sqrt{(1-\rho_1^2)(1-\rho_2^2)},\ \rho_1\rho_2+\sqrt{(1-\rho_1^2)(1-\rho_2^2)}\Big]$。

**几何直觉**：把标准化后的变量看成单位向量，相关系数是夹角的余弦。$\arccos0.8\approx36.9°$，$A$、$C$ 的夹角最多 $73.7°$，$\cos73.7°=0.28$。

</details>

### A3. 两两相关系数的下界 `中等` `经典·绿皮书`

三个随机变量两两之间的相关系数都是 $\rho$，$\rho$ 最小可以是多少？$n$ 个呢？

<details>
<summary>答案与解析</summary>

**答案**：三个：$-\frac12$；$n$ 个：$-\frac1{n-1}$

**解析**：$\mathrm{Var}(X_1+\cdots+X_n)=n+n(n-1)\rho\ge0$（各变量标准化），所以 $\rho\ge-\frac1{n-1}$。
也可以看相关矩阵的特征值：$1+(n-1)\rho$（一个）和 $1-\rho$（$n-1$ 个），都要 $\ge0$。

</details>

### A4. 不相关但不独立 `基础` `押题`

举一个两个随机变量不相关但不独立的例子。

<details>
<summary>答案与解析</summary>

**答案**：$X\sim N(0,1)$，$Y=X^2$。

**解析**：$\mathrm{Cov}(X,X^2)=E[X^3]-E[X]E[X^2]=0$，不相关。但 $Y$ 完全由 $X$ 决定。
**要点**：相关系数只衡量**线性**关系。对联合正态分布，不相关才等价于独立。
**量化含义**：因子与收益的线性 IC 为 0，不代表因子没有用，可能存在非线性关系（例如两端都有效的"U 形"因子）。

</details>

### A5. 偏度的计算与含义 `中等` `往年·诚奇`

用 Python 计算一只股票日收益率的偏度，并解释偏度为正或负分别意味着什么。

<details>
<summary>答案与解析</summary>

**代码**：
```python
import numpy as np
import pandas as pd

def skewness(r: pd.Series) -> float:
    r = r.dropna()
    n = len(r)
    z = (r - r.mean()) / r.std(ddof=0)
    return (z ** 3).mean()          # 有偏版本；pandas 的 r.skew() 做了小样本修正

# 直接用库：r.skew() 或 scipy.stats.skew(r, bias=False)
```

**解释**：
- **负偏**：左尾长，大部分时间小涨，偶尔暴跌。股票指数、卖出期权、短期反转/套利类策略常见负偏。
- **正偏**：右尾长，大部分时间小跌，偶尔暴涨。个股（尤其小盘、题材股）、买入期权、趋势跟踪策略常见正偏。
- 投资者普遍偏好正偏（"彩票偏好"），所以高正偏的股票往往被高估、未来收益偏低——这本身就是一个因子（特质偏度因子）。

**易错点**：偏度对极端值极其敏感（三次方），样本量小时估计很不稳定。

</details>

### A6. 峰度与厚尾 `基础` `押题`

正态分布的峰度是多少？金融收益率的峰度通常怎样？这对风险管理有什么影响？

<details>
<summary>答案与解析</summary>

**答案**：正态分布峰度为 3（超额峰度 0）。金融收益普遍尖峰厚尾，日收益的超额峰度常常远大于 0。

**影响**：
- 极端事件比正态模型预测的频繁得多。按正态分布，单日 −5σ 的下跌几千年一遇，但实际市场里并不罕见。
- 用正态假设计算的 VaR 会低估尾部风险，应改用历史模拟、t 分布或极值理论（EVT），并看 CVaR（尾部平均损失）。

</details>

### A7. Pearson 与 Spearman `基础` `押题`

Pearson 和 Spearman 相关系数有什么区别？量化中为什么常用 Rank IC 而不是普通 IC？

<details>
<summary>答案与解析</summary>

**面试回答要点**：
1. Pearson 衡量线性关系，对极端值敏感；Spearman 先转成排名再算 Pearson，衡量单调关系，稳健。
2. 股票收益和因子值都有大量极端值（涨跌停、财务数据异常），一两只股票就可能主导 Pearson IC。
3. 选股本质上是**排序**问题：我们关心"因子高的股票是否收益更高"，而不是精确的线性关系。
4. Rank IC 对因子做任何单调变换都不变，省去了对因子分布的处理。

</details>

### A8. 和与差的相关 `基础` `押题`

$X,Y$ 独立同分布，方差有限。求 $\mathrm{Corr}(X+Y,\ X-Y)$ 和 $\mathrm{Corr}(X,\ X+Y)$。

<details>
<summary>答案与解析</summary>

**答案**：$0$；$\dfrac1{\sqrt2}$

**解析**：
- $\mathrm{Cov}(X+Y,X-Y)=\mathrm{Var}X-\mathrm{Var}Y=0$。
- $\mathrm{Cov}(X,X+Y)=\sigma^2$，$\mathrm{Var}(X+Y)=2\sigma^2$，相关系数 $\frac{\sigma^2}{\sigma\cdot\sqrt2\sigma}=\frac1{\sqrt2}$。

**推广**：$\mathrm{Corr}(X_1,\ X_1+\cdots+X_n)=\frac1{\sqrt n}$。一只股票与 $n$ 只等权组合的相关性就是这样随 $n$ 下降的（独立情形）。

</details>

### A9. 相关系数在变换下 `基础` `押题`

$\mathrm{Corr}(X,Y)=\rho$。$\mathrm{Corr}(2X+3,\ -Y+1)$ 是多少？

<details>
<summary>答案与解析</summary>

**答案**：$-\rho$

**解析**：相关系数对正的线性变换不变，乘以负数会改变符号，加常数不影响。

</details>

### A10. 辛普森悖论 `中等` `押题`

某因子在**每一个行业内部**与未来收益都是正相关，但在全市场混合计算时却是负相关。这可能吗？为什么？

<details>
<summary>答案与解析</summary>

**答案**：可能。这是辛普森悖论。

**解析**：如果行业之间的因子均值和收益均值呈相反关系（例如因子值普遍高的行业恰好整体收益低），混合计算时**行业间的差异**会压倒行业内的关系。
数值示意：行业 A 因子值 1–2、收益 5%–6%；行业 B 因子值 8–9、收益 0%–1%。每个行业内都是正相关，但混合后整体是负相关。

**对策**：做**行业中性化**——在行业内标准化因子，或对行业哑变量回归取残差。这正是讲解第 7.9 节 FWL 定理的应用。

</details>

### A11. 均值与中位数 `基础` `押题`

一个策略的日收益均值是正的，但中位数是负的。这说明了什么？

<details>
<summary>答案与解析</summary>

**答案**：收益分布右偏（正偏）：大部分日子小亏，少数日子大赚。

**解析**：典型例子是趋势跟踪或买期权类策略。这类策略对**少数大赚日**极度依赖，回测时要检查：去掉收益最高的几天后，策略是否还赚钱？如果大赚日集中在样本中的个别事件，那么结果很可能不可复制。

</details>

### A12. 标准化与去极值 `基础` `押题（因子预处理）`

因子预处理中常见"MAD 去极值"和"Z-score 标准化"，具体怎么做？为什么用 MAD 而不是标准差去极值？

<details>
<summary>答案与解析</summary>

**做法**：
1. **MAD 去极值**：$\text{MAD}=\text{median}(|x_i-\text{median}(x)|)$。把超出 $\text{median}\pm k\cdot1.4826\cdot\text{MAD}$（$k$ 常取 3 或 5）的值截断到边界。
2. **Z-score**：$z_i=\frac{x_i-\bar x}{s}$，每个截面（每天）单独做。

**为什么用 MAD**：极端值会把标准差本身拉大，"均值 ± 3 倍标准差"的边界被极端值撑宽，就去不掉极端值了。中位数和 MAD 不受极端值影响。（1.4826 使正态分布下 MAD 与标准差可比。）

**注意**：顺序是先去极值、再中性化、再标准化；且**只能用当天截面的数据**，不能用未来数据。

</details>

---

## B. 点估计

> 对应讲解第 4 节。

### B1. 正态分布的 MLE `基础` `押题（高频）`

$X_1,\dots,X_n$ 独立同分布于 $N(\mu,\sigma^2)$，推导 $\mu$ 和 $\sigma^2$ 的 MLE，并判断是否无偏。

<details>
<summary>答案与解析</summary>

**答案**：$\hat\mu=\bar X$（无偏）；$\hat\sigma^2=\frac1n\sum(X_i-\bar X)^2$（有偏，期望为 $\frac{n-1}n\sigma^2$）

**解析**：
$$\ell(\mu,\sigma^2)=-\frac n2\ln(2\pi)-\frac n2\ln\sigma^2-\frac1{2\sigma^2}\sum(x_i-\mu)^2$$
- $\frac{\partial\ell}{\partial\mu}=\frac1{\sigma^2}\sum(x_i-\mu)=0$ ⇒ $\hat\mu=\bar x$
- $\frac{\partial\ell}{\partial\sigma^2}=-\frac n{2\sigma^2}+\frac1{2\sigma^4}\sum(x_i-\mu)^2=0$ ⇒ $\hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$

偏差见 A1。MLE 有偏但一致（$n\to\infty$ 时偏差消失）。

</details>

### B2. 伯努利与泊松的 MLE `基础` `押题`

(a) $n$ 次独立伯努利试验中成功 $k$ 次，求 $p$ 的 MLE。(b) 泊松样本 $x_1,\dots,x_n$，求 $\lambda$ 的 MLE。

<details>
<summary>答案与解析</summary>

**答案**：(a) $\hat p=\frac kn$；(b) $\hat\lambda=\bar x$

**解析**：
(a) $\ell=k\ln p+(n-k)\ln(1-p)$，$\ell'=\frac kp-\frac{n-k}{1-p}=0$。
(b) $\ell=\sum x_i\ln\lambda-n\lambda-\sum\ln x_i!$，$\ell'=\frac{\sum x_i}{\lambda}-n=0$。

</details>

### B3. 指数分布的 MLE `中等` `经典·C&B`

$X_1,\dots,X_n$ 独立同分布于 Exp($\lambda$)（均值 $1/\lambda$）。求 $\lambda$ 的 MLE，它无偏吗？

<details>
<summary>答案与解析</summary>

**答案**：$\hat\lambda=\frac1{\bar X}$，**有偏**：$E[\hat\lambda]=\frac{n}{n-1}\lambda$。无偏版本是 $\frac{n-1}{n\bar X}$。

**解析**：$\ell=n\ln\lambda-\lambda\sum x_i$，$\ell'=\frac n\lambda-\sum x_i=0$。
$S=\sum X_i\sim\text{Gamma}(n,\lambda)$，$E[1/S]=\frac{\lambda}{n-1}$，所以 $E[n/S]=\frac{n\lambda}{n-1}$。
**要点**：即使 $\bar X$ 对 $1/\lambda$ 无偏，$1/\bar X$ 对 $\lambda$ 也不是无偏的（Jensen 不等式：$1/x$ 是凸函数）。**MLE 有不变性，但无偏性在非线性变换下不保持。**

</details>

### B4. 均匀分布的三个估计 `困难` `经典·C&B`

$X_1,\dots,X_n$ 独立同分布于 $U(0,\theta)$。比较三个估计量：矩估计 $2\bar X$、MLE $X_{(n)}=\max X_i$、修正的 $\frac{n+1}nX_{(n)}$。

<details>
<summary>答案与解析</summary>

**答案**：

| 估计量 | 偏差 | 方差 | MSE |
|---|---|---|---|
| $2\bar X$ | 0 | $\dfrac{\theta^2}{3n}$ | $\dfrac{\theta^2}{3n}$ |
| $X_{(n)}$ | $-\dfrac{\theta}{n+1}$ | $\dfrac{n\theta^2}{(n+1)^2(n+2)}$ | $\dfrac{2\theta^2}{(n+1)(n+2)}$ |
| $\frac{n+1}nX_{(n)}$ | 0 | $\dfrac{\theta^2}{n(n+2)}$ | $\dfrac{\theta^2}{n(n+2)}$ |

**解析**：
- $\mathrm{Var}(2\bar X)=4\cdot\frac{\theta^2/12}{n}=\frac{\theta^2}{3n}$。
- $X_{(n)}/\theta\sim\text{Beta}(n,1)$，$E=\frac{n}{n+1}\theta$，方差 $\frac{n}{(n+1)^2(n+2)}\theta^2$。

**结论**：修正后的 MLE 方差是 $O(1/n^2)$，远小于矩估计的 $O(1/n)$。原因是 $\max$ 直接利用了"边界"信息。$n\ge2$ 时修正 MLE 都更好。

</details>

### B5. 德国坦克问题 `中等` `押题`

敌方坦克编号为 $1,2,\dots,N$。你随机缴获了 $k$ 辆，最大编号为 $m$。估计 $N$。

<details>
<summary>答案与解析</summary>

**答案**：$\hat N=m+\frac mk-1$（最小方差无偏估计）

**解析**：直观上，$k$ 个样本把 $1$ 到 $N$ 分成 $k+1$ 段，平均每段长约 $\frac{m}{k}$，所以在 $m$ 之后再加一段"平均间隔"。这是 B4 的离散版本：$\frac{k+1}{k}m-1$。
二战中盟军用这个方法估计德军坦克月产量，结果与战后档案非常接近，远比情报人员的估计准确。

</details>

### B6. 偏差-方差分解 `基础` `押题（高频）`

证明 $\mathrm{MSE}(\hat\theta)=\mathrm{Bias}^2+\mathrm{Var}$，并举一个"有偏但更好"的估计的例子。

<details>
<summary>答案与解析</summary>

**解析**：
$$E[(\hat\theta-\theta)^2]=E\big[(\hat\theta-E\hat\theta)+(E\hat\theta-\theta)\big]^2=\mathrm{Var}(\hat\theta)+(E\hat\theta-\theta)^2$$
交叉项 $2(E\hat\theta-\theta)E[\hat\theta-E\hat\theta]=0$。

**有偏但更好的例子**：
1. 正态分布方差：$\frac1{n+1}\sum(X_i-\bar X)^2$ 的 MSE 比无偏的 $s^2$ 更小。
2. Ridge 回归：引入偏差，大幅降低共线性下的方差。
3. 协方差矩阵收缩、James-Stein 估计。

**量化含义**：金融数据噪声大，方差往往是主要矛盾，所以"收缩"几乎无处不在。

</details>

### B7. 无偏与一致 `基础` `押题`

分别举一个"无偏但不一致"和"有偏但一致"的估计量的例子。

<details>
<summary>答案与解析</summary>

**答案**：
- **无偏但不一致**：用第一个观测值 $X_1$ 估计 $\mu$。$E[X_1]=\mu$，但它不随样本量增加而变准。
- **有偏但一致**：正态方差的 MLE $\frac1n\sum(X_i-\bar X)^2$；B3 中的 $1/\bar X$。

**要点**：一致性通常比无偏性更重要——数据足够多时能得到正确答案。

</details>

### B8. 标准差的估计有偏 `中等` `押题`

$s^2$ 是 $\sigma^2$ 的无偏估计，那么 $s$ 是 $\sigma$ 的无偏估计吗？

<details>
<summary>答案与解析</summary>

**答案**：不是，$E[s]<\sigma$。

**解析**：$\sqrt{\cdot}$ 是凹函数，由 Jensen 不等式 $E[s]=E[\sqrt{s^2}]\le\sqrt{E[s^2]}=\sigma$，等号仅在 $s^2$ 为常数时成立。
**量化联系**：用历史波动率估计 Sharpe 的分母，同样存在这种小样本偏差。

</details>

### B9. 合并两个估计 `中等` `押题（信号合成）`

$\hat\theta_1$、$\hat\theta_2$ 是 $\theta$ 的两个独立无偏估计，方差分别为 $\sigma_1^2$、$\sigma_2^2$。怎样线性组合使方差最小？最小方差是多少？

<details>
<summary>答案与解析</summary>

**答案**：权重与方差成反比，$w_1=\dfrac{1/\sigma_1^2}{1/\sigma_1^2+1/\sigma_2^2}$；最小方差 $\dfrac{1}{1/\sigma_1^2+1/\sigma_2^2}$

**解析**：$\hat\theta=w\hat\theta_1+(1-w)\hat\theta_2$ 无偏。$\mathrm{Var}=w^2\sigma_1^2+(1-w)^2\sigma_2^2$，对 $w$ 求导得到上式。
**推论**：合并后的"精度"（方差倒数）等于两者精度之和。
**量化联系**：多个 alpha 信号合成的最基础形式就是"按精度加权"；如果信号相关，需要用协方差矩阵的逆（类似最小方差组合）。

</details>

### B10. Fisher 信息与 Cramér-Rao 下界 `中等` `经典·C&B`

伯努利($p$) 样本，求 Fisher 信息量和无偏估计方差的下界。$\bar X$ 达到下界吗？

<details>
<summary>答案与解析</summary>

**答案**：$I(p)=\frac1{p(1-p)}$；下界 $\frac{p(1-p)}{n}$；$\bar X$ 达到下界。

**解析**：$\ln f=x\ln p+(1-x)\ln(1-p)$，$\frac{\partial^2}{\partial p^2}\ln f=-\frac x{p^2}-\frac{1-x}{(1-p)^2}$。
$I(p)=-E[\cdot]=\frac1p+\frac1{1-p}=\frac1{p(1-p)}$。CRLB $=\frac{1}{nI(p)}=\frac{p(1-p)}n=\mathrm{Var}(\bar X)$。

</details>

### B11. 胜率估计需要多少样本 `基础` `押题`

一个策略的真实胜率是 55%。要让胜率估计的标准误不超过 1%，需要多少笔交易？

<details>
<summary>答案与解析</summary>

**答案**：约 2500 笔

**解析**：$\mathrm{SE}=\sqrt{\frac{p(1-p)}n}=\sqrt{\frac{0.2475}n}\le0.01$ ⇒ $n\ge2475$。
**含义**：区分 55% 和 50% 的胜率（差 5 个百分点）至少需要几百笔交易才能有 2 个标准误的把握。**这就是为什么高频策略容易被验证，而低频策略很难被统计验证。**

</details>

### B12. MLE 的不变性 `基础` `押题`

已知正态样本，求 $\sigma$ 的 MLE 和 $P(X>0)$ 的 MLE。

<details>
<summary>答案与解析</summary>

**答案**：$\hat\sigma=\sqrt{\hat\sigma^2_{MLE}}$；$\widehat{P(X>0)}=\Phi(\hat\mu/\hat\sigma)$

**解析**：MLE 的不变性：$g(\theta)$ 的 MLE 是 $g(\hat\theta)$。$P(X>0)=\Phi(\mu/\sigma)$，直接代入。

</details>

---

## C. 假设检验与多重检验

> 对应讲解第 5、6 节。

### C1. 硬币公平吗 `基础` `经典·绿皮书`

抛硬币 100 次得到 60 次正面。在 5% 显著性水平下，能否认为硬币不公平？

<details>
<summary>答案与解析</summary>

**答案**：正态近似下勉强拒绝（双侧 p ≈ 0.046）；精确二项检验双侧 p ≈ 0.057，不拒绝。处于边界。

**解析**：$H_0$：$p=0.5$。$z=\frac{60-50}{\sqrt{100\cdot0.25}}=2$，双侧 p 值 $\approx0.0455$。
精确计算 $P(X\ge60)\approx0.0284$，双侧约 0.057。差异来自连续性修正：用 $59.5$ 代替 $60$ 时 $z=1.9$，p ≈ 0.057。

**面试加分**：主动说明"正态近似和精确检验在边界附近会给出不同结论，这种情况下不应只看 p 值是否小于 0.05"。

</details>

### C2. p 值是什么 `基础` `押题（高频）`

用一句话解释 p 值，并指出两个常见误解。

<details>
<summary>答案与解析</summary>

**答案**：p 值是**在原假设成立的前提下**，得到至少和观察值一样极端的结果的概率。

**常见误解**：
1. "p 值是原假设为真的概率"——错，那是 $P(H_0\mid\text{数据})$，需要贝叶斯和先验才能算。
2. "p 值小说明效应大"——错，样本量大时微小的效应也能很显著。
3. "p > 0.05 说明原假设成立"——错，只是证据不足。

</details>

### C3. 两类错误与功效 `基础` `押题`

什么是第一类、第二类错误？在因子研究中，哪类错误代价更大？

<details>
<summary>答案与解析</summary>

**答案**：
- 第一类错误：$H_0$ 为真却拒绝（把无效因子当成有效）。
- 第二类错误：$H_0$ 为假却没有拒绝（错过有效因子）。
- 功效 $=1-P(\text{第二类错误})$。

**因子研究中**：通常**第一类错误代价更大**——无效因子上线会真金白银亏钱，还会挤占风险预算；错过一个因子只是机会成本。所以量化研究对显著性的要求比学术论文更严格（t > 3）。

</details>

### C4. 100 个无效因子 `基础` `押题（高频）`

你测试了 100 个**完全无效**的因子，每个都在 5% 水平下做检验（假设相互独立）。期望有几个显著？至少一个显著的概率？

<details>
<summary>答案与解析</summary>

**答案**：期望 5 个；至少一个显著的概率 $1-0.95^{100}\approx99.4\%$

**要点**：这就是多重检验问题。对策见 C5、C6 和讲解第 6 节。

</details>

### C5. Bonferroni 修正 `基础` `押题`

对 C4 的情形做 Bonferroni 修正，每个检验的显著性门槛是多少？对应的 t 值约为多少？

<details>
<summary>答案与解析</summary>

**答案**：$\alpha/m=0.05/100=0.0005$；双侧检验对应 $|t|\approx3.48$

**解析**：Bonferroni 保证"至少一个假阳性"的概率 $\le\alpha$（由并集上界 $P(\cup A_i)\le\sum P(A_i)$）。
**缺点**：很保守，检验之间相关时尤其如此（因子之间往往高度相关）。

</details>

### C6. BH 方法 `中等` `押题`

10 个因子的 p 值从小到大为：0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216。用 Bonferroni 和 Benjamini-Hochberg（FDR = 5%）分别能拒绝几个原假设？

<details>
<summary>答案与解析</summary>

**答案**：Bonferroni 拒绝 1 个；BH 拒绝 2 个。不做修正时拒绝 5 个。

**解析**：
- Bonferroni 门槛 $0.05/10=0.005$：只有 0.001 通过。
- BH：第 $k$ 个 p 值与 $\frac{k}{10}\times0.05=0.005k$ 比较，找最大的满足 $p_{(k)}\le0.005k$ 的 $k$：
  - $k=1$：$0.001\le0.005$，成立；$k=2$：$0.008\le0.010$，成立；$k=3$：$0.039>0.015$，不成立；之后都，不成立。
  - 所以拒绝前 2 个。

**区别**：Bonferroni 控制"犯任何一个错误"的概率（FWER）；BH 控制"所有发现中错误发现的比例"（FDR），更适合大规模因子筛选。

</details>

### C7. 检测微小优势需要多少样本 `中等` `押题`

一枚硬币正面概率实际是 0.51。大约要抛多少次，才能以约 2 个标准误的把握把它和公平硬币区分开？

<details>
<summary>答案与解析</summary>

**答案**：约 10000 次

**解析**：$\hat p$ 的标准误 $\approx\frac{0.5}{\sqrt n}$。需要 $0.01\ge2\times\frac{0.5}{\sqrt n}$，即 $\sqrt n\ge100$。
**含义**：量化交易的优势往往就是 51% 这个量级。**优势越小，验证所需的样本量按平方增长**。这也是高频策略（样本多）和低频策略（样本少）在统计上的根本区别。

</details>

### C8. 置信区间的含义 `基础` `押题`

"某因子 IC 均值的 95% 置信区间是 [0.02, 0.06]"，下面哪个解读正确？
(a) 真实 IC 均值有 95% 的概率落在 [0.02, 0.06] 内；
(b) 如果反复抽样并用同样方法构造区间，约 95% 的区间会包含真实 IC 均值。

<details>
<summary>答案与解析</summary>

**答案**：(b)

**解析**：频率学派中真实 IC 是固定的数，区间才是随机的。(a) 是贝叶斯"可信区间"的解读，需要先验。面试中能区分两者是加分项，但也可以补充："实践中两者在数值上常常接近。"

</details>

### C9. t 检验与 z 检验 `基础` `押题`

什么时候用 t 检验，什么时候用 z 检验？t 分布和正态分布有什么区别？

<details>
<summary>答案与解析</summary>

**答案**：总体方差已知（或样本很大）用 z；方差未知、用样本标准差估计时用 t。
t 分布尾部比正态厚，因为分母 $s$ 本身也有估计误差。自由度越大越接近正态，自由度 30 以上差别已经很小。
**量化中**：样本通常有几百上千个，t 和 z 几乎没区别；真正的问题是**样本不独立**（自相关、重叠），需要用 Newey-West 调整。

</details>

### C10. 两个策略谁更好 `中等` `押题`

策略 A 年化 Sharpe 1.2，策略 B 年化 Sharpe 1.0，回测期相同。能说 A 显著优于 B 吗？怎样检验？

<details>
<summary>答案与解析</summary>

**答案**：不能直接下结论。

**做法**：
1. 对**日收益之差** $d_t=r_{A,t}-r_{B,t}$ 做 t 检验（配对检验），利用两者的相关性。如果两者高度相关，差值的波动小，更容易检验出差异。
2. 直接比较 Sharpe 可以用 Jobson-Korkie / Memmel 检验，或 bootstrap 两者 Sharpe 之差的分布。
3. 参考 H3：回测几年时 Sharpe 的标准误约为 0.5，0.2 的差异通常不显著。

</details>

### C11. Sharpe 比率的 t 值 `基础` `押题（高频）`

一个策略年化 Sharpe 1.5，回测 2 年。它的平均收益显著不为 0 吗？年化 Sharpe 0.8 需要回测多少年才能达到 $t=3$？

<details>
<summary>答案与解析</summary>

**答案**：$t=1.5\sqrt2\approx2.1$，勉强显著（5% 水平）；$0.8\sqrt{T}=3$ ⇒ $T\approx14$ 年

**解析**：$t=\frac{\bar r}{s/\sqrt{n}}=SR_{日}\sqrt{n_{日}}=SR_{年}\sqrt{\text{年数}}$。
**含义**：中低 Sharpe 的策略几乎不可能在合理的时间内被统计证明。所以研究员还需要依靠经济逻辑、多市场验证、样本外表现来判断。

</details>

### C12. 为什么新因子要求 $t>3$ `中等` `经典·Harvey, Liu & Zhu (2016)`

为什么学术界建议新因子的 t 值要超过 3，而不是传统的 2？

<details>
<summary>答案与解析</summary>

**面试回答要点**：
1. 学术界和业界已经检验了几百上千个因子（"因子动物园"），这本身是大规模的多重检验。
2. 发表偏差：只有显著的结果被发表，失败的尝试不被报告，实际尝试次数远多于看到的。
3. Harvey, Liu & Zhu 用多重检验框架（Bonferroni、Holm、BHY）估计，考虑到已有尝试次数，合理的门槛约为 $t>3$。
4. 业内自己做研究时同理：**你自己尝试的因子数也要算进去**。

</details>

---

## D. 线性回归：推导与性质

> 对应讲解第 7 节。**D1–D3 必须能闭卷手推。**

### D1. 一元回归推导 `基础` `押题（高频）`

推导一元线性回归 $y=\beta_0+\beta_1x+\varepsilon$ 的 OLS 估计。

<details>
<summary>答案与解析</summary>

**答案**：$\hat\beta_1=\dfrac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}$，$\hat\beta_0=\bar y-\hat\beta_1\bar x$

**解析**：最小化 $S=\sum(y_i-\beta_0-\beta_1x_i)^2$。
- $\frac{\partial S}{\partial\beta_0}=-2\sum(y_i-\beta_0-\beta_1x_i)=0$ ⇒ $\bar y=\beta_0+\beta_1\bar x$
- $\frac{\partial S}{\partial\beta_1}=-2\sum x_i(y_i-\beta_0-\beta_1x_i)=0$

把第一式代入第二式：$\sum x_i\big[(y_i-\bar y)-\beta_1(x_i-\bar x)\big]=0$。注意 $\sum x_i(y_i-\bar y)=\sum(x_i-\bar x)(y_i-\bar y)$，$\sum x_i(x_i-\bar x)=\sum(x_i-\bar x)^2$，得到结果。

**记忆方式**：$\hat\beta_1=\frac{\mathrm{Cov}(x,y)}{\mathrm{Var}(x)}=r\frac{s_y}{s_x}$。

</details>

### D2. 矩阵形式推导 `中等` `押题（高频）`

推导 $\hat\beta=(X^\top X)^{-1}X^\top y$，并说明需要什么条件。

<details>
<summary>答案与解析</summary>

**解析**：
$$S(\beta)=(y-X\beta)^\top(y-X\beta)=y^\top y-2\beta^\top X^\top y+\beta^\top X^\top X\beta$$
用 $\frac{\partial(a^\top\beta)}{\partial\beta}=a$、$\frac{\partial(\beta^\top A\beta)}{\partial\beta}=2A\beta$（$A$ 对称）：
$$\frac{\partial S}{\partial\beta}=-2X^\top y+2X^\top X\beta=0$$
条件：$X^\top X$ 可逆 ⇔ $X$ 列满秩（没有完全共线性，且 $n\ge p+1$）。
二阶条件：Hessian $2X^\top X$ 正定，所以是最小值。

</details>

### D3. 无偏性与方差 `中等` `押题（高频）`

在高斯-马尔可夫假设下，证明 $\hat\beta$ 无偏，并求 $\mathrm{Var}(\hat\beta\mid X)$。

<details>
<summary>答案与解析</summary>

**解析**：$\hat\beta=(X^\top X)^{-1}X^\top(X\beta+\varepsilon)=\beta+(X^\top X)^{-1}X^\top\varepsilon$。
- $E[\hat\beta\mid X]=\beta+(X^\top X)^{-1}X^\top E[\varepsilon\mid X]=\beta$（用到 $E[\varepsilon\mid X]=0$）
- $\mathrm{Var}(\hat\beta\mid X)=(X^\top X)^{-1}X^\top\mathrm{Var}(\varepsilon\mid X)X(X^\top X)^{-1}=\sigma^2(X^\top X)^{-1}$（用到 $\mathrm{Var}(\varepsilon\mid X)=\sigma^2I$）

**面试回答要点**：说清楚**无偏只用到外生性**，**方差公式用到同方差和无自相关**。这两点直接对应"违反假设的后果"。

</details>

### D4. 两条回归线的斜率 `中等` `往年·衍复` `往年·幻方`

$y$ 对 $x$ 回归的斜率为 0.5，$x$ 对 $y$ 回归的斜率为 1.2。求 $R^2$。斜率能否分别是 2 和 1？

<details>
<summary>答案与解析</summary>

**答案**：$R^2=0.6$；不能是 2 和 1。

**解析**：$b_{y|x}=r\frac{s_y}{s_x}$，$b_{x|y}=r\frac{s_x}{s_y}$，所以 $b_{y|x}\cdot b_{x|y}=r^2=R^2$。
$0.5\times1.2=0.6$。而 $2\times1=2>1$，不可能。
另外两个斜率必须**同号**（都与 $r$ 同号）。

**追问**：为什么 $b_{x|y}\ne1/b_{y|x}$？因为两次回归最小化的是不同方向的误差（竖直距离 vs 水平距离）。只有 $|r|=1$ 时两条线重合。

</details>

### D5. $R^2$ 的几种等价形式 `基础` `押题`

证明一元回归中 $R^2=r^2$，并说明多元回归中 $R^2$ 与什么相关系数有关。

<details>
<summary>答案与解析</summary>

**解析**：一元回归 $\hat y_i-\bar y=\hat\beta_1(x_i-\bar x)$，
$$R^2=\frac{\sum(\hat y_i-\bar y)^2}{\sum(y_i-\bar y)^2}=\hat\beta_1^2\frac{\sum(x_i-\bar x)^2}{\sum(y_i-\bar y)^2}=r^2\frac{s_y^2}{s_x^2}\cdot\frac{s_x^2}{s_y^2}=r^2$$
多元回归（有截距）中，$R^2=\mathrm{Corr}(y,\hat y)^2$。

</details>

### D6. 数据复制一份 `中等` `押题（高频）`

把数据集原样复制一份（样本量翻倍）后重新回归，系数、标准误、t 值、$R^2$ 怎么变？

<details>
<summary>答案与解析</summary>

**答案**：系数不变；标准误约缩小为原来的 $\frac1{\sqrt2}$；t 值约放大 $\sqrt2$ 倍；$R^2$ 不变。

**解析**：$X^\top X$ 和 $X^\top y$ 都翻倍，$\hat\beta$ 不变。残差平方和翻倍，$\hat\sigma^2=\frac{2RSS}{2n-p-1}\approx\frac{RSS}{n-p-1}$ 基本不变，而 $(X^\top X)^{-1}$ 减半，所以方差减半。

**要点**：复制数据没有带来新信息，t 值增大是**虚假的精确**。量化中的类似情况：用重叠窗口构造样本（如每天计算未来 20 日收益），样本不独立，不修正标准误就会高估显著性。

</details>

### D7. 变量的缩放与平移 `基础` `押题`

把 $x$ 乘以 100（例如从小数改成百分数），回归系数、标准误、t 值、$R^2$ 怎么变？把 $y$ 加上常数 5 呢？

<details>
<summary>答案与解析</summary>

**答案**：
- $x\to100x$：斜率变为 $\frac1{100}$，标准误同样变为 $\frac1{100}$，**t 值和 $R^2$ 不变**；截距不变。
- $y\to y+5$：只有截距加 5，斜率、标准误、t 值、$R^2$ 都不变。

**要点**：t 值和 $R^2$ 对线性变换不变，是"无量纲"的；系数大小取决于单位，不能直接比较不同变量的系数大小（要比较就先标准化）。

</details>

### D8. 增加变量对 $R^2$ 的影响 `基础` `押题`

在回归中加入一个与 $y$ 完全无关的随机变量，$R^2$ 会怎样？调整 $R^2$ 呢？

<details>
<summary>答案与解析</summary>

**答案**：$R^2$ 不会下降（几乎总会略微上升）；调整 $R^2$ 可能下降。

**解析**：多一个变量，最小化的可行范围变大（新变量系数取 0 就回到原模型），所以 RSS 不增。调整 $R^2=1-\frac{RSS/(n-p-1)}{TSS/(n-1)}$ 对变量个数加了惩罚。
**极端情形**：变量个数等于样本量时 $R^2=1$（完美过拟合）。

</details>

### D9. 残差的性质 `中等` `押题`

有截距的 OLS 回归中，残差有什么性质？残差与 $y$ 相关吗？

<details>
<summary>答案与解析</summary>

**答案**：
1. $\sum e_i=0$；
2. 残差与每个自变量正交：$X^\top e=0$；
3. 残差与拟合值正交：$\hat y^\top e=0$；
4. 但**残差与 $y$ 正相关**：$\mathrm{Corr}(e,y)=\sqrt{1-R^2}$。

**解析（第 4 点）**：$y=\hat y+e$，$\hat y\perp e$，所以 $\mathrm{Cov}(e,y)=\mathrm{Var}(e)$，$\mathrm{Corr}(e,y)=\frac{\mathrm{Var}(e)}{\sigma_e\sigma_y}=\frac{\sigma_e}{\sigma_y}=\sqrt{1-R^2}$。
**陷阱**：画"残差 vs $y$"的散点图一定会看到正斜率，这不说明模型有问题。诊断应该画"残差 vs $\hat y$"或"残差 vs $x$"。

</details>

### D10. 回归到均值 `中等` `押题`

$x$ 和 $y$ 都已标准化（均值 0、方差 1），相关系数 $r=0.5$。某个样本的 $x=2$，回归预测的 $y$ 是多少？这说明了什么现象？

<details>
<summary>答案与解析</summary>

**答案**：$\hat y=1$

**解析**：标准化后斜率就是 $r$，$\hat y=rx=1$。$x$ 偏离均值 2 个标准差，但预测的 $y$ 只偏离 1 个标准差——**回归到均值**（Galton 发现：高个子父母的孩子平均也高，但没有父母那么高）。

**量化例子**：
- 去年业绩最好的基金经理，今年业绩通常仍高于平均，但没有去年那么好。
- 因子值极端的股票，下期因子值往往会向均值回归。
- 回测中表现最好的参数，样本外表现通常会变差（选择偏差 + 回归到均值）。

</details>

### D11. 遗漏变量与系数关系 `中等` `经典·Wooldridge`

用 $y$ 对 $x_1$ 回归得到系数 $\tilde\beta_1$；用 $y$ 对 $x_1,x_2$ 回归得到 $\hat\beta_1,\hat\beta_2$；用 $x_2$ 对 $x_1$ 回归得到 $\hat\delta$。三者有什么关系？

<details>
<summary>答案与解析</summary>

**答案**：$\tilde\beta_1=\hat\beta_1+\hat\beta_2\hat\delta$（在样本中**精确成立**）

**解析**：把长回归 $y=\hat\beta_0+\hat\beta_1x_1+\hat\beta_2x_2+e$ 代入短回归的公式。$e$ 与 $x_1$ 正交，$x_2$ 对 $x_1$ 回归的斜率是 $\hat\delta$，所以 $\tilde\beta_1=\hat\beta_1+\hat\beta_2\hat\delta$。
**含义**：短回归系数 = 直接效应 + 通过 $x_2$ 传导的间接效应。只有 $\hat\beta_2=0$ 或 $\hat\delta=0$ 时二者相等。

</details>

### D12. Frisch-Waugh-Lovell 定理 `中等` `押题（中性化原理）`

解释 FWL 定理，并说明它和因子中性化的关系。

<details>
<summary>答案与解析</summary>

**答案**：多元回归中 $x_1$ 的系数，等于"$y$ 对其他变量回归的残差"对"$x_1$ 对其他变量回归的残差"做回归的系数。

**解析（矩阵形式）**：设 $M_2=I-X_2(X_2^\top X_2)^{-1}X_2^\top$ 为"剔除 $X_2$"的投影矩阵，则 $\hat\beta_1=(X_1^\top M_2X_1)^{-1}X_1^\top M_2y$。

**与中性化的关系**：因子中性化是把因子对行业哑变量和对数市值回归，取残差作为新因子。由 FWL，用中性化后的因子预测收益，等价于在多元回归里控制了行业和市值。它回答的问题是：**在相同行业、相同市值的股票之间，这个因子还有没有区分能力？**

</details>

### D13. 多重共线性与 VIF `中等` `押题`

什么是方差膨胀因子（VIF）？两个因子相关系数 0.95，同时放进回归会怎样？

<details>
<summary>答案与解析</summary>

**答案**：$\mathrm{VIF}_j=\dfrac{1}{1-R_j^2}$，$R_j^2$ 是 $x_j$ 对其他自变量回归的 $R^2$。$\mathrm{Var}(\hat\beta_j)=\dfrac{\sigma^2}{\sum(x_{ij}-\bar x_j)^2}\cdot\mathrm{VIF}_j$。

两个变量时 $R_j^2=r^2=0.9025$，$\mathrm{VIF}\approx10.3$，系数的标准差放大约 3.2 倍。

**表现**：两个系数可能一正一负、绝对值很大；换一段样本就符号反转；单个都不显著但联合显著（见 D15）。
**注意**：共线性不影响**预测**（两者之和估计得很准），只影响对**单个系数**的解释。
**对策**：只保留一个；做正交化；合成一个因子；用 Ridge。

</details>

### D14. 无截距回归 `中等` `押题`

强行去掉截距做回归会有什么后果？什么时候可以不要截距？

<details>
<summary>答案与解析</summary>

**答案**：
- 如果真实截距不为 0，斜率估计**有偏**（回归线被强迫过原点）；
- 残差之和不再为 0，TSS = ESS + RSS 的分解不成立，软件报告的 $R^2$ 通常改用未中心化的定义，数值虚高，不能和有截距的 $R^2$ 比较。

**可以不要截距的情形**：理论上要求过原点，例如 CAPM 检验中有时把截距（alpha）单独研究；或者自变量里已经包含了完整的哑变量组（如全部行业哑变量），此时加截距反而会完全共线（"哑变量陷阱"）。

</details>

### D15. 单个不显著、联合显著 `中等` `押题`

回归中 $x_1$ 和 $x_2$ 的 t 检验都不显著，但 F 检验显示二者联合显著。这可能吗？为什么？

<details>
<summary>答案与解析</summary>

**答案**：可能，典型原因是 $x_1$、$x_2$ 高度相关。

**解析**：t 检验问的是"在控制了另一个变量之后，这个变量还有没有额外作用"。两个变量高度相关时，任何一个都能替代另一个，所以单独看都不显著；但两者至少有一个有用，所以联合显著。
反过来也可能：单个显著、联合不显著（较少见，与检验功效有关）。

</details>

### D16. OLS 与 MLE `中等` `押题（高频）`

证明：若 $\varepsilon\sim N(0,\sigma^2I)$，$\beta$ 的 MLE 等于 OLS 估计。

<details>
<summary>答案与解析</summary>

**解析**：
$$\ell(\beta,\sigma^2)=-\frac n2\ln(2\pi\sigma^2)-\frac{1}{2\sigma^2}(y-X\beta)^\top(y-X\beta)$$
对任何固定的 $\sigma^2$，最大化 $\ell$ 关于 $\beta$ ⇔ 最小化 $(y-X\beta)^\top(y-X\beta)$，即 OLS。
$\sigma^2$ 的 MLE 是 $\frac{RSS}{n}$（有偏），无偏估计是 $\frac{RSS}{n-p-1}$。

**推广**：误差服从拉普拉斯分布时，MLE 是**最小绝对偏差回归**（LAD，即中位数回归），对离群值更稳健。

</details>

### D17. BLUE 的含义 `基础` `押题`

高斯-马尔可夫定理说 OLS 是 BLUE，这需要误差服从正态分布吗？"最优"是在什么范围内比较的？

<details>
<summary>答案与解析</summary>

**答案**：
- **不需要**正态假设，只需要线性、满秩、外生性、同方差无自相关。
- "最优"是在**线性无偏**估计量中方差最小。有偏的估计（如 Ridge）或非线性估计可能有更小的 MSE。
- 正态假设的作用是让小样本下的 t、F 检验精确成立；大样本时靠中心极限定理。

</details>

### D18. 帽子矩阵 `中等` `押题`

帽子矩阵 $H=X(X^\top X)^{-1}X^\top$ 有哪些性质？它的对角元素有什么含义？

<details>
<summary>答案与解析</summary>

**答案**：
- 对称：$H^\top=H$；幂等：$H^2=H$；特征值只能是 0 或 1。
- $\mathrm{tr}(H)=\mathrm{rank}(X)=p+1$。
- $I-H$ 也是投影矩阵，$e=(I-H)y$。
- 对角元素 $h_{ii}$ 叫**杠杆值**：第 $i$ 个观测对自己拟合值的影响，$0\le h_{ii}\le1$，平均值 $\frac{p+1}n$。$h_{ii}$ 远大于平均值的点是高杠杆点（$x$ 远离中心）。

**证明 $\mathrm{tr}(H)=p+1$**：$\mathrm{tr}(X(X^\top X)^{-1}X^\top)=\mathrm{tr}((X^\top X)^{-1}X^\top X)=\mathrm{tr}(I_{p+1})=p+1$。

</details>

### D19. $R^2$ 能否为负 `基础` `押题`

$R^2$ 可以是负数吗？

<details>
<summary>答案与解析</summary>

**答案**：
- 有截距的 OLS，**样本内** $R^2\in[0,1]$。
- **样本外** $R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$ 可以为负：说明模型预测还不如直接用历史均值。在收益预测中这非常常见。
- 无截距回归或非 OLS 方法（如带约束的拟合）用这个公式也可能为负。

</details>

### D20. 低 $R^2$ 也可以很显著 `中等` `押题（量化直觉）`

用某因子做日频收益的一元回归，$R^2=1\%$，样本 2500 天。斜率显著吗？$R^2$ 这么低，因子有用吗？

<details>
<summary>答案与解析</summary>

**答案**：$t\approx5.0$，非常显著；而且很有用。

**解析**：一元回归中 $t^2=\frac{(n-2)R^2}{1-R^2}$，$t\approx\sqrt{2500\times0.01/0.99}\approx5.0$。
**量化含义**：$R^2=1\%$ 对应相关系数约 0.1。由主动管理基本定律 $IR\approx IC\sqrt{\text{Breadth}}$，在几千只股票上反复使用，IC 0.05–0.1 的信号就可以支撑很高的 Sharpe。**金融里不能用"$R^2$ 低"来否定一个信号。**

</details>

---

## E. 违反回归假设

> 对应讲解第 8 节。

### E1. 异方差 `中等` `押题（高频）`

什么是异方差？对 OLS 有什么影响？如何检测和处理？

<details>
<summary>答案与解析</summary>

**面试回答要点**：
1. **定义**：误差方差随观测变化，$\mathrm{Var}(\varepsilon_i\mid X)=\sigma_i^2$。例：大市值股票收益波动小，小市值波动大。
2. **影响**：$\hat\beta$ 仍然无偏、一致，但不再是 BLUE；**常规标准误公式错误**，t 检验和置信区间不可靠。
3. **检测**：残差图呈喇叭形；Breusch-Pagan 检验；White 检验。
4. **处理**：
   - 用 **White 稳健标准误**（HC0–HC3），系数不变、只修正标准误；
   - 若知道方差结构，用**加权最小二乘**（WLS），权重 $1/\sigma_i^2$，重新获得有效性；
   - 对 $y$ 做变换（如取对数）。

</details>

### E2. 自相关与重叠样本 `中等` `押题（高频）`

用每天的因子值预测**未来 20 日收益**，在日频样本上做回归。会有什么问题？怎样修正？

<details>
<summary>答案与解析</summary>

**答案**：相邻样本的 20 日收益重叠 19 天，残差高度自相关，常规 t 值严重高估。

**解析**：若日收益独立，相隔 $j$ 天的两个 20 日收益的相关系数为 $\frac{20-j}{20}$（$j<20$）。有效样本量大约只有原来的 $\frac1{20}$，t 值大约被高估 $\sqrt{20}\approx4.5$ 倍（粗略量级，取决于残差结构）。

**修正**：
1. Newey-West 标准误，滞后阶数至少取 19；
2. 只用不重叠的样本（每 20 天取一次），代价是样本量减少；
3. 用 Hansen-Hodrick 标准误。

**机器学习中的同类问题**：标签有重叠时做交叉验证，要用 Purged K-Fold + Embargo（AFML 第 7 章）。

</details>

### E3. 遗漏变量偏误 `中等` `押题（高频）`

真实模型 $y=\beta_1x_1+\beta_2x_2+\varepsilon$，$\beta_1=1$、$\beta_2=2$，$x_2$ 对 $x_1$ 回归的斜率 $\delta=0.5$。只用 $x_1$ 回归，$\hat\beta_1$ 的期望是多少？举一个量化中的例子。

<details>
<summary>答案与解析</summary>

**答案**：$E[\tilde\beta_1]=\beta_1+\beta_2\delta=1+2\times0.5=2$

**解析**：见讲解 8.1 节。偏误方向取决于 $\beta_2$ 和 $\delta$ 的符号。

**量化例子**：
- "低股价股票收益高"——可能只是因为低股价股票市值小，真正起作用的是**市值**。
- "某因子在 2015–2020 年很有效"——可能因为它暴露了**小市值风格**，而这段时间小盘股表现好。
- 所以研究新因子时，要控制已知的风格因子（市值、价值、动量、波动率、流动性、行业），看**增量**信息。

</details>

### E4. 衰减偏误 `中等` `经典·Wooldridge`

真实模型 $y=\beta x^*+\varepsilon$，但我们观察到的是 $x=x^*+u$，$u$ 是独立噪声，且 $\mathrm{Var}(u)=\mathrm{Var}(x^*)$。OLS 估计的斜率趋于多少？

<details>
<summary>答案与解析</summary>

**答案**：$\dfrac\beta2$

**解析**：$\mathrm{plim}\,\hat\beta=\frac{\mathrm{Cov}(x,y)}{\mathrm{Var}(x)}=\frac{\beta\mathrm{Var}(x^*)}{\mathrm{Var}(x^*)+\mathrm{Var}(u)}=\beta\cdot\frac{1}{2}$。
系数被压向 0，比例为"信号占总方差的比例"（可靠性比率）。

**对比**：$y$ 有测量误差时，误差并入 $\varepsilon$，$\hat\beta$ 仍然无偏，只是方差变大。

**量化含义**：因子是对"真实特征"的有噪估计（例如用历史数据估计的 beta、波动率），回归出的因子溢价会被低估。Fama-MacBeth 研究中常用组合代替个股，就是为了降低这种测量误差。

</details>

### E5. 反向因果 `中等` `押题`

研究发现"分析师覆盖越多的股票，收益越高"。能否得出"增加分析师覆盖能提高收益"的结论？

<details>
<summary>答案与解析</summary>

**答案**：不能。可能存在反向因果和遗漏变量。

**解析**：
- **反向因果**：表现好的股票吸引了更多分析师关注。
- **遗漏变量**：大市值、高流动性、机构持股多的股票既有更多分析师覆盖，也可能有不同的收益特征。
- 这都是内生性问题，OLS 有偏。处理：控制变量、工具变量（例如券商合并导致的分析师覆盖外生减少，这是学术文献中的经典识别策略）、事件研究。

**面试要点**：区分**预测**和**因果**。量化选股只需要预测关系（只要它稳定存在），但理解因果有助于判断这种关系会不会持续。

</details>

### E6. 伪回归 `中等` `经典·Tsay`

把两个**相互独立**的随机游走序列（例如两只毫无关系的股票的价格）做回归，会得到什么结果？

<details>
<summary>答案与解析</summary>

**答案**：常常得到很高的 $R^2$ 和非常显著的 t 值，但这种关系是虚假的（伪回归，Granger & Newbold 1974）。

**解析**：非平稳序列都有随机趋势，在有限样本中常常"碰巧"同向漂移。常规 t 检验的假设（平稳）被破坏，t 统计量不收敛到 t 分布，样本越大越容易"显著"。
**诊断**：残差高度自相关（Durbin-Watson 统计量很低），一个经验法则是 $R^2>DW$ 就要警惕。
**对策**：对收益率（差分）回归；或者检验协整。

</details>

### E7. 离群值 `基础` `押题`

回归中有几个极端值，怎么处理？

<details>
<summary>答案与解析</summary>

**面试回答要点**：
1. **先查原因**：数据错误（复权错误、单位错误）就修正或删除；真实的极端行情不能随便删。
2. **去极值**：MAD 去极值或按分位数截尾（winsorize，例如截到 1% 和 99% 分位数）。
3. **用排名代替原值**：如 Rank IC、对排名做回归。
4. **稳健回归**：Huber 损失、LAD 回归。
5. **诊断**：看杠杆值和 Cook 距离，判断哪些点对系数影响最大。

</details>

### E8. 幸存者偏差 `基础` `押题`

用"当前沪深 300 成分股"过去 10 年的数据回测一个选股策略，有什么问题？

<details>
<summary>答案与解析</summary>

**答案**：同时有**幸存者偏差**和**前视偏差**。

**解析**：现在的成分股是"过去 10 年表现好、市值长大"的股票；被调出、退市的股票都不在样本里。回测相当于提前知道了哪些股票会成功，结果会严重高估。
**正确做法**：使用**历史时点**的成分股名单（point-in-time 数据），包括已退市的股票。

</details>

### E9. 横截面回归的权重 `中等` `押题`

每天做一次"股票收益 ~ 因子暴露"的横截面回归（Fama-MacBeth / Barra 风格），为什么常用市值平方根作为权重？

<details>
<summary>答案与解析</summary>

**答案**：处理异方差，并让结果更贴近可投资的部分。

**解析**：小市值股票的特质波动远大于大市值股票，这是典型的异方差。WLS 的最优权重是 $1/\sigma_i^2$。经验上特质方差大致与市值的平方根成反比，所以用 $\sqrt{\text{市值}}$ 作为权重。另外，直接用市值作为权重会让少数超大市值股票主导回归，平方根是一种折中。

</details>

### E10. Fama-MacBeth 回归 `中等` `押题`

简述 Fama-MacBeth 两步法，以及它解决了什么问题。

<details>
<summary>答案与解析</summary>

**步骤**：
1. 每个时期 $t$ 做一次横截面回归：$r_{i,t+1}=a_t+\lambda_tx_{i,t}+e_{i,t}$，得到因子收益序列 $\lambda_t$。
2. 对 $\lambda_t$ 序列求均值，用其标准误 $\frac{s(\lambda_t)}{\sqrt T}$ 做 t 检验（通常用 Newey-West 修正）。

**解决的问题**：同一天的股票收益高度相关（横截面相关），把所有股票-时间样本混在一起做混合回归，会严重低估标准误。Fama-MacBeth 把每期回归作为一个观测，自然考虑了横截面相关。
**量化中**：$\lambda_t$ 就是"因子收益率"，它的均值和 t 值是评价因子的核心指标之一，和 IC 分析互为补充。

</details>

---

## F. 协方差矩阵、PCA、正则化

> 对应讲解第 9、10 节。

### F1. 合法的相关矩阵 `中等` `经典·绿皮书`

下面的矩阵可以是相关矩阵吗？
$$\begin{pmatrix}1&0.9&0.9\\0.9&1&-0.9\\0.9&-0.9&1\end{pmatrix}$$

<details>
<summary>答案与解析</summary>

**答案**：不可以，它不是半正定的（特征值为 $-0.8,\,1.9,\,1.9$）

**解析**：直观上，A 与 B 高度正相关、A 与 C 高度正相关，那么 B 与 C 也应该正相关，不可能是 −0.9（见 A2：下界 $0.81-0.19=0.62$）。
严格验证：取 $w=(1,-1,-1)$，$w^\top Mw=3-2(0.9)-2(0.9)+2(-0.9)=3-1.8-1.8-1.8=-2.4<0$，所以不是半正定的。

**方法提示**：判断半正定，可以找一个让 $w^\top Mw<0$ 的向量，或算行列式和特征值。

</details>

### F2. 协方差矩阵半正定 `基础` `押题（高频）`

证明协方差矩阵一定是半正定的。什么时候不是正定的？

<details>
<summary>答案与解析</summary>

**解析**：对任何向量 $w$：$w^\top\Sigma w=\mathrm{Var}(w^\top X)\ge0$。
不是正定 ⇔ 存在非零 $w$ 使 $\mathrm{Var}(w^\top X)=0$ ⇔ 某个线性组合是常数 ⇔ 变量之间存在完全线性关系。
**量化例子**：多空完全对冲的组合、ETF 与其成分股组合、或样本量不足时的样本协方差矩阵（见 F3）。

</details>

### F3. 样本协方差矩阵的秩 `中等` `押题`

用 $T$ 天的收益数据估计 $N$ 只股票的样本协方差矩阵，$N>T$ 时会有什么问题？

<details>
<summary>答案与解析</summary>

**答案**：样本协方差矩阵的秩至多为 $T-1$，**奇异、不可逆**，无法直接做均值-方差优化。

**解析**：去均值后的数据矩阵是 $T\times N$，秩至多 $T-1$（去均值损失 1）。$S=\frac{1}{T-1}\tilde R^\top\tilde R$ 的秩不超过 $T-1<N$。
即使 $N<T$ 但二者接近，最小特征值也会严重低估，优化器会在这些方向上加大杠杆（**误差最大化**）。
**对策**：收缩估计（Ledoit-Wolf）、因子模型、PCA 降维、加约束。

</details>

### F4. 等相关矩阵的主成分 `中等` `押题`

$n$ 只股票方差都是 1，两两相关系数都是 $\rho>0$。求协方差矩阵的特征值，第一主成分是什么？它解释了多少方差？

<details>
<summary>答案与解析</summary>

**答案**：特征值 $1+(n-1)\rho$（1 个）和 $1-\rho$（$n-1$ 个）。第一主成分是等权组合方向 $\frac1{\sqrt n}(1,\dots,1)$，解释比例 $\frac{1+(n-1)\rho}{n}\to\rho$。

**解析**：$\Sigma=(1-\rho)I+\rho\mathbf{1}\mathbf{1}^\top$。$\mathbf 1\mathbf 1^\top$ 的特征值为 $n$（特征向量 $\mathbf 1$）和 0（其余正交方向）。
**量化含义**：第一主成分 = 市场因子。$\rho=0.3$、$n=1000$ 时，市场因子解释约 30% 的方差。

</details>

### F5. Ridge 与 Lasso 的闭式解 `困难` `经典·ISLR`

假设 $X^\top X=I$（自变量正交且标准化），OLS 解为 $\hat\beta^{OLS}$。求 Ridge（$\lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert_2^2$）和 Lasso（$\lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert_1$）的解。

<details>
<summary>答案与解析</summary>

**答案**：
- Ridge：$\hat\beta_j=\dfrac{\hat\beta_j^{OLS}}{1+\lambda}$（**等比例收缩**）
- Lasso：$\hat\beta_j=\mathrm{sign}(\hat\beta_j^{OLS})\Big(|\hat\beta_j^{OLS}|-\dfrac\lambda2\Big)_+$（**软阈值**）

**解析**：$X^\top X=I$ 时，目标函数可以拆成每个 $\beta_j$ 独立的一维问题：$(\beta_j-\hat\beta_j^{OLS})^2+\text{惩罚}$（加上与 $\beta$ 无关的常数）。
- Ridge：$(\beta_j-b)^2+\lambda\beta_j^2$，求导得 $\beta_j=\frac{b}{1+\lambda}$。
- Lasso：$(\beta_j-b)^2+\lambda|\beta_j|$。若 $b>\frac\lambda2$，解为 $b-\frac\lambda2$；若 $|b|\le\frac\lambda2$，解为 0。

**结论**：Lasso 会把小系数**直接压成 0**，所以能做变量选择；Ridge 只按比例缩小，不会产生 0。

</details>

### F6. 正则化前为什么要标准化 `基础` `押题`

做 Ridge 或 Lasso 之前，为什么要先把自变量标准化？

<details>
<summary>答案与解析</summary>

**答案**：惩罚项对所有系数一视同仁，而系数的大小取决于变量的单位。不标准化的话，单位小（数值大）的变量系数小、受惩罚少，单位大的变量受惩罚多，结果依赖于单位的选择，不合理。

</details>

### F7. 因子模型的参数个数 `基础` `押题`

3000 只股票，直接估计样本协方差矩阵需要多少个参数？用 40 个因子的 Barra 型模型（因子暴露由股票特征直接给出）呢？

<details>
<summary>答案与解析</summary>

**答案**：样本协方差：$\frac{3000\times3001}{2}=4501500$ 个；因子模型：因子协方差 $\frac{40\times41}{2}=820$ 个 + 特质方差 3000 个 = 3820 个。

**解析**：$\Sigma=B\Sigma_fB^\top+D$。Barra 模型中暴露 $B$ 由行业、市值、动量等特征直接计算，不需要估计；只需估计 $\Sigma_f$ 和对角阵 $D$。
**好处**：参数大幅减少，估计更稳定，而且矩阵一定正定（$D$ 对角元为正时）。

</details>

### F8. 收缩估计 `中等` `经典·Ledoit-Wolf`

什么是协方差矩阵的收缩估计？为什么有效？

<details>
<summary>答案与解析</summary>

**答案**：$\hat\Sigma=\delta F+(1-\delta)S$，$S$ 为样本协方差（无偏但方差大），$F$ 为结构化目标（如常相关矩阵、单因子模型、单位阵的倍数；有偏但方差小），$\delta\in[0,1]$ 为收缩强度。

**为什么有效**：偏差-方差权衡。Ledoit-Wolf 给出了使期望 Frobenius 范数误差最小的 $\delta$ 的估计公式。收缩把过大的特征值拉小、过小的特征值拉大，矩阵条件数改善，组合优化更稳定。

</details>

---

## G. 时间序列

> 对应讲解第 11 节。

### G1. AR(1) 的性质 `基础` `押题（高频）`

$X_t=0.9X_{t-1}+\varepsilon_t$，$\varepsilon_t$ 独立同分布、均值 0、方差 1。求平稳均值、方差、自相关函数和半衰期。

<details>
<summary>答案与解析</summary>

**答案**：均值 0；方差 $\frac{1}{1-0.81}\approx5.26$；$\rho(k)=0.9^k$；半衰期 $\frac{\ln0.5}{\ln0.9}\approx6.6$ 期

**解析**：
- 平稳时 $\mathrm{Var}(X_t)=\phi^2\mathrm{Var}(X_{t-1})+\sigma^2$，且两者相等，得 $\frac{\sigma^2}{1-\phi^2}$。
- $\mathrm{Cov}(X_t,X_{t-k})=\phi^k\mathrm{Var}(X)$。
- 偏离均值后，期望偏离按 $\phi^k$ 衰减，衰减一半需要 $k=\frac{\ln0.5}{\ln\phi}$。

**量化应用**：配对交易价差的半衰期决定持仓周期；半衰期太长（$\phi$ 接近 1）意味着回归很慢、资金占用久、风险大。

</details>

### G2. 平稳性判断 `基础` `押题`

下列哪些序列是平稳的？(a) 白噪声；(b) 随机游走 $X_t=X_{t-1}+\varepsilon_t$；(c) $X_t=0.5X_{t-1}+\varepsilon_t$；(d) $X_t=t+\varepsilon_t$；(e) 股票日收益率；(f) 股票价格。

<details>
<summary>答案与解析</summary>

**答案**：平稳：(a)(c)，(e) 近似平稳；非平稳：(b)(d)(f)。

**解析**：
- (b) $\mathrm{Var}(X_t)=t\sigma^2$ 随时间增大（单位根）；差分后平稳。
- (d) 均值随时间变化（确定性趋势）；去趋势后平稳。
- (e) 收益率的均值和方差大体稳定，但波动率聚集使其条件方差变化，严格来说只是近似平稳。

</details>

### G3. MA(1) 的自相关 `中等` `押题`

$X_t=\varepsilon_t+\theta\varepsilon_{t-1}$。求 1 阶自相关系数，它最大可能是多少？

<details>
<summary>答案与解析</summary>

**答案**：$\rho(1)=\dfrac{\theta}{1+\theta^2}$，最大值 $\frac12$（$\theta=1$ 时），$k\ge2$ 时 $\rho(k)=0$。

**解析**：$\mathrm{Var}=(1+\theta^2)\sigma^2$，$\mathrm{Cov}(X_t,X_{t-1})=\theta\sigma^2$。$\frac{\theta}{1+\theta^2}\le\frac12$ 由 $1+\theta^2\ge2\theta$ 得到。

**量化例子**：**买卖价差反弹**（bid-ask bounce）会让高频收益呈现负的 1 阶自相关，近似 MA(1)。

</details>

### G4. 用 ACF 和 PACF 识别模型 `基础` `经典·Tsay`

某序列的 ACF 缓慢指数衰减，PACF 在 2 阶后截尾。应该用什么模型？反过来呢？

<details>
<summary>答案与解析</summary>

**答案**：AR(2)。反过来（ACF 在 $q$ 阶后截尾，PACF 缓慢衰减）用 MA($q$)。两者都缓慢衰减考虑 ARMA。

</details>

### G5. GARCH 的长期波动率 `中等` `押题`

日收益的 GARCH(1,1) 参数为 $\omega=2\times10^{-6}$，$\alpha=0.1$，$\beta=0.88$。长期日波动率和年化波动率是多少？波动率冲击的持续性如何？

<details>
<summary>答案与解析</summary>

**答案**：长期日方差 $\frac{\omega}{1-\alpha-\beta}=\frac{2\times10^{-6}}{0.02}=10^{-4}$，日波动率 1%，年化约 15.9%。$\alpha+\beta=0.98$，持续性强，冲击的半衰期约 $\frac{\ln0.5}{\ln0.98}\approx34$ 天。

**解析**：GARCH 的方差预测 $E[\sigma_{t+k}^2]-\bar\sigma^2=(\alpha+\beta)^k(\sigma_t^2-\bar\sigma^2)$，按 $\alpha+\beta$ 的幂衰减回长期水平。

</details>

### G6. EWMA 波动率 `基础` `押题`

RiskMetrics 用 $\lambda=0.94$ 的指数加权移动平均估计日波动率。这相当于多长的"半衰期"？它和 GARCH 是什么关系？

<details>
<summary>答案与解析</summary>

**答案**：半衰期 $\frac{\ln0.5}{\ln0.94}\approx11.2$ 天；EWMA 是 $\omega=0$、$\alpha+\beta=1$ 的 GARCH(1,1)（IGARCH），没有均值回复，没有长期方差水平。

</details>

### G7. 相关与协整 `中等` `押题（高频）`

两只股票的日收益相关系数是 0.9，能否用它们做配对交易？相关性和协整有什么区别？

<details>
<summary>答案与解析</summary>

**答案**：不一定。配对交易需要**价格协整**，而不仅是收益相关。

**解析**：
- **相关**描述**收益**的短期同步性：今天一起涨、一起跌。
- **协整**描述**价格**的长期均衡：价差平稳、会回归。
- 收益高度相关的两只股票，价格仍可能长期分离（比如一只每年多涨 5%，价差会持续扩大）。反之，收益相关性不高的两个序列也可能协整。
**检验**：Engle-Granger（回归取残差做 ADF 检验，注意要用专门的临界值，因为 $\beta$ 是估计出来的）或 Johansen。

</details>

### G8. 重叠收益的自相关 `中等` `押题`

日收益独立同分布。每天计算一次"过去 20 日累计收益"，这个序列的 1 阶、10 阶、20 阶自相关系数各是多少？

<details>
<summary>答案与解析</summary>

**答案**：$\frac{19}{20}=0.95$；$\frac{10}{20}=0.5$；0

**解析**：两个窗口相隔 $j$ 天，共享 $20-j$ 个日收益，$\mathrm{Cov}=(20-j)\sigma^2$，方差 $20\sigma^2$。
**含义**：由重叠窗口构造的特征（如 20 日动量）和标签（未来 20 日收益）天然高度自相关，统计检验和交叉验证都要处理（见 E2）。

</details>

### G9. 方差比检验 `中等` `押题`

什么是方差比（Variance Ratio）检验？$VR(5)=0.8$ 说明什么？

<details>
<summary>答案与解析</summary>

**答案**：$VR(q)=\dfrac{\mathrm{Var}(q\text{ 期收益})}{q\cdot\mathrm{Var}(1\text{ 期收益})}$。随机游走下 $VR=1$。$VR<1$ 说明收益存在负自相关（均值回复），$VR>1$ 说明正自相关（动量）。$VR(5)=0.8$ 说明 5 日尺度上有均值回复。

**解析**：$VR(q)=1+2\sum_{k=1}^{q-1}\big(1-\frac kq\big)\rho(k)$，是自相关系数的加权和（Lo & MacKinlay 1988）。

</details>

### G10. 单位根检验 `基础` `押题`

ADF 检验的原假设是什么？得到 p 值 0.3，结论是什么？

<details>
<summary>答案与解析</summary>

**答案**：原假设是**存在单位根（非平稳）**。p = 0.3 不能拒绝原假设，没有证据表明序列平稳。

**注意**：ADF 的功效较低，$\phi=0.98$ 这种接近单位根的平稳序列也常常拒绝不了。可以配合 KPSS 检验（原假设是平稳）一起看。

</details>

---

## H. 量化应用与押题

> 对应讲解第 14 节。

### H1. IC 的显著性 `基础` `押题（高频）`

某因子 120 个月的月度 Rank IC 均值 0.04，标准差 0.10。ICIR 是多少？IC 均值显著吗？

<details>
<summary>答案与解析</summary>

**答案**：ICIR $=0.4$；$t=0.4\times\sqrt{120}\approx4.4$，显著

**解析**：$t=\frac{\overline{IC}}{s_{IC}/\sqrt T}=ICIR\times\sqrt T$。年化 ICIR（月频）$=0.4\times\sqrt{12}\approx1.39$。
**面试追问**：IC 显著不代表能赚钱，还要看：分层回测是否单调、多空收益扣费后是否为正、换手率、因子容量、与已有因子的相关性。

</details>

### H2. 主动管理基本定律 `中等` `经典·Grinold & Kahn`

一个选股模型的 IC 为 0.05，每月对 400 只股票做独立的预测。年化信息比率大约是多少？

<details>
<summary>答案与解析</summary>

**答案**：约 $0.05\times\sqrt{400\times12}\approx3.5$（理想情况）

**解析**：$IR\approx IC\times\sqrt{\text{Breadth}}$，Breadth 为每年的独立下注次数：$400\times12=4800$，$\sqrt{4800}\approx69$。
**现实折扣**：股票之间并不独立（行业、风格相关）、有交易成本和约束（转移系数 TC < 1），实际 IR 会显著低于这个数。但它说明了**为什么量化要做广度**：单只股票的预测能力很弱，靠大量独立下注积累优势。

</details>

### H3. Sharpe 比率的不确定性 `中等` `经典·Lo (2002)`

一个策略回测 5 年，年化 Sharpe 为 1.0。Sharpe 估计值的标准误大约是多少？

<details>
<summary>答案与解析</summary>

**答案**：约 0.55（年化单位）

**解析**：独立同分布正态收益下，$\mathrm{SE}(\widehat{SR})\approx\sqrt{\frac{1+SR^2/2}{T}}$，$SR$ 与 $T$ 用相同频率。以年为单位：$\sqrt{\frac{1.5}5}\approx0.55$。
所以 95% 置信区间大约是 $[-0.1,\ 2.1]$——5 年的回测几乎不能精确区分 Sharpe 0.5 和 1.5。
**注意**：收益存在负偏、厚尾或自相关时，标准误更大。

</details>

### H4. 参数搜索后的最大 Sharpe `困难` `经典·AFML` `押题`

你在 5 年的数据上回测了 200 组参数，最好的一组年化 Sharpe 为 2.0。假设所有参数组合其实都没有 alpha（真实 Sharpe 为 0），且相互独立，纯靠运气能得到的最大 Sharpe 期望是多少？2.0 可信吗？

<details>
<summary>答案与解析</summary>

**答案**：纯噪声下最大 Sharpe 的期望约 1.2。2.0 比这个高，但不能只凭这一点下结论。

**解析**：每个策略的 Sharpe 估计值近似 $N(0,\ 1/T)$，$T=5$ 年，标准差约 0.45。200 个独立标准正态的最大值期望约 2.75（粗略上界 $\sqrt{2\ln200}\approx3.26$）。所以纯噪声下的最大 Sharpe 约 $0.45\times2.75\approx1.23$。
2.0 约比这个高 1.7 个标准差，有一定证据，但：
1. 参数组合之间高度相关，有效尝试次数小于 200；
2. 但如果之前还尝试过别的策略思路，有效尝试次数可能远大于 200；
3. 需要看参数邻域是否平滑（参数稍变 Sharpe 是否崩塌）、样本外表现、经济逻辑。

**参考**：López de Prado 的 **Deflated Sharpe Ratio** 正是为修正这种"多次尝试后的最大值"而设计的。

</details>

### H5. 回测与实盘的落差 `中等` `押题（项目追问高频）`

回测年化 Sharpe 2.5，实盘只有 0.8。列出可能的原因。

<details>
<summary>答案与解析</summary>

**面试回答要点**（按"统计 → 数据 → 执行 → 市场"组织）：
1. **统计**：过拟合、多重检验、参数选择偏差；回测期太短或行情单一。
2. **数据**：前视偏差（财报实际发布时间、复权方式、指数成分）、幸存者偏差、数据清洗用了全样本信息。
3. **执行**：低估了交易成本（冲击成本随规模增大）、涨跌停和停牌无法成交、成交价假设过于乐观（用收盘价成交）、延迟。
4. **市场**：因子拥挤、市场结构变化（监管、参与者结构）、alpha 衰减。
5. **风险**：回测期没有出现过的风格暴露在实盘中亏损。

**加分**：说明你会怎么定位——拆分实盘收益和回测收益的差异（归因），看是选股层面、组合层面还是执行层面的问题。

</details>

### H6. 估计 Beta `基础` `押题`

股票年化波动率 30%，市场年化波动率 20%，二者相关系数 0.6。求该股票的 beta，以及收益方差中被市场解释的比例。

<details>
<summary>答案与解析</summary>

**答案**：$\beta=0.6\times\frac{0.30}{0.20}=0.9$；被市场解释的比例 $R^2=0.36$

**解析**：$\beta=\frac{\mathrm{Cov}(r_i,r_m)}{\mathrm{Var}(r_m)}=\rho\frac{\sigma_i}{\sigma_m}$，就是一元回归的斜率。系统性方差 $\beta^2\sigma_m^2=0.81\times0.04=0.0324$，占总方差 $0.09$ 的 36%，等于 $\rho^2$。
**追问**：估计 beta 时要注意窗口长度、频率（日频受非同步交易影响，小盘股 beta 会被低估）、以及向 1 收缩（Blume 调整：$0.67\hat\beta+0.33$）。

</details>

---

## 参考来源

- Wooldridge,《Introductory Econometrics》（D 部分、E 部分大多数题的背景）
- Casella & Berger,《Statistical Inference》（B 部分估计理论）
- James et al.,《An Introduction to Statistical Learning》（F5、F6 正则化）
- Tsay,《Analysis of Financial Time Series》（G 部分时间序列、E6 伪回归）
- Xinfeng Zhou,《A Practical Guide to Quantitative Finance Interviews》（A2、A3、F1 相关矩阵）
- López de Prado,《Advances in Financial Machine Learning》（E2 标签重叠、H4 回测过拟合）
- Harvey, Liu & Zhu (2016)（C12）；Lo (2002)（H3）；Lo & MacKinlay (1988)（G9）；Ledoit & Wolf (2004)（F8）；Grinold & Kahn《Active Portfolio Management》（H2）
- GitHub · [SoYuCry/awesome-quant-interview](https://github.com/SoYuCry/awesome-quant-interview)（Q5–Q12、Q29–Q40）
- 国内往年考点：衍复、幻方（回归斜率与 $R^2$）[一亩三分地](https://www.1point3acres.com/bbs/thread-993494-1-1.html)、[一亩三分地](https://www.1point3acres.com/bbs/thread-1097054-1-1.html)、[知乎](https://www.zhihu.com/question/313149375/answer/610309132)；诚奇（收益偏度）[知乎](https://www.zhihu.com/question/62031149)、[CSDN](https://blog.csdn.net/gitblog_09797/article/details/142943590)
