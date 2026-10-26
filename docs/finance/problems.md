# 金融常识题库

配套讲解见[讲解](notes.md)。代码均为 Python 3。

| 标签 | 含义 |
|---|---|
| `往年·公司名` | 公开面经中记录的往年笔试或面试题；面经只记录考点时，题干按考点整理并在题前注明 |
| `经典·书名` | 出自面试书或教材。Hull = *Options, Futures, and Other Derivatives*（第 10 版）；Shreve = *Stochastic Calculus for Finance II*；Grinold & Kahn = *Active Portfolio Management*；Barra USE4 = MSCI USE4 方法论文档；论文以“作者 年份”标注；其后数字为章节号 |
| `押题` | 依据往年考点整理的高频题型 |
| `基础` `中等` `困难` | 难度 |

---

## A. A 股交易制度

> 对应讲解第 1 节。

### A1. T+1 的含义与例外 `基础` `押题`

(1) 什么是 T+1？(2) 卖出股票的资金当天能否再买入？(3) 哪些品种可以 T+0？

<details>
<summary>答案与解析</summary>

(1) 当日买入的股票，次一交易日才能卖出。
(2) 可以。卖出所得资金当日可用于买入（但当日不能转出到银行）。
(3) 跨境 ETF、债券 ETF、黄金 ETF、货币 ETF、可转债实行 T+0；股指期货、商品期货也可以日内开平仓。

**追问**：T+1 下怎么做日内交易？——用昨日底仓先卖后买（或先买后卖），收盘时仓位不变，即 T0 策略（F7）。

</details>

### A2. 各板块涨跌幅限制 `基础` `押题`

写出沪深主板、主板 ST 股、创业板、科创板、北交所的日涨跌幅限制，以及新股上市初期的规定。

<details>
<summary>答案与解析</summary>

| 板块 | 日涨跌幅 | 新股 |
|---|---|---|
| 沪深主板 | ±10% | 上市前 5 个交易日不设涨跌幅（2023-04-10 起） |
| 主板 ST、*ST | 原为 ±5%，2026-07-06 起为 ±10% | — |
| 创业板、科创板 | ±20% | 上市前 5 个交易日不设涨跌幅 |
| 北交所 | ±30% | 上市首日不设涨跌幅 |

涨跌停价按前收盘价乘以 $(1\pm\text{幅度})$ 后四舍五入到 0.01 元。

</details>

### A3. 计算涨跌停价 `基础` `押题`

写函数：输入前收盘价和板块（`main`、`st`、`gem`、`bse`），返回涨停价和跌停价（四舍五入到分）。分别计算前收盘价 5.35 元的主板股票和 9.99 元的创业板股票。注意浮点误差。

<details>
<summary>答案与解析</summary>

```python
from decimal import Decimal, ROUND_HALF_UP

LIMIT = {"main": "0.10", "st": "0.10", "gem": "0.20", "bse": "0.30"}   # st 按 2026-07-06 新规

def price_limits(prev_close, board):
    p, r = Decimal(str(prev_close)), Decimal(LIMIT[board])
    q = Decimal("0.01")
    up = (p * (1 + r)).quantize(q, rounding=ROUND_HALF_UP)
    down = (p * (1 - r)).quantize(q, rounding=ROUND_HALF_UP)
    return float(up), float(down)

print(price_limits(5.35, "main"), price_limits(9.99, "gem"))
print(round(5.35 * 1.1, 2), 5.35 * 1.1)                  # 浮点直接计算
```

输出：

```
(5.89, 4.82) (11.99, 7.99)
5.88 5.885
```

 主板涨停价 5.89（$5.35\times1.1=5.885$）、跌停价 4.82（$4.815$）；创业板涨停价 11.99、跌停价 7.99。

**注意**：直接用浮点数计算，`5.35 * 1.1` 在二进制下略小于 5.885，`round` 得到 5.88，与交易所规则不符（Python 的 `round` 对恰好为 5 的情形还采用"银行家舍入"）。涉及价格的计算应使用 `Decimal` 或以"分"为单位的整数。

</details>

### A4. 集合竞价的撮合规则 `中等` `押题`

(1) 写出沪深交易所开盘、收盘集合竞价的时段与撤单规则。(2) 集合竞价按什么原则确定成交价？(3) 给定买卖申报，写函数求开盘价。

<details>
<summary>答案与解析</summary>

(1) 开盘集合竞价 9:15–9:25，其中 9:15–9:20 可撤单，9:20–9:25 不可撤单；收盘集合竞价 14:57–15:00，不可撤单。
(2) 成交量最大原则：在所有候选价格中，选使成交量最大的价格；成交量相同时，选未成交量（买卖不平衡）最小的价格；仍相同时，取最接近前收盘价（或中间价）的价格。所有成交按同一价格。

```python
def call_auction(bids, asks, prev_close):
    """bids/asks: [(价格, 数量)]；返回 (成交价, 成交量)"""
    prices = sorted({p for p, _ in bids} | {p for p, _ in asks})
    best = None
    for p in prices:
        demand = sum(q for bp, q in bids if bp >= p)      # 愿意以不低于 p 买入
        supply = sum(q for ap, q in asks if ap <= p)      # 愿意以不高于 p 卖出
        vol, imb = min(demand, supply), abs(demand - supply)
        key = (vol, -imb, -abs(p - prev_close))
        if best is None or key > best[0]:
            best = (key, p, vol)
    return best[1], best[2]

bids = [(10.05, 300), (10.03, 500), (10.00, 400), (9.98, 200)]
asks = [(9.97, 200), (10.00, 300), (10.02, 400), (10.04, 600)]
print(call_auction(bids, asks, prev_close=10.00))
```

输出：

```
(10.02, 800)
```

 开盘价 10.02，成交 800 股：价格 10.02 时买方需求为 300+500=800（10.05、10.03 两档），卖方供给为 200+300+400=900，成交量 800，是所有候选价格中最大的。

</details>

### A5. 交易成本估算 `基础` `押题`

某机构股票账户佣金为万分之 1.2（含经手费，双向），过户费为成交金额的 0.001%（双向），印花税为卖出金额的 0.05%。(1) 买入并卖出一次的显性成本是多少？(2) 某策略单边日换手 15%，一年 242 个交易日，年显性成本是多少？

<details>
<summary>答案与解析</summary>

```python
commission, transfer, stamp = 1.2e-4, 1e-5, 5e-4
round_trip = 2 * (commission + transfer) + stamp
daily_turnover, days = 0.15, 242
annual = round_trip * daily_turnover * days
print(f"买卖一次 {round_trip * 1e4:.2f} bp，年显性成本 {annual:.2%}")
```

输出：

```
买卖一次 7.60 bp，年显性成本 2.76%
```

 买卖一次约 7.6 bp；单边日换手 15% 意味着每天有 15% 的仓位完成一次"卖出 + 买入"，年显性成本约 2.76%。再加上冲击成本（G 章），总成本通常是显性成本的 1–2 倍。

</details>

### A6. 印花税调整对量化的影响 `基础` `押题`

2023-08-28 印花税从 0.1% 减半到 0.05%，对高换手量化策略有什么影响？

<details>
<summary>答案与解析</summary>

- 买卖一次的显性成本从约 12.6 bp 降到约 7.6 bp（按 A5 的佣金假设），降幅约 40%。
- 对单边日换手 15% 的策略，年成本降低约 $0.05\%\times0.15\times242\approx1.8\%$。
- 高换手策略（T0、高频、短周期选股）受益最大：原来因成本无法覆盖的弱信号变得可用，最优换手率上升。
- 降低成本也会吸引更多资金进入同类策略，提高拥挤度。

</details>

### A7. 回测中的涨跌停与停牌处理 `中等` `押题`

日频回测中应该如何处理涨停、跌停和停牌？如果忽略会有什么后果？

<details>
<summary>答案与解析</summary>

| 情形 | 处理 | 忽略的后果 |
|---|---|---|
| 目标买入的股票开盘即涨停（一字板） | 无法买入，保留现金或顺延 | 收益虚高：涨停股次日往往继续上涨 |
| 目标卖出的股票跌停 | 无法卖出，继续持有 | 低估亏损：跌停股往往继续下跌 |
| 停牌 | 不能交易，按停牌前价格估值，复牌后按复牌价成交 | 低估风险：复牌后可能补跌 |

**细节**：
- 判断"能否成交"要用**交易时点**的状态：以开盘价成交就看开盘是否涨停；以 VWAP 成交就看全天是否有成交机会（例如是否全天封板）。
- 不能用当天收盘才知道的信息（例如"今天收盘涨停"）来决定今天是否买入，否则引入前视偏差。

</details>

### A8. 融资融券与转融券 `基础` `押题`

(1) 融资、融券、转融券分别是什么？(2) 为什么国内量化产品主要用股指期货而不是融券对冲？

<details>
<summary>答案与解析</summary>

(1) 融资：投资者向券商借钱买股票。融券：向券商借股票卖出，到期买回归还。转融券：证金公司从机构借入股票，再出借给券商用于融券。
(2) 原因：
- 券源少、成本高、期限短，不能覆盖数千只股票的空头需求；
- 2023-10 起融券保证金比例由 50% 提高到 80%，2024 年起限售股出借暂停、转融券暂停（2024-07-11 起），融券保证金比例提高到 100%（私募 120%）；
- 股指期货流动性好、成本低，可以对冲整体市场风险，但只能对冲指数，不能对冲个股。

</details>

### A9. 程序化交易监管 `中等` `押题`

(1) 程序化交易监管的主要文件和施行时间？(2) 高频交易的认定标准？(3) 对量化机构有什么影响？

<details>
<summary>答案与解析</summary>

(1) 证监会《证券市场程序化交易管理规定（试行）》2024-05-15 发布、2024-10-08 施行；沪深北交易所《程序化交易管理实施细则》2025-04-03 发布、2025-07-07 施行。
(2) 单账户每秒申报、撤单合计最高达到 300 笔以上，或单日申报、撤单合计最高达到 20000 笔以上。
(3) 影响：
- 须事先报告账户和策略信息，高频交易报告要求更严、收费更高；
- 交易所重点监控瞬时申报速率异常、频繁瞬时撤单、频繁拉抬打压、短时间大额成交等行为；
- 高频和 T0 策略的成本上升、容量受限，策略向中低频迁移；
- 开盘、收盘集合竞价阶段的大额集中申报受到更多关注（参考灵均事件，I2）。

</details>

### A10. 对股市的看法 `基础` `往年·诚奇`

> 诚奇笔试版本一包含"对股市看法"一题。面经未给出具体题干，以下为作答框架。

谈谈你对 A 股市场的看法。

<details>
<summary>答案与解析</summary>

开放题，避免空谈涨跌，从量化研究者的角度组织：

| 维度 | 可谈的要点 |
|---|---|
| 市场结构 | 散户成交占比高，非理性交易多，这是量价因子 alpha 的重要来源；机构化程度在提高，alpha 逐渐衰减 |
| 制度 | T+1、涨跌停、做空受限，导致定价效率与美股不同；股指期货长期贴水 |
| 监管趋势 | 新国九条、程序化交易监管、融券收紧，量化的边界更清晰，高频空间收窄 |
| 风格 | 大小盘风格切换剧烈（2024 年初微盘股危机），风格控制是指增的核心 |
| 结论 | 量化需要在 alpha 衰减和监管趋严的环境下，靠更多数据、更好的模型和更严格的风控取得超额 |

回答时给出一两个具体例子（数据、事件、日期），比泛泛而谈更有说服力。

</details>

---

---

## B. 金融工具与期货基差

> 对应讲解第 2 节。

### B1. 股票、ETF、期货、期权的区别 `基础` `押题`

从本质、杠杆、做空、量化用途四个方面比较。

<details>
<summary>答案与解析</summary>

| 工具 | 本质 | 杠杆 | 做空 | 量化用途 |
|---|---|---|---|---|
| 股票 | 公司所有权 | 无（融资除外） | 受限 | 选股策略多头端 |
| ETF | 跟踪指数的场内基金 | 无 | 部分可融券 | 指数暴露、ETF 套利、部分品种 T+0 |
| 期货 | 标准化远期合约，保证金交易，每日结算 | 有 | 自由 | 对冲、CTA |
| 期权 | 权利而非义务 | 天然杠杆 | 可卖出 | 波动率交易、做市、尾部对冲 |

期货与远期的区别：期货标准化、交易所交易、每日盯市结算、有保证金，信用风险小；远期为场外合约，无每日结算。

</details>

### B2. 股指期货品种与合约价值 `基础` `押题`

(1) 列出 IF、IH、IC、IM 的标的和合约乘数。(2) IM 报价 6200 点时一手合约价值多少？(3) 一个 1 亿元、相对中证 1000 的 Beta 为 1.05 的股票组合需要做空多少手 IM？

<details>
<summary>答案与解析</summary>

(1) IF：沪深 300，每点 300 元；IH：上证 50，每点 300 元；IC：中证 500，每点 200 元；IM：中证 1000，每点 200 元（2022-07-22 上市）。

```python
V, beta, price, mult = 1e8, 1.05, 6200, 200
notional = price * mult
print(f"一手价值 {notional / 1e4:.0f} 万元，需做空 {beta * V / notional:.1f} 手")
```

输出：

```
一手价值 124 万元，需做空 84.7 手
```

 一手 124 万元，需做空约 84.7 手，实际取 85 手。

</details>

### B3. 持有成本模型 `中等` `经典·Hull 5`

(1) 推导无股息时远期价格 $F=Se^{rT}$。(2) 有连续股息率 $q$ 时呢？(3) 若市场上 $F>Se^{(r-q)T}$，如何套利？

<details>
<summary>答案与解析</summary>

(1) 构造组合：借入 $S$ 买入股票，同时做空远期。到期交割股票得 $F$，偿还借款 $Se^{rT}$。期初成本为 0，到期收益 $F-Se^{rT}$ 是确定的，无套利要求其为 0。
(2) 持有股票期间收到股息，相当于持有成本降低，$F=Se^{(r-q)T}$。
(3) **正向套利**（期现套利）：借钱买入现货（或 ETF），做空期货，到期交割或平仓，锁定 $F-Se^{(r-q)T}$ 的利润。反之 $F$ 过低时做反向套利：融券卖出现货、买入期货，但 A 股融券受限，所以反向套利难以执行，这也是贴水长期存在的原因之一。

</details>

### B4. 基差与年化贴水 `基础` `押题`

IM 当月合约剩余 25 个自然日，期货 5880 点，现货指数 5950 点。(1) 基差和年化贴水率是多少？(2) 做空 1 手并持有到期，若到期时指数不变，空头盈亏多少？

<details>
<summary>答案与解析</summary>

```python
F, S, days, mult = 5880, 5950, 25, 200
basis = F - S
ann = (S - F) / S * 365 / days
print(f"基差 {basis} 点，年化贴水 {ann:.2%}，空头到期盈亏 {(F - S) * mult:.0f} 元")
```

输出：

```
基差 -70 点，年化贴水 17.18%，空头到期盈亏 -14000 元
```

 基差 −70 点，年化贴水约 17.2%。到期时期货收敛到现货 5950，空头在 5880 卖出、5950 买回，亏损 70 点 × 200 = 14000 元。这就是中性产品的对冲成本。

</details>

### B5. 股指期货为什么长期贴水 `中等` `押题`

国内股指期货（尤其 IC、IM）为什么长期贴水？贴水对指增和中性产品分别有什么影响？

<details>
<summary>答案与解析</summary>

**原因**：
1. **分红**：5–8 月是分红集中期，期货价格要扣除合约期内的预期分红，这部分贴水是"理论贴水"。
2. **对冲需求**：中性产品、雪球等结构化产品的对冲盘长期做空期货，压低期货价格。
3. **套利受限**：贴水时的反向套利需要融券卖出现货，A 股融券困难，套利力量无法把基差拉回理论值。
4. **情绪**：市场下跌、恐慌时贴水加深（2024 年初）。

**影响**：
- **中性产品**：做空贴水的期货，到期时空头亏损贴水部分，是对冲成本；贴水越深，新建仓成本越高。
- **指增产品**：可以用"买入贴水期货 + 现金管理"代替现货持有指数（期货替代），赚取贴水收敛的收益，相当于额外增强。

</details>

### B6. 期货展期 `中等` `押题`

中性产品做空当月合约，到期前需要展期（平当月、开下月）。若下月合约贴水更深，对产品有什么影响？如何选择对冲合约？

<details>
<summary>答案与解析</summary>

- 展期时平掉当月空头、开下月空头。下月合约的贴水更深，意味着新空头的建仓价格更低，未来收敛时的亏损更大，即对冲成本更高；展期本身的价差就是一笔交易成本。
- **合约选择**：比较各合约的年化贴水率，选择年化成本最低的合约（考虑流动性和冲击成本）；远月合约年化贴水通常较低，但流动性差、换月次数少。
- 季节性：分红季前远月贴水含分红因素，比较时应扣除预期分红。
- 也可以动态择时：贴水较深时降低对冲比例或改用期权，但这引入了方向性风险。

</details>

### B7. 期权基本术语 `基础` `经典·Hull 10`

解释：看涨、看跌、欧式、美式、实值、平值、虚值、内在价值、时间价值。行权价 100 的看涨期权，标的价格 105、期权价格 7，内在价值和时间价值分别是多少？

<details>
<summary>答案与解析</summary>

| 术语 | 含义 |
|---|---|
| 看涨 / 看跌 | 以行权价买入 / 卖出标的的权利 |
| 欧式 / 美式 | 只能到期行权 / 到期前任何时间可以行权 |
| 实值 / 平值 / 虚值 | 立即行权收益为正 / $S\approx K$ / 立即行权收益为 0 |
| 内在价值 | $\max(S-K,0)$（看涨） |
| 时间价值 | 期权价格 − 内在价值 |

例：内在价值 $105-100=5$，时间价值 $7-5=2$。

</details>

### B8. 国内场内期权品种 `基础` `押题`

列出国内主要的金融期权品种和上市时间。

<details>
<summary>答案与解析</summary>

| 品种 | 交易所 | 上市时间 |
|---|---|---|
| 50ETF 期权 | 上交所 | 2015-02-09 |
| 沪深 300ETF 期权 | 上交所、深交所 | 2019-12-23 |
| 沪深 300 股指期权 | 中金所 | 2019-12-23 |
| 中证 500ETF 期权 | 上交所、深交所 | 2022-09-19 |
| 中证 1000 股指期权 | 中金所 | 2022-07-22 |
| 上证 50 股指期权 | 中金所 | 2022-12-19 |

ETF 期权为欧式、实物交割；股指期权为欧式、现金交割。此外商品交易所有大量商品期权。

</details>

### B9. 期货保证金与杠杆 `基础` `押题`

期货保证金比例为 12%，合约价值上涨 3%，多头保证金的收益率是多少？若保证金不足会发生什么？

<details>
<summary>答案与解析</summary>

杠杆约为 $1/0.12\approx8.3$ 倍，保证金收益率约 $3\%/12\%=25\%$；同理下跌 3% 亏损 25%。

期货每日结算（盯市）：当日盈亏计入保证金账户。保证金低于维持水平时，期货公司要求追加保证金；不能追加时强行平仓。中性产品因此必须预留现金（股票端通常只占资金的 75%–85%），以应对市场大涨时空头端的亏损。

</details>

---

---

## C. 收益与风险指标

> 对应讲解第 3 节。

### C1. 计算年化 Sharpe `基础` `押题`

给定日收益序列（pandas Series），无风险利率年化 2%，写函数计算年化 Sharpe。用模拟数据（年化收益 15%、年化波动 20%，共 5 年）验证。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
import pandas as pd

def annual_sharpe(ret, rf=0.02, periods=252):
    excess = ret - rf / periods
    return excess.mean() / excess.std(ddof=1) * np.sqrt(periods)

rng = np.random.default_rng(0)
ret = pd.Series(rng.normal(0.15 / 252, 0.20 / np.sqrt(252), 252 * 5))
print(f"样本 Sharpe {annual_sharpe(ret):.3f}，理论值 {(0.15 - 0.02) / 0.20:.3f}")
```

输出：

```
样本 Sharpe 0.078，理论值 0.650
```

 理论值 0.65。样本 Sharpe 与理论值的差异来自估计误差：5 年日频数据的年化 Sharpe 标准误约 $\sqrt{252/1260}\approx0.45$（C4），该样本值 0.078 与理论值相差约 1.3 个标准误。5 年的回测都不足以精确估计 Sharpe。

</details>

### C2. 计算最大回撤 `基础` `押题`

写函数：输入净值序列，返回最大回撤、回撤开始日（峰值）和最低点日期。要求 $O(n)$。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
import pandas as pd

def max_drawdown(nav):
    peak = nav.cummax()
    dd = 1 - nav / peak
    trough = dd.idxmax()
    start = nav.loc[:trough].idxmax()
    return dd.max(), start, trough

nav = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 1.4],
                index=pd.date_range("2024-01-01", periods=7))
mdd, s, t = max_drawdown(nav)
print(f"{mdd:.4f} {s.date()} {t.date()}")
```

输出：

```
0.2500 2024-01-02 2024-01-03
```

 最大回撤 25%，从 1.2（01-02）跌到 0.9（01-03）。1.3 到 1.0 的回撤为 23.1%，较小。

**注意**：回撤用净值（累计收益）计算，而不是对日收益求和；累计收益应用 `(1 + r).cumprod()`。

</details>

### C3. 信息比率与跟踪误差 `基础` `押题`

某指增产品与基准的日收益已知，写函数计算年化超额收益、年化跟踪误差和信息比率。模拟：日超额均值 0.06%、日超额标准差 0.4%，共 3 年。

<details>
<summary>答案与解析</summary>

```python
import numpy as np

def ir_stats(rp, rb, periods=252):
    active = rp - rb
    ann_ex = active.mean() * periods
    te = active.std(ddof=1) * np.sqrt(periods)
    return ann_ex, te, ann_ex / te

rng = np.random.default_rng(1)
rb = rng.normal(0.0003, 0.015, 252 * 3)
rp = rb + rng.normal(0.0006, 0.004, 252 * 3)
ex, te, ir = ir_stats(rp, rb)
print(f"年化超额 {ex:.2%}，跟踪误差 {te:.2%}，IR {ir:.2f}")
print(f"理论：超额 {0.0006 * 252:.2%}，跟踪误差 {0.004 * np.sqrt(252):.2%}，IR {0.0006 / 0.004 * np.sqrt(252):.2f}")
```

输出：

```
年化超额 13.31%，跟踪误差 6.64%，IR 2.00
理论：超额 15.12%，跟踪误差 6.35%，IR 2.38
```

 理论年化超额 15.12%、跟踪误差 6.35%、IR 2.38；3 年样本的估计值为 13.31%、6.64%、2.00，IR 的标准误约 $1/\sqrt3\approx0.58$，差异在误差范围内。

</details>

### C4. Sharpe 的标准误 `中等` `经典·Lo 2002`

(1) 收益独立同分布、正态时，Sharpe 估计量的渐近标准误是多少？(2) 年化 Sharpe 为 1，需要多少年数据才能在 5% 水平上显著大于 0？(3) 用模拟验证 (1)。

<details>
<summary>答案与解析</summary>

(1) Lo (2002)：$\widehat{\mathrm{SR}}$ 的渐近方差为 $(1+\mathrm{SR}^2/2)/n$，$\mathrm{SR}$ 为单期 Sharpe、$n$ 为期数。年化后标准误约为 $\sqrt{(1+\mathrm{SR}_{\text{日}}^2/2)\,252/n}\approx1/\sqrt{\text{年数}}$。

(2) 单边检验 $\mathrm{SR}_{\text{年}}\sqrt{T}>1.645$，即 $T>2.7$ 年；双边 $1.96$ 时 $T>3.8$ 年。

```python
import numpy as np

rng = np.random.default_rng(0)
sr_d, n, sims = 0.1, 500, 4000                     # 单期 Sharpe 0.1，500 期
est = []
for _ in range(sims):
    r = rng.normal(sr_d, 1, n)
    est.append(r.mean() / r.std(ddof=1))
print(f"模拟标准误 {np.std(est):.4f}，公式 {np.sqrt((1 + sr_d ** 2 / 2) / n):.4f}")
print(f"年化 Sharpe 1 时显著所需年数：单边 {1.645 ** 2:.1f}，双边 {1.96 ** 2:.1f}")
```

输出：

```
模拟标准误 0.0450，公式 0.0448
年化 Sharpe 1 时显著所需年数：单边 2.7，双边 3.8
```

 模拟标准误与公式一致。结论：短期回测的 Sharpe 噪声很大，1 年的业绩几乎不能说明问题。

</details>

### C5. 收益自相关与 Sharpe 年化 `中等` `经典·Lo 2002`

日收益存在一阶自相关 $\rho$ 时，用 $\sqrt{252}$ 年化 Sharpe 会有什么偏差？用 AR(1) 模拟比较"日 Sharpe × $\sqrt{252}$"与"年收益的真实 Sharpe"。

<details>
<summary>答案与解析</summary>

$q$ 期收益之和的方差为 $\sigma^2\big[q+2\sum_{k=1}^{q-1}(q-k)\rho^k\big]$。$\rho>0$ 时方差大于 $q\sigma^2$，所以 $\sqrt q$ 年化会**高估** Sharpe；$\rho<0$ 时低估。

```python
import numpy as np

def true_ratio(rho, q=252):
    k = np.arange(1, q)
    return np.sqrt(q / (q + 2 * np.sum((q - k) * rho ** k)))

rng = np.random.default_rng(0)
for rho in [0.0, 0.1, 0.3]:
    yearly = []
    for _ in range(3000):
        e = rng.standard_normal(252)
        x = np.empty(252); x[0] = e[0] / np.sqrt(1 - rho ** 2)
        for t in range(1, 252):
            x[t] = rho * x[t - 1] + e[t]
        yearly.append(x.sum())
    var_ratio = np.var(yearly) / (252 / (1 - rho ** 2))
    print(f"rho={rho}: 年收益方差 / (252×日方差) = {var_ratio:.2f}，"
          f"sqrt(252) 年化高估倍数（理论）{1 / true_ratio(rho):.2f}")
```

输出：

```
rho=0.0: 年收益方差 / (252×日方差) = 1.01，sqrt(252) 年化高估倍数（理论）1.00
rho=0.1: 年收益方差 / (252×日方差) = 1.19，sqrt(252) 年化高估倍数（理论）1.11
rho=0.3: 年收益方差 / (252×日方差) = 1.81，sqrt(252) 年化高估倍数（理论）1.36
```

 $\rho=0.1$ 时年收益方差约为 $252\sigma^2$ 的 1.2 倍（模拟 1.19，理论 1.22），用 $\sqrt{252}$ 年化高估约 11%；$\rho=0.3$ 时约 1.8 倍，高估约 36%。流动性差的资产（私募股权、对冲基金的平滑净值）收益常有正自相关，报告的 Sharpe 偏高。

</details>

### C6. 换手率的计算 `基础` `押题`

给定每日目标权重矩阵（日期 × 股票），写函数计算日单边换手率。考虑股价变动导致的权重漂移。

<details>
<summary>答案与解析</summary>

调仓前的实际权重是昨日权重随当日收益漂移后的结果：$\tilde w_{i,t}=w_{i,t-1}(1+r_{i,t})/\sum_jw_{j,t-1}(1+r_{j,t})$。单边换手 $=\frac12\sum_i\lvert w_{i,t}-\tilde w_{i,t}\rvert$。

```python
import numpy as np
import pandas as pd

def turnover(w, ret):
    drift = w.shift(1) * (1 + ret)
    drift = drift.div(drift.sum(axis=1), axis=0)
    return 0.5 * (w - drift).abs().sum(axis=1).iloc[1:]

idx = pd.date_range("2024-01-01", periods=3)
w = pd.DataFrame([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]], index=idx, columns=list("abc"))
ret = pd.DataFrame([[0.0, 0.0, 0.0], [0.10, -0.10, 0.0], [0.0, 0.0, 0.0]], index=idx, columns=list("abc"))
print(turnover(w, ret).round(4).tolist())
```

输出：

```
[0.05, 0.5]
```

 第 2 天：股票 a 涨 10%、b 跌 10%，漂移后权重为 0.55 / 0.45，调回 0.5 / 0.5 需要换手 5%；第 3 天卖出 a、买入 c，换手 50%。不考虑漂移会把第 2 天的换手算成 0。

</details>

### C7. 几何年化与算术年化 `基础` `押题`

某产品两年收益分别为 +50% 和 −50%。(1) 算术平均和几何平均年化收益各是多少？(2) 解释"波动拖累"$\approx\sigma^2/2$。

<details>
<summary>答案与解析</summary>

(1) 算术平均 0%；几何平均 $\sqrt{1.5\times0.5}-1=-13.4\%$，两年后净值为 0.75。
(2) 对数收益 $\ln(1+R)\approx R-R^2/2$，取期望得 $\mathbb E[\ln(1+R)]\approx\mu-(\sigma^2+\mu^2)/2\approx\mu-\sigma^2/2$。几何收益约等于算术收益减去方差的一半，波动越大，长期复合收益越低。这也是杠杆不能无限加的原因之一：杠杆 $L$ 下几何收益约为 $L\mu-L^2\sigma^2/2$，在 $L^*=\mu/\sigma^2$ 时最大（Kelly 准则）。

</details>

### C8. Sortino 与 Calmar `基础` `押题`

写出 Sortino 比率和 Calmar 比率的定义。什么情况下它们比 Sharpe 更合适？

<details>
<summary>答案与解析</summary>

- **Sortino**：$\dfrac{\mathbb E[R]-\text{MAR}}{\sigma_{\text{下行}}}$，$\sigma_{\text{下行}}=\sqrt{\mathbb E[\min(R-\text{MAR},0)^2]}$，只惩罚下行波动。
- **Calmar**：年化收益 / 最大回撤（通常取近 3 年）。

Sharpe 把上行和下行波动同等对待，假设收益对称。收益分布偏斜时（例如卖期权策略：大部分时间小赚、偶尔大亏，负偏），Sharpe 会高估其质量，Sortino、Calmar 和偏度、峰度更能反映尾部风险。反之，买期权等正偏策略的 Sharpe 会被低估。

</details>

### C9. 用回测指标评价策略 `中等` `押题`

一个策略：年化收益 30%、Sharpe 3.5、最大回撤 4%、单边日换手 40%、回测期 2 年、主要持有中证 2000 成分股。你会如何评价？需要追问哪些问题？

<details>
<summary>答案与解析</summary>

| 关注点 | 问题 |
|---|---|
| 成本 | 收益是扣费前还是扣费后？单边日换手 40% 的显性成本约每年 $7.6\text{bp}\times0.4\times242\approx7.4\%$，再加冲击成本 |
| 容量 | 小盘股流动性差，按成交量占比 5% 估算能容纳多少资金？ |
| 成交假设 | 用什么价格成交？是否处理涨跌停、停牌？ |
| 样本长度 | 2 年回测的 Sharpe 标准误约 0.7（C4），且只覆盖一种市场环境 |
| 风格暴露 | 是否大量暴露于小市值、低流动性？2024 年初这类暴露回撤巨大 |
| 过拟合 | 尝试了多少参数组合？样本外表现如何？ |
| 收益来源 | 超额来自选股还是风格？做归因 |

**结论**：高 Sharpe、高换手、小盘的组合往往容量很小、对成本假设极其敏感；需要扣费后、样本外、考虑容量的结果才能判断价值。

</details>

### C10. 胜率与盈亏比 `基础` `押题`

某策略胜率 40%，平均盈利是平均亏损的 2 倍。每次交易的期望收益为正吗？要使期望为正，胜率 40% 时盈亏比至少多少？

<details>
<summary>答案与解析</summary>

期望 $=0.4\times2L-0.6\times L=0.2L>0$（$L$ 为平均亏损），为正。
盈亏比 $b$ 满足 $0.4b-0.6>0$，即 $b>1.5$。一般地，胜率 $p$ 时盈亏比须大于 $(1-p)/p$。

趋势跟踪（CTA）通常胜率低、盈亏比高；均值回复和做市通常胜率高、盈亏比低。

</details>

### C11. 收益偏度的计算与含义 `基础` `往年·诚奇`

> 诚奇笔试版本二：编程计算股票收益偏度并解释正负。完整实现见[编程与算法题库](../programming/problems.md) M10，此处侧重金融含义。

偏度为正和为负分别说明什么？哪些策略或资产通常是负偏的？

<details>
<summary>答案与解析</summary>

样本偏度 $\hat\gamma=\dfrac{\frac1n\sum(r_i-\bar r)^3}{\big(\frac1n\sum(r_i-\bar r)^2\big)^{3/2}}$。

- **正偏**：右尾长，偶尔出现大幅上涨。例如小盘股、彩票型股票、买入期权、趋势跟踪策略。
- **负偏**：左尾长，偶尔大幅下跌。例如股票指数（暴跌比暴涨更多见）、卖出期权、套利与中性策略（平时稳定、危机时回撤，如 2024 年初）、高杠杆策略。
- **量化含义**：投资者偏好正偏资产，愿意为"彩票"支付溢价，所以高偏度股票未来收益偏低（特质偏度因子）；负偏策略的 Sharpe 会高估其质量（C8）。

</details>

### C12. 年化波动率的换算 `基础` `押题`

年化波动 16%，日波动约多少？周、月呢？日波动 2% 的股票，年化波动约多少？

<details>
<summary>答案与解析</summary>

按 $\sqrt t$ 缩放（假设收益独立）：日波动 $16\%/\sqrt{252}\approx1.0\%$（$\sqrt{256}=16$ 便于心算）；周 $16\%/\sqrt{52}\approx2.2\%$；月 $16\%/\sqrt{12}\approx4.6\%$。日波动 2% 时年化约 $2\%\times15.9\approx31.7\%$。

</details>

---

## D. CAPM、多因子与 Barra

> 对应讲解第 4 节。

### D1. CAPM 与 Beta `基础` `经典·Hull`

(1) 写出 CAPM。(2) Beta 的定义和估计方法？(3) 用模拟数据估计 Beta 和 Jensen's alpha。

<details>
<summary>答案与解析</summary>

(1) $\mathbb E[R_i]-R_f=\beta_i(\mathbb E[R_m]-R_f)$。
(2) $\beta_i=\operatorname{Cov}(R_i,R_m)/\operatorname{Var}(R_m)$；用超额收益做时间序列 OLS：$R_i-R_f=\alpha_i+\beta_i(R_m-R_f)+\varepsilon$。

```python
import numpy as np

rng = np.random.default_rng(0)
n = 750
rm = rng.normal(0.0004, 0.012, n)
ri = 0.0002 + 1.3 * rm + rng.normal(0, 0.015, n)       # 真实 alpha 日 2bp，beta 1.3
X = np.column_stack([np.ones(n), rm])
alpha, beta = np.linalg.lstsq(X, ri, rcond=None)[0]
beta_cov = np.cov(ri, rm)[0, 1] / np.var(rm, ddof=1)
print(f"beta {beta:.3f}（协方差法 {beta_cov:.3f}），年化 alpha {alpha * 252:.2%}")
```

输出：

```
beta 1.303（协方差法 1.303），年化 alpha 5.42%
```

 OLS 斜率与协方差公式给出的 Beta 完全相同。alpha 的估计误差大：日特质波动 1.5%、3 年数据，年化 alpha 的标准误约 $0.015\times252/\sqrt{750}\approx13.8\%$，远大于真实 alpha 5%。**Beta 容易估准，alpha 很难估准。**

</details>

### D2. 系统性风险与特质风险 `基础` `押题`

某股票 Beta 为 1.2，市场年化波动 20%，股票年化波动 40%。系统性风险和特质风险各占总方差的多少？

<details>
<summary>答案与解析</summary>

系统性方差 $\beta^2\sigma_m^2=1.44\times0.04=0.0576$；总方差 $0.16$；特质方差 $0.1024$。系统性占 36%，特质占 64%，$R^2=0.36$。

单只股票的大部分风险是特质风险，可以通过分散化消除（讲解 §5.2）；CAPM 认为只有系统性风险能获得补偿。

</details>

### D3. Fama-French 三因子与五因子 `基础` `经典·Fama & French 1993`

(1) 写出三因子模型，说明 SMB、HML 的构造。(2) 五因子加了什么？(3) A 股中规模因子有什么特点？

<details>
<summary>答案与解析</summary>

(1) $R_i-R_f=\alpha_i+b_i\,\text{MKT}+s_i\,\text{SMB}+h_i\,\text{HML}+\varepsilon_i$。按市值中位数分大小两组、按账面市值比 30%/70% 分位分高中低三组，得 6 个组合：$\text{SMB}$ = 三个小盘组合的平均收益 − 三个大盘组合的平均收益；$\text{HML}$ = 两个高 B/M 组合平均 − 两个低 B/M 组合平均。
(2) 五因子加盈利 RMW（盈利稳健减疲弱）和投资 CMA（投资保守减激进）；Carhart 四因子加动量 UMD。
(3) A 股小市值效应长期显著，部分原因是"壳价值"（注册制前小公司被借壳上市的期权价值，Liu, Stambaugh & Yuan 2019 构造 CH-3 模型时剔除了市值最小的 30% 股票）；2017 年后和 2024 年初，小盘风格出现过大幅回撤。

</details>

### D4. 时间序列回归与截面回归 `中等` `押题`

时间序列回归与 Fama-MacBeth 截面回归分别回答什么问题？写出 Fama-MacBeth 的步骤。

<details>
<summary>答案与解析</summary>

| | 时间序列回归 | Fama-MacBeth 截面回归 |
|---|---|---|
| 回归对象 | 单只股票或组合的收益对因子收益 | 每期所有股票的收益对因子暴露 |
| 得到 | 暴露 $\beta$、截距 $\alpha$ | 每期的因子收益（风险溢价） |
| 回答 | 已知因子能否解释该资产收益（$\alpha=0$？） | 该特征是否有溢价 |

**Fama-MacBeth 步骤**：
1. 每期 $t$ 做截面回归 $r_{i,t}=\gamma_{0,t}+\gamma_t^\top x_{i,t-1}+\varepsilon_{i,t}$，得到 $\hat\gamma_t$；
2. 溢价估计 $\bar\gamma=\frac1T\sum_t\hat\gamma_t$，标准误 $\operatorname{std}(\hat\gamma_t)/\sqrt T$；
3. 因子收益序列存在自相关时用 Newey-West 调整标准误。

量化中常用的"因子 IC 序列的均值和 t 值"本质上是 Fama-MacBeth 的变体（IC 是秩相关，相当于标准化后的截面回归系数）。

</details>

### D5. Barra 模型的结构与估计 `中等` `经典·Barra USE4`

(1) 写出 Barra 多因子模型的收益分解。(2) 截面回归为什么用 $\sqrt{\text{市值}}$ 加权？(3) 国家因子与行业因子共线怎么处理？

<details>
<summary>答案与解析</summary>

(1) $r_n=f_c+\sum_iX_{n,i}f_i+\sum_sX_{n,s}f_s+u_n$：国家因子、行业因子（0/1 暴露）、风格因子（标准化暴露）、特质收益。
(2) 特质收益方差与市值负相关（小盘股特质波动大），WLS 权重应与特质方差成反比；经验上特质方差约与 $1/\sqrt{\text{市值}}$ 成正比，所以权重取 $\sqrt{\text{市值}}$。这同时降低了小盘股的影响。
(3) 每只股票恰好属于一个行业，所有行业暴露之和等于国家因子暴露（全为 1），设计矩阵不满秩。USE4 加约束 $\sum_iw_if_i=0$（$w_i$ 为行业的市值权重），即行业因子收益是相对市场的超额，国家因子收益约等于市值加权的市场收益。

</details>

### D6. 风格因子的标准化 `基础` `经典·Barra USE4`

Barra 风格因子暴露如何标准化？为什么用市值加权均值而不是等权均值？

<details>
<summary>答案与解析</summary>

先去极值，然后 $X_{n,s}=\dfrac{\text{raw}_{n,s}-\mu_s^{\text{cap}}}{\sigma_s^{\text{eq}}}$：减去**市值加权**均值，除以**等权**标准差。

用市值加权均值，使市值加权的市场组合对每个风格因子的暴露为 0，即"市场组合是风格中性的"，风格因子收益可以解释为相对市场的超额。除以等权标准差，使暴露的尺度在不同因子之间可比。

</details>

### D7. 组合风险分解 `中等` `经典·Barra USE4`

给定因子暴露矩阵 $X$（$N\times K$）、因子协方差 $F$、特质方差 $\Delta$、组合权重 $\mathbf w$，写代码计算组合总风险、因子风险、特质风险，以及各因子的风险贡献。

<details>
<summary>答案与解析</summary>

组合方差 $\sigma_p^2=\mathbf x^\top F\mathbf x+\mathbf w^\top\Delta\mathbf w$，$\mathbf x=X^\top\mathbf w$。第 $k$ 个因子的风险贡献为 $x_k(F\mathbf x)_k$，各因子贡献之和等于因子方差。

```python
import numpy as np

rng = np.random.default_rng(0)
N, K = 200, 4
X = rng.standard_normal((N, K)); X[:, 0] = 1.0              # 第 0 列为国家因子
A = rng.standard_normal((K, K)) * 0.05
F = A @ A.T + np.diag([0.02, 0.003, 0.002, 0.001])           # 年化因子协方差
delta = rng.uniform(0.04, 0.16, N)                           # 年化特质方差
w = np.full(N, 1 / N)

x = X.T @ w
factor_var = x @ F @ x
spec_var = np.sum(w ** 2 * delta)
contrib = x * (F @ x)
print(f"总风险 {np.sqrt(factor_var + spec_var):.4f}，因子方差 {factor_var:.5f}，特质方差 {spec_var:.6f}")
print("各因子方差贡献", contrib.round(5), "合计", round(contrib.sum(), 5))
```

输出：

```
总风险 0.1497，因子方差 0.02189，特质方差 0.000503
各因子方差贡献 [ 2.225e-02  1.000e-05 -1.700e-04 -2.000e-04] 合计 0.02189
```

 各因子贡献之和等于因子方差。等权 200 只股票的特质方差只有单只股票的约 $1/200$，组合风险几乎全部来自因子（尤其国家因子）。

</details>

### D8. 业绩归因 `中等` `押题`

指增产品今年超额 10%。如何判断超额来自选股能力还是风格暴露？

<details>
<summary>答案与解析</summary>

基于 Barra 的归因：每期主动暴露 $\mathbf x^{a}_t=X_t^\top(\mathbf w_{p,t}-\mathbf w_{b,t})$，

$$
R_{p,t}-R_{b,t}=\underbrace{\sum_ix^a_{t,i}f_{t,i}}_{\text{行业}}+\underbrace{\sum_sx^a_{t,s}f_{t,s}}_{\text{风格}}+\underbrace{\sum_n(w_{p,n}-w_{b,n})u_{n,t}}_{\text{特质（选股）}}.
$$

将各项按期累加。如果超额主要来自风格（例如大幅超配小市值，而小盘风格今年上涨），说明是风格 beta 而非 alpha，风格反转时超额会回撤；特质部分占比高且稳定，才说明有选股能力。

**追问**：风格暴露是有意的还是无意的？各风格的暴露是否在风控限制内？

</details>

### D9. 因子中性化 `中等` `押题`

为什么新因子要做行业、市值中性化？写出中性化的方法。

<details>
<summary>答案与解析</summary>

**原因**：很多因子与市值、行业高度相关（例如低价股因子、换手率因子与市值相关）。不中性化时，因子的 IC 可能只是市值效应的重复，组合也会暴露于市值风格。

**方法**：每个截面上把因子值对行业哑变量和对数市值（可再加其他 Barra 风格）做回归，取残差作为中性化后的因子：

$$
f_n=\sum_i\beta_iI_{n,i}+\gamma\ln\text{MV}_n+\varepsilon_n,\qquad f_n^{\text{neu}}=\hat\varepsilon_n.
$$

残差与行业、市值正交。实现见[编程与算法题库](../programming/problems.md) M 章；中性化后再检验 IC，看因子是否有独立的信息。

</details>

---

---

## E. 组合理论

> 对应讲解第 5 节。

### E1. 两资产最小方差组合 `基础` `经典·Hull`

两资产波动 20%、30%，相关系数 0.2。求最小方差组合的权重和波动。

<details>
<summary>答案与解析</summary>

$w_1=\dfrac{\sigma_2^2-\rho\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}$。

```python
import numpy as np

s1, s2, rho = 0.2, 0.3, 0.2
w1 = (s2 ** 2 - rho * s1 * s2) / (s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
vol = np.sqrt(w1 ** 2 * s1 ** 2 + (1 - w1) ** 2 * s2 ** 2 + 2 * w1 * (1 - w1) * rho * s1 * s2)
Sigma = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
w = np.linalg.solve(Sigma, np.ones(2)); w /= w.sum()
print(f"w1 = {w1:.4f}（矩阵法 {w[0]:.4f}），组合波动 {vol:.4f}")
```

输出：

```
w1 = 0.7358（矩阵法 0.7358），组合波动 0.1806
```

 $w_1\approx0.7358$，组合波动约 18.1%，低于两个资产中较低的 20%。

</details>

### E2. 切点组合 `中等` `经典·Grinold & Kahn`

证明最大 Sharpe 组合权重 $\mathbf w\propto\Sigma^{-1}(\boldsymbol\mu-R_f\mathbf 1)$，并说明其最大 Sharpe 平方等于 $(\boldsymbol\mu-R_f\mathbf1)^\top\Sigma^{-1}(\boldsymbol\mu-R_f\mathbf1)$。

<details>
<summary>答案与解析</summary>

记超额收益 $\mathbf m=\boldsymbol\mu-R_f\mathbf1$。Sharpe $S(\mathbf w)=\mathbf w^\top\mathbf m/\sqrt{\mathbf w^\top\Sigma\mathbf w}$ 对 $\mathbf w$ 的缩放不变。等价问题：$\max\mathbf w^\top\mathbf m-\frac\lambda2\mathbf w^\top\Sigma\mathbf w$，一阶条件 $\mathbf m=\lambda\Sigma\mathbf w$，得 $\mathbf w=\Sigma^{-1}\mathbf m/\lambda$。

代入：$S^2=\dfrac{(\mathbf m^\top\Sigma^{-1}\mathbf m)^2}{\mathbf m^\top\Sigma^{-1}\Sigma\Sigma^{-1}\mathbf m}=\mathbf m^\top\Sigma^{-1}\mathbf m$。

或由 Cauchy-Schwarz：令 $\mathbf v=\Sigma^{1/2}\mathbf w$，$S=\dfrac{\mathbf v^\top\Sigma^{-1/2}\mathbf m}{\lVert\mathbf v\rVert}\le\lVert\Sigma^{-1/2}\mathbf m\rVert$，等号在 $\mathbf v\propto\Sigma^{-1/2}\mathbf m$ 时成立。

这与[机器学习题库](../machine-learning/problems.md) L3 中"最大化 ICIR 的因子权重 $\propto\Sigma_{IC}^{-1}\overline{\mathrm{IC}}$"是同一个结论。

</details>

### E3. 分散化的极限 `基础` `押题`

$n$ 只股票等权，每只波动 30%，两两相关系数 0.3。$n=10$、$100$、$\infty$ 时组合波动分别是多少？

<details>
<summary>答案与解析</summary>

$\sigma_p=\sigma\sqrt{\rho+(1-\rho)/n}$。

```python
import numpy as np

s, rho = 0.3, 0.3
for n in [10, 100, 10 ** 9]:
    print(n, round(s * np.sqrt(rho + (1 - rho) / n), 4))
```

输出：

```
10 0.1825
100 0.1662
1000000000 0.1643
```

 10 只约 18.3%，100 只约 16.6%，极限 16.4%（$0.3\sqrt{0.3}$）。超过几十只后，增加股票数几乎不再降低风险，剩下的是系统性风险。

</details>

### E4. 均值-方差优化的问题 `中等` `押题`

为什么直接用样本均值和样本协方差做均值-方差优化，实践中效果很差？有哪些改进方法？

<details>
<summary>答案与解析</summary>

**问题**：
- **期望收益估计误差大**：均值的标准误为 $\sigma/\sqrt T$，10 年数据下年化收益的标准误仍约 $20\%/\sqrt{10}\approx6\%$，与收益本身同一量级。
- **协方差矩阵病态**：$N$ 接近 $T$ 时样本协方差矩阵的小特征值被严重低估，$\Sigma^{-1}$ 放大估计误差；$N>T$ 时不可逆。
- 优化器会把权重集中在估计误差最大的方向上（"误差最大化器"，Michaud 1989），结果权重极端、不稳定、换手高。

**改进**：
- 协方差：因子模型（Barra）、Ledoit-Wolf 收缩、去噪（随机矩阵理论）；
- 期望收益：用模型预测的 alpha，并按置信度收缩；Black-Litterman；
- 约束：权重上下限、行业和风格暴露约束、换手约束、跟踪误差约束；
- 稳健优化、重抽样。

</details>

### E5. 风险平价 `中等` `押题`

风险平价组合的定义是什么？两资产波动分别为 10% 和 30%、相关系数为 0 时，权重是多少？

<details>
<summary>答案与解析</summary>

**定义**：每个资产对组合风险的贡献相等：$w_i(\Sigma\mathbf w)_i=w_j(\Sigma\mathbf w)_j$。

相关系数为 0 时，贡献为 $w_i^2\sigma_i^2$，相等要求 $w_i\propto1/\sigma_i$，即 $w_1:w_2=3:1$，权重 75% 和 25%。

```python
import numpy as np
from scipy.optimize import minimize

sig = np.array([0.1, 0.3]); Sigma = np.diag(sig ** 2)
def obj(w):
    rc = w * (Sigma @ w)
    return np.sum((rc - rc.mean()) ** 2)
res = minimize(obj, x0=[0.5, 0.5], constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
               bounds=[(0, 1)] * 2, tol=1e-14)
print(res.x.round(4))
```

输出：

```
[0.75 0.25]
```

 数值解为 (0.75, 0.25)。股债风险平价通常给低波动的债券更高权重，并用杠杆提高整体收益。

</details>

### E6. 主动管理基本定律 `中等` `经典·Grinold & Kahn`

(1) 写出基本定律。(2) 选股策略 IC 为 0.03，每月调仓、每次在 1000 只股票上独立下注，IR 是多少？(3) 定律的局限是什么？

<details>
<summary>答案与解析</summary>

(1) $\mathrm{IR}\approx\mathrm{IC}\sqrt{\mathrm{BR}}$；考虑约束时 $\mathrm{IR}\approx\mathrm{TC}\cdot\mathrm{IC}\sqrt{\mathrm{BR}}$。
(2) $\mathrm{BR}=12\times1000=12000$，$\mathrm{IR}\approx0.03\times\sqrt{12000}\approx3.3$。
(3) 局限：
- 假设各次下注独立，实际上股票收益截面相关、因子之间相关，有效 BR 远小于名义值，实际 IR 低于计算值；
- 假设 IC 在各次下注中恒定，实际 IC 时变；
- 未考虑交易成本：提高调仓频率增加 BR，但成本也随之增加。

</details>

### E7. VaR 与 CVaR `基础` `押题`

(1) 定义 95% VaR 和 CVaR。(2) 日收益服从均值 0、波动 1% 的正态分布，持仓 1 亿元，两者各是多少？(3) 为什么 CVaR 更好？

<details>
<summary>答案与解析</summary>

(1) VaR：在 95% 置信水平下的最大损失，即损失分布的 95% 分位数。CVaR（Expected Shortfall）：损失超过 VaR 时的平均损失。

```python
from scipy.stats import norm

sigma, V, a = 0.01, 1e8, 0.95
var = norm.ppf(a) * sigma * V
cvar = norm.pdf(norm.ppf(a)) / (1 - a) * sigma * V
print(f"VaR {var / 1e4:.1f} 万元，CVaR {cvar / 1e4:.1f} 万元")
```

输出：

```
VaR 164.5 万元，CVaR 206.3 万元
```

 VaR 约 164.5 万元，CVaR 约 206.3 万元（正态下 $\mathrm{CVaR}=\sigma\varphi(z_\alpha)/(1-\alpha)$）。

(3) VaR 不反映尾部损失的大小，且不满足次可加性（合并两个组合的 VaR 可能大于各自 VaR 之和，违背分散化）；CVaR 是一致性风险度量，满足次可加性，且反映尾部。

</details>

---

---

## F. 量化策略与收益拆解

> 对应讲解第 6 节。

### F1. 指数增强与市场中性 `基础` `押题`

指数增强和市场中性的区别是什么？各自适合什么样的投资者？

<details>
<summary>答案与解析</summary>

| | 指数增强 | 市场中性 |
|---|---|---|
| 持仓 | 满仓股票，跟踪基准 | 多股票 + 空股指期货 |
| 收益 | 指数收益 + 超额 | 超额 − 对冲成本 |
| 市场风险 | 完全承担 | 基本对冲 |
| 波动与回撤 | 与指数相当 | 低 |
| 额外成本 | 无 | 基差、保证金占用 |
| 适合 | 看好市场、希望获得超过指数的收益 | 追求绝对收益、不愿承担市场风险 |

两者共用同一套选股 alpha，差别只在于是否对冲贝塔。

</details>

### F2. 中性产品收益拆解 `中等` `押题`

某中性产品：股票仓位占资金 80%，股票组合相对中证 1000 的年化超额 15%，期货对冲的年化贴水 8%，保证金和备付金的现金收益 2%，管理费和业绩报酬合计约 3%。估算产品年化收益。

<details>
<summary>答案与解析</summary>

```python
stock_w, excess, basis, cash_ret, fee = 0.80, 0.15, 0.08, 0.02, 0.03
ret = stock_w * excess - stock_w * basis + (1 - stock_w) * cash_ret - fee
print(f"{ret:.2%}")
```

输出：

```
3.00%
```

 约 3.0%：超额 $0.8\times15\%=12\%$，对冲成本 $0.8\times8\%=6.4\%$，现金收益 0.4%，费用 3%。

**结论**：深贴水时期，中性产品的大部分超额被对冲成本吃掉。这也是 2025 年中性产品收益远低于指增的原因之一（指增的超额加上指数上涨）。

</details>

### F3. 指增的超额来源 `中等` `押题`

中证 1000 指增产品的超额收益可能来自哪些方面？哪些是可持续的？

<details>
<summary>答案与解析</summary>

| 来源 | 说明 | 可持续性 |
|---|---|---|
| 选股 alpha | 量价、基本面、另类数据因子 | 核心来源，但随竞争衰减 |
| 风格暴露 | 偏小市值、高流动性、高 Beta | 风格切换时回撤（2024 年初） |
| 成分股外选股 | 在中证 1000 之外选更小的股票 | 本质上也是风格暴露 |
| 交易能力 | 低冲击执行、T0 增强 | 可持续，但容量有限 |
| 期货替代 | 用贴水期货替代部分现货 | 取决于贴水水平 |
| 打新 | 新股申购收益 | 规模越大贡献越小 |

面试时要强调：好的指增应控制风格和行业暴露，使超额主要来自选股和交易，而不是押注风格。

</details>

### F4. 跟踪误差与超额的关系 `基础` `押题`

两个中证 500 指增产品：A 年化超额 12%、跟踪误差 4%；B 年化超额 15%、跟踪误差 10%。哪个更好？

<details>
<summary>答案与解析</summary>

信息比率：A 为 3.0，B 为 1.5。A 的风险调整后超额更高，超额更稳定；B 的超额更高，但很可能来自更大的风格偏离，风格反转时超额回撤更大。

对于看重相对基准稳定性的投资者，A 更好；B 可以理解为"A + 更多风格暴露"。若 A 能加杠杆或放宽跟踪误差到 10%，在 IR 不变时超额可达 30%（理论上）。

</details>

### F5. DMA 与杠杆中性 `中等` `押题`

什么是 DMA 产品？为什么它在 2024 年初受到很大冲击？

<details>
<summary>答案与解析</summary>

**DMA**（Direct Market Access）：私募通过券商的场外收益互换获得杠杆，在券商提供的交易通道中做多股票、做空股指期货，即**加杠杆的中性策略**，常被称为"空气指增"（指增产品的超额 + 杠杆，但不承担指数收益）。

**2024 年初的冲击**：
1. 小盘股暴跌，DMA 产品的多头偏小盘，超额大幅回撤；
2. 期货贴水急剧加深，空头端的对冲成本上升；
3. 杠杆放大亏损，产品净值触及平仓线后被券商强制平仓，被动卖出小盘股，进一步加剧下跌；
4. 结果部分产品回撤超过 40%。此后监管要求券商 DMA 业务不得新增规模、存量逐步压降。

</details>

### F6. CTA 策略 `基础` `押题`

CTA 策略的收益来源有哪些？为什么 CTA 被认为有"危机 alpha"？

<details>
<summary>答案与解析</summary>

**收益来源**：
- **时序趋势**：价格持续上涨（下跌）时做多（做空），赚取趋势延续的收益（Moskowitz, Ooi & Pedersen 2012）；
- **截面策略**：期限结构（做多贴水最大的品种、做空升水最大的品种，赚取展期收益）、截面动量、基本面（库存、基差）；
- **短周期**：日内、高频量价。

**危机 alpha**：市场危机（如 2008 年）往往伴随持续的单边行情，趋势策略能在空头方向获利，与股票收益负相关或低相关，在股票组合中起分散作用。代价是震荡市中频繁止损、长期回撤。

</details>

### F7. T0 策略 `基础` `押题`

T0 策略如何在 T+1 制度下实现日内交易？收益和风险分别是什么？

<details>
<summary>答案与解析</summary>

**做法**：持有底仓（例如指增产品的持仓）。日内预测价格走势：预期下跌时先卖出部分底仓、低点买回；预期上涨时先买入、高点卖出昨日底仓。收盘时持仓数量恢复，日内实现价差收益。

**收益**：约等于"日内价差收益 × 交易额 − 交易成本"，与底仓的方向无关，可叠加在指增产品上增强超额（通常每年数个百分点）。

**风险与限制**：
- 判断错误时收盘前被迫平仓，产生亏损；
- 受底仓规模和个股流动性限制，容量有限；
- 印花税降低后成本下降；但 2024 年起严禁利用融券进行日内回转，程序化交易监管也限制了高频申报。

</details>

### F8. 高频策略 `中等` `押题`

国内股票高频策略主要有哪些类型？面试官问"你怎么看 tick 数据"时应该谈什么？

<details>
<summary>答案与解析</summary>

**类型**：
- 日内 T0（基于底仓）；
- 做市（ETF、期权、期货做市，赚取买卖价差和交易所返还）；
- 短周期统计套利（期现、ETF 与成分股、相关品种价差）；
- 微观结构 alpha：订单簿不平衡、成交方向、大单追踪。

**tick 数据要点**：A 股 Level-1 行情为每 3 秒一次快照（五档），Level-2 提供十档行情和逐笔委托、逐笔成交。常用特征：
- 订单簿不平衡 $\dfrac{V^{\text{bid}}-V^{\text{ask}}}{V^{\text{bid}}+V^{\text{ask}}}$；
- 中间价、加权中间价（microprice）；
- 主动买卖成交量；
- 撤单率、大单占比。

**难点**：数据量大、需要处理时间戳对齐和异常值；回测须模拟排队和成交概率，否则高估收益。

> 宽德高频岗面试关注"tick 数据与微观结构"，以上为按考点整理的要点。

</details>

### F9. 容量估算 `中等` `押题`

某策略每天调仓，平均持仓 300 只股票，持仓股日均成交额 2 亿元，单边日换手 20%。要求每只股票的交易额不超过日成交额的 5%。容量大约多少？

<details>
<summary>答案与解析</summary>

每天交易额 = 规模 × 换手（单边）× 2（买卖）。等权持仓时，每只股票的单边交易额 = 规模 × 20% / 300。要求每只股票的买入（或卖出）额 ≤ 2 亿 × 5% = 1000 万：

```python
n, adv, cap_ratio, turnover = 300, 2e8, 0.05, 0.20
aum = adv * cap_ratio * n / turnover
print(f"容量约 {aum / 1e8:.0f} 亿元")
```

输出：

```
容量约 150 亿元
```

 约 150 亿元。这是粗略上限：实际持仓不等权、成交额分布不均，小票更早触及上限；而且接近上限时冲击成本已经很高（G3），扣费后超额会显著下降。

</details>

---

---

## G. 交易成本与冲击成本

> 对应讲解第 7 节。

### G1. 交易成本的构成 `基础` `押题`

交易成本包括哪些部分？什么是实施缺口？

<details>
<summary>答案与解析</summary>

| 类型 | 内容 |
|---|---|
| 显性成本 | 佣金、印花税、过户费 |
| 价差成本 | 市价成交支付的半个买卖价差 |
| 冲击成本 | 大单推动价格向不利方向移动 |
| 机会成本 | 未成交或延迟成交导致错过的收益 |

**实施缺口**（Perold 1988）：纸面组合（决策时以决策价格立即成交）与实际组合的收益之差，包含以上全部成本。

</details>

### G2. 平方根冲击模型 `中等` `押题`

(1) 写出平方根冲击模型。(2) 日波动 2.5%、交易量占日成交量 4%，冲击成本约多少？(3) 交易量加倍，单位成本和总成本如何变化？

<details>
<summary>答案与解析</summary>

(1) $\text{冲击}\approx Y\sigma\sqrt{Q/V}$，$Y$ 量级为 1。

```python
import numpy as np

Y, sigma, part = 1.0, 0.025, 0.04
impact = Y * sigma * np.sqrt(part)
print(f"冲击成本 {impact * 1e4:.0f} bp；交易量加倍后单位成本 ×{np.sqrt(2):.3f}，总成本 ×{2 * np.sqrt(2):.3f}")
```

输出：

```
冲击成本 50 bp；交易量加倍后单位成本 ×1.414，总成本 ×2.828
```

 约 50 bp。交易量加倍，单位成本变为 $\sqrt2\approx1.41$ 倍，总成本变为 $2\sqrt2\approx2.83$ 倍。

(3) 的含义：冲击成本随规模超线性增长，是策略容量的根本约束；拆单、延长执行时间可以降低冲击，但会增加价格风险。

</details>

### G3. 扣费后超额与规模 `中等` `押题`

某策略扣费前年化超额 20%，单边日换手 20%，显性成本每次买卖 7.6 bp。持仓股票日波动 2.5%、日成交额 3 亿元、等权持有 300 只。用平方根模型（$Y=1$）计算规模为 10、50、100、200 亿元时的扣费后超额。

<details>
<summary>答案与解析</summary>

单只股票每天单边交易额 $=\text{AUM}\times0.2/300$，占日成交额的比例 $p$，每次交易的冲击为 $\sigma\sqrt p$。每天买入、卖出各一次，年成本 $=$ 换手 × 242 × (显性成本 + 2 × 冲击)。

```python
import numpy as np

alpha, to, explicit, sigma, adv, n = 0.20, 0.20, 7.6e-4, 0.025, 3e8, 300
for aum in [10e8, 50e8, 100e8, 200e8]:
    p = aum * to / n / adv
    cost = to * 242 * (explicit + 2 * sigma * np.sqrt(p))
    print(f"{aum / 1e8:>4.0f} 亿: 成交量占比 {p:.2%}, 年成本 {cost:.2%}, 扣费后超额 {alpha - cost:.2%}")
```

输出：

```
  10 亿: 成交量占比 0.22%, 年成本 15.09%, 扣费后超额 4.91%
  50 亿: 成交量占比 1.11%, 年成本 29.19%, 扣费后超额 -9.19%
 100 亿: 成交量占比 2.22%, 年成本 39.75%, 扣费后超额 -19.75%
 200 亿: 成交量占比 4.44%, 年成本 54.70%, 扣费后超额 -34.70%
```

 10 亿时成交量占比 0.22%，年成本约 15.1%，扣费后超额 4.9%；50 亿时成交量占比 1.1%，扣费后超额已为负（−9.2%）。可见高换手策略的超额对规模极其敏感。这里取 $Y=1$，并按"每次交易一次性冲击"计算，偏保守；实际拆单执行、用限价单提供流动性时冲击更低，实证中的 $Y$ 常在 0.5–1 之间。

</details>

### G4. VWAP 与 TWAP `基础` `押题`

(1) VWAP 和 TWAP 算法的区别？(2) 写函数计算一组成交的 VWAP，以及执行均价相对市场 VWAP 的滑点（bp）。

<details>
<summary>答案与解析</summary>

(1) TWAP 按时间均匀拆单；VWAP 按预测的成交量分布拆单，使执行均价接近全天 VWAP。VWAP 更贴近市场的流动性分布（A 股开盘和收盘成交量大，日内呈 U 形）。

```python
import numpy as np

def vwap(prices, volumes):
    prices, volumes = np.asarray(prices, float), np.asarray(volumes, float)
    return (prices * volumes).sum() / volumes.sum()

market_p = [10.00, 10.05, 10.10, 10.08, 10.02]
market_v = [5000, 3000, 2000, 3000, 7000]
fill_p, fill_v = [10.03, 10.09, 10.04], [1000, 500, 1500]
mv, fv = vwap(market_p, market_v), vwap(fill_p, fill_v)
print(f"市场 VWAP {mv:.4f}，执行均价 {fv:.4f}，买入滑点 {(fv / mv - 1) * 1e4:.1f} bp")
```

输出：

```
市场 VWAP 10.0365，执行均价 10.0450，买入滑点 8.5 bp
```

 市场 VWAP 10.0365，执行均价 10.0450，买入时执行均价高于市场 VWAP，滑点约 8.5 bp。

</details>

### G5. Almgren-Chriss 的权衡 `中等` `经典·Almgren & Chriss 2000`

最优执行问题中，快速执行和缓慢执行各有什么代价？最优执行路径的形状由什么决定？

<details>
<summary>答案与解析</summary>

- **快速执行**：临时冲击大（短时间内消耗大量流动性）。
- **缓慢执行**：价格风险大（持有未完成头寸的时间长，价格随机波动）。

Almgren-Chriss 模型最小化 $\mathbb E[\text{成本}]+\lambda\operatorname{Var}[\text{成本}]$。线性临时冲击下最优持仓路径为

$$
x(t)=X\frac{\sinh\big(\kappa(T-t)\big)}{\sinh(\kappa T)},\qquad\kappa\approx\sqrt{\frac{\lambda\sigma^2}{\eta}},
$$

$\eta$ 为临时冲击系数。风险厌恶 $\lambda$ 越大、波动 $\sigma$ 越大，$\kappa$ 越大，执行越前置（前期交易更多）；$\lambda\to0$ 时退化为匀速执行（TWAP）。

</details>

### G6. 回测中的成本建模 `基础` `押题`

回测时如何设置交易成本？成本设得过低或过高分别有什么后果？

<details>
<summary>答案与解析</summary>

**设置**：
- 显性成本按实际费率：佣金、过户费双向，印花税只在卖出时收；
- 冲击成本按成交量占比（平方根模型或分档线性），或使用历史实盘的成交滑点统计；
- 成交价假设：下一时点的开盘价、VWAP 或收盘价，不能使用信号计算时已知的价格；
- 涨跌停、停牌不能成交（A7）。

**后果**：
- 成本过低：高换手策略看起来很好，实盘收益大幅低于回测；模型倾向于过度交易；
- 成本过高：错过有价值的信号，策略过于保守。

**实践**：用不同成本假设做敏感性分析，并在优化目标中显式加入成本项（换手惩罚）。

</details>

---

## H. 期权

> 对应讲解第 8 节。H1–H8 是期权做市方向的必会题。

### H1. 推导平价公式 `基础` `经典·Hull 11`

推导欧式期权的 Put-Call Parity（无股息和连续股息率 $q$ 两种情形）。

<details>
<summary>答案与解析</summary>

**无股息**：组合 A = 一份看涨 + 现金 $Ke^{-rT}$；组合 B = 一份看跌 + 一股股票。到期时两者价值都是 $\max(S_T,K)$。无套利要求当前价值相等：

$$
C+Ke^{-rT}=P+S\ \Longleftrightarrow\ C-P=S-Ke^{-rT}.
$$

**连续股息率 $q$**：持有 $e^{-qT}$ 股股票并将股息再投资，到期恰为 1 股，所以把 $S$ 换为 $Se^{-qT}$：$C-P=Se^{-qT}-Ke^{-rT}$。

**等价理解**：多看涨 + 空看跌 = 行权价为 $K$ 的远期多头。

</details>

### H2. 平价公式套利 `中等` `押题`

$S=50$，$K=50$，$r=5\%$，$T=0.5$ 年，无股息。看涨期权 4.5 元，看跌期权 2.0 元。是否存在套利？如何操作？每份套利利润多少？

<details>
<summary>答案与解析</summary>

```python
import numpy as np

S, K, r, T, C, P = 50, 50, 0.05, 0.5, 4.5, 2.0
lhs, rhs = C - P, S - K * np.exp(-r * T)
print(f"C - P = {lhs:.4f}，S - K e^(-rT) = {rhs:.4f}，偏差 {lhs - rhs:.4f}")
```

输出：

```
C - P = 2.5000，S - K e^(-rT) = 1.2345，偏差 1.2655
```

 $C-P=2.5>1.2345$，看涨相对看跌**偏贵**。操作：卖出看涨、买入看跌、买入股票、借入 $Ke^{-rT}$。期初净收入 $C-P-S+Ke^{-rT}=1.2655$；到期时"空看涨 + 多看跌 + 多股票"的价值恒为 $K$，恰好偿还借款 $K$。无风险利润 1.2655 元（现值）。

</details>

### H3. 期权价格的界限 `中等` `经典·Hull 11`

证明：(1) $C\ge\max(S-Ke^{-rT},0)$；(2) $C\le S$；(3) 无股息美式看涨期权不应提前行权。

<details>
<summary>答案与解析</summary>

(1) 组合"看涨 + 现金 $Ke^{-rT}$"到期价值 $\max(S_T,K)\ge S_T$，所以当前价值 $C+Ke^{-rT}\ge S$。又 $C\ge0$。
(2) 股票到期价值 $S_T\ge\max(S_T-K,0)$，所以 $S\ge C$。
(3) 由 (1)，美式看涨价值 $\ge C_{\text{欧}}\ge S-Ke^{-rT}>S-K$（$r>0$），即持有或卖出期权总比行权得到 $S-K$ 更好。直觉：提前行权要提前支付 $K$（损失利息），并放弃了下跌保护。

**看跌**：深度实值的美式看跌可能提前行权，因为提前拿到 $K$ 可以赚取利息，而期权剩余的时间价值很小。

</details>

### H4. 实现 Black-Scholes 公式 `基础` `经典·Hull 15`

实现欧式看涨、看跌期权的 BS 定价（含连续股息率）。计算 $S=K=100$、$r=3\%$、$\sigma=20\%$、$T=1$ 的价格，并验证平价公式。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, q=0.0, call=True):
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

C = bs_price(100, 100, 1, 0.03, 0.2)
P = bs_price(100, 100, 1, 0.03, 0.2, call=False)
print(f"C = {C:.4f}, P = {P:.4f}, C - P = {C - P:.4f}, S - K e^(-rT) = {100 - 100 * np.exp(-0.03):.4f}")
```

输出：

```
C = 9.4134, P = 6.4580, C - P = 2.9554, S - K e^(-rT) = 2.9554
```

 看涨 9.4134，看跌 6.4580，平价公式成立。

</details>

### H5. 平值期权的近似价格 `基础` `押题`

(1) 推导平值期权近似公式 $C\approx0.4S\sigma\sqrt T$。(2) 某股票 100 元，年化波动 30%，3 个月平值看涨期权约多少钱？与 BS 精确值比较。

<details>
<summary>答案与解析</summary>

(1) $r=0$、$S=K$ 时 $d_{1,2}=\pm\sigma\sqrt T/2$，$C=S[N(\sigma\sqrt T/2)-N(-\sigma\sqrt T/2)]\approx S\cdot\varphi(0)\cdot\sigma\sqrt T=\frac{S\sigma\sqrt T}{\sqrt{2\pi}}\approx0.4S\sigma\sqrt T$。

```python
import numpy as np
from scipy.stats import norm

S, sigma, T = 100, 0.3, 0.25
d1 = sigma * np.sqrt(T) / 2
exact = S * (norm.cdf(d1) - norm.cdf(-d1))
print(f"近似 {0.4 * S * sigma * np.sqrt(T):.4f}，1/sqrt(2pi) 近似 {S * sigma * np.sqrt(T) / np.sqrt(2 * np.pi):.4f}，精确 {exact:.4f}")
```

输出：

```
近似 6.0000，1/sqrt(2pi) 近似 5.9841，精确 5.9785
```

 近似 6.00，精确 5.97。$\sigma\sqrt T$ 较小时近似很准。平值看跌期权（$r=0$）价格相同。

</details>

### H6. Greeks 的计算 `中等` `经典·Hull 19`

实现看涨期权的 Delta、Gamma、Vega、Theta、Rho 解析公式，用有限差分验证，参数同 H4。

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from scipy.stats import norm

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        "vega": S * norm.pdf(d1) * np.sqrt(T),
        "theta": -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
        "rho": K * T * np.exp(-r * T) * norm.cdf(d2),
    }

S, K, T, r, sig, h = 100, 100, 1.0, 0.03, 0.2, 1e-4
g = greeks(S, K, T, r, sig)
c = lambda S=S, T=T, r=r, sig=sig: bs_price(S, K, T, r, sig)     # 沿用 H4 的 bs_price
fd = {
    "delta": (c(S=S + h) - c(S=S - h)) / (2 * h),
    "gamma": (c(S=S + h) - 2 * c() + c(S=S - h)) / h ** 2,
    "vega": (c(sig=sig + h) - c(sig=sig - h)) / (2 * h),
    "theta": -(c(T=T + h) - c(T=T - h)) / (2 * h),
    "rho": (c(r=r + h) - c(r=r - h)) / (2 * h),
}
for k in g:
    print(f"{k:>5}: 解析 {g[k]:9.4f}  差分 {fd[k]:9.4f}")
```

输出：

```
delta: 解析    0.5987  差分    0.5987
gamma: 解析    0.0193  差分    0.0193
 vega: 解析   38.6668  差分   38.6668
theta: 解析   -5.3804  差分   -5.3804
  rho: 解析   50.4572  差分   50.4572
```

 五个 Greeks 的解析值与中心差分在 4 位小数上一致。Rho 50.46 表示利率上升 1 个百分点，价格上升约 0.50。Theta 为每年的值，除以 365 得每天约 $-0.015$；Vega 为波动率变化 1（即 100 个百分点）的价格变化，常用"每 1 个百分点"约 0.387。

</details>

### H7. Greeks 的性质 `中等` `押题`

回答：(1) 看涨和看跌的 Gamma、Vega 有什么关系？(2) 平值期权临近到期时 Gamma 和 Theta 如何变化？(3) 深度实值看涨期权的 Delta 和 Gamma 约为多少？

<details>
<summary>答案与解析</summary>

(1) 由平价公式 $C-P=S-Ke^{-rT}$，右边对 $S$ 的二阶导和对 $\sigma$ 的导数都为 0，所以看涨、看跌的 Gamma 相同、Vega 相同；Delta 相差 1（$\Delta_P=\Delta_C-1$，有股息时为 $\Delta_C-e^{-qT}$）。
(2) 平值 Gamma $\approx\dfrac{\varphi(0)}{S\sigma\sqrt T}\propto1/\sqrt T$，临近到期时急剧增大；Theta $\approx-\dfrac{S\varphi(0)\sigma}{2\sqrt T}$ 也按 $1/\sqrt T$ 增大（时间衰减加速）。这就是到期日附近平值期权的"Gamma 风险"。
(3) 深度实值看涨 $N(d_1)\to1$，Delta 约 1、Gamma 约 0，行为接近股票；深度虚值时 Delta、Gamma 都约 0。

</details>

### H8. Delta 对冲与 Gamma 损益 `困难` `经典·Hull 19`

(1) 买入一份平值看涨期权并做 Delta 对冲，组合的盈亏来自哪里？(2) 用模拟验证：实现波动率高于（低于）隐含波动率时，每日 Delta 对冲的多头期权组合盈利（亏损）。

<details>
<summary>答案与解析</summary>

(1) 对冲后组合对 $S$ 的一阶敏感度为 0，盈亏来自二阶项与时间衰减：$\text{P\&L}\approx\frac12\Gamma(\Delta S)^2+\Theta\Delta t$。由 BS 方程（$r=0$）$\Theta=-\frac12\sigma_{\text{隐含}}^2S^2\Gamma$，所以

$$
\text{P\&L}\approx\tfrac12\Gamma S^2\Big[\big(\tfrac{\Delta S}{S}\big)^2-\sigma_{\text{隐含}}^2\Delta t\Big],
$$

即"实现方差 − 隐含方差"按 Gamma 加权的累积。多 Gamma 赚实现波动、付 Theta。

```python
import numpy as np
from scipy.stats import norm

def hedge_pnl(sigma_real, sigma_impl=0.2, S0=100, K=100, T=0.25, n=63, paths=20000, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n
    S = np.full(paths, float(S0))
    delta_of = lambda S, tau: norm.cdf((np.log(S / K) + sigma_impl ** 2 / 2 * tau) / (sigma_impl * np.sqrt(tau)))
    delta = delta_of(S, T)
    cash = -bs_price(S0, K, T, 0, sigma_impl) + delta * S       # 买入期权（r=0），卖出 delta 股对冲
    for i in range(1, n + 1):
        S = S * np.exp(-sigma_real ** 2 / 2 * dt + sigma_real * np.sqrt(dt) * rng.standard_normal(paths))
        tau = T - i * dt
        if tau > 1e-12:                                         # 每日调整对冲
            new = delta_of(S, tau)
            cash += (new - delta) * S
            delta = new
    pnl = cash + np.maximum(S - K, 0) - delta * S               # 期权到期价值，买回空头股票
    return pnl.mean(), pnl.std()

for sr in [0.15, 0.20, 0.25]:
    m, s = hedge_pnl(sr)
    print(f"实现波动 {sr:.0%}：平均盈亏 {m:+.3f}，标准差 {s:.3f}")
print(f"理论近似：Vega × 波动率差 = {100 * norm.pdf(0.025) * 0.5:.2f} × (σ_real − 0.20)")
```

输出：

```
实现波动 15%：平均盈亏 -0.994，标准差 0.455
实现波动 20%：平均盈亏 +0.003，标准差 0.437
实现波动 25%：平均盈亏 +0.999，标准差 0.690
理论近似：Vega × 波动率差 = 19.94 × (σ_real − 0.20)
```

 实现波动等于隐含波动（20%）时平均盈亏约为 0；实现波动 25% 时盈利、15% 时亏损，幅度与 Vega × 波动率差（约 $19.9\times0.05\approx1.0$）同一量级。离散对冲带来路径上的随机误差（标准差一列）。

</details>

### H9. 隐含波动率的求解 `中等` `押题`

已知看涨期权市场价格 10.0 元（$S=K=100$、$r=3\%$、$T=1$），用牛顿法求隐含波动率。为什么牛顿法在这里收敛很快？什么时候需要改用二分法？

<details>
<summary>答案与解析</summary>

```python
import numpy as np
from scipy.stats import norm

def implied_vol(price, S, K, T, r, sigma0=0.3, tol=1e-10):
    sigma = sigma0
    for i in range(100):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        diff = bs_price(S, K, T, r, sigma) - price
        if abs(diff) < tol:
            return sigma, i
        sigma -= diff / (S * norm.pdf(d1) * np.sqrt(T))       # 牛顿步，导数为 Vega
    raise RuntimeError("未收敛")

iv, it = implied_vol(10.0, 100, 100, 1, 0.03)
print(f"隐含波动率 {iv:.6f}，迭代 {it} 次，回代价格 {bs_price(100, 100, 1, 0.03, iv):.8f}")
```

输出：

```
隐含波动率 0.215164，迭代 3 次，回代价格 10.00000000
```

 隐含波动率约 21.5%，几次迭代即收敛。

**原因**：BS 价格关于 $\sigma$ 单调递增，平值附近接近线性，Vega 较大。**需要二分法的情形**：深度虚值或实值、临近到期时 Vega 接近 0，牛顿步长 $\text{diff}/\text{Vega}$ 会过大甚至跳到负值；此时用二分法或 Brent 法（有界、保证收敛），或先用二分法找到初值再用牛顿法。价格超出无套利界限时隐含波动率不存在。

</details>

### H10. 波动率微笑 `基础` `经典·Hull 20`

(1) 什么是波动率微笑和偏斜？(2) 为什么股指期权通常呈现左偏（低行权价隐含波动率更高）？(3) 这说明 BS 模型的哪个假设不成立？

<details>
<summary>答案与解析</summary>

(1) 同一到期日、不同行权价的隐含波动率不相等。外汇期权常见对称的"微笑"，股票和股指期权常见单调递减的"偏斜"。
(2) 原因：
- 市场下跌时波动率上升（杠杆效应、恐慌），收益分布左尾更厚；
- 投资者需要虚值看跌期权保护下行风险，需求推高其价格；
- 1987 年股灾后偏斜显著出现。
(3) BS 假设对数收益正态、波动率常数。微笑说明真实分布有厚尾和偏斜、波动率随机且与价格相关。改进模型：局部波动率（Dupire）、随机波动率（Heston）、跳跃扩散（Merton）。

</details>

### H11. 期权组合的损益 `基础` `押题`

写出以下组合的到期损益并说明市场观点：(1) 跨式；(2) 牛市看涨价差；(3) 蝶式。若蝶式价差的价格为负，如何套利？

<details>
<summary>答案与解析</summary>

| 组合 | 构成 | 到期损益 | 观点 |
|---|---|---|---|
| 跨式 | 多 $C_K$ + 多 $P_K$ | $\lvert S_T-K\rvert$ | 大幅波动 |
| 牛市价差 | 多 $C_{K_1}$ + 空 $C_{K_2}$（$K_1<K_2$） | $\min(\max(S_T-K_1,0),K_2-K_1)$ | 温和上涨 |
| 蝶式 | 多 $C_{K-h}$ + 空 2 $C_K$ + 多 $C_{K+h}$ | 在 $K$ 处取最大值 $h$ 的三角形，其他地方为 0 | 价格停留在 $K$ 附近 |

蝶式的到期损益非负，所以其价格必须非负，即 $C_{K-h}-2C_K+C_{K+h}\ge0$（看涨价格是 $K$ 的凸函数）。若价格为负，则买入蝶式：期初收到现金，到期损益非负，构成套利。

</details>

### H12. 风险中性定价与 $N(d_2)$ `中等` `经典·Shreve`

(1) 为什么期权价格与股票的期望收益 $\mu$ 无关？(2) 用蒙特卡洛在风险中性测度下为欧式看涨定价，与 BS 公式比较；并验证 $N(d_2)$ 是风险中性下到期实值的概率。

<details>
<summary>答案与解析</summary>

(1) 期权可以用股票和无风险资产动态复制（Delta 对冲），复制组合的成本不依赖 $\mu$。等价地，在风险中性测度 $\mathbb Q$ 下所有资产的期望收益都是 $r$，价格 $=e^{-rT}\mathbb E^{\mathbb Q}[\text{payoff}]$。$\mu$ 只影响真实测度下的行权概率，不影响价格。

```python
import numpy as np
from scipy.stats import norm

S, K, T, r, sig = 100, 105, 0.5, 0.03, 0.25
rng = np.random.default_rng(0)
Z = rng.standard_normal(1_000_000)
ST = S * np.exp((r - sig ** 2 / 2) * T + sig * np.sqrt(T) * Z)
mc = np.exp(-r * T) * np.maximum(ST - K, 0)
d2 = (np.log(S / K) + (r - sig ** 2 / 2) * T) / (sig * np.sqrt(T))
print(f"MC {mc.mean():.4f} ± {1.96 * mc.std() / np.sqrt(len(Z)):.4f}，BS {bs_price(S, K, T, r, sig):.4f}")
print(f"P(S_T > K) = {(ST > K).mean():.4f}，N(d2) = {norm.cdf(d2):.4f}")
```

输出：

```
MC 5.5896 ± 0.0203，BS 5.5760
P(S_T > K) = 0.3904，N(d2) = 0.3899
```

 蒙特卡洛价格与 BS 一致（在 95% 置信区间内）；实值频率与 $N(d_2)$ 一致。

</details>

### H13. 布朗运动与 Itô 引理 `中等` `经典·Shreve`

(1) 用 Itô 引理求 $d(\ln S_t)$，其中 $dS=\mu S\,dt+\sigma S\,dW$。(2) 证明 $\int_0^tW_s\,dW_s=(W_t^2-t)/2$。(3) 用模拟验证 $\mathbb E[W_t^4]=3t^2$ 和 (2)。

<details>
<summary>答案与解析</summary>

(1) $f=\ln S$，$f_S=1/S$，$f_{SS}=-1/S^2$：$d\ln S=\frac{dS}{S}-\frac12\frac{(dS)^2}{S^2}=(\mu-\tfrac12\sigma^2)dt+\sigma\,dW$。所以 $S_T=S_0\exp\big((\mu-\sigma^2/2)T+\sigma W_T\big)$。
(2) 对 $f(W)=W^2$ 用 Itô 引理：$d(W_t^2)=2W_t\,dW_t+dt$，积分得 $W_t^2=2\int_0^tW_s\,dW_s+t$。

```python
import numpy as np

rng = np.random.default_rng(0)
t, n, paths = 2.0, 1000, 20000
dW = rng.standard_normal((paths, n)) * np.sqrt(t / n)
W = np.cumsum(dW, axis=1)
W_prev = np.hstack([np.zeros((paths, 1)), W[:, :-1]])
ito = np.sum(W_prev * dW, axis=1)                      # 左端点求和：Itô 积分
print(f"E[W_t^4] = {np.mean(W[:, -1] ** 4):.3f}，3t^2 = {3 * t ** 2:.1f}")
print(f"Itô 和与 (W_t^2 - t)/2 的平均绝对差 {np.mean(np.abs(ito - (W[:, -1] ** 2 - t) / 2)):.4f}")
```

输出：

```
E[W_t^4] = 12.082，3t^2 = 12.0
Itô 和与 (W_t^2 - t)/2 的平均绝对差 0.0358
```

 $\mathbb E[W_t^4]$ 与 $3t^2=12$ 一致；左端点黎曼和与 $(W_t^2-t)/2$ 的差随步数增加趋于 0（差值等于 $\frac12[t-\sum(\Delta W)^2]$，即二次变差的离散误差）。若取右端点，极限变为 $(W_t^2+t)/2$，说明随机积分依赖取点方式。

</details>

### H14. 期权做市的风险管理 `中等` `押题`

期权做市商持有大量期权头寸，需要管理哪些风险？Delta 中性的组合还有什么风险？

<details>
<summary>答案与解析</summary>

| 风险 | 管理方式 |
|---|---|
| Delta | 用标的（ETF、期货）实时对冲 |
| Gamma | Delta 对冲后的主要风险；通过买卖其他期权控制净 Gamma，临近到期的平值头寸重点关注 |
| Vega | 隐含波动率整体变化；按到期日分桶管理 |
| 偏斜和期限结构 | 不同行权价、到期日的隐含波动率相对变化 |
| 跳跃 | 隔夜或事件跳空，Delta 对冲来不及调整 |
| Pin risk | 到期时标的价格恰好在行权价附近，不确定是否会被行权 |
| 流动性 | 极端行情下难以对冲或平仓 |

做市商的收益主要来自买卖价差和交易所做市返还，而不是方向性观点；核心能力是准确的定价（波动率曲面）和快速的风险对冲。

</details>

---

---

## I. 行业事件与开放题

> 对应讲解第 9 节。事件题要说准时间和数字，并注明出处。

### I1. 2024 年初小盘股流动性危机 `中等` `押题`

2024 年 1–2 月量化中性和 DMA 产品为什么大幅回撤？从拥挤、杠杆、对冲工具、流动性四个方面回答。

<details>
<summary>答案与解析</summary>

**事实**：截至 2024-02-07，万得微盘股指数年内跌幅超过四成；部分中性和 DMA 产品回撤超过 40%（证券时报、中国证券报 2024-02 报道）。

| 方面 | 机制 |
|---|---|
| 拥挤 | 大量量化产品在小市值、低流动性、反转等相似因子上有暴露，持仓高度重叠 |
| 杠杆 | DMA 产品杠杆高，净值下跌触发券商平仓，被动卖出小盘股，形成踩踏 |
| 对冲工具 | 雪球产品敲入后的对冲盘卖出 IC、IM，贴水急剧加深；中性产品股票端与期货端同时亏损 |
| 流动性 | 小盘股在集中卖压下流动性枯竭，冲击成本剧增，止损困难 |

**反思**：控制风格暴露（尤其市值）、控制杠杆、做压力测试、监测行业拥挤度。随后的监管措施：DMA 业务压降、融券收紧、程序化交易监管。

</details>

### I2. 灵均事件 `基础` `押题`

2024-02-19 灵均投资事件的经过和监管处理是什么？对量化机构有什么启示？

<details>
<summary>答案与解析</summary>

**经过**：2024-02-19 开盘后 1 分钟内，灵均旗下产品大量卖出：沪市 9:30–9:31 卖出 11.95 亿元，深市 9:30:00–9:30:42 卖出 13.72 亿元，合计 25.67 亿元，导致指数快速下跌。
**处理**：沪深交易所对相关账户在 2024-02-20 至 02-22 采取限制交易措施，并启动公开谴责程序。公司回应称当日全天实际净买入 1.87 亿元。
**来源**：上交所纪律处分决定书〔2024〕55 号；澎湃新闻、每日经济新闻 2024-02-20。

**启示**：在市场敏感时期，短时间集中大额申报会影响市场稳定；量化机构需要控制单位时间的成交占比和申报速率，执行算法要考虑对市场的影响；这一事件也推动了后续的程序化交易监管。

</details>

### I3. 融券收紧的时间线 `基础` `押题`

按时间顺序说明 2023-10 至 2024-07 融券和转融券的主要政策。

<details>
<summary>答案与解析</summary>

| 日期 | 措施 |
|---|---|
| 2023-10-14 | 融券保证金比例下限由 50% 提高到 80%，私募基金提高到 100%（10-30 起施行）；限制战略投资者配售股份出借 |
| 2024-01-28 | 全面暂停限售股出借；转融券市场化约定申报由实时可用改为次日可用（2024-03-18 起实施） |
| 2024-02-06 | 暂停新增转融券规模；严禁利用融券进行日内回转交易（变相 T+0） |
| 2024-07-10 | 公告自 2024-07-11 起暂停转融券业务，存量合约最迟 2024-09-30 了结；融券保证金比例由 80% 提高到 100%，私募基金由 100% 提高到 120%，2024-07-22 起施行 |

**影响**：基于融券的个股对冲和日内回转基本无法开展，量化产品的空头工具几乎只剩股指期货和场外衍生品。

</details>

### I4. 程序化交易新规对策略的影响 `中等` `押题`

2025-07-07 交易所程序化交易实施细则施行后，高频策略和中低频策略分别受到什么影响？

<details>
<summary>答案与解析</summary>

- **高频策略**：达到每秒 300 笔或每日 20000 笔申报撤单的账户被认定为高频交易，须提交更多报告、承担差异化收费，频繁瞬时撤单等行为受到重点监控；部分以高频撤单为主的策略收益下降、容量受限。
- **中低频策略**：直接影响较小，但大额集中申报（尤其开盘、收盘集合竞价）需要更谨慎地拆单，执行算法需要遵守申报速率限制。
- **行业趋势**：策略向中低频、基本面与量价结合迁移；对交易系统的合规监控（申报速率、撤单率）成为基础设施的一部分。

**说明**：2026 年网上流传的"高频认定标准下调到每秒 15 笔"等说法，目前未见交易所正式文件，回答时不宜作为事实引用。

</details>

### I5. 2025 年指增与中性产品的业绩 `基础` `押题`

2025 年指增产品和中性产品的业绩如何？为什么差异较大？

<details>
<summary>答案与解析</summary>

**数据**（私募排排网）：2025 年全部指增产品平均收益 45.08%、平均超额 16.75%；中证 1000 指增平均收益 49.78%、平均超额 17.49%。中性产品的平均收益明显低于指增（上海证券报 2025-11-25 报道）。

**原因**：
- 指增 = 指数收益 + 超额，2025 年中小盘指数大幅上涨，指数收益贡献很大；
- 中性只赚超额，且需扣除对冲成本；指数上涨、贴水收敛时，期货空头亏损，侵蚀超额（F2）。

</details>

### I6. 量化行业格局 `基础` `押题`

目前国内有多少家百亿量化私募？头部量化私募的业务布局有什么特点？

<details>
<summary>答案与解析</summary>

**数据**：截至 2026-06-30，百亿私募 142 家，其中百亿量化 71 家（私募排排网）。

**特点**（可结合目标公司准备）：
- 产品线以指数增强（沪深 300、中证 500、中证 1000、中证 A500 等）为主，辅以中性、CTA、多策略；
- 研究向 AI 和深度学习倾斜（幻方、九坤等设有 AI 实验室或 AI 方向岗位）；
- 投入大量算力和数据，交易系统自研；
- 规模扩张后注重容量，策略向中低频迁移；
- 监管趋严，合规与风控岗位需求增加。

</details>

### I7. 为什么做量化 `基础` `押题`

面试官："你为什么想做量化研究？你怎么看量化在 A 股的未来？"

<details>
<summary>答案与解析</summary>

**作答框架**：
1. **动机**：结合自身背景（数学、编程、机器学习），说明量化研究是能把这些能力直接转化为可检验结果的领域；举一个具体经历（项目、竞赛、实习）。
2. **对行业的理解**：A 股散户占比高、定价效率仍有提升空间，量化有生存空间；但 alpha 在衰减、监管趋严、2024 年初的回撤说明风险控制和容量管理同样重要。
3. **个人定位**：说明想做的方向（因子研究、机器学习建模、高频），以及为此做的准备。

避免空洞的"热爱金融"或只谈薪资；用具体事实和数字支撑观点。

</details>

---

## 参考来源

- John Hull，*Options, Futures, and Other Derivatives*（第 10 版），2018：第 5 章（远期与期货定价）、第 10–11 章（期权市场与性质）、第 15 章（BS 模型）、第 19 章（Greeks）、第 20 章（波动率微笑）
- Sheldon Natenberg，*Option Volatility and Pricing*（第 2 版），2014
- Steven Shreve，*Stochastic Calculus for Finance II*，2004：第 3–5 章
- Grinold & Kahn，*Active Portfolio Management*（第 2 版），1999
- Menchero, Orr & Wang，*The Barra US Equity Model (USE4): Methodology Notes*，MSCI，2011
- 论文：Fama & French (1993, 2015)；Carhart (1997)；Liu, Stambaugh & Yuan (2019) *Size and Value in China*；Lo (2002) *The Statistics of Sharpe Ratios*；Michaud (1989)；Perold (1988) *The Implementation Shortfall*；Almgren & Chriss (2000)；Moskowitz, Ooi & Pedersen (2012) *Time Series Momentum*
- GitHub · [SoYuCry/awesome-quant-interview](https://github.com/SoYuCry/awesome-quant-interview)：Q35–Q40（组合、回测指标、成本）
- 交易所与监管：上交所、深交所《交易规则》；中金所股指期货、股指期权合约规则与手续费公告；中国期货业协会基础知识；证监会《证券市场程序化交易管理规定（试行）》；沪深北交易所《程序化交易管理实施细则》；上交所纪律处分决定书〔2024〕55 号
- 媒体报道与数据：见[讲解第 11 节](notes.md#11-参考来源)（均注明日期）
- 往年真题：诚奇“对股市的看法”“编程计算收益偏度”（[知乎](https://www.zhihu.com/question/62031149)、[CSDN](https://blog.csdn.net/gitblog_09797/article/details/142943590)）；宽德高频岗“tick 数据与微观结构”（[牛客](https://www.nowcoder.com/discuss/382695299372896256)、[牛客](https://www.nowcoder.com/discuss/393797475918774272)）；九坤量化开发岗“金融计算编程题”，题干未公开。A10、C11 标为 `往年·诚奇`，F8 按宽德考点整理。
