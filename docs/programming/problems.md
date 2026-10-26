# 编程与算法题库

配套讲解见[讲解](notes.md)。代码均为 Python 3。

| 标签 | 含义 |
|---|---|
| `往年·公司名` | 公开面经中记录的往年笔试或面试题，题干经过整理 |
| `经典·书名` | 出自面试书或教材。LC = LeetCode（附题号，Hot 100 中的题标 `LC·Hot100`）；AQI = awesome-quant-interview 题号；CSAPP = *Computer Systems: A Programmer's Perspective*；EMC = *Effective Modern C++* |
| `押题` | 依据往年考点整理的高频题型 |
| `基础` `中等` `困难` | 难度 |

---

## A. 数组、哈希、前缀和

> 对应讲解第 2 节。

### A1. 两数之和 `基础` `经典·LC1·Hot100`

给定整数数组 `nums` 和目标值 `target`，返回和为 `target` 的两个元素的下标（保证恰有一个解，同一元素不能用两次）。

<details>
<summary>答案与解析</summary>

**思路**：边遍历边用哈希表记录"值 → 下标"，对每个 `x` 查 `target - x` 是否出现过。

```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

**追问**：数组有序时可以用对撞双指针做到 $O(1)$ 空间；如果要返回所有不重复的数对，见 B1 三数之和的去重方法。

</details>

### A2. 和为 K 的子数组 `中等` `经典·LC560·Hot100`

给定整数数组 `nums`（可含负数）和整数 `k`，求和为 `k` 的连续子数组个数。

<details>
<summary>答案与解析</summary>

**思路**：子数组 $(j,i]$ 的和 $=S_i-S_j$。遍历到 $i$ 时，答案增加"之前出现过多少个前缀和等于 $S_i-k$"。

```python
from collections import defaultdict

def subarray_sum(nums, k):
    cnt = defaultdict(int)
    cnt[0] = 1                       # 空前缀
    s = ans = 0
    for x in nums:
        s += x
        ans += cnt[s - k]
        cnt[s] += 1
    return ans
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

**为什么不能用滑动窗口**：有负数时，窗口和不随右端扩张单调增加，"和太大就收缩左端"的逻辑不成立。

</details>

### A3. 最长连续序列 `中等` `经典·LC128·Hot100`

给定未排序的整数数组，找出数字连续的最长序列的长度（不要求在原数组中连续），要求 $O(n)$。

<details>
<summary>答案与解析</summary>

**思路**：放进集合。只从"序列起点"（`x - 1` 不在集合中）开始向上数，每个数最多被访问两次。

```python
def longest_consecutive(nums):
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 not in s:          # x 是某段的起点
            y = x
            while y + 1 in s:
                y += 1
            best = max(best, y - x + 1)
    return best
```

**复杂度**：时间 $O(n)$（内层 while 总共只走 $n$ 步），空间 $O(n)$。

</details>

### A4. 除自身以外数组的乘积 `中等` `经典·LC238·Hot100`

返回数组 `ans`，`ans[i]` 等于 `nums` 中除 `nums[i]` 外其余元素的乘积。不能用除法，$O(n)$ 时间。

<details>
<summary>答案与解析</summary>

**思路**：`ans[i]` = 左侧前缀积 × 右侧后缀积。先正序填前缀积，再倒序乘上后缀积。

```python
def product_except_self(nums):
    n = len(nums)
    ans = [1] * n
    for i in range(1, n):
        ans[i] = ans[i - 1] * nums[i - 1]
    right = 1
    for i in range(n - 1, -1, -1):
        ans[i] *= right
        right *= nums[i]
    return ans
```

**复杂度**：时间 $O(n)$，额外空间 $O(1)$（输出数组不计）。

**为什么不用除法**：数组含 0 时除法失效。

</details>

### A5. 多数元素 `基础` `经典·LC169·Hot100`

找出数组中出现次数超过 $\lfloor n/2\rfloor$ 的元素，要求 $O(1)$ 额外空间。

<details>
<summary>答案与解析</summary>

**思路**：Boyer-Moore 投票。不同的两个数互相"抵消"，多数元素抵消不完，最后剩下的一定是它。

```python
def majority_element(nums):
    cand, cnt = None, 0
    for x in nums:
        if cnt == 0:
            cand = x
        cnt += 1 if x == cand else -1
    return cand
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

</details>

### A6. 合并区间 `中等` `经典·LC56·Hot100`

给定若干区间 `[start, end]`，合并所有重叠的区间。

<details>
<summary>答案与解析</summary>

**思路**：按左端点排序；当前区间左端 $\le$ 上一个区间右端时合并（右端取最大）。

```python
def merge_intervals(intervals):
    res = []
    for s, e in sorted(intervals):
        if res and s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res
```

**复杂度**：时间 $O(n\log n)$，空间 $O(n)$。

**量化联系**：合并停牌区间、合并持仓期间、把多个交易时段拼成连续时段。

</details>

### A7. 可以读通讯稿的组数 `中等` `往年·九坤（2022 力扣九坤专场第 1 题）`

给定非负整数数组 `nums`，`rev(x)` 表示把 `x` 的十进制数字反转（如 `rev(120) = 21`）。统计满足 $i<j$ 且 `nums[i] + rev(nums[j]) == nums[j] + rev(nums[i])` 的下标对数，结果对 $10^9+7$ 取模。（题意根据题解博客整理，与 LeetCode 1814 "统计一个数组中好对子的数目" 相同。）

<details>
<summary>答案与解析</summary>

**思路**：移项得 `nums[i] - rev(nums[i]) == nums[j] - rev(nums[j])`。以 $x-\text{rev}(x)$ 为键计数，大小为 $c$ 的组贡献 $\binom c2$ 对。

```python
from collections import Counter

def count_nice_pairs(nums):
    MOD = 10**9 + 7
    def rev(x):
        return int(str(x)[::-1])
    cnt = Counter(x - rev(x) for x in nums)
    return sum(c * (c - 1) // 2 for c in cnt.values()) % MOD
```

**复杂度**：时间 $O(n\cdot d)$（$d$ 为数字位数），空间 $O(n)$。

**方法提示**：看到"两个下标的等式"，先尝试**把 $i$ 和 $j$ 分离到等式两边**，变成"同一个函数值相等"，就能用哈希分组计数。

</details>

### A8. 二维区域和检索 `中等` `经典·LC304`

给定矩阵 `M`，多次查询子矩阵 $(r_1,c_1)$–$(r_2,c_2)$ 的元素和，要求每次查询 $O(1)$。

<details>
<summary>答案与解析</summary>

**思路**：二维前缀和 $S[i+1][j+1]=$ 左上角 $(0,0)$ 到 $(i,j)$ 的和。查询用容斥。

```python
class NumMatrix:
    def __init__(self, M):
        m, n = len(M), len(M[0])
        self.S = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.S[i+1][j+1] = self.S[i][j+1] + self.S[i+1][j] - self.S[i][j] + M[i][j]

    def sum_region(self, r1, c1, r2, c2):
        S = self.S
        return S[r2+1][c2+1] - S[r1][c2+1] - S[r2+1][c1] + S[r1][c1]
```

**复杂度**：预处理 $O(mn)$，查询 $O(1)$。

**量化联系**：日期 × 股票的收益矩阵上，快速求"某段时间、某组股票"的累计收益。

</details>

---

## B. 双指针与滑动窗口

> 对应讲解第 2.3–2.4 节、3.2 节。

### B1. 三数之和 `中等` `经典·LC15·Hot100`

找出数组中所有和为 0 且不重复的三元组。

<details>
<summary>答案与解析</summary>

**思路**：排序后固定第一个数 `nums[i]`，在右侧用对撞指针找两数之和为 `-nums[i]`。三处去重：`i` 跳过重复值，找到解后 `l`、`r` 都跳过重复值。

```python
def three_sum(nums):
    nums = sorted(nums)
    n, res = len(nums), []
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1
                r -= 1
    return res
```

**复杂度**：时间 $O(n^2)$，空间 $O(1)$（不计排序和输出）。

</details>

### B2. 盛最多水的容器 `中等` `经典·LC11·Hot100`

给定高度数组 `h`，选两条线与 x 轴构成容器，求最大面积。

<details>
<summary>答案与解析</summary>

**思路**：对撞指针，每次移动**较矮**的一边（讲解 2.3 节有正确性证明）。

```python
def max_area(h):
    l, r, best = 0, len(h) - 1, 0
    while l < r:
        best = max(best, (r - l) * min(h[l], h[r]))
        if h[l] < h[r]:
            l += 1
        else:
            r -= 1
    return best
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

</details>

### B3. 接雨水 `困难` `经典·LC42·Hot100`

给定柱子高度数组，计算下雨后能接多少雨水。

<details>
<summary>答案与解析</summary>

**思路**：位置 $i$ 的水量 $=\min(\text{左侧最高},\text{右侧最高})-h_i$。双指针：哪边的历史最高更矮，那边的水量就已经确定。

```python
def trap(h):
    l, r = 0, len(h) - 1
    lmax = rmax = water = 0
    while l < r:
        if h[l] < h[r]:
            lmax = max(lmax, h[l])
            water += lmax - h[l]
            l += 1
        else:
            rmax = max(rmax, h[r])
            water += rmax - h[r]
            r -= 1
    return water
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

**为什么正确**：指针总是从较矮的一侧移动，所以较高的柱子会一直"停"在原地。当 $h_l<h_r$ 时，`lmax`（左侧历史最高）不会超过 $h_r$——否则当初那根最高的左柱会比右边所有经过的柱子都高，左指针就不会越过它。于是位置 $l$ 右侧一定有不低于 `lmax` 的柱子，它的水位由 `lmax` 决定。

</details>

### B4. 无重复字符的最长子串 `中等` `经典·LC3·Hot100`

求字符串中不含重复字符的最长子串长度。

<details>
<summary>答案与解析</summary>

**思路**：滑动窗口。记录每个字符最后出现的位置；遇到重复字符时，左端跳到它上次出现位置的下一位。

```python
def length_of_longest_substring(s):
    last = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best
```

**复杂度**：时间 $O(n)$，空间 $O(\Sigma)$（字符集大小）。

</details>

### B5. 最小覆盖子串 `困难` `经典·LC76·Hot100`

给定字符串 `s` 和 `t`，找 `s` 中包含 `t` 所有字符（含重复）的最短子串；不存在返回 `""`。

<details>
<summary>答案与解析</summary>

**思路**：右指针扩张直到窗口覆盖 `t`，然后左指针尽量收缩；用 `need` 计数和 `missing`（还差几个字符）判断是否覆盖。

```python
from collections import Counter

def min_window(s, t):
    need = Counter(t)
    missing = len(t)
    left = 0
    best = (float("inf"), 0, 0)
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:                       # 窗口已覆盖 t，尝试收缩
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right + 1)
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    return s[best[1]:best[2]] if best[0] < float("inf") else ""
```

**复杂度**：时间 $O(|s|+|t|)$，空间 $O(\Sigma)$。

</details>

### B6. 滑动窗口最大值 `困难` `经典·LC239·Hot100`

给定数组和窗口大小 `k`，返回每个窗口的最大值。

<details>
<summary>答案与解析</summary>

**思路**：单调递减双端队列存下标（讲解 3.2 节）。

```python
from collections import deque

def max_sliding_window(nums, k):
    dq, res = deque(), []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:     # 比 x 小的永远不会再成为最大值
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                  # 队首已滑出窗口
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

**复杂度**：时间 $O(n)$（每个下标进出队各一次），空间 $O(k)$。

**量化联系**：滚动最高价（唐奇安通道上轨）、滚动最大回撤都要用它；`rolling().max()` 的底层也是类似算法。

</details>

### B7. $O(1)$ 更新的滚动均值与方差 `中等` `押题`

实现一个类，支持 `push(x)` 加入新数据，并在 $O(1)$ 时间内返回最近 `w` 个数据的均值和样本方差（除以 $w-1$）。

<details>
<summary>答案与解析</summary>

**思路**：维护窗口内的 $\sum x$ 和 $\sum x^2$，新数据进来时加，老数据出去时减。$\text{Var}=\frac{\sum x^2-(\sum x)^2/w}{w-1}$。

```python
from collections import deque

class RollingStats:
    def __init__(self, w):
        self.w, self.q = w, deque()
        self.s = self.s2 = 0.0

    def push(self, x):
        self.q.append(x)
        self.s += x
        self.s2 += x * x
        if len(self.q) > self.w:
            y = self.q.popleft()
            self.s -= y
            self.s2 -= y * y

    def mean(self):
        return self.s / len(self.q)

    def var(self):
        n = len(self.q)
        return (self.s2 - self.s * self.s / n) / (n - 1)
```

**复杂度**：每次操作 $O(1)$，空间 $O(w)$。

**追问：数值稳定性。** 当均值远大于标准差（如价格约 3000、日波动约 10）时，$\sum x^2$ 和 $(\sum x)^2/n$ 是两个很接近的大数相减，会损失精度（灾难性抵消）。更稳健的做法是 **Welford 算法**（维护均值和离差平方和 $M_2$ 的增量更新），或者先减去一个参考值（如第一个价格）再计算。对收益率序列（均值接近 0）这个问题不明显。

</details>

---

## C. 栈与单调栈

> 对应讲解第 3.1 节。

### C1. 有效的括号 `基础` `经典·LC20·Hot100`

判断只含 `()[]{}` 的字符串括号是否合法。

<details>
<summary>答案与解析</summary>

```python
def is_valid(s):
    pair = {")": "(", "]": "[", "}": "{"}
    st = []
    for ch in s:
        if ch in pair:
            if not st or st[-1] != pair[ch]:
                return False
            st.pop()
        else:
            st.append(ch)
    return not st
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

</details>

### C2. 最小栈 `基础` `经典·LC155·Hot100` `往年·诚奇（编程考栈）`

设计一个栈，支持 `push`、`pop`、`top`，并能在 $O(1)$ 时间内取得最小元素。

<details>
<summary>答案与解析</summary>

**思路**：每个位置同时存"当前值"和"到这里为止的最小值"。

```python
class MinStack:
    def __init__(self):
        self.st = []                      # 元素为 (值, 当前最小值)

    def push(self, x):
        cur_min = min(x, self.st[-1][1]) if self.st else x
        self.st.append((x, cur_min))

    def pop(self):
        self.st.pop()

    def top(self):
        return self.st[-1][0]

    def get_min(self):
        return self.st[-1][1]
```

**复杂度**：所有操作 $O(1)$，空间 $O(n)$。

（诚奇的面经只说"栈"，没有给出具体题目，这里选最常考的一道。）

</details>

### C3. 每日温度 `中等` `经典·LC739·Hot100`

对每一天，求要等几天才会出现更高的温度；之后都不会更高则为 0。

<details>
<summary>答案与解析</summary>

**思路**：单调递减栈存"还没等到更高温度"的下标。

```python
def daily_temperatures(T):
    ans = [0] * len(T)
    st = []
    for i, t in enumerate(T):
        while st and T[st[-1]] < t:
            j = st.pop()
            ans[j] = i - j
        st.append(i)
    return ans
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

</details>

### C4. 柱状图中最大的矩形 `困难` `经典·LC84·Hot100`

给定柱子高度，求能勾勒出的最大矩形面积。

<details>
<summary>答案与解析</summary>

**思路**：单调递增栈。柱子 $j$ 被弹出时，右边界是当前下标 $i$（第一个更矮的），左边界是新栈顶（左边第一个更矮的），宽度 $i-\text{栈顶}-1$。末尾加一个高度 0 的哨兵，把栈中剩余的柱子全部弹出。

```python
def largest_rectangle_area(heights):
    h = heights + [0]
    st = [-1]                                   # 哨兵下标
    best = 0
    for i, x in enumerate(h):
        while st[-1] != -1 and h[st[-1]] > x:
            j = st.pop()
            best = max(best, h[j] * (i - st[-1] - 1))
        st.append(i)
    return best
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

</details>

### C5. 股票价格跨度 `中等` `经典·LC901`

设计 `StockSpanner`，每天输入当日股价，返回"从今天往前连续多少天的股价都 $\le$ 今天"（含今天）。

<details>
<summary>答案与解析</summary>

**思路**：单调递减栈存 `(价格, 跨度)`；新价格到来时把栈顶所有 $\le$ 它的弹出，跨度累加。

```python
class StockSpanner:
    def __init__(self):
        self.st = []

    def next(self, price):
        span = 1
        while self.st and self.st[-1][0] <= price:
            span += self.st.pop()[1]
        self.st.append((price, span))
        return span
```

**复杂度**：均摊 $O(1)$，空间 $O(n)$。

**量化联系**：这就是"创 N 日新高"信号的在线计算——跨度 $\ge N$ 即为 N 日新高。

</details>

---

## D. 堆与 Top-K

> 对应讲解第 3.3 节。

### D1. 数组中的第 K 个最大元素 `中等` `经典·LC215·Hot100`

<details>
<summary>答案与解析</summary>

**方法一：大小为 K 的小顶堆**，堆顶就是第 K 大。$O(n\log k)$。

**方法二：快速选择**，期望 $O(n)$。随机选主元，按"大于 / 等于 / 小于"三路划分，只递归进答案所在的那一部分。

```python
import heapq, random

def kth_largest_heap(nums, k):
    h = nums[:k]
    heapq.heapify(h)
    for x in nums[k:]:
        if x > h[0]:
            heapq.heapreplace(h, x)
    return h[0]

def kth_largest_quickselect(nums, k):
    pivot = random.choice(nums)
    big = [x for x in nums if x > pivot]
    eq = [x for x in nums if x == pivot]
    small = [x for x in nums if x < pivot]
    if k <= len(big):
        return kth_largest_quickselect(big, k)
    if k <= len(big) + len(eq):
        return pivot
    return kth_largest_quickselect(small, k - len(big) - len(eq))
```

**复杂度**：堆 $O(n\log k)$ 时间、$O(k)$ 空间；快速选择期望 $O(n)$，最坏 $O(n^2)$（三路划分保证有大量重复值时也不会退化）。

**量化联系**：每天选因子值最高的 K 只股票，用 `np.argpartition(-x, k)[:k]`，就是快速选择。

</details>

### D2. 前 K 个高频元素 `中等` `经典·LC347·Hot100`

<details>
<summary>答案与解析</summary>

**思路**：计数后取前 K。用堆 $O(n\log k)$；或**桶排序**：出现次数作为桶下标，$O(n)$。

```python
from collections import Counter

def top_k_frequent(nums, k):
    cnt = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for x, c in cnt.items():
        buckets[c].append(x)
    res = []
    for c in range(len(buckets) - 1, 0, -1):
        res.extend(buckets[c])
        if len(res) >= k:
            return res[:k]
    return res
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

</details>

### D3. 数据流的中位数 `困难` `经典·LC295·Hot100`

设计数据结构，支持 `add_num(x)` 和 `find_median()`。

<details>
<summary>答案与解析</summary>

**思路**：大顶堆 `lo` 存较小的一半（Python 存负数），小顶堆 `hi` 存较大的一半，保持 `len(lo) == len(hi)` 或多 1 个。

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo, self.hi = [], []            # lo 存负数，模拟大顶堆

    def add_num(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))   # 保证 lo 的最大值 <= hi 的最小值
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

**复杂度**：插入 $O(\log n)$，查询 $O(1)$。

**追问**：如果数据都在 0–100 之间？用计数数组，查询 $O(100)$。如果 99% 的数据在 0–100 之间？计数数组 + 两端溢出计数。

</details>

### D4. 合并 K 个升序链表 `困难` `经典·LC23·Hot100`

（这里用 Python 列表代替链表，思路相同。）给定 K 个升序列表，合并为一个升序列表。

<details>
<summary>答案与解析</summary>

**思路**：堆中存 `(当前值, 列表编号, 下标)`，每次弹出最小值并把该列表的下一个元素放入堆。

```python
import heapq

def merge_k_sorted(lists):
    h = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(h)
    res = []
    while h:
        v, i, j = heapq.heappop(h)
        res.append(v)
        if j + 1 < len(lists[i]):
            heapq.heappush(h, (lists[i][j + 1], i, j + 1))
    return res
```

**复杂度**：时间 $O(N\log K)$（$N$ 为总元素数），空间 $O(K)$。

**量化联系**：把多个交易所、多只股票按时间戳排好序的 tick 流合并成一条全局有序的事件流（回测引擎的事件驱动核心）。标准库 `heapq.merge` 就是这个算法。

</details>

### D5. 滑动窗口中位数 `困难` `经典·LC480`

给定数组和窗口大小 `k`，返回每个窗口的中位数。

<details>
<summary>答案与解析</summary>

**思路**：双堆 + 延迟删除。窗口滑动时要删除的元素不立即从堆里删（堆不支持删除任意元素），而是记在 `delayed` 里，等它到达堆顶时再真正弹出。用 `balance` 记录两堆"有效元素"个数之差。

```python
import heapq
from collections import defaultdict

def median_sliding_window(nums, k):
    lo, hi = [], []                       # lo: 大顶堆（存负数），hi: 小顶堆
    delayed = defaultdict(int)

    def prune(h, sign):                   # 弹出堆顶那些已被删除的元素
        while h and delayed[sign * h[0]]:
            delayed[sign * h[0]] -= 1
            heapq.heappop(h)

    def median():
        return float(-lo[0]) if k % 2 else (-lo[0] + hi[0]) / 2

    for x in nums[:k]:
        heapq.heappush(lo, -x)
    for _ in range(k // 2):
        heapq.heappush(hi, -heapq.heappop(lo))
    res = [median()]

    for i in range(k, len(nums)):
        x, out = nums[i], nums[i - k]
        balance = 0                        # lo 有效个数 - 目标个数 的变化量
        if x <= -lo[0]:
            heapq.heappush(lo, -x); balance += 1
        else:
            heapq.heappush(hi, x); balance -= 1
        delayed[out] += 1
        if out <= -lo[0]:
            balance -= 1
        else:
            balance += 1
        if balance > 0:                    # lo 多了，移一个到 hi
            heapq.heappush(hi, -heapq.heappop(lo))
        elif balance < 0:                  # hi 多了，移一个到 lo
            heapq.heappush(lo, -heapq.heappop(hi))
        prune(lo, -1)
        prune(hi, 1)
        res.append(median())
    return res
```

**复杂度**：时间 $O(n\log n)$（延迟删除使堆中最多累积 $O(n)$ 个元素），空间 $O(n)$。

**说明**：`balance` 两次调整后只可能是 $-2,0,2$，所以一次移动即可恢复平衡；移动前堆顶不会是已删除元素，因为每轮结束都做了 `prune`。

**量化联系**：MAD 去极值需要滚动中位数；`pandas.rolling().median()` 在 C 层用的是类似的跳表 / 双堆算法。面试时能讲清"延迟删除"就够了，不要求一次写对。

</details>

---

## E. 二分查找

> 对应讲解第 4.1 节。

### E1. `bisect` 的复杂度 `基础` `往年·九坤（2023 暑期实习笔试选择题）`

Python 中对长度为 $n$ 的有序列表调用 `bisect.bisect_left` 和 `bisect.insort_left`，时间复杂度分别是多少？

<details>
<summary>答案与解析</summary>

**答案**：`bisect_left` 为 $O(\log n)$；`insort_left` 为 $O(n)$。

**解析**：`insort` 先用二分找位置（$O(\log n)$），再调用 `list.insert`，插入点之后的元素都要后移一位，是 $O(n)$。Python 官方文档也明确写了这一点。

**追问**：需要频繁插入并保持有序怎么办？用平衡树（C++ `std::set`/`map`）、跳表，或者 Python 第三方库 `sortedcontainers.SortedList`（插入约 $O(\log n)$）。

</details>

### E2. 在排序数组中查找元素的第一个和最后一个位置 `中等` `经典·LC34·Hot100`

<details>
<summary>答案与解析</summary>

**思路**：两次 lower_bound：第一个 $\ge x$ 的位置和第一个 $\ge x+1$ 的位置（整数数组）。

```python
def lower_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

def search_range(nums, target):
    l = lower_bound(nums, target)
    if l == len(nums) or nums[l] != target:
        return [-1, -1]
    return [l, lower_bound(nums, target + 1) - 1]
```

**复杂度**：时间 $O(\log n)$，空间 $O(1)$。

</details>

### E3. 搜索旋转排序数组 `中等` `经典·LC33·Hot100`

升序数组（元素互不相同）在某处被旋转（如 `[4,5,6,7,0,1,2]`），在 $O(\log n)$ 时间内查找 `target` 的下标，不存在返回 $-1$。

<details>
<summary>答案与解析</summary>

**思路**：`mid` 把数组分成两半，**至少有一半是有序的**。判断 `target` 是否落在有序的那一半里，决定往哪边走。

```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                  # 左半有序
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                      # 右半有序
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

**复杂度**：时间 $O(\log n)$，空间 $O(1)$。

</details>

### E4. 寻找两个正序数组的中位数 `困难` `经典·LC4·Hot100`

两个有序数组长度为 $m$、$n$，要求 $O(\log(\min(m,n)))$ 求合并后的中位数。

<details>
<summary>答案与解析</summary>

**思路**：在较短数组 `A` 上二分切分点 `i`，`B` 的切分点 `j = (m + n + 1) // 2 - i`，使左半部分恰好有一半元素。合法切分满足 `A[i-1] <= B[j]` 且 `B[j-1] <= A[i]`。

```python
def find_median_sorted_arrays(A, B):
    if len(A) > len(B):
        A, B = B, A
    m, n = len(A), len(B)
    half = (m + n + 1) // 2
    lo, hi = 0, m
    INF = float("inf")
    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i
        a_left = A[i - 1] if i > 0 else -INF
        a_right = A[i] if i < m else INF
        b_left = B[j - 1] if j > 0 else -INF
        b_right = B[j] if j < n else INF
        if a_left <= b_right and b_left <= a_right:
            if (m + n) % 2:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2
        if a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1
```

**复杂度**：时间 $O(\log\min(m,n))$，空间 $O(1)$。

</details>

### E5. 分割数组的最大值（答案二分） `困难` `经典·LC410`

把非负整数数组分成 `k` 个连续非空子数组，使各子数组和的**最大值最小**，返回这个最小值。

<details>
<summary>答案与解析</summary>

**思路**：给定上限 `cap`，贪心地从左往右装，超过 `cap` 就新开一段，看段数是否 $\le k$。`cap` 越大越可行（单调），所以对 `cap` 二分，范围 $[\max(\text{nums}),\sum\text{nums}]$。

```python
def split_array(nums, k):
    def feasible(cap):
        groups, cur = 1, 0
        for x in nums:
            if cur + x > cap:
                groups += 1
                cur = 0
            cur += x
        return groups <= k

    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

**复杂度**：时间 $O(n\log S)$（$S$ 为数组和），空间 $O(1)$。

**方法提示**：看到"最大值最小""最小值最大"，先想**答案二分 + 贪心检查**。

</details>

---

## F. 链表

> 对应讲解第 4.2 节。以下题目统一使用这个节点类：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(vals):                       # 列表 → 链表（测试用）
    dummy = cur = ListNode()
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head):                     # 链表 → 列表（测试用）
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out
```

### F1. 反转链表 `基础` `经典·LC206·Hot100`

<details>
<summary>答案与解析</summary>

```python
def reverse_list(head):
    prev, cur = None, head
    while cur:
        cur.next, prev, cur = prev, cur, cur.next
    return prev

def reverse_list_rec(head):            # 递归版
    if not head or not head.next:
        return head
    new_head = reverse_list_rec(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

**复杂度**：迭代版时间 $O(n)$、空间 $O(1)$；递归版空间 $O(n)$（调用栈）。

**注意**：Python 的多重赋值 `a, b, c = x, y, z` 是先算右边再**从左到右**赋值，`cur.next` 必须写在 `cur` 之前，否则会改到新的 `cur`。

</details>

### F2. 环形链表 II `中等` `经典·LC142·Hot100`

返回链表开始入环的第一个节点；无环返回 `None`。要求 $O(1)$ 空间。

<details>
<summary>答案与解析</summary>

**思路**：快慢指针相遇后，一个指针回到头部，两个指针同速前进，相遇点就是入环点（推导见讲解 4.2 节：$a=kc-b$）。

```python
def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            p = head
            while p is not slow:
                p, slow = p.next, slow.next
            return p
    return None
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

</details>

### F3. 合并两个有序链表 `基础` `经典·LC21·Hot100`

<details>
<summary>答案与解析</summary>

```python
def merge_two_lists(a, b):
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next
```

**复杂度**：时间 $O(m+n)$，空间 $O(1)$。

</details>

### F4. 删除链表的倒数第 N 个结点 `中等` `经典·LC19·Hot100`

<details>
<summary>答案与解析</summary>

**思路**：哑节点 + 快指针先走 $n$ 步，然后快慢同走，快指针到尾时慢指针停在待删节点的前一个。

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return dummy.next
```

**复杂度**：时间 $O(L)$，空间 $O(1)$。

**为什么用哑节点**：要删除的可能是头节点，哑节点让这种情况不用特判。

</details>

### F5. K 个一组翻转链表 `困难` `经典·LC25·Hot100`

每 `k` 个节点一组进行翻转，最后不足 `k` 个的保持原样。

<details>
<summary>答案与解析</summary>

```python
def reverse_k_group(head, k):
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = group_prev                       # 找到本组第 k 个节点
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next
        group_next = kth.next
        prev, cur = group_next, group_prev.next   # 反转本组，尾部接上 group_next
        while cur is not group_next:
            cur.next, prev, cur = prev, cur, cur.next
        first = group_prev.next                # 反转前的组首 = 反转后的组尾
        group_prev.next = kth
        group_prev = first
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

</details>

---

## G. 二叉树

> 对应讲解第 4.3 节。统一使用：

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build_tree(vals):                  # 层序列表（None 表示空）→ 树（测试用）
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    q, i = deque([root]), 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i]); q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i]); q.append(node.right)
        i += 1
    return root
```

### G1. 二叉树的中序遍历（迭代） `基础` `经典·LC94·Hot100`

<details>
<summary>答案与解析</summary>

**思路**：一路向左压栈；弹出时访问，再转向右子树。

```python
def inorder(root):
    res, st, node = [], [], root
    while st or node:
        while node:
            st.append(node)
            node = node.left
        node = st.pop()
        res.append(node.val)
        node = node.right
    return res
```

**复杂度**：时间 $O(n)$，空间 $O(h)$（$h$ 为树高）。

**为什么面试要求迭代**：递归深度过大时 Python 会触发 `RecursionError`（默认上限约 1000），退化成链的树就会出问题。

</details>

### G2. 按层打印二叉树 `基础` `经典·LC102·Hot100` `往年·诚奇（编程考二叉树打印）`

返回二叉树的层序遍历结果，每层一个列表。

<details>
<summary>答案与解析</summary>

```python
def level_order(root):
    if not root:
        return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):           # 当前队列长度 = 本层节点数
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res
```

**复杂度**：时间 $O(n)$，空间 $O(w)$（$w$ 为最大层宽）。

**变体**：之字形打印（奇数层反转）、右视图（每层最后一个）、每层最大值——都只改 `level` 的处理方式。（诚奇面经只写了"二叉树打印"，未给出具体要求，这里选最常见的层序版本。）

</details>

### G3. 二叉树的直径 `基础` `经典·LC543·Hot100`

求任意两节点间最长路径的边数。

<details>
<summary>答案与解析</summary>

**思路**：后序遍历。函数返回"以当前节点为端点向下的最长链（节点数）"，全局答案用"左链 + 右链"更新。

```python
def diameter_of_binary_tree(root):
    best = 0
    def depth(node):
        nonlocal best
        if not node:
            return 0
        l, r = depth(node.left), depth(node.right)
        best = max(best, l + r)
        return 1 + max(l, r)
    depth(root)
    return best
```

**复杂度**：时间 $O(n)$，空间 $O(h)$。

</details>

### G4. 验证二叉搜索树 `中等` `经典·LC98·Hot100`

<details>
<summary>答案与解析</summary>

**思路**：递归时传递当前节点允许的取值区间 $(lo,hi)$。**常见错误**：只比较节点和它的直接孩子——这样无法发现"右子树里的某个孙子小于根"。

```python
def is_valid_bst(root):
    def ok(node, lo, hi):
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return ok(node.left, lo, node.val) and ok(node.right, node.val, hi)
    return ok(root, float("-inf"), float("inf"))
```

**复杂度**：时间 $O(n)$，空间 $O(h)$。

**另一种方法**：中序遍历结果严格递增。

</details>

### G5. 二叉树的最近公共祖先 `中等` `经典·LC236·Hot100`

<details>
<summary>答案与解析</summary>

**思路**：在子树中找 `p` 或 `q`。若左右子树各找到一个，当前节点就是答案；否则返回找到的那一边。

```python
def lowest_common_ancestor(root, p, q):
    if root is None or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```

**复杂度**：时间 $O(n)$，空间 $O(h)$。

</details>

### G6. 二叉树中的最大路径和 `困难` `经典·LC124·Hot100`

路径是任意节点序列（相邻节点有边相连，每个节点最多出现一次，不必经过根），求路径上节点值之和的最大值。节点值可以为负。

<details>
<summary>答案与解析</summary>

**思路**：与 G3 同一个套路。函数返回"从当前节点向下的一条链的最大和"，**负的子链直接丢弃**（取 0）；全局答案用"左链 + 自己 + 右链"更新。

```python
def max_path_sum(root):
    best = float("-inf")
    def gain(node):
        nonlocal best
        if not node:
            return 0
        l = max(gain(node.left), 0)
        r = max(gain(node.right), 0)
        best = max(best, node.val + l + r)
        return node.val + max(l, r)
    gain(root)
    return best
```

**复杂度**：时间 $O(n)$，空间 $O(h)$。（全负数的树也正确：答案为最大的单个节点）

</details>

### G7. 从前序与中序遍历序列构造二叉树 `中等` `经典·LC105·Hot100`

节点值互不相同。

<details>
<summary>答案与解析</summary>

**思路**：前序的第一个是根；在中序中找到根的位置，左边是左子树、右边是右子树。用哈希表 $O(1)$ 找位置，用下标而不是切片避免复制。

```python
def build_from_pre_in(preorder, inorder):
    pos = {v: i for i, v in enumerate(inorder)}
    it = iter(preorder)
    def build(lo, hi):                     # 用中序区间 [lo, hi) 建树
        if lo >= hi:
            return None
        v = next(it)
        node = TreeNode(v)
        node.left = build(lo, pos[v])
        node.right = build(pos[v] + 1, hi)
        return node
    return build(0, len(inorder))
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

**关键**：前序是"根-左-右"，所以必须**先建左子树再建右子树**，迭代器才会按正确顺序吐出节点。

</details>

---

## H. 图

> 对应讲解第 4.4 节。

### H1. 岛屿数量 `中等` `经典·LC200·Hot100`

`'1'` 是陆地、`'0'` 是水，上下左右相连的陆地构成一个岛屿，求岛屿数量。

<details>
<summary>答案与解析</summary>

**思路**：遍历网格，遇到未访问的陆地就 BFS 把整座岛标记掉，计数加一。

```python
from collections import deque

def num_islands(grid):
    g = [list(row) for row in grid]          # 复制一份，避免修改输入
    m, n = len(g), len(g[0])
    cnt = 0
    for i in range(m):
        for j in range(n):
            if g[i][j] == "1":
                cnt += 1
                g[i][j] = "0"
                q = deque([(i, j)])
                while q:
                    x, y = q.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = x + dx, y + dy
                        if 0 <= a < m and 0 <= b < n and g[a][b] == "1":
                            g[a][b] = "0"
                            q.append((a, b))
    return cnt
```

**复杂度**：时间 $O(mn)$，空间 $O(\min(m,n))$（BFS 队列）。

**为什么用 BFS 而不是递归 DFS**：大网格（如 $1000\times1000$ 全是陆地）递归深度可达 $10^6$，Python 会栈溢出。

</details>

### H2. 池塘计数（八连通） `中等` `往年·九坤（2022 力扣九坤专场第 2 题）`

字符网格中 `'W'` 是水、`'.'` 是陆地。水格子在**八个方向**（含对角线）上相邻即属于同一个池塘，求池塘个数。（题意根据题解博客整理。）

<details>
<summary>答案与解析</summary>

**思路**：和 H1 完全相同，方向数组改成 8 个。这里用并查集写一遍，作为另一种方法的练习。

```python
def count_ponds(field):
    m, n = len(field), len(field[0])
    parent = list(range(m * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]        # 路径压缩（隔代）
            x = parent[x]
        return x

    water = 0
    for i in range(m):
        for j in range(n):
            if field[i][j] != "W":
                continue
            water += 1
            # 只需检查已经扫过的 4 个邻居：左、左上、上、右上
            for di, dj in ((0, -1), (-1, -1), (-1, 0), (-1, 1)):
                a, b = i + di, j + dj
                if 0 <= a < m and 0 <= b < n and field[a][b] == "W":
                    ra, rb = find(i * n + j), find(a * n + b)
                    if ra != rb:
                        parent[ra] = rb
                        water -= 1                # 每合并一次，连通块少一个
    return water
```

**复杂度**：时间 $O(mn\,\alpha(mn))$，近似线性；空间 $O(mn)$。

</details>

### H3. 任务依赖排序（拓扑排序） `中等` `往年·九坤（笔试编程题）` `经典·LC210`

共 `task_num` 个任务，`prerequisites` 中 `[i, j]` 表示任务 `i` 依赖任务 `j`（`j` 要先做）。返回一个可行的执行顺序；若存在循环依赖返回空列表。
示例：`task_num = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]`，原题示例输出 `[0, 2, 1, 3]`（`[0, 1, 2, 3]` 也合法）。

<details>
<summary>答案与解析</summary>

**思路**：Kahn 算法（讲解 4.4 节）。

```python
from collections import deque

def task_order(task_num, prerequisites):
    graph = [[] for _ in range(task_num)]
    indeg = [0] * task_num
    for i, j in prerequisites:
        graph[j].append(i)                   # j 完成后才能做 i
        indeg[i] += 1
    q = deque(v for v in range(task_num) if indeg[v] == 0)
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for w in graph[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return order if len(order) == task_num else []
```

**复杂度**：时间 $O(V+E)$，空间 $O(V+E)$。

**追问**：
- 要求字典序最小的顺序？把队列换成小顶堆，$O((V+E)\log V)$。原题示例输出 `[0, 2, 1, 3]` 不是字典序最小，说明原题接受任意合法顺序（具体判题方式面经未说明）。
- 只判断有没有环？DFS 三色标记（白/灰/黑），遇到灰色节点即有环。

**量化联系**：因子计算框架里，因子之间有依赖（如"残差动量"依赖"收益"和"市场收益"），调度器就是按拓扑序计算。

</details>

### H4. 腐烂的橘子（多源 BFS） `中等` `经典·LC994·Hot100`

网格中 0 为空、1 为新鲜橘子、2 为腐烂橘子。每分钟腐烂橘子会使上下左右的新鲜橘子腐烂。求所有橘子腐烂的最少分钟数，不可能则返回 $-1$。

<details>
<summary>答案与解析</summary>

**思路**：把**所有**腐烂橘子同时作为 BFS 起点，按层扩散，层数就是分钟数。

```python
from collections import deque

def oranges_rotting(grid):
    g = [row[:] for row in grid]
    m, n = len(g), len(g[0])
    q = deque((i, j) for i in range(m) for j in range(n) if g[i][j] == 2)
    fresh = sum(row.count(1) for row in g)
    minutes = 0
    while q and fresh:
        for _ in range(len(q)):
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = x + dx, y + dy
                if 0 <= a < m and 0 <= b < n and g[a][b] == 1:
                    g[a][b] = 2
                    fresh -= 1
                    q.append((a, b))
        minutes += 1
    return minutes if fresh == 0 else -1
```

**复杂度**：时间 $O(mn)$，空间 $O(mn)$。

</details>

### H5. 网络延迟时间（Dijkstra） `中等` `经典·LC743`

`times[i] = (u, v, w)` 表示从 `u` 到 `v` 的信号传递需要 `w`（非负）。从节点 `k` 发出信号，求所有节点（编号 1..n）都收到信号的最短时间；有节点收不到返回 $-1$。

<details>
<summary>答案与解析</summary>

```python
import heapq
from collections import defaultdict

def network_delay_time(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    dist = {}
    h = [(0, k)]
    while h:
        d, u = heapq.heappop(h)
        if u in dist:                     # 已确定最短距离，跳过旧记录
            continue
        dist[u] = d
        for v, w in graph[u]:
            if v not in dist:
                heapq.heappush(h, (d + w, v))
    return max(dist.values()) if len(dist) == n else -1
```

**复杂度**：时间 $O(E\log E)$，空间 $O(V+E)$。

**为什么要求非负权**：Dijkstra 假设"弹出时距离已最短"，负权边会打破这一点；有负权用 Bellman-Ford（$O(VE)$），它还能**检测负环**。

**量化联系：汇率套利。** 汇率 $r_{ij}$ 取边权 $-\ln r_{ij}$，存在套利机会 $\Leftrightarrow$ 存在一个环使 $\prod r>1$ $\Leftrightarrow$ $\sum(-\ln r)<0$，即负环。

</details>

### H6. 除法求值（带权并查集） `困难` `经典·LC399·Hot100`

给定等式 `a / b = 2.0`、`b / c = 3.0` 等，回答查询 `a / c = ?`；无法确定返回 $-1.0$。

<details>
<summary>答案与解析</summary>

**思路**：带权并查集。`weight[x]` 表示 `x / parent[x]` 的值；路径压缩时把权重沿路径相乘。

```python
def calc_equation(equations, values, queries):
    parent, weight = {}, {}

    def find(x):
        if parent[x] != x:
            root = find(parent[x])
            weight[x] *= weight[parent[x]]   # x/root = x/parent * parent/root
            parent[x] = root
        return parent[x]

    def union(a, b, v):                       # a / b = v
        for x in (a, b):
            if x not in parent:
                parent[x], weight[x] = x, 1.0
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            weight[ra] = v * weight[b] / weight[a]   # ra/rb = (a/b)·(b/rb)/(a/ra)

    for (a, b), v in zip(equations, values):
        union(a, b, v)
    res = []
    for a, b in queries:
        if a not in parent or b not in parent or find(a) != find(b):
            res.append(-1.0)
        else:
            res.append(weight[a] / weight[b])
    return res
```

**复杂度**：近似 $O((E+Q)\,\alpha(n))$，空间 $O(n)$。

**量化联系**：多币种汇率换算、复权因子的链式换算（各次除权的比例连乘）都是"带权连通"问题。

</details>

---

## I. 动态规划

> 对应讲解第 5.1–5.3 节。

### I1. 打家劫舍 `基础` `经典·LC198·Hot100`

不能偷相邻的两间房，求能偷到的最大金额。

<details>
<summary>答案与解析</summary>

**状态**：$dp[i]$ = 前 $i$ 间房的最大金额。**转移**：$dp[i]=\max(dp[i-1],\ dp[i-2]+a_i)$（不偷 / 偷第 $i$ 间）。只依赖前两项，压缩成两个变量。

```python
def rob(nums):
    prev2 = prev1 = 0
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

</details>

### I2. 最大子数组和 `基础` `经典·LC53·Hot100`

<details>
<summary>答案与解析</summary>

**状态**：$f_i$ = 以 $i$ 结尾的最大子数组和。**转移**：$f_i=\max(f_{i-1}+a_i,\ a_i)$（Kadane 算法）。

```python
def max_sub_array(nums):
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(cur + x, x)
        best = max(best, cur)
    return best
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

**量化联系**：对日收益序列求最大子数组和 = "事后看最赚钱的一段持有期"；把收益取负就得到**最大累计亏损区间**，和最大回撤（M6）是同一类问题（回撤用的是乘法净值）。

</details>

### I3. 最长递增子序列 `中等` `经典·LC300·Hot100`

求严格递增子序列的最大长度，要求 $O(n\log n)$。

<details>
<summary>答案与解析</summary>

**思路**：`tails[k]` = 长度为 $k+1$ 的递增子序列的**最小**结尾。`tails` 单调递增，所以每个新数用二分找到第一个 $\ge x$ 的位置替换（或追加）。

```python
import bisect

def length_of_lis(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

**复杂度**：时间 $O(n\log n)$，空间 $O(n)$。

**注意**：`tails` 本身**不是**一个真实的递增子序列，只是长度对。若要求"非严格递增"，改用 `bisect_right`。

</details>

### I4. 零钱兑换 `中等` `经典·LC322·Hot100`

给定硬币面额（每种无限个）和总金额，求凑成总金额的最少硬币数；凑不出返回 $-1$。

<details>
<summary>答案与解析</summary>

**状态**：$dp[a]$ = 凑出金额 $a$ 的最少硬币数。**转移**：$dp[a]=\min_c dp[a-c]+1$。这是完全背包，金额**正序**遍历。

```python
def coin_change(coins, amount):
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] <= amount else -1
```

**复杂度**：时间 $O(\text{amount}\times|\text{coins}|)$，空间 $O(\text{amount})$。

**为什么不能贪心**：面额 `[1, 3, 4]` 凑 6，贪心先拿 4 得 4+1+1（3 枚），最优是 3+3（2 枚）。

</details>

### I5. 分割等和子集（0-1 背包） `中等` `经典·LC416·Hot100`

判断正整数数组能否分成和相等的两个子集。

<details>
<summary>答案与解析</summary>

**思路**：等价于"能否选出若干数，和为 $S/2$"。$dp[j]$ = 能否凑出 $j$。每个数只能用一次，所以容量**倒序**遍历（避免同一个数被重复使用）。

```python
def can_partition(nums):
    s = sum(nums)
    if s % 2:
        return False
    target = s // 2
    dp = [True] + [False] * target
    for x in nums:
        for j in range(target, x - 1, -1):
            dp[j] = dp[j] or dp[j - x]
    return dp[target]
```

**复杂度**：时间 $O(n\cdot S)$，空间 $O(S)$。

**另一种写法**：用一个大整数做位集合，`bits |= bits << x`，Python 下快得多。

</details>

### I6. 最长公共子序列 `中等` `经典·LC1143·Hot100`

<details>
<summary>答案与解析</summary>

**状态**：$dp[i][j]$ = `a` 的前 $i$ 个与 `b` 的前 $j$ 个字符的 LCS 长度。
**转移**：$a_i=b_j$ 时 $dp[i][j]=dp[i-1][j-1]+1$，否则 $\max(dp[i-1][j],dp[i][j-1])$。

```python
def longest_common_subsequence(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

**复杂度**：时间 $O(mn)$，空间 $O(mn)$，可压缩到 $O(\min(m,n))$。

</details>

### I7. 编辑距离 `困难` `经典·LC72·Hot100`

把 `a` 变成 `b` 的最少操作数（插入、删除、替换一个字符）。

<details>
<summary>答案与解析</summary>

**状态**：$dp[i][j]$ = `a` 前 $i$ 个变成 `b` 前 $j$ 个的最少操作数。
**转移**：$a_i=b_j$ 时 $dp[i][j]=dp[i-1][j-1]$；否则 $1+\min(\underbrace{dp[i-1][j]}_{\text{删}},\ \underbrace{dp[i][j-1]}_{\text{插}},\ \underbrace{dp[i-1][j-1]}_{\text{换}})$。边界 $dp[i][0]=i$，$dp[0][j]=j$。

```python
def min_distance(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n]
```

**复杂度**：时间 $O(mn)$，空间 $O(n)$（滚动数组）。

</details>

### I8. 买卖股票的最佳时机 I、II `基础` `往年·天演（2024 OA 考 LeetCode 股票买卖）` `经典·LC121·LC122`

(a) 只能买卖一次；(b) 可以买卖任意多次（任何时候最多持有一股）。求最大利润。

<details>
<summary>答案与解析</summary>

```python
def max_profit_once(prices):              # LC121
    low, best = float("inf"), 0
    for p in prices:
        low = min(low, p)
        best = max(best, p - low)
    return best

def max_profit_unlimited(prices):         # LC122
    return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices)))
```

**复杂度**：都是时间 $O(n)$，空间 $O(1)$。

**(b) 的贪心为什么对**：任何一段持有 $[i,j]$ 的利润 $p_j-p_i$ 都可以拆成每日涨跌之和；只吃正的日涨幅不会比任何方案差，而且"每天涨了就持有"本身就是一个合法方案。

（天演面经只写了"LeetCode 股票买卖"，没说明是哪一版，I8–I10 覆盖了这个系列的全部常见版本。）

</details>

### I9. 买卖股票的最佳时机 IV（最多 k 笔） `困难` `往年·天演（股票买卖系列）` `经典·LC188·LC123`

最多完成 `k` 笔交易（买入 + 卖出算一笔），求最大利润。`k = 2` 即 LC123。

<details>
<summary>答案与解析</summary>

**状态**：`buy[j]` = 进行到第 $j$ 笔买入后（持有中）的最大收益，`sell[j]` = 完成第 $j$ 笔卖出后的最大收益。
**转移**（对每个价格 $p$）：`buy[j] = max(buy[j], sell[j-1] - p)`，`sell[j] = max(sell[j], buy[j] + p)`。

```python
def max_profit_k(k, prices):
    if not prices or k == 0:
        return 0
    if k >= len(prices) // 2:                 # k 足够大，等价于不限次数
        return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices)))
    buy = [float("-inf")] * (k + 1)
    sell = [0] * (k + 1)
    for p in prices:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j - 1] - p)
            sell[j] = max(sell[j], buy[j] + p)
    return sell[k]
```

**复杂度**：时间 $O(nk)$，空间 $O(k)$。

**细节**：同一天内先更新 `buy[j]` 再用它更新 `sell[j]`，相当于允许"当天买当天卖"，利润为 0，不影响答案。

</details>

### I10. 含冷冻期、含手续费的股票买卖 `中等` `往年·天演（股票买卖系列）` `经典·LC309·LC714`

(a) 不限次数，但卖出后第二天不能买入（冷冻期 1 天）；(b) 不限次数，每笔交易收手续费 `fee`。

<details>
<summary>答案与解析</summary>

```python
def max_profit_cooldown(prices):          # LC309
    hold, sold, rest = float("-inf"), 0, 0  # 持有 / 今天刚卖 / 空仓且可买
    for p in prices:
        hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)
    return max(sold, rest)

def max_profit_fee(prices, fee):          # LC714
    hold, free = float("-inf"), 0
    for p in prices:
        hold, free = max(hold, free - p), max(free, hold + p - fee)
    return free
```

**复杂度**：都是时间 $O(n)$，空间 $O(1)$。

**冷冻期的状态机**：`rest`（可以买）只能来自 `rest` 或昨天的 `sold`，所以刚卖出的第二天不能直接买。

**量化联系**：手续费越高，最优解的交易次数越少——与"因子换手越高，扣费后收益越差，要降低调仓频率"是同一个道理。

</details>

### I11. 鸡蛋掉落 `困难` `往年·九坤（2023 暑期实习笔试编程题）` `经典·LC887`

`k` 个相同的鸡蛋、`n` 层楼。存在临界楼层 $f$（$0\le f\le n$）：从高于 $f$ 的楼层扔下会碎，不高于 $f$ 不会碎；没碎的蛋可以重复使用。求最坏情况下确定 $f$ 的最少扔蛋次数。九坤原题：$1\le k\le100$，$1\le n\le10^4$，时限 2 秒。

<details>
<summary>答案与解析</summary>

**思路**：反过来定义状态（讲解 5.3 节）。$f[t][k]$ = 扔 $t$ 次、有 $k$ 个蛋时**最多能确定的楼层数**：
$$f[t][k]=f[t-1][k-1]+f[t-1][k]+1$$
（这一扔碎了 → 往下还能确定 $f[t-1][k-1]$ 层；没碎 → 往上还能确定 $f[t-1][k]$ 层；再加上这一层。）求最小的 $t$ 使 $f[t][k]\ge n$。

```python
def super_egg_drop(k, n):
    f = [0] * (k + 1)                 # f[j]：当前 t 次、j 个蛋最多确定的层数
    t = 0
    while f[k] < n:
        t += 1
        for j in range(k, 0, -1):     # 倒序：f[j-1] 仍是上一轮（t-1）的值
            f[j] = f[j] + f[j - 1] + 1
    return t
```

**复杂度**：时间 $O(k\cdot t)$，其中 $t\le n$；$k\ge2$ 时 $t=O(\sqrt n)$，$k=100,n=10^4$ 时 $t=14$，瞬间完成。空间 $O(k)$。

**原始 DP 及其问题**：$dp[k][n]=1+\min_{1\le x\le n}\max(dp[k-1][x-1],\ dp[k][n-x])$，$O(kn^2)=10^{10}$，必然超时（这正是复盘作者超时的原因）。中间方案：$dp[k-1][x-1]$ 关于 $x$ 递增、$dp[k][n-x]$ 关于 $x$ 递减，可以对 $x$ 二分，降到 $O(kn\log n)\approx 1.3\times10^7$，C++ 能过。

**闭式解（$k=2$）**：$f[t][2]=\frac{t(t+1)}2$，与数学专题 7.7 节一致。

</details>

### I12. 单词拆分 `中等` `经典·LC139·Hot100`

判断字符串 `s` 能否被拆分为字典 `words` 中的单词（单词可以重复使用）。

<details>
<summary>答案与解析</summary>

**状态**：$dp[i]$ = `s[:i]` 能否被拆分。**转移**：存在 $j<i$ 使 $dp[j]$ 为真且 `s[j:i]` 在字典中。只需枚举不超过最长单词长度的 $j$。

```python
def word_break(s, words):
    wordset = set(words)
    maxlen = max(map(len, wordset), default=0)
    dp = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - maxlen), i):
            if dp[j] and s[j:i] in wordset:
                dp[i] = True
                break
    return dp[len(s)]
```

**复杂度**：时间 $O(n\cdot L^2)$（$L$ 为最长单词长度，切片和哈希各 $O(L)$），空间 $O(n)$。

</details>

---

## J. 贪心、位运算与随机算法

> 对应讲解第 5.4–5.6 节。

### J1. 跳跃游戏 I、II `中等` `经典·LC55·LC45·Hot100`

`nums[i]` 是在位置 $i$ 最多能跳的步数。(a) 能否到达最后一个位置？(b) 保证能到达时，最少跳几次？

<details>
<summary>答案与解析</summary>

```python
def can_jump(nums):                       # LC55
    reach = 0
    for i, x in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + x)
    return True

def min_jumps(nums):                      # LC45：按"层"做 BFS
    jumps = cur_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == cur_end:                  # 走完当前这一跳能覆盖的范围
            jumps += 1
            cur_end = farthest
    return jumps
```

**复杂度**：都是时间 $O(n)$，空间 $O(1)$。

</details>

### J2. 加油站 `中等` `经典·LC134`

环路上有 $n$ 个加油站，`gas[i]` 是站 $i$ 的油量，`cost[i]` 是从站 $i$ 开到 $i+1$ 的耗油。油箱无限大、初始为空，返回能绕一圈的出发站（唯一），否则返回 $-1$。

<details>
<summary>答案与解析</summary>

**思路**：总油量 $<$ 总消耗则无解；否则从起点开始累计 `gas - cost`，一旦累计值为负，说明从当前起点到这里之间的任何站出发都不行，起点改为下一站。

```python
def can_complete_circuit(gas, cost):
    if sum(gas) < sum(cost):
        return -1
    start = tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start, tank = i + 1, 0
    return start
```

**复杂度**：时间 $O(n)$，空间 $O(1)$。

**为什么"中间站都不行"**：设从 $s$ 出发在 $i$ 处首次为负。对 $s<m\le i$，从 $s$ 到 $m-1$ 的累计值 $\ge0$，所以从 $m$ 出发到 $i$ 的累计值 $\le$ 从 $s$ 出发的累计值 $<0$。这和数学专题里"车队油箱"智力题（九坤考过）是同一个结论。

</details>

### J3. 只出现一次的数字 `基础` `经典·LC136·Hot100`

除了一个元素只出现一次外，其余元素都恰好出现两次，找出它，要求 $O(1)$ 空间。追问：其余元素都出现**三次**呢（LC137）？

<details>
<summary>答案与解析</summary>

```python
from functools import reduce
from operator import xor

def single_number(nums):                  # 出现两次：全部异或
    return reduce(xor, nums, 0)

def single_number_iii(nums):              # 出现三次：逐位统计 1 的个数，对 3 取模
    res = 0
    for b in range(32):
        cnt = sum((x >> b) & 1 for x in nums)
        if cnt % 3:
            res |= 1 << b
    return res - (1 << 32) if res >= 1 << 31 else res   # 还原 32 位补码负数
```

**复杂度**：时间 $O(n)$ 和 $O(32n)$，空间 $O(1)$。

**注意**：Python 整数无限精度，负数右移得到的是无限个 1 的补码，所以 32 位循环后要手动还原符号。

</details>

### J4. 比特位计数 `基础` `经典·LC338`

对 $0\le i\le n$，返回每个 $i$ 的二进制中 1 的个数，要求 $O(n)$。

<details>
<summary>答案与解析</summary>

**思路**：`i & (i - 1)` 去掉最低位的 1，所以 `bits[i] = bits[i & (i-1)] + 1`。

```python
def count_bits(n):
    bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits[i] = bits[i & (i - 1)] + 1
    return bits
```

**复杂度**：时间 $O(n)$，空间 $O(n)$。

</details>

### J5. 用 rand7 实现 rand10 `中等` `经典·LC470`

已知 `rand7()` 等概率返回 1–7，实现等概率返回 1–10 的 `rand10()`，并求每次调用 `rand10` 平均调用 `rand7` 的次数。

<details>
<summary>答案与解析</summary>

**思路**：拒绝采样。`(rand7() - 1) * 7 + rand7()` 在 1–49 上均匀；只接受 1–40（10 的倍数），对 10 取模。

```python
import random

def rand7():
    return random.randint(1, 7)

def rand10():
    while True:
        x = (rand7() - 1) * 7 + rand7()       # 1..49 均匀
        if x <= 40:
            return (x - 1) % 10 + 1
```

**期望调用次数**：每轮 2 次 `rand7`，接受概率 $\frac{40}{49}$，轮数服从几何分布，期望 $\frac{49}{40}$ 轮，共 $2\times\frac{49}{40}=2.45$ 次。（模拟 $10^6$ 次：卡方检验不拒绝均匀，平均调用次数 2.45）

**优化**：被拒绝的 41–49 还剩 9 种均匀结果，可以和下一次 `rand7` 组合成 1–63 继续用，期望次数降到约 2.19。

</details>

### J6. 蓄水池抽样 `中等` `经典（Knuth TAOCP 第 2 卷 Algorithm R）`

从长度未知的数据流中等概率抽取 `k` 个元素，只能遍历一次，空间 $O(k)$。写代码并证明每个元素被选中的概率都是 $k/n$。

<details>
<summary>答案与解析</summary>

```python
import random

def reservoir_sample(stream, k):
    pool = []
    for i, x in enumerate(stream, start=1):
        if i <= k:
            pool.append(x)
        else:
            j = random.randint(1, i)          # 以 k/i 的概率选中 x
            if j <= k:
                pool[j - 1] = x
    return pool
```

**证明**：第 $i$ 个元素（$i>k$）在第 $i$ 步被放入的概率是 $\frac ki$；之后第 $m$ 步（$m>i$）它被替换掉的概率是 $\frac km\cdot\frac1k=\frac1m$，保留概率 $\frac{m-1}{m}$。所以最终留在池中的概率
$$\frac ki\prod_{m=i+1}^{n}\frac{m-1}{m}=\frac ki\cdot\frac in=\frac kn$$
前 $k$ 个元素同理（初始概率为 1）。

**复杂度**：时间 $O(n)$，空间 $O(k)$。（$n=10,k=3$ 模拟 $2\times10^5$ 次，每个元素被选中的频率都在 $0.300\pm0.003$ 内）

</details>

### J7. Fisher-Yates 洗牌 `中等` `经典（Knuth TAOCP 第 2 卷 Algorithm P）`

写出等概率随机打乱数组的算法，并指出下面这个"错误洗牌"的问题：`for i in range(n): j = randint(0, n-1); swap(a[i], a[j])`。

<details>
<summary>答案与解析</summary>

```python
import random

def shuffle(a):
    a = list(a)
    for i in range(len(a) - 1, 0, -1):
        j = random.randint(0, i)              # 注意是 [0, i]，包含 i 本身
        a[i], a[j] = a[j], a[i]
    return a
```

**正确性**：最后一个位置从 $n$ 个元素中等概率选一个，倒数第二个位置从剩下 $n-1$ 个中选……共 $n!$ 种等概率结果，每种排列概率 $\frac1{n!}$。

**错误版本的问题**：它有 $n^n$ 种等概率的执行路径，但 $n^n$ 一般**不能被 $n!$ 整除**（如 $n=3$：$27$ 不能被 $6$ 整除），所以各排列不可能等概率。（$n=3$ 模拟 $6\times10^5$ 次：正确版本每种排列约 $\frac16$；错误版本各排列频率为 $\frac4{27},\frac5{27}$ 两种，与精确枚举一致）

**复杂度**：时间 $O(n)$，空间 $O(1)$（原地版本）。

</details>

### J8. 数字默契考验 `中等` `往年·九坤（2022 力扣九坤专场第 3 题）`

有 $N$ 个正整数，每次操作可以把其中一个数乘以 2 或乘以 3。求让所有数相等的最少操作总次数；做不到返回 $-1$。示例：`[50, 75, 100]` → 5；`[10, 14]` → $-1$。（题意根据题解博客整理。）

<details>
<summary>答案与解析</summary>

**思路**：最终的数 $T$ 必须是所有数的公倍数，并且每个 $T/a_i$ 都只含因子 2 和 3。取 $T=\text{lcm}$ 时操作数最少（$T$ 再乘以 2 或 3，每个数都要多操作一次）。所以：求 lcm，检查每个 $\text{lcm}/a_i$ 是否形如 $2^p3^q$，答案为 $\sum(p+q)$。

```python
from math import gcd

def min_operations(nums):
    L = 1
    for x in nums:
        L = L * x // gcd(L, x)
    total = 0
    for x in nums:
        r = L // x
        for p in (2, 3):
            while r % p == 0:
                r //= p
                total += 1
        if r != 1:
            return -1
    return total
```

**复杂度**：时间 $O(N\log M)$（$M$ 为 lcm 大小），空间 $O(1)$。

**更稳妥的写法**：lcm 可能非常大（C++ 会溢出，题解作者特别提醒要用 `long long`）。等价做法是把每个数的 2、3 因子全部除掉，剩下的部分必须全部相同；答案为 $\sum_i(\max p-p_i)+\sum_i(\max q-q_i)$。不需要大整数。

</details>

---

## K. 数据结构设计

### K1. LRU 缓存 `中等` `经典·LC146·Hot100` `押题（九坤复盘的相关推荐里也有华为笔试 LRU）`

设计容量为 `capacity` 的 LRU（最近最少使用）缓存，`get(key)` 和 `put(key, value)` 都要 $O(1)$。面试一般要求**手写双向链表**，不能直接用 `OrderedDict`。

<details>
<summary>答案与解析</summary>

**思路**：哈希表（key → 节点）+ 双向链表（按使用时间排序，头部最新、尾部最旧）。访问时把节点移到头部；超容量时删尾部。用头尾两个哨兵节点省去空指针判断。

```python
class Node:
    __slots__ = ("key", "val", "prev", "next")
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = Node(), Node()        # 哨兵
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _add_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_front(node)
        return node.val

    def put(self, key, value):
        if key in self.map:
            node = self.map[key]
            node.val = value
            self._remove(node)
            self._add_front(node)
            return
        if len(self.map) == self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]                   # 节点里存 key 就是为了这一步
        node = Node(key, value)
        self.map[key] = node
        self._add_front(node)
```

**复杂度**：`get`、`put` 都是 $O(1)$，空间 $O(\text{capacity})$。

**快速版本**（面试官允许时）：`OrderedDict` 的 `move_to_end(key)` 和 `popitem(last=False)`。

**量化联系**：行情 / 因子数据的内存缓存；Python 的 `functools.lru_cache` 就是这个结构。

</details>

### K2. 实现 Trie（前缀树） `中等` `经典·LC208·Hot100`

支持 `insert(word)`、`search(word)`、`starts_with(prefix)`。

<details>
<summary>答案与解析</summary>

```python
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True                     # 单词结束标记

    def _walk(self, s):
        node = self.root
        for ch in s:
            if ch not in node:
                return None
            node = node[ch]
        return node

    def search(self, word):
        node = self._walk(word)
        return node is not None and "$" in node

    def starts_with(self, prefix):
        return self._walk(prefix) is not None
```

**复杂度**：每个操作 $O(L)$（$L$ 为字符串长度），空间为所有单词的字符总数。

**量化联系**：股票代码 / 合约代码的前缀匹配（如找出所有以 `IC` 开头的合约）。

</details>

### K3. 简易限价订单簿 `困难` `押题（宽德高频岗关注微观结构）`

实现一个只含单一股票的撮合引擎：
- `add(order_id, side, price, qty)`：新限价单。按**价格优先、时间优先**与对手方撮合，未成交部分挂单；返回成交列表 `[(对手单号, 成交价, 成交量), ...]`，成交价取挂单方价格。
- `cancel(order_id)`：撤单。
- `best_bid()`、`best_ask()`：返回最优买价 / 卖价，没有则返回 `None`。

<details>
<summary>答案与解析</summary>

**思路**：买方用大顶堆（价格取负）、卖方用小顶堆，堆元素为 `(价格键, 序号, 单号)`，序号保证同价位时间优先。撤单用**延迟删除**：只在字典里删除，堆顶遇到已撤或已成交的订单再弹出。

```python
import heapq, itertools

class OrderBook:
    def __init__(self):
        self.bids, self.asks = [], []        # bids 存 (-price, seq, id)，asks 存 (price, seq, id)
        self.orders = {}                     # id -> [side, price, 剩余量]
        self.seq = itertools.count()

    def _clean(self, heap):
        while heap and heap[0][2] not in self.orders:
            heapq.heappop(heap)

    def best_bid(self):
        self._clean(self.bids)
        return -self.bids[0][0] if self.bids else None

    def best_ask(self):
        self._clean(self.asks)
        return self.asks[0][0] if self.asks else None

    def add(self, oid, side, price, qty):
        trades = []
        book = self.asks if side == "buy" else self.bids
        crosses = (lambda p: p <= price) if side == "buy" else (lambda p: p >= price)
        while qty > 0:
            self._clean(book)
            if not book:
                break
            key, _, rid = book[0]
            rprice = key if side == "buy" else -key
            if not crosses(rprice):
                break
            resting = self.orders[rid]
            fill = min(qty, resting[2])
            trades.append((rid, rprice, fill))
            qty -= fill
            resting[2] -= fill
            if resting[2] == 0:
                del self.orders[rid]
                heapq.heappop(book)
        if qty > 0:
            self.orders[oid] = [side, price, qty]
            if side == "buy":
                heapq.heappush(self.bids, (-price, next(self.seq), oid))
            else:
                heapq.heappush(self.asks, (price, next(self.seq), oid))
        return trades

    def cancel(self, oid):
        return self.orders.pop(oid, None) is not None
```

**复杂度**：下单 $O((1+m)\log n)$（$m$ 为成交的对手单数），撤单 $O(1)$，查询最优价均摊 $O(\log n)$。

**追问与改进**：
- 真实撮合引擎按**价位**组织：`价格 → 该价位的订单队列`，价格用有序结构（C++ `std::map`），或在价格范围有限时用以最小变动价位为下标的数组，取最优价 $O(1)$。
- 价格用**整数**（以最小变动价位为单位）存储，避免浮点比较问题（见 O2）。
- 加上市价单、IOC/FOK 单、改单（通常视为撤单 + 新单，失去时间优先）。

</details>

---

## L. Python 语言

> 对应讲解第 6 节。这类题常以"写出输出"的形式出现。

### L1. 深拷贝与浅拷贝 `基础` `经典·AQI Q16`

写出输出：
```python
import copy
a = [1, [2, 3], {"k": 4}]
b = a
c = a.copy()
d = copy.deepcopy(a)
a[0] = 100
a[1].append(99)
a[2]["k"] = 0
print(b, c, d, sep="\n")
```

<details>
<summary>答案与解析</summary>

**答案**：
```text
[100, [2, 3, 99], {'k': 0}]
[1, [2, 3, 99], {'k': 0}]
[1, [2, 3], {'k': 4}]
```


**解析**：
- `b` 和 `a` 是同一个对象，所有修改都可见；
- `c` 是浅拷贝：外层是新列表，所以 `a[0] = 100`（替换外层元素）不影响 `c`；但内层的列表和字典是共享的，`append` 和字典修改都可见；
- `d` 是深拷贝，完全独立。

**量化中的坑**：
```python
base = {"window": 20, "universe": ["000001", "600000"]}
cfg = base.copy()
cfg["universe"].append("300750")     # base["universe"] 也被改了
```
批量跑参数网格时，每组参数要 `copy.deepcopy(base)`。

</details>

### L2. 可变默认参数 `基础` `押题`

写出输出，并给出正确写法：
```python
def add_signal(name, signals=[]):
    signals.append(name)
    return signals
print(add_signal("mom"))
print(add_signal("rev"))
print(add_signal("vol", []))
print(add_signal("liq"))
```

<details>
<summary>答案与解析</summary>

**答案**：
```text
['mom']
['mom', 'rev']
['vol']
['mom', 'rev', 'liq']
```


**解析**：默认值 `[]` 在 `def` 执行时**只求值一次**，存放在 `add_signal.__defaults__` 中，之后每次不传参都用同一个列表。第三次调用传入了新列表，所以不受影响；第四次又回到那个共享的默认列表。

**正确写法**：
```python
def add_signal_fixed(name, signals=None):
    if signals is None:
        signals = []
    signals.append(name)
    return signals
```

</details>

### L3. 闭包的延迟绑定 `中等` `押题`

写出输出，并修改代码使其输出 `[0, 10, 20]`：
```python
multipliers = [lambda x: i * x for i in range(3)]
print([f(10) for f in multipliers])
```

<details>
<summary>答案与解析</summary>

**答案**：`[20, 20, 20]`

**解析**：`lambda` 捕获的是变量 `i` 本身（闭包查找的是外层作用域中的名字），而不是创建时的值。调用时循环已结束，`i == 2`。

**修正**（任选一种）：
```python
multipliers_fixed = [lambda x, i=i: i * x for i in range(3)]      # 默认参数在定义时求值
import functools, operator
multipliers_partial = [functools.partial(operator.mul, i) for i in range(3)]
```

**量化中的坑**：循环里批量生成因子函数（如 `[lambda df: df.close.pct_change(w) for w in (5, 10, 20)]`），所有函数都会用最后一个窗口 20。

</details>

### L4. GIL 与并行 `中等` `经典·AQI Q18`

(a) GIL 是什么？(b) 用 4 个线程计算 4 段纯 Python 循环求和，会比单线程快吗？下载 4 个文件呢？(c) 在量化中如何实现真正的并行？

<details>
<summary>答案与解析</summary>

**答案**：
(a) CPython 的全局解释器锁，任一时刻只有一个线程执行 Python 字节码，用来保护引用计数等解释器内部状态。
(b) **纯 Python 计算不会变快**（线程轮流持有 GIL，还多了切换开销）；**下载会变快**，因为线程在等待网络 I/O 时会释放 GIL。
(c) 方案：

| 方案 | 适用 |
|---|---|
| `multiprocessing` / `ProcessPoolExecutor` | 按股票、按日期切分的 CPU 密集任务（每个进程有自己的解释器和 GIL） |
| NumPy / pandas 向量化 | 大部分数值计算在 C 层执行，并且会释放 GIL |
| Numba `@njit(nogil=True, parallel=True)` | 无法向量化的循环（如逐笔撮合、路径依赖的计算） |
| Cython、C++ 扩展 | 性能关键模块 |
| `threading` / `asyncio` | 数据下载、数据库读写等 I/O 密集任务 |

**示例**（按股票并行计算因子，Windows 下必须放在 `if __name__ == "__main__":` 里，因为子进程会重新导入主模块）：
```python
from concurrent.futures import ProcessPoolExecutor

def calc_one(code):
    ...                                   # 读取单只股票数据，计算因子
    return code

if __name__ == "__main__":
    codes = ["000001", "600000", "300750"]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(calc_one, codes))
```

**补充**：进程间传递数据要序列化（pickle），大数组的传输开销可能超过计算本身；可以让每个进程自己读文件，或使用共享内存（`multiprocessing.shared_memory`）。Python 3.13 起有实验性的无 GIL 构建（PEP 703），默认版本仍有 GIL。

</details>

### L5. 写一个带参数的重试装饰器 `中等` `经典·AQI Q20`

实现装饰器 `retry(times, exceptions)`：被装饰函数抛出指定异常时最多重试 `times` 次，全部失败则抛出最后一次的异常；要求保留原函数的 `__name__`。

<details>
<summary>答案与解析</summary>

**思路**：带参数的装饰器是**三层**函数：`retry(...)` 返回装饰器，装饰器接收函数返回 `wrapper`。

```python
import functools

def retry(times=3, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
        return wrapper
    return decorator

calls = {"n": 0}

@retry(times=2, exceptions=(ConnectionError,))
def fetch_price(code):
    """模拟取数：前两次失败，第三次成功"""
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("timeout")
    return 10.5
```

**测试结果**：`fetch_price("600000")` 返回 `10.5`，共调用 3 次；`fetch_price.__name__ == "fetch_price"`；若改成 `times=1`，第二次失败后抛出 `ConnectionError`。

**为什么要 `functools.wraps`**：不加的话 `__name__`、`__doc__` 都变成 `wrapper`，日志、调试、序列化（pickle 按名字找函数）都会出问题。

</details>

### L6. 用生成器产出 walk-forward 区间 `中等` `经典·AQI Q21`

给定交易日列表 `dates`，写一个生成器 `walk_forward(dates, train, test, step)`，依次产出 `(训练集日期, 测试集日期)`：训练窗口长 `train`，测试窗口长 `test`，每次向前滚动 `step` 天，测试集不足 `test` 天时停止。

<details>
<summary>答案与解析</summary>

```python
def walk_forward(dates, train, test, step):
    start = 0
    while start + train + test <= len(dates):
        yield dates[start:start + train], dates[start + train:start + train + test]
        start += step
```

**测试**：`dates = list(range(10))`，`train=4, test=2, step=2` 产出 3 组：`([0..3],[4,5])`、`([2..5],[6,7])`、`([4..7],[8,9])`。

**为什么用生成器**：
- 惰性求值：调用方 `for tr, te in walk_forward(...)` 时才逐个计算，不需要一次性生成所有区间；
- 如果每次产出的是**数据本身**（例如训练特征矩阵）而不是日期，生成器避免了同时在内存中保留所有窗口的数据。

**量化联系**：这是 walk-forward 验证的骨架；实际使用时训练集和测试集之间还要留出 **embargo** 间隔，防止标签重叠造成泄漏（见[机器学习讲解](../machine-learning/notes.md)第 3 节）。

</details>

### L7. 几个"写出输出"的小题 `基础` `押题`

```python
grid = [[0] * 3] * 3
grid[0][0] = 1
print(grid)                       # (1)

x = 0.1 + 0.2
print(x == 0.3, round(x, 10) == 0.3)   # (2)

a = [1, 2, 3]
for v in a:
    if v == 2:
        a.remove(v)
print(a)                          # (3)

t = (1, [2])
try:
    t[1] += [3]
except TypeError:
    pass
print(t)                          # (4)
```

<details>
<summary>答案与解析</summary>

**答案**：
```text
[[1, 0, 0], [1, 0, 0], [1, 0, 0]]
False True
[1, 3]
(1, [2, 3])
```


**解析**：
1. `[[0]*3]*3` 是**同一个内层列表**重复 3 次。正确写法：`[[0]*3 for _ in range(3)]`。
2. 浮点误差（见 O1、O2）。
3. 恰好看起来正常：删除 2 后，3 移到下标 1，迭代器下一步访问下标 2，越界结束——3 被**跳过**了。若列表是 `[1, 2, 2, 3]`，结果是 `[1, 2, 3]`，第二个 2 没删掉。**遍历时不要修改列表**，用列表推导式生成新列表。
4. 经典陷阱：`t[1] += [3]` 先执行 `t[1].__iadd__([3])`（**原地修改成功**），再执行 `t[1] = ...`（元组不能赋值，抛 `TypeError`）。所以既报错、又改了。

</details>

### L8. pandas 的链式赋值与 `apply` 性能 `中等` `经典·AQI Q17·Q19`

(a) 下面的代码能把负收益改成 0 吗？正确写法是什么？
```python
import pandas as pd
df = pd.DataFrame({"ret": [0.01, -0.02, 0.03]})
df[df["ret"] < 0]["ret"] = 0
```
(b) 对一个 100 万行的列计算 `x * 2 + 1`，`apply`、`map`、向量化三种写法速度如何？

<details>
<summary>答案与解析</summary>

**(a)** **不能**。`df[df["ret"] < 0]` 先产生一个新对象，再对它的 `ret` 列赋值，`df` 本身不变。
- pandas 2.x 及以前：会给出 `SettingWithCopyWarning`（有时也会"碰巧"生效，取决于中间结果是视图还是副本）；
- **pandas 3.0 起默认启用写时复制（Copy-on-Write）**：链式赋值**永远不会**修改原 DataFrame，并给出 `ChainedAssignmentError` 警告。

正确写法：
```python
df.loc[df["ret"] < 0, "ret"] = 0
# 或者
df["ret"] = df["ret"].clip(lower=0)
```
（pandas 3.0.6 实测：链式写法后 `df` 不变；`.loc` 写法得到 `[0.01, 0.0, 0.03]`）

**(b)** 速度：**向量化 ≫ `map` ≈ `apply`**。`apply`/`map` 对每个元素调用一次 Python 函数；向量化在 C 层一次算完。本机实测（100 万行，pandas 3.0）：向量化约 5 ms，`apply` 和 `map` 约 180 ms，相差约 35 倍（具体数字因机器而异）。

**规则**：能用向量化运算、`np.where`、`groupby().transform` 的就不要 `apply`；必须写循环时用 Numba。

</details>

---

## M. NumPy 与 pandas 量化场景题

> 对应讲解第 7–8 节。本章所有代码都可以直接运行。先运行下面的**数据准备代码**，生成一个模拟的 A 股面板（250 个交易日 × 300 只股票，含行业、市值和一个带有微弱预测力的因子），后面每道题都用这份数据。

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
T, N = 250, 300
dates = pd.bdate_range("2024-01-02", periods=T)
codes = [f"{i:06d}" for i in range(1, N + 1)]
industry = pd.Series(rng.choice(["银行", "医药", "电子", "化工", "食品"], N), index=codes)
log_mcap = pd.Series(rng.normal(23, 1, N), index=codes)          # 对数市值

mkt = rng.normal(0.0003, 0.01, T)                                 # 市场收益
beta_true = rng.uniform(0.5, 1.5, N)
signal = rng.normal(0, 1, (T, N))                                 # t 日可观测的因子
noise = rng.normal(0, 0.02, (T, N))
ret_arr = np.empty((T, N))
ret_arr[0] = mkt[0] * beta_true + noise[0]
ret_arr[1:] = mkt[1:, None] * beta_true + 0.002 * signal[:-1] + noise[1:]   # t 日因子预测 t+1 日收益

ret = pd.DataFrame(ret_arr, index=dates, columns=codes)          # 宽表：日收益
close = 10 * (1 + ret).cumprod()                                  # 宽表：收盘价
factor = pd.DataFrame(signal, index=dates, columns=codes)         # 宽表：因子值
factor = factor + 0.3 * (log_mcap - log_mcap.mean()).values       # 掺入市值暴露，留给中性化去掉

long = (factor.stack().rename("factor").to_frame()
        .join(ret.shift(-1).stack().rename("fwd_ret"))            # 下一日收益作为标签
        .rename_axis(["date", "code"]).reset_index())
long["industry"] = long["code"].map(industry)
long["log_mcap"] = long["code"].map(log_mcap)
```

### M1. 收益率、年化指标与 Sharpe `基础` `押题`

价格序列 `p = [100, 102, 99, 103, 104]`（日频）。(a) 求简单收益与对数收益序列，验证对数收益之和等于 $\ln(p_T/p_0)$；(b) 对 M 章数据中第一只股票，计算年化收益、年化波动率和 Sharpe（无风险利率取 0，一年 252 个交易日）。

<details>
<summary>答案与解析</summary>

```python
p = pd.Series([100, 102, 99, 103, 104], dtype=float)
simple = p.pct_change().dropna()
logret = np.log(p).diff().dropna()
assert np.isclose(logret.sum(), np.log(p.iloc[-1] / p.iloc[0]))     # 对数收益可加
assert np.isclose((1 + simple).prod(), p.iloc[-1] / p.iloc[0])      # 简单收益可乘

def annualized_stats(r, periods=252):
    ann_ret = (1 + r).prod() ** (periods / len(r)) - 1               # 几何年化
    ann_vol = r.std() * np.sqrt(periods)
    sharpe = r.mean() / r.std() * np.sqrt(periods)                  # 用算术均值
    return ann_ret, ann_vol, sharpe

ann_ret, ann_vol, sharpe = annualized_stats(ret.iloc[:, 0])
```

**(a) 数值**：简单收益 $[2.00\%,-2.94\%,4.04\%,0.97\%]$；对数收益 $[1.98\%,-2.99\%,3.96\%,0.97\%]$，和为 $\ln1.04=0.03922$。

**要点**：
- 年化波动乘 $\sqrt{252}$ 的前提是**日收益独立同分布**；有自相关时会低估或高估。
- Sharpe 用**算术均值**；年化收益用**几何复利**。二者的差约为 $\frac{\sigma^2}{2}$（"波动拖累"）。
- **对数收益在时间上可加、简单收益在截面上可加**（组合收益 = 权重 × 个股简单收益）。所以组合计算用简单收益，跨期累加用对数收益。

</details>

### M2. 截面排名与截面标准化 `基础` `押题`

对宽表 `factor` 每天做 (a) 百分位排名；(b) z-score。再用长表 `long` 的 `groupby` 写一遍，验证两种写法结果一致。

<details>
<summary>答案与解析</summary>

```python
# 宽表：axis=1 表示沿股票方向（截面）
rank_wide = factor.rank(axis=1, pct=True)
z_wide = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)

# 长表：按日期分组，用 transform 保持原形状
long["rank"] = long.groupby("date")["factor"].rank(pct=True)
long["z"] = long.groupby("date")["factor"].transform(lambda s: (s - s.mean()) / s.std())

# 检查两种写法一致
chk = long.set_index(["date", "code"])
assert np.allclose(chk["rank"], rank_wide.stack().reindex(chk.index))
assert np.allclose(chk["z"], z_wide.stack().reindex(chk.index))
```


**易错点**：
- `factor.sub(series, axis=0)`：按行对齐（每行减去该日的均值）。直接写 `factor - factor.mean(axis=1)` 会把日期序列按**列标签**（股票代码）对齐，标签全部对不上，得到一张全是 NaN、列数还变多了的表。
- `transform("mean")` 比 `transform(lambda s: s.mean())` 快很多（前者走 Cython 路径）。更快的 z-score 写法：`g = long.groupby("date")["factor"]; (long.factor - g.transform("mean")) / g.transform("std")`。
- rank 与 z-score 的选择：rank 对异常值稳健，但丢掉了"距离"信息；z-score 保留距离，但必须先去极值（M3）。

</details>

### M3. MAD 去极值 `中等` `押题`

实现 MAD 去极值：每天把因子截断在 $[\text{中位数}-5\times1.4826\times\text{MAD},\ \text{中位数}+5\times1.4826\times\text{MAD}]$ 之内，其中 MAD 为"各值与中位数之差的绝对值"的中位数。说明 1.4826 的来历。

<details>
<summary>答案与解析</summary>

```python
def mad_clip(df, n=5):
    """df: 宽表（日期 × 股票），逐日截面去极值"""
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1)
    lo = med - n * 1.4826 * mad
    hi = med + n * 1.4826 * mad
    return df.clip(lower=lo, upper=hi, axis=0)

dirty = factor.copy()
dirty.iloc[0, :3] = [50, -80, 1e6]                    # 人为加入异常值
clean = mad_clip(dirty)
```

**测试**：异常值被截到当日边界；未超出边界的值保持不变；`clip` 前后非异常值完全相同。

**1.4826 的来历**：对正态分布 $N(\mu,\sigma^2)$，$\text{MAD}=\sigma\cdot\Phi^{-1}(0.75)=0.6745\sigma$，所以 $\sigma\approx\frac{\text{MAD}}{0.6745}=1.4826\,\text{MAD}$。也就是说 "$5\times1.4826\times$MAD" 约等于"5 倍稳健标准差"。（`1 / scipy.stats.norm.ppf(0.75) = 1.48260`）

**为什么不用 $3\sigma$**：均值和标准差本身会被极端值拉偏（一个 $10^6$ 能让标准差变得巨大，于是什么都截不掉）；中位数和 MAD 的**崩溃点**是 50%。

</details>

### M4. 行业与市值中性化 `困难` `押题`

对每个交易日，把因子对"行业哑变量 + 对数市值"做截面回归，取残差作为中性化后的因子。验证残差与对数市值的相关系数为 0、各行业内残差均值为 0。

<details>
<summary>答案与解析</summary>

**思路**：每天一次 OLS：$f_i=\sum_k\beta_k D_{ik}+\gamma\ln\text{mcap}_i+\varepsilon_i$。行业哑变量全部放进去时**不再加截距**（否则完全共线）。用 `np.linalg.lstsq` 求解。

```python
def neutralize(df, factor_col="factor"):
    def one_day(g):
        X = pd.get_dummies(g["industry"], dtype=float)
        X["log_mcap"] = g["log_mcap"]
        y = g[factor_col].to_numpy()
        beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
        return pd.Series(y - X.to_numpy() @ beta, index=g.index)
    return df.groupby("date", group_keys=False)[["industry", "log_mcap", factor_col]].apply(one_day)

long["factor_neu"] = neutralize(long)

day = long[long["date"] == long["date"].iloc[0]]
print(np.corrcoef(day["factor"], day["log_mcap"])[0, 1])        # 中性化前：约 0.29
print(np.corrcoef(day["factor_neu"], day["log_mcap"])[0, 1])    # 中性化后：0（数值误差内）
print(day.groupby("industry")["factor_neu"].mean().abs().max()) # 各行业残差均值：0
```

**结果**：中性化前因子与对数市值的相关系数约 0.29（数据准备时故意掺入了市值暴露）；中性化后为 $10^{-16}$ 量级；各行业残差均值也为 $10^{-16}$ 量级。

**原理**：OLS 残差与每个回归变量正交（$X^\top e=0$）。行业哑变量的正交性 ⇒ 每个行业内残差和为 0；市值的正交性 ⇒ 残差与市值的样本协方差为 0。这就是 FWL 定理。

**实践细节**：
- 先去极值、标准化，再中性化；中性化后通常再标准化一次。
- 性能：`groupby().apply` 每天建一次矩阵较慢。全市场 10 年数据可以按日期并行，或者用"组内去均值"处理行业哑变量（FWL），只对市值做一元回归。
- 加权：Barra 风格的回归用市值平方根加权（WLS），减少小盘股的影响。

</details>

### M5. IC、RankIC 与 ICIR `中等` `押题`

用长表 `long` 计算每日 IC（因子与下一日收益的 Pearson 相关）和 RankIC（Spearman 相关），以及 IC 均值、ICIR、IC 的 t 值和 IC 胜率。

<details>
<summary>答案与解析</summary>

```python
def ic_series(df, factor_col, ret_col="fwd_ret", method="pearson"):
    d = df.dropna(subset=[factor_col, ret_col])
    if method == "spearman":                         # Spearman = 秩的 Pearson 相关
        d = d.assign(**{factor_col: d.groupby("date")[factor_col].rank(),
                        ret_col: d.groupby("date")[ret_col].rank()})
    return d.groupby("date")[[factor_col, ret_col]].apply(
        lambda g: g[factor_col].corr(g[ret_col]))

def ic_summary(ic):
    return pd.Series({
        "IC均值": ic.mean(),
        "IC标准差": ic.std(),
        "ICIR": ic.mean() / ic.std(),
        "t值": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "IC>0占比": (ic > 0).mean(),
    })

ic = ic_series(long, "factor_neu")
rank_ic = ic_series(long, "factor_neu", method="spearman")
print(ic_summary(ic).round(3))
print(ic_summary(rank_ic).round(3))
```

**结果**（模拟数据，249 个有效交易日）：IC 均值约 0.093，ICIR 约 1.67，t 值约 26，IC>0 占比约 95%；RankIC 均值约 0.087。（RankIC 与 `scipy.stats.spearmanr` 逐日结果一致）

**解读**：
- 模拟数据的信号故意设得很强。真实 A 股日频因子 |IC| 均值 0.03–0.05 就有价值；**IC 超过 0.1 先怀疑有未来函数**。
- t 值 $=\overline{IC}/s_{IC}\times\sqrt T$ 假设各期 IC 独立。若预测周期是 20 日而每天计算一次，相邻 IC 高度重叠，t 值会被高估约 $\sqrt{20}$ 倍，要用 Newey-West 调整或每 20 天取一次。
- 最后一个交易日的 `fwd_ret` 是 NaN，必须先 `dropna`。

</details>

### M6. 分层回测：五分组、多空净值与换手率 `困难` `押题`

每天按中性化因子把股票分成 5 组（`qcut`），计算各组下一日的等权平均收益、各组累计净值、多空组合（第 5 组减第 1 组）的年化收益与 Sharpe，以及第 5 组的日均单边换手率。

<details>
<summary>答案与解析</summary>

```python
d = long.dropna(subset=["fwd_ret"]).copy()
d["group"] = d.groupby("date")["factor_neu"].transform(
    lambda s: pd.qcut(s, 5, labels=False) + 1)

grp_ret = d.groupby(["date", "group"])["fwd_ret"].mean().unstack()     # 日期 × 组
nav = (1 + grp_ret).cumprod()
ls = grp_ret[5] - grp_ret[1]                                            # 多空组合日收益
ls_ann = ls.mean() * 252
ls_sharpe = ls.mean() / ls.std() * np.sqrt(252)

# 换手率：第 5 组等权持仓，相邻两天权重变化绝对值之和的一半
w = (d[d["group"] == 5].assign(w=1.0)
     .pivot(index="date", columns="code", values="w").fillna(0))
w = w.div(w.sum(axis=1), axis=0)
turnover = w.diff().abs().sum(axis=1).iloc[1:] / 2
print(nav.iloc[-1].round(3))
print(round(ls_ann, 3), round(ls_sharpe, 2), round(turnover.mean(), 3))
```

**结果**（模拟数据）：期末净值从第 1 组到第 5 组单调递增（0.46、0.66、0.90、1.05、1.75）；多空年化收益（算术）约 134%、Sharpe 约 23（模拟信号很强，且因子每天完全重新抽样）；第 5 组日均单边换手约 80%（实测 0.797）。

**要点**：
- 检查**单调性**比只看多空收益更重要：只有两端好、中间乱，说明因子可能只在极端值上有效。
- 换手率定义：单边换手 $=\frac12\sum_i|w_{i,t}-w_{i,t-1}|$。这里因子每天独立重新生成，所以换手极高；真实因子有自相关，换手低得多。
- 扣费：日均单边换手 80%、单边成本 0.1%（佣金 + 冲击，卖出另加 0.05% 印花税），每天成本约 $0.8\times2\times0.1\%$ 级别，年化约 40%——**高换手因子的大部分收益会被成本吃掉**。
- 严格的回测还需要：剔除停牌、涨跌停无法成交的股票，用 $t$ 日收盘后信号、$t+1$ 日的成交价（如 VWAP）计算收益。

</details>

### M7. 最大回撤及其起止日期 `中等` `押题`

给定净值序列，计算最大回撤、回撤开始（前高点）日期和谷底日期，分别用 pandas 和纯 Python 的 $O(n)$ 单次遍历实现。

<details>
<summary>答案与解析</summary>

```python
def max_drawdown(nav):
    """nav: 以日期为索引的净值 Series；返回 (最大回撤, 峰值日, 谷底日)"""
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    return dd.min(), peak, trough

def max_drawdown_loop(values):
    """O(n) 单次遍历，返回 (最大回撤, 峰值下标, 谷底下标)"""
    peak_i, best, best_pair = 0, 0.0, (0, 0)
    for i, v in enumerate(values):
        if v > values[peak_i]:
            peak_i = i
        dd = v / values[peak_i] - 1
        if dd < best:
            best, best_pair = dd, (peak_i, i)
    return best, best_pair[0], best_pair[1]

nav_toy = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3, 0.78, 1.0],
                    index=pd.bdate_range("2024-01-01", periods=7))
print(max_drawdown(nav_toy))          # (-0.4, 2024-01-05, 2024-01-08)
print(max_drawdown_loop(nav_toy.tolist()))
```

**答案**：示例净值的最大回撤为 $0.78/1.3-1=-40\%$，峰值在第 5 天（1.3），谷底在第 6 天（0.78）。注意它**不是**从 1.2 到 0.9 的那次（$-25\%$）。（两种实现在 1000 条随机净值上结果一致）

**追问**：
- Calmar 比率 = 年化收益 / |最大回撤|。
- 回撤持续期（从前高到恢复前高的天数）也常被问到：记录每个点距离上一次创新高的天数，取最大值。
- 最大回撤是**路径依赖**的，样本越长期望越大，不同长度的回测之间不能直接比较。

</details>

### M8. 滚动 Beta `中等` `押题`

用 60 日滚动窗口估计 M 章数据中每只股票对市场收益的 beta：$\beta=\frac{\text{Cov}(r_i,r_m)}{\text{Var}(r_m)}$。用等权市场组合代理市场收益，并用 `np.polyfit` 验证某一个窗口的结果。

<details>
<summary>答案与解析</summary>

```python
mkt_ret = ret.mean(axis=1)                                   # 等权市场组合
beta_roll = ret.rolling(60).cov(mkt_ret).div(mkt_ret.rolling(60).var(), axis=0)

# 验证：第一只股票在最后 60 天上的 OLS 斜率
y, x = ret.iloc[-60:, 0], mkt_ret.iloc[-60:]
slope = np.polyfit(x, y, 1)[0]
assert np.isclose(beta_roll.iloc[-1, 0], slope)
```

**结果**：`rolling().cov(Series)` 会把 Series 与 DataFrame 的每一列配对，一次算出 300 只股票的滚动协方差；与 `polyfit` 的斜率相同。最后一个窗口上 300 只股票的估计 beta 与真实 beta 的相关系数约 0.76（60 个日度样本，估计误差不小）。

**要点**：
- 一元回归斜率 = 协方差 / 方差，所以不需要每个窗口都跑一次回归，复杂度 $O(TN)$。
- **Vasicek / Blume 收缩**：beta 估计噪声大，常用 $\beta_{\text{adj}}=0.67\hat\beta+0.33\times1$（Bloomberg 调整 beta）或向行业均值收缩。
- 用指数加权（`ewm(halflife=...)`）代替等权窗口，可以更快反映 beta 的变化，Barra 模型的 beta 因子就是这样做的。

</details>

### M9. 财报数据的 point-in-time 对齐 `困难` `押题`

财报表 `fin` 含 `code, report_period（报告期）, ann_date（公告日）, eps`；交易日表 `px` 含 `date, code`。把每只股票在每个交易日**已经公告**的最新 EPS 合并到交易日表中。说明为什么不能按报告期对齐。

<details>
<summary>答案与解析</summary>

```python
fin = pd.DataFrame({
    "code": ["A", "A", "A", "B"],
    "report_period": pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30", "2024-03-31"]),
    "ann_date": pd.to_datetime(["2024-03-28", "2024-04-25", "2024-08-20", "2024-04-29"]),
    "eps": [1.0, 0.3, 0.7, 0.5],
})
px = pd.DataFrame({
    "date": pd.to_datetime(["2024-04-01", "2024-04-26", "2024-04-26", "2024-07-01", "2024-08-21"]),
    "code": ["A", "A", "B", "A", "A"],
})

merged = pd.merge_asof(
    px.sort_values("date"),
    fin.sort_values("ann_date"),
    left_on="date", right_on="ann_date", by="code",
    direction="backward",               # 只向过去找：公告日 <= 交易日
)
print(merged[["date", "code", "report_period", "eps"]])
```

**输出**：
```text
        date code report_period  eps
0 2024-04-01    A    2023-12-31  1.0
1 2024-04-26    A    2024-03-31  0.3
2 2024-04-26    B           NaT  NaN
3 2024-07-01    A    2024-03-31  0.3
4 2024-08-21    A    2024-06-30  0.7
```


**解读**：
- 2024-07-01 时半年报（报告期 6-30）还**没有公告**（8-20 才公告），所以只能用一季报。若按报告期对齐，7-01 就会"提前"用到半年报——这是最常见的**未来函数**。
- B 在 4-26 时一季报还没公告（4-29），所以是 NaN。
- `merge_asof` 要求两边都按 `on` 的键排序；`by` 指定按股票分别匹配。
- 更严格的处理：财报会**更正**（同一报告期有多个版本），要保留每个版本的公告日；公告时间若在盘后，信号应从下一个交易日起生效。

</details>

### M10. 计算股票收益的偏度并解释正负 `中等` `往年·诚奇（笔试编程题：计算股票收益偏度并解释正负）`

对 M 章数据中的每只股票计算日收益的样本偏度（不使用 `scipy`），与 `pandas` 的结果核对；解释偏度为正、为负分别意味着什么，A 股个股收益通常是正偏还是负偏。

<details>
<summary>答案与解析</summary>

**公式**：总体偏度 $g_1=\frac{m_3}{m_2^{3/2}}$，$m_k=\frac1n\sum(x_i-\bar x)^k$；pandas 的 `skew()` 用的是无偏调整版 $G_1=\frac{\sqrt{n(n-1)}}{n-2}g_1$。

```python
def skewness(x, bias=False):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    d = x - x.mean()
    m2, m3 = (d ** 2).mean(), (d ** 3).mean()
    g1 = m3 / m2 ** 1.5
    return g1 if bias else g1 * np.sqrt(n * (n - 1)) / (n - 2)

sk = ret.apply(skewness)                       # 每列（每只股票）一个偏度
assert np.allclose(sk, ret.skew())             # 与 pandas 一致

# 一个右偏的例子：多数日子小跌，偶尔大涨
lottery = np.r_[np.full(95, -0.01), np.full(5, 0.15)]
print(round(skewness(lottery), 2))             # 4.19（正偏）
```

**结果**：手写版与 `pandas.DataFrame.skew()`、`scipy.stats.skew(bias=False)` 一致；"彩票型"收益的偏度为 $+4.19$（总体偏度 $g_1=4.13$）。

**解释**：
- **正偏**（右尾长）：大部分时候小亏或小赚，偶尔大涨——"彩票型"股票（小市值、高波动、题材股）。
- **负偏**（左尾长）：大部分时候稳定小赚，偶尔大跌——如卖出期权、套利策略、市场指数。
- **经验事实**：个股日收益通常正偏（个股可以涨很多倍，跌最多 100%；涨跌停限制会截断两侧尾部），**指数收益通常负偏**（下跌时相关性上升，个股一起跌）。
- **量化联系**：投资者偏好正偏股票（彩票偏好）→ 正偏股票被高估、未来收益偏低。"特质偏度"、"最大日收益（MAX）"因子都基于这一现象。

</details>

---

## N. C++ 基础

> 对应讲解第 9 节。九坤（QD 笔试 C++ 占一半）、宽德（笔试含 C++ 题）、灵均（C++/Linux 基础）考过 C++，但面经未给出具体题目，本章为押题。本章的 C++ 代码均用 g++ 13.2（`-std=c++17`）编译运行，输出与答案一致的标。

### N1. 指针与引用 `基础` `经典·AQI Q22`

写出输出，并说明指针与引用的区别：
```cpp
#include <iostream>
void inc_val(int x)  { x++; }
void inc_ref(int& x) { x++; }
void inc_ptr(int* x) { (*x)++; }
int main() {
    int a = 1;
    inc_val(a); inc_ref(a); inc_ptr(&a);
    int& r = a;
    int b = 10;
    r = b;            // 这一行做了什么？
    b = 20;
    std::cout << a << " " << r << " " << b << std::endl;
}
```

<details>
<summary>答案与解析</summary>

**答案**：`10 10 20`

**解析**：
- `inc_val` 传值，修改的是副本；`inc_ref`、`inc_ptr` 各使 `a` 加 1，此时 `a == 3`；
- `r = b` **不是让 `r` 改为引用 `b`**（引用一旦绑定不能更改），而是把 `b` 的值赋给 `r` 所引用的 `a`，所以 `a == 10`；
- 之后修改 `b` 与 `a`、`r` 无关。

**区别**：引用必须初始化、不能为空、不能改绑定；指针可以为 `nullptr`、可以改指向、有自己的地址和 `sizeof`。引用在底层通常用指针实现，但语义上是"别名"。

</details>

### N2. 虚函数、多态与虚析构函数 `中等` `经典·AQI Q23`

写出输出；如果把 `Base` 的析构函数前的 `virtual` 去掉，会发生什么？
```cpp
#include <iostream>
struct Base {
    virtual void name() const { std::cout << "Base\n"; }
    void plain() const { std::cout << "Base::plain\n"; }
    virtual ~Base() { std::cout << "~Base\n"; }
};
struct Derived : Base {
    void name() const override { std::cout << "Derived\n"; }
    void plain() const { std::cout << "Derived::plain\n"; }
    ~Derived() override { std::cout << "~Derived\n"; }
};
int main() {
    Base* p = new Derived();
    p->name();
    p->plain();
    delete p;
}
```

<details>
<summary>答案与解析</summary>

**答案**：
```text
Derived
Base::plain
~Derived
~Base
```


**解析**：
- `name` 是虚函数，通过基类指针调用时按**对象的实际类型**（运行时）分派 → `Derived`；
- `plain` 不是虚函数，按**指针的静态类型**（编译时）决定 → `Base::plain`；
- 析构是虚的，所以先调用派生类析构、再调用基类析构。

**去掉 `virtual ~Base()` 的 `virtual`**：`delete p` 只调用 `~Base`，这是**未定义行为**，派生类持有的资源（内存、文件句柄）会泄漏。规则：**只要类会被多态地使用（有虚函数），析构函数就要声明为虚函数。**

**实现原理**：含虚函数的类有一张虚函数表（vtable），对象头部有一个虚表指针（vptr，64 位下 8 字节）。虚调用 = 取 vptr → 查表 → 间接调用，通常无法内联。低延迟代码的热路径上常用模板（CRTP）代替。

</details>

### N3. 智能指针与循环引用 `中等` `押题`

写出输出，说明为什么 `A`、`B` 的析构函数没有被调用，并修改代码修复。
```cpp
#include <iostream>
#include <memory>
struct B;
struct A { std::shared_ptr<B> b; ~A() { std::cout << "~A\n"; } };
struct B { std::shared_ptr<A> a; ~B() { std::cout << "~B\n"; } };
int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();
        a->b = b;
        b->a = a;
        std::cout << a.use_count() << " " << b.use_count() << "\n";
    }
    std::cout << "end\n";
}
```

<details>
<summary>答案与解析</summary>

**答案**：
```text
2 2
end
```
（`~A`、`~B` 都没有输出）

**解析**：每个对象被两个 `shared_ptr` 持有（局部变量 + 对方的成员）。离开作用域时局部变量销毁，计数各减到 1，永远不会到 0 → 内存泄漏。

**修复**：把其中一方改为 `std::weak_ptr`（不增加引用计数）：
```cpp
struct B { std::weak_ptr<A> a; ~B() { std::cout << "~B\n"; } };
```
修复后输出 `1 2`、`~A`、`~B`、`end`。
（修复后 `A` 只被局部变量 `a` 持有，计数为 1；`B` 被局部变量 `b` 和 `a->b` 持有，计数为 2。离开作用域时局部变量按声明的逆序销毁：先销毁 `b`，`B` 的计数降为 1；再销毁 `a`，`A` 的计数降为 0，`A` 析构，析构时释放成员 `a->b`，`B` 的计数降为 0，随之析构。）

**三种智能指针**：`unique_ptr` 独占所有权、只能移动、零开销（首选）；`shared_ptr` 引用计数（原子操作，有开销）；`weak_ptr` 观察而不拥有，使用前 `lock()` 得到 `shared_ptr`。

</details>

### N4. 移动语义 `中等` `经典·AQI Q24` `经典·EMC 条款 23`

写出输出，并解释 `std::move` 做了什么：
```cpp
#include <iostream>
#include <string>
#include <utility>
#include <vector>
struct Buf {
    std::vector<double> data;
    Buf(size_t n) : data(n) {}
    Buf(const Buf& o) : data(o.data) { std::cout << "copy\n"; }
    Buf(Buf&& o) noexcept : data(std::move(o.data)) { std::cout << "move\n"; }
};
int main() {
    Buf a(1000000);
    Buf b = a;
    Buf c = std::move(a);
    std::cout << a.data.size() << " " << b.data.size() << " " << c.data.size() << "\n";
}
```

<details>
<summary>答案与解析</summary>

**答案**：
```text
copy
move
0 1000000 1000000
```
（`a.data.size()` 为 0 是 libstdc++ 的实际行为；标准只保证被移动的对象处于"有效但未指定"的状态）

**解析**：
- `Buf b = a`：`a` 是左值，调用拷贝构造，复制 100 万个 double（$O(n)$）；
- `Buf c = std::move(a)`：`std::move` **本身不移动任何东西**，只是 `static_cast<Buf&&>(a)`，让重载决议选中移动构造；移动构造把 `vector` 内部的指针"偷"过来（$O(1)$）。
- 移动构造标记 `noexcept` 很重要：`std::vector<Buf>` 扩容时，只有移动构造是 `noexcept` 才会用移动代替拷贝（为了保证强异常安全）。

**量化联系**：在回测 / 交易系统中传递大块行情数据、订单对象时，移动语义避免了不必要的深拷贝。

</details>

### N5. STL 容器的选择 `中等` `经典·AQI Q25`

(a) 为订单簿的买卖价位选容器，要求按价格有序、能快速找到最优价，用什么？(b) 按订单号查订单，用什么？(c) `vector` 在 `push_back` 时，之前保存的迭代器和指针还能用吗？(d) `vector` 的 `reserve` 和 `resize` 有什么区别？

<details>
<summary>答案与解析</summary>

**答案**：
(a) `std::map<Price, Level>`（红黑树，有序；买方用 `std::greater` 比较器或取 `rbegin()`），插入删除 $O(\log n)$，最优价 $O(1)$（`begin()`）。价格范围有限时，用以最小变动价位为下标的数组更快。
(b) `std::unordered_map<OrderId, Order*>`，期望 $O(1)$；需要时 `reserve` 预留桶，避免交易时段内 rehash。
(c) **不一定**：若 `push_back` 触发扩容（`size() == capacity()`），元素被搬到新内存，**所有迭代器、指针、引用失效**；未扩容时，只有 `end()` 失效。
(d) `reserve(n)` 只分配容量，不改变 `size()`，不构造元素；`resize(n)` 改变 `size()`，会构造或销毁元素。已知元素个数时先 `reserve`，避免多次扩容。

**容器复杂度表**见讲解 9.5 节。默认首选 `vector`：连续内存、缓存友好，即使中间插入是 $O(n)$，在元素较少时也常常比 `list` 快。

</details>

### N6. `sizeof` 与内存对齐 `中等` `押题`

在 64 位 Linux / Windows（g++）下，写出各 `sizeof` 的值：
```cpp
#include <iostream>
struct S1 { char c; int i; char d; };
struct S2 { int i; char c; char d; };
struct S3 { char c; double x; };
struct S4 { virtual void f() {} int i; };
struct S5 {};
int main() {
    std::cout << sizeof(S1) << " " << sizeof(S2) << " " << sizeof(S3) << " "
              << sizeof(S4) << " " << sizeof(S5) << " " << sizeof(int*) << "\n";
}
```

<details>
<summary>答案与解析</summary>

**答案**：`12 8 16 16 1 8`

**解析**：
- **对齐规则**：每个成员的起始偏移是其对齐值的倍数；结构体总大小是最大对齐值的倍数。
- `S1`：`c` 在 0，`int` 要对齐到 4 → `i` 在 4–7，`d` 在 8，总大小补齐到 4 的倍数 → 12；
- `S2`：`i` 0–3，`c` 4，`d` 5，补齐到 8；**同样的成员，按从大到小排列更省空间**；
- `S3`：`double` 对齐 8 → `x` 在 8–15，共 16；
- `S4`：虚表指针 8 字节 + `int` 4 字节，补齐到 8 的倍数 → 16；
- `S5`：空类大小为 1（保证不同对象地址不同）；
- 指针 8 字节。

**量化联系**：tick 数据结构体按字段大小排序，可以显著减少内存和缓存占用；多线程写的计数器按 64 字节对齐（`alignas(64)`）可避免伪共享（见 O6）。

</details>

---

## O. 计算机基础

> 对应讲解第 10 节。

### O1. float32 能精确表示哪个数 `基础` `往年·九坤（2023 暑期实习笔试选择题）`

`0.1`、`0.3`、`0.8`、`0.5` 中，哪个能被 float32 精确表示？

<details>
<summary>答案与解析</summary>

**答案**：`0.5`。

**解析**：浮点数能精确表示的只有 $\frac{m}{2^k}$ 形式的数（尾数位数也要够用）。$0.5=2^{-1}$。而 $0.1$ 的二进制是 $0.0\overline{0011}$（无限循环），$0.3$、$0.8$ 同理（分母含因子 5）。

**判断方法**：把小数写成最简分数，**分母是 2 的幂**才可能被精确表示。所以 $0.25$、$0.375=\frac38$、$0.625$ 可以；$0.2$、$0.6$ 不行。

</details>

### O2. 浮点误差的三个坑与 Kahan 求和 `中等` `经典·CSAPP 第 2 章`

(a) `0.1 + 0.2 == 0.3` 的结果？如何正确比较？(b) 在 float64 下 `1e16 + 1 - 1e16` 等于多少？(c) 把 1000 万个 `0.1` 依次相加，结果与 $10^6$ 差多少？写出 Kahan 补偿求和。

<details>
<summary>答案与解析</summary>

**答案**：
(a) `False`。比较用 `math.isclose(a, b, rel_tol=1e-9)` 或 `abs(a - b) < eps`。
(b) `0.0`。float64 有 52 位尾数，$10^{16}>2^{53}\approx9.007\times10^{15}$，相邻可表示数的间隔为 2，`1e16 + 1` 被舍入回 `1e16`（**大数吃小数**）。
(c) 朴素求和结果为 `999999.9998389754`，误差约 $1.6\times10^{-4}$；Kahan 求和与 `math.fsum` 的结果都是 `1000000.0`（误差 $<10^{-9}$）。

```python
import math

def kahan_sum(xs):
    s = c = 0.0                 # c 累积每一步被舍掉的低位
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y         # (t - s) 是实际加上去的部分，减去 y 得到舍入误差
        s = t
    return s

xs = [0.1] * 10_000_000
naive = sum(xs)                  # 注意：Python 3.12 起内置 sum 对浮点也做了补偿求和
```

**注意**：Python 3.12 起，内置 `sum()` 对浮点数**也使用了补偿求和**，所以 `sum([0.1]*10**7)` 在新版本中已经很准；上面的"朴素求和"数字来自显式的 `for` 循环 `s += x`。NumPy 的 `np.sum` 用**两两求和（pairwise summation）**，误差为 $O(\log n\cdot\varepsilon)$。

**量化联系**：累计成千上万笔小额盈亏、计算长期累计收益时，要注意累计误差；价格比较不要用 `==`，交易系统中价格用整数（以最小变动价位为单位）存储。

</details>

### O3. 复杂度判断 `基础` `押题`

写出时间复杂度：
(a) `for i in range(n): j = 1; while j < n: j *= 2`
(b) 归并排序；(c) 用数组建堆（`heapify`）；(d) `T(n) = 2T(n/2) + O(1)`；(e) Python 中 `x in list` 与 `x in set`；(f) 对长度为 $n$ 的列表反复执行 `lst.pop(0)` 直到为空。

<details>
<summary>答案与解析</summary>

**答案**：
(a) $O(n\log n)$；
(b) $O(n\log n)$（主定理：$a=2,b=2,d=1$，$d=\log_ba$）；
(c) $O(n)$（自底向上下沉，高度为 $h$ 的节点约 $n/2^{h+1}$ 个，$\sum_h h\cdot\frac n{2^{h+1}}=O(n)$），而逐个插入是 $O(n\log n)$；
(d) $O(n)$（$d=0<\log_22=1$，结果 $O(n^{\log_ba})=O(n)$，例如二叉树遍历）；
(e) $O(n)$ 与期望 $O(1)$；
(f) $O(n^2)$：每次 `pop(0)` 要把后面所有元素前移。用 `collections.deque.popleft()` 是 $O(1)$。

</details>

### O4. 进程与线程 `中等` `往年·宽德（面试问进程通信）`

(a) 进程和线程的区别？(b) 进程间通信（IPC）有哪些方式？(c) 一个 Python 回测程序要同时跑 100 组参数，用多线程还是多进程？

<details>
<summary>答案与解析</summary>

**答案**：
(a)

| | 进程 | 线程 |
|---|---|---|
| 本质 | 资源分配的单位 | CPU 调度的单位 |
| 地址空间 | 独立 | 共享同一进程的内存 |
| 创建、切换开销 | 大（要切换页表等） | 小 |
| 通信 | 需要 IPC | 直接读写共享变量（需同步） |
| 健壮性 | 一个崩溃不影响其他 | 一个线程崩溃可能拖垮整个进程 |

(b) 管道（pipe，含命名管道 FIFO）、消息队列、**共享内存**（最快，需要自己同步）、信号量、信号（signal）、套接字（socket，可跨机器）、内存映射文件（mmap）。
(c) **多进程**：回测是 CPU 密集任务，受 GIL 限制，多线程不能并行（见 L4）。每组参数相互独立，用 `ProcessPoolExecutor` 即可。

**追问**：
- 线程间同步：互斥锁、读写锁、条件变量、信号量、原子操作。
- 协程（`asyncio`）：用户态调度，单线程内切换，适合大量并发 I/O（如同时订阅几百个行情连接）。

（宽德面经只说"进程通信、CPU、Cache 等基础问题"，具体问法未公开。）

</details>

### O5. 生产者-消费者模型 `中等` `往年·宽德（CTO 面问生产者-消费者模型）`

用 Python 实现：一个生产者线程模拟行情推送（产生 1–100 共 100 个 tick），两个消费者线程从有界缓冲区（容量 10）取数据并累加；全部处理完后主线程打印总和。要求正确结束所有线程。

<details>
<summary>答案与解析</summary>

**思路**：`queue.Queue(maxsize)` 内部用"互斥锁 + 两个条件变量"实现：队列满时 `put` 阻塞，空时 `get` 阻塞。用**哨兵值** `None` 通知消费者结束，每个消费者一个。

```python
import threading, queue

def run_pipeline(n_ticks=100, n_consumers=2, capacity=10):
    buf = queue.Queue(maxsize=capacity)
    total = [0]
    lock = threading.Lock()                 # 保护共享的 total

    def producer():
        for tick in range(1, n_ticks + 1):
            buf.put(tick)                   # 满了会阻塞
        for _ in range(n_consumers):
            buf.put(None)                   # 每个消费者一个结束信号

    def consumer():
        while True:
            item = buf.get()                # 空了会阻塞
            if item is None:
                break
            with lock:
                total[0] += item

    threads = [threading.Thread(target=producer)]
    threads += [threading.Thread(target=consumer) for _ in range(n_consumers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return total[0]
```

**结果**：`run_pipeline()` 返回 5050；改为 4 个消费者、容量 1 也正确，不会死锁。

**要点**：
- 为什么 `total += item` 要加锁：`+=` 不是原子操作（读、加、写三步），两个线程交错执行会丢失更新。
- 有界缓冲区的意义：**背压**——消费者跟不上时生产者被阻塞，而不是无限占用内存。
- C++ 版本：`std::mutex` + `std::condition_variable`，`wait` 时要用**谓词**（`cv.wait(lock, [&]{ return !q.empty(); })`）防止虚假唤醒。
- 低延迟场景（行情线程 → 策略线程）用无锁的单生产者单消费者环形队列（SPSC ring buffer），避免锁的开销。

</details>

### O6. 缓存友好与伪共享 `中等` `往年·宽德（一面 cache miss 场景题）`

(a) 一个 $5000\times5000$ 的 float64 矩阵（C 顺序存储），按行求和与按列求和（都用纯循环），哪个快？为什么？(b) 两个线程分别对两个相邻的计数器做自增，为什么可能比单线程还慢？如何解决？

<details>
<summary>答案与解析</summary>

**答案**：
(a) **按行快**。C 顺序（行优先）下同一行的元素在内存中连续，读一个元素会把同一缓存行（64 字节 = 8 个 double）一起读进 L1，后面 7 次访问都命中缓存。按列访问时相邻两次访问相隔 $5000\times8=40000$ 字节，几乎每次都缓存未命中。差距通常在数倍以上。NumPy 中可以直接对比：
```python
import time
import numpy as np
A = np.random.rand(5000, 5000)                 # C 顺序
t0 = time.perf_counter(); s1 = sum(A[i, :].sum() for i in range(5000)); t1 = time.perf_counter()
s2 = sum(A[:, j].sum() for j in range(5000)); t2 = time.perf_counter()
print(f"按行 {t1 - t0:.3f}s，按列 {t2 - t1:.3f}s")
```
本机实测按行约 0.02 s，按列约 0.15 s（具体数字因机器而异；两种求和结果相同）。

(b) **伪共享（false sharing）**：两个计数器位于同一个 64 字节缓存行中。线程 1 写计数器 A 时，按缓存一致性协议（MESI），线程 2 所在核心的这条缓存行失效；线程 2 写 B 时又反过来。缓存行在两个核心之间反复传递，性能大幅下降。
**解决**：让两个变量位于不同的缓存行——C++ 用 `alignas(64)` 或在中间填充 56 字节；Java 用 `@Contended`；或者每个线程先在局部变量上累加，最后再合并。

**宽德原题背景**：面经提到一面围绕期货交易系统项目给出 cache miss 场景题，二面讨论"用一个较大的结构体存行情字段，因为 L3 缓存足够大"——说明面试官关心数据布局对缓存的影响。

</details>

### O7. 死锁 `中等` `押题`

(a) 死锁的四个必要条件？(b) 下面的代码为什么可能死锁？怎样修改？
```python
import threading
lock_a, lock_b = threading.Lock(), threading.Lock()
def transfer_1():
    with lock_a:
        with lock_b:
            pass
def transfer_2():
    with lock_b:
        with lock_a:
            pass
```

<details>
<summary>答案与解析</summary>

**答案**：
(a) **互斥**（资源一次只能被一个线程使用）、**持有并等待**（持有一个资源的同时等待另一个）、**不可剥夺**（资源不能被强行收回）、**循环等待**（存在等待环）。
(b) 线程 1 持有 `lock_a` 等 `lock_b`，线程 2 持有 `lock_b` 等 `lock_a`，形成循环等待。
**修改**：所有线程按**固定顺序**加锁（例如总是先 `lock_a` 后 `lock_b`，或按锁的 `id()` 排序），破坏循环等待：
```python
def acquire_in_order(*locks):
    return sorted(locks, key=id)

def transfer_safe(l1, l2):
    first, second = acquire_in_order(l1, l2)
    with first:
        with second:
            return True
```
（两个线程分别以 `(lock_a, lock_b)` 和 `(lock_b, lock_a)` 调用 `transfer_safe` 各 10000 次，全部完成，无死锁）

**其他方法**：`lock.acquire(timeout=...)` 超时放弃并重试（破坏"持有并等待"）；一次性申请所有资源；用更粗粒度的单一锁。

</details>

---

## 参考来源

- LeetCode Hot 100（[力扣热题 100](https://leetcode.cn/studyplan/top-100-liked/)）：A–K 章题号为 LeetCode 原题编号
- GitHub · [SoYuCry/awesome-quant-interview](https://github.com/SoYuCry/awesome-quant-interview)：Q16（深浅拷贝）、Q17（apply 性能）、Q18（GIL）、Q19（SettingWithCopyWarning）、Q20（装饰器）、Q21（生成器）、Q22–Q25（C++ 指针引用、虚函数、移动语义、STL 容器）
- Donald Knuth，《The Art of Computer Programming》第 2 卷 3.4.2 节（蓄水池抽样 Algorithm R、洗牌 Algorithm P）
- Scott Meyers，《Effective Modern C++》条款 23（`std::move` 与 `std::forward`）、条款 19–21（智能指针）
- Randal Bryant & David O'Hallaron，《Computer Systems: A Programmer's Perspective》第 2 章（浮点数）、第 6 章（缓存）、第 12 章（并发）
- Python 官方文档：[`bisect`](https://docs.python.org/3/library/bisect.html)（insort 的 $O(n)$ 插入）、[`sum` 在 3.12 中的变化](https://docs.python.org/3/library/functions.html#sum)；[pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html)
- 国内往年题：
  - 九坤：[2023 暑期实习笔试复盘](https://blog.csdn.net/a_beatiful_knife/article/details/130834869)（鸡蛋掉落 $k\le100,n\le10^4$、任务依赖排序及示例、bisect 复杂度、float32）；九坤投资力扣专场竞赛 2022-08-21（[题解 1](https://blog.csdn.net/m0_50176535/article/details/126469812)、[题解 2](https://blog.csdn.net/Strengthennn/article/details/126473338)；题意根据题解整理）
  - 宽德：[牛客 宽德笔试到终面](https://www.nowcoder.com/discuss/393797475918774272)（进程通信、CPU、Cache、生产者-消费者、CTP）
  - 天演：[一亩三分地](https://www.1point3acres.com/bbs/thread-1047719-1-1.html)；诚奇：[知乎](https://www.zhihu.com/question/62031149)、[CSDN](https://blog.csdn.net/gitblog_09797/article/details/142943590)；灵均：[一亩三分地](https://www.1point3acres.com/bbs/thread-1025604-1-1.html)
