# CuTe Layout Algebra 详细教程

## 目录

1. [概述](#1-概述)
2. [核心概念](#2-核心概念)
3. [Layout 基础操作](#3-layout-基础操作)
4. [高级操作](#4-高级操作)
5. [实际应用场景](#5-实际应用场景)
6. [编程接口和示例](#6-编程接口和示例)

---

## 1. 概述

### 1.1 什么是 CuTe Layout？

**CuTe Layout** 是 CUTLASS 3.x 中用于描述数据布局和访问模式的核心抽象。本质上，**Layout 是一个从整数到整数的函数**：

```
Layout: ℕⁿ → ℕ
```

这个函数将**逻辑坐标**（如矩阵的行列索引）映射到**物理地址**（内存中的线性偏移）。

### 1.2 为什么需要 Layout Algebra？

在 GPU 编程中，我们需要处理：
- **复杂的内存布局**：行主序、列主序、分块布局等
- **数据重组**：转置、reshape、tiling
- **线程协作**：将数据分配给不同线程和线程块
- **访存优化**：合并访存、避免 bank conflict

Layout Algebra 提供了一套**数学化的工具**来描述和变换这些模式，使代码更清晰、可组合、可验证。

### 1.3 Layout Algebra 的核心思想

Layout Algebra 提供了一组操作，将复杂的数据布局变换分解为简单操作的组合：

```
复杂变换 = 简单操作₁ ∘ 简单操作₂ ∘ ... ∘ 简单操作ₙ
```

这种代数化的方法带来：
- **可组合性**：小的 Layout 可以组合成复杂的 Layout
- **可验证性**：每个操作都有明确的数学性质和后置条件
- **编译时优化**：利用 C++ 模板元编程在编译期计算

---

## 2. 核心概念

### 2.1 Shape 和 Stride

**Layout** 由两个核心组件定义：

#### Shape（形状）

**定义**：每个维度的大小/范围

**表示法**：
```cpp
Shape = (S₀, S₁, S₂, ...)
```

**重要说明**：
- 对于矩阵，通常第 0 维是行（M），第 1 维是列（N）
- Shape `(M, N)` 表示 M 行 N 列的矩阵
- 这与数学中的表示一致：C[M, N] = A[M, K] × B[K, N]

**示例**：
- `(6, 4)`：6 行 4 列的矩阵
- `(128, 8, 16)`：三维数据，大小为 128×8×16

#### Stride（步长）

**定义**：在每个维度上移动一个单位时，物理地址的增量

**表示法**：
```cpp
Stride = (D₀, D₁, D₂, ...)
```

**示例**：
- `(4, 1)`：行主序，行方向步长为 4，列方向步长为 1（同一行元素连续）
- `(1, 6)`：列主序，行方向步长为 1，列方向步长为 6（同一列元素连续）

**解释**：
- **行主序**：在行方向（mode-0）移动一步，跳过整行的元素（N 个），所以 stride₀ = N
- **列主序**：在列方向（mode-1）移动一步，跳过整列的元素（M 个），所以 stride₁ = M

#### Layout 的完整表示

**记号**：`Shape:Stride`

**示例**：
```
(6, 4):(4, 1)  // 6×4 矩阵，行主序（行内连续）
(6, 4):(1, 6)  // 6×4 矩阵，列主序（列内连续）
(8, 2):(2, 1)  // 8×2 矩阵，行主序
```

### 2.2 Layout 作为函数

#### 扁平 Layout 的映射

给定**扁平** Layout `(S₀, S₁, ...):(D₀, D₁, ...)`，其对应的函数为：

```
Layout(i₀, i₁, ...) = i₀ × D₀ + i₁ × D₁ + ...
```

其中：
- `0 ≤ iₖ < Sₖ`（坐标在 Shape 范围内）

#### 层次化 Layout 的递归映射

对于**层次化** Layout，映射是**递归定义**的。每个模态可以是：
1. **标量**：直接返回值
2. **元组**：递归地应用映射

**递归定义**：
```
Layout(shape, stride)(coord) =
  if shape is scalar:
    coord × stride
  else:  // shape and stride are tuples
    sum(Layout(shape[i], stride[i])(coord[i]) for i in modes)
```

**通用公式**：
```
Layout((S₀, S₁, ...), (D₀, D₁, ...))(i₀, i₁, ...) =
  Layout(S₀, D₀)(i₀) + Layout(S₁, D₁)(i₁) + ...
```

其中每个 `Layout(Sₖ, Dₖ)(iₖ)` 可能进一步递归展开。

**示例 1：行主序矩阵**
```
Layout: (4, 3):(3, 1)

计算 Layout(2, 1)：
= 2 × 3 + 1 × 1
= 7

物理内存布局：
[0, 1, 2,  3, 4, 5,  6, 7, 8,  9, 10, 11]
 ------   ------   -------   ---------
  row 0    row 1    row 2     row 3
```

**示例 2：列主序矩阵**
```
Layout: (4, 3):(1, 4)

计算 Layout(2, 1)：
= 2 × 1 + 1 × 4
= 6

物理内存布局：
[0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11]
 ---------   ---------   -----------
   col 0       col 1        col 2
```

### 2.3 多模态 Layout（Hierarchical Layout）

CuTe 支持**嵌套的 Shape 和 Stride**，用于表达层次化的结构：

**语法**：
```cpp
Shape:  ((S₀₀, S₀₁), S₁, S₂, ...)
Stride: ((D₀₀, D₀₁), D₁, D₂, ...)
```

**示例**：分块矩阵
```cpp
// 将 8×8 矩阵（行主序）分成 2×2 个 4×4 块
Layout: ((4, 4), (2, 2)):((8, 1), (32, 4))
         ^^^^^   ^^^^^    ^^^^^   ^^^^^^
        块内4×4  2×2个块  块内步长 块间步长
```

**递归映射计算**：

给定坐标 `((i₀, i₁), (j₀, j₁))`，递归地计算：

```
Layout(((4,4), (2,2)), ((8,1), (32,4)))((i₀,i₁), (j₀,j₁)) =
  Layout((4,4), (8,1))(i₀, i₁) + Layout((2,2), (32,4))(j₀, j₁)
  = [i₀×8 + i₁×1] + [j₀×32 + j₁×4]
  = 8i₀ + i₁ + 32j₀ + 4j₁
```

**示例计算**：访问 Block(1,1) 的元素 [2,3]
```
坐标: ((2, 3), (1, 1))
offset = 8×2 + 1×3 + 32×1 + 4×1
       = 16 + 3 + 32 + 4
       = 55

验证：原矩阵 8×8 中，第 7 行第 8 列（0-indexed: [6,7]）
     = 6×8 + 7 = 55 ✓
```

**理解**：
- **内层 `(4, 4):(8, 1)`**：每个块是 4×4，行主序
  - 块内行步长 = 8（原矩阵的列数）
  - 块内列步长 = 1
- **外层 `(2, 2):(32, 4)`**：有 2×2 = 4 个块
  - 行方向块间距 = 32（4行 × 8列）
  - 列方向块间距 = 4（4列）

**可视化**：
```
8×8 矩阵分块：
Block(0,0): rows [0-3], cols [0-3]
Block(0,1): rows [0-3], cols [4-7]
Block(1,0): rows [4-7], cols [0-3]
Block(1,1): rows [4-7], cols [4-7]
```

---

## 3. Layout 基础操作

### 3.1 Coalesce（合并）

#### 定义

**Coalesce** 简化 Layout，通过合并相邻模态（mode），但**不改变数学函数**。

#### 合并规则

1. **消除大小为 1 的模态**
   ```
   (6, 1, 4):(2, X, 12) → (6, 4):(2, 12)
   ```

2. **合并相邻模态**（当满足条件时）
   - 条件：`stride₁ = size₀ × stride₀`
   - 结果：`(size₀ × size₁):(stride₀)`

   ```
   (4, 3):(1, 4) → (12):(1)
   // 因为 stride₁(4) = size₀(4) × stride₀(1)
   ```

#### 示例

**示例 1：消除单元素维度**
```cpp
原始: (8, 1, 16):(4, X, 32)
合并后: (8, 16):(4, 32)
// 中间的大小为 1 的维度被消除
```

**示例 2：合并连续维度**
```cpp
原始: (4, 8):(1, 4)
合并后: (32):(1)

验证：
- 原始 Layout(3, 2) = 3×1 + 2×4 = 11
- 合并后 Layout(11) = 11×1 = 11 ✓
```

**示例 3：无法合并的情况**
```cpp
原始: (4, 8):(2, 4)
无法合并！

原因：stride₁(4) ≠ size₀(4) × stride₀(2) = 8
```

#### 后置条件

- **大小不变**：`size(coalesce(layout)) == size(layout)`
- **深度最小**：`depth(coalesce(layout)) ≤ 1`（尽可能扁平）
- **函数等价**：对所有坐标 `c`，`coalesce(layout)(c) == layout(c)`

#### CuTe 代码

```cpp
#include <cute/layout.hpp>

using namespace cute;

// 定义 Layout
auto layout = make_layout(make_shape(4, 8), make_stride(1, 4));
// (4,8):(1,4)

// Coalesce
auto coalesced = coalesce(layout);
// (32):(1)

// 验证
assert(size(coalesced) == size(layout));
assert(coalesced(11) == layout(3, 2));
```

---

### 3.2 Composition（组合）

#### 定义

**Composition** 是函数复合操作：给定两个 Layout `A` 和 `B`，组合 `R := A ∘ B` 定义为：

```
R(c) := A(B(c))
```

即先应用 `B`，再应用 `A`。

#### 语义

Composition 将 `B` 的**逻辑坐标空间**映射到 `A` 的**物理地址空间**，通过 `B` 的映射结果作为 `A` 的输入。

#### 数学计算

给定：
- `A = (Shape_A):(Stride_A)`
- `B = (Shape_B):(Stride_B)`

计算 `R = A ∘ B` 的步骤：

1. **对每个 `B` 的 stride**，在 `A` 中"除掉"（factor out）
2. **对每个 `B` 的 shape**，在结果中"取模"（mod out）
3. 组合结果

**伪代码**：
```
for each (shape_b, stride_b) in B:
    layout_a, remainder = divide(layout_a, stride_b)
    result = concatenate(result, layout_a % shape_b)
    layout_a = remainder
return concatenate(result, layout_a)
```

#### 示例

**示例 1：使用 Composition 重新索引**

Composition 更常用于重新索引数据访问模式，而非简单的 reshape。

```cpp
// 原始 Layout：16 个元素，stride=2（隔一个取一个）
A = (16):(2)
// 值域：{0, 2, 4, 6, ..., 30}

// 逻辑索引：连续的 0-7
B = (8):(1)

// Composition
R = A ∘ B
  = (8):(2)

// 解释：
// B 生成连续索引 0,1,2,...,7
// A 将这些索引映射到 {0,2,4,...,14}
// R(i) = A(B(i)) = A(i) = i×2
```

**验证**：
```
R(3) = A(B(3)) = A(3) = 3×2 = 6 ✓
```

**注意**：对于 reshape 操作，推荐使用 `logical_divide` 而非 composition。Reshape 示例见 3.4 节。

**示例 2：转置**

```cpp
// 行主序 4×8 矩阵
A = (4, 8):(8, 1)

// 转置坐标：(i, j) → (j, i)
B = (8, 4):(1, 8)  // 交换维度和步长

R = A ∘ B
  = (8, 4):(1, 8)  // 列主序访问原矩阵
```

#### CuTe 代码

```cpp
#include <cute/layout.hpp>

using namespace cute;

// 定义原始 Layout
auto layout_a = make_layout(make_shape(10, 2), make_stride(2, 1));

// 定义组合 Layout（reshape 到 5×4）
auto layout_b = make_layout(make_shape(5, 4), make_stride(1, 5));

// Composition
auto composed = composition(layout_a, layout_b);

// 结果是嵌套的 Layout
print(composed);  // ((5,(2,2)):((2,(1,10))))
```

---

### 3.3 Complement（补集）

#### 定义

**Complement** 找到一个 Layout 的**补集**——在目标地址空间中，未被原 Layout 覆盖的元素。

给定：
- Layout `B`
- 目标 codomain 大小 `N`（可选，默认从 `B` 推断）

计算 `B* := complement(B, N)`，使得：
- `B` 和 `B*` 的值域**不相交**
- `B` 和 `B*` 的值域**并集**完全覆盖 `[0, N)`
- `B*` 是**有序的**（步长为正且递增）

#### 应用场景

1. **数据分区**：将数据分给不同线程后，找到剩余数据
2. **Tiling**：将矩阵分块后，找到块之间的关系
3. **内存布局**：找到 padding 或对齐的空隙

#### 示例

**示例 1：简单补集**

```cpp
// 原始 Layout：每 2 个元素取 1 个
B = (4):(2)
// 值域：{0, 2, 4, 6}

// 目标空间大小：8
N = 8

// 补集
B* = complement(B, 8)
   = (2):(1)
// 值域：{1, 3, 5, 7}（填补空隙）

// 验证：{0,2,4,6} ∪ {1,3,5,7} = [0, 8) ✓
```

**示例 2：二维补集**

```cpp
// 2×2 块在 4×4 矩阵中的位置
B = (2, 2):(4, 1)
// 值域：{0,1,4,5}（左上角 2×2 块）

// 目标：4×4 矩阵
N = 16

// 补集
B* = complement(B, 16)
   = ((2,2), 2):(((2,8), 4))
// 值域：剩余的 12 个元素，按某种顺序排列
```

#### 后置条件

- **不相交**：`B` 和 `B*` 的值域无交集
- **完全覆盖**：`B` 和 `B*` 的值域并集为 `[0, N)`
- **有序性**：`B*` 的所有步长为正
- **大小约束**：`size(B*) ≤ N`

#### CuTe 代码

```cpp
#include <cute/layout.hpp>

using namespace cute;

// 定义 Layout：每 2 个取 1 个，共 4 个元素
auto layout = make_layout(make_shape(4), make_stride(2));

// 计算补集（目标空间大小为 8）
auto comp = complement(layout, 8);

print(comp);  // (2):(1) 或类似结构

// 验证不相交
for (int i = 0; i < size(layout); ++i) {
    for (int j = 0; j < size(comp); ++j) {
        assert(layout(i) != comp(j));
    }
}
```

---

### 3.4 Division（除法 / Tiling）

#### 定义

**Division** 将 Layout `A` 按照"tiler" `B` 进行**分块**，产生两部分：

```
R := A ÷ B := A ∘ (B, B*)
```

其中：
- `B`：选择每个 tile 内的元素
- `B*`：选择不同 tile 的索引（complement）

结果 `R` 是一个**二层嵌套的 Layout**：
- **内层**：tile 内的布局
- **外层**：tile 之间的布局

#### 语义

Division 回答了问题：

> "如果我用 `B` 这个模式去切分 `A`，每个 tile 内部是什么样子？tile 之间又是什么关系？"

#### 示例

**示例 1：一维分块**

```cpp
// 原始 Layout：16 个元素，线性排列
A = (16):(1)

// Tiler：每次取 4 个元素
B = (4):(1)

// Division
R = A ÷ B
  = ((4), (4)):((1), (4))
     ^^   ^^    ^^   ^^
   tile内 tile数 tile内步长 tile间步长

// 解释：
// - 内层 (4):(1)：每个 tile 有 4 个连续元素
// - 外层 (4):(4)：有 4 个 tile，每个 tile 起始位置相差 4
```

**可视化**：
```
原始：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

分块后：
Tile 0: [0, 1, 2, 3]
Tile 1: [4, 5, 6, 7]
Tile 2: [8, 9, 10, 11]
Tile 3: [12, 13, 14, 15]
```

**示例 2：二维分块**

```cpp
// 原始 Layout：8×8 矩阵，行主序
A = (8, 8):(8, 1)

// Tiler：每个块取 2×2 元素（使用 Shape 作为 tiler）
B = (2, 2)  // CuTe 自动推导为 (2, 2):(8, 1)

// Division
R = A ÷ B
  = ((2, 2), (4, 4)):((8, 1), (16, 2))
     ^^^^^   ^^^^^    ^^^^^   ^^^^^^
    块内2×2  4×4个块  块内步长 块间步长
```

**Stride 计算详解**：

1. **块内 stride `(8, 1)`**：
   - 在一个 2×2 块内移动
   - 行方向：stride = 8（原矩阵的列数，跳到下一行）
   - 列方向：stride = 1（连续）

2. **块间 stride `(16, 2)`**：
   - 从一个块跳到相邻块
   - 行方向：跨 2 行 = 2×8 = 16 个元素
   - 列方向：跨 2 列 = 2 个元素

**验证计算**：访问 Block(1, 2) 的元素 [1, 0]
```
块索引: (1, 2) → 1×16 + 2×2 = 20（块起始位置）
块内索引: (1, 0) → 1×8 + 0×1 = 8
总偏移: 20 + 8 = 28

验证：原矩阵 [3, 4]（第4行第5列）= 3×8 + 4 = 28 ✓
```

**可视化**：
```
原始 8×8 矩阵分成 4×4 个 2×2 块：

+----+----+----+----+
|B00 |B01 |B02 |B03 |
+----+----+----+----+
|B10 |B11 |B12 |B13 |
+----+----+----+----+
|B20 |B21 |B22 |B23 |
+----+----+----+----+
|B30 |B31 |B32 |B33 |
+----+----+----+----+

每个 B_ij 是一个 2×2 的块
```

#### 应用场景

1. **Thread Block Tiling**：将矩阵分配给不同 CTA
2. **Warp Tiling**：将 CTA 的 tile 分配给不同 warp
3. **Thread Tiling**：将 warp 的 tile 分配给不同 thread
4. **Register Tiling**：每个 thread 处理多个元素

#### CuTe 代码

```cpp
#include <cute/layout.hpp>

using namespace cute;

// 原始 Layout：16 个元素
auto layout = make_layout(make_shape(16), make_stride(1));

// Tiler：每 4 个一组
auto tiler = make_layout(make_shape(4), make_stride(1));

// Division (Tiling)
auto tiled = logical_divide(layout, tiler);

print(tiled);  // ((4),(4)):((1),(4))

// 访问：tile i 中的元素 j
int tile_idx = 2;
int elem_idx = 3;
int physical_addr = tiled(elem_idx, tile_idx);
// = 2*4 + 3 = 11
```

---

### 3.5 Product（乘积 / Inverse Tiling）

#### 定义

**Product** 是 Division 的**逆操作**，将一个 Layout 按照另一个 Layout 的结构进行**重复排列**。

```
R := A × B
```

语义：用 `B` 的形状和步长去"重复" `A`。

#### 示例

**示例 1：一维重复**

```cpp
// 基础 Layout：4 个元素
A = (4):(2)  // {0, 2, 4, 6}

// 重复模式：重复 3 次
B = (3):(8)

// Product
R = A × B
  = (4, 3):(2, 8)

// 值域：
// B=0: {0, 2, 4, 6}
// B=1: {8, 10, 12, 14}
// B=2: {16, 18, 20, 22}
```

**示例 2：二维重复（Tile 到矩阵）**

```cpp
// 基础 tile：2×2
A = (2, 2):(4, 1)

// 重复成 4×4 个 tile
B = (4, 4):(8, 64)

// Product
R = A × B
  = ((2, 2), (4, 4)):((4, 1), (8, 64))
     ^^^^^   ^^^^^    ^^^^^   ^^^^^^^^
     tile形状 重复次数 tile步长 重复步长
```

#### Product vs Division

**关系**：
```
A × B ≈ inverse of (A ÷ B)
```

- **Division**：大 Layout → 切分成小 tiles
- **Product**：小 tile → 重复成大 Layout

#### CuTe 代码

```cpp
#include <cute/layout.hpp>

using namespace cute;

// 基础 Layout
auto base = make_layout(make_shape(4), make_stride(2));

// 重复模式
auto repeat = make_layout(make_shape(3), make_stride(8));

// Product
auto product = logical_product(base, repeat);

print(product);  // (4,3):(2,8)
```

---

## 4. 高级操作

### 4.1 Tiler（分块器）

#### 定义

**Tiler** 是一个广义的 Layout，可以递归地应用到多模态 Layout 的每个模态上。

#### Tiler 的形式

1. **单个 Layout**：对所有模态应用同一个 Layout
2. **Shape**：解释为 stride-1 的 Layout
3. **Tuple of Tilers**：对每个模态应用不同的 Tiler（递归）

#### 语法

```cpp
Tiler = Layout | Shape | (Tiler, Tiler, ...)
```

#### 示例

**示例 1：统一 Tiler**

```cpp
// Layout：4×8 矩阵
A = (4, 8):(8, 1)

// Tiler：每个维度取 2 个
T = (2, 2):(1, 1)

// 应用
R = tile(A, T)
  = ((2, 2), (2, 4)):((8, 1), (16, 2))
     ^^^^^   ^^^^^
     tile形状  tile数量
```

**示例 2：不同模态用不同 Tiler**

```cpp
// Layout：8×16 矩阵
A = (8, 16):(16, 1)

// Tiler：M 方向取 4 个，N 方向取 8 个
T = <(4):(1), (8):(1)>

// 应用
R = tile(A, T)
  = ((4, 8), (2, 2)):((16, 1), (64, 8))
     ^^^^^   ^^^^^
     4×8 tile  2×2 tiles
```

**示例 3：递归 Tiler**

```cpp
// 三层嵌套：CTA → Warp → Thread

// Layout：128×128 矩阵
A = (128, 128):(128, 1)

// Tiler：
// - CTA: 64×64
// - Warp: 32×32 per CTA
// - Thread: 8×8 per Warp
T = <
    <(8, 8), (4, 4)>,   // M 方向：先 8×4=32，再 2
    <(8, 8), (4, 4)>    // N 方向：同上
>

// 结果是深度嵌套的 Layout
R = tile(A, T)
```

#### CuTe 代码

```cpp
#include <cute/layout.hpp>

using namespace cute;

// Layout
auto layout = make_layout(make_shape(8, 16), make_stride(16, 1));

// Tiler（使用 Shape，自动转换为 stride-1 Layout）
auto tiler = make_shape(4, 8);

// Tile
auto tiled = logical_divide(layout, tiler);

print(tiled);
```

---

### 4.2 By-Mode Operations（按模态操作）

#### 定义

大多数 Layout 操作都有**按模态**的变体，允许只对特定维度进行操作，而保持其他维度不变。

#### 操作列表

- `coalesce(layout, mode)`：只合并指定模态
- `composition(layout_a, layout_b, mode)`：只在指定模态上组合
- `complement(layout, target, mode)`：只在指定模态上求补集
- `logical_divide(layout, tiler, mode)`：只在指定模态上分块
- `logical_product(layout, repeat, mode)`：只在指定模态上重复

#### 示例

**示例 1：只在第 0 模态上 coalesce**

```cpp
// Layout：(4, 2, 8):(1, 4, 8)
A = ((4, 2), 8):((1, 4), 8)

// Coalesce mode-0
R = coalesce(A, Int<0>{})
  = (8, 8):(1, 8)
  // 只合并了 (4,2) → 8
```

**示例 2：只在第 1 模态上分块**

```cpp
// Layout：8×16 矩阵
A = (8, 16):(16, 1)

// 只在 N 方向（mode-1）分块，每块 4 个
T = 4

R = logical_divide(A, T, Int<1>{})
  = (8, (4, 4)):(16, (1, 4))
     ^   ^^^^^
   不变  分成4×4
```

#### 应用场景

- **部分转置**：只转置某些维度
- **选择性 tiling**：只对 M 或 N 方向分块
- **局部重组**：保持整体结构，只优化某一层

---

### 4.3 Static vs Dynamic Layout

#### 编译时 Layout（Static）

**特点**：
- Shape 和 Stride 在编译时已知
- 使用 C++ 整型常量（如 `Int<16>`）
- 编译器可以极致优化（展开循环、消除分支）

**示例**：
```cpp
using namespace cute;

// 编译时 Layout：16×16，行主序
auto static_layout = make_layout(
    make_shape(Int<16>{}, Int<16>{}),
    make_stride(Int<16>{}, Int<1>{})
);

// 编译时计算
constexpr int offset = static_layout(8, 4);  // 编译期常量
```

#### 运行时 Layout（Dynamic）

**特点**：
- Shape 和 Stride 在运行时确定
- 使用普通整型变量（`int`, `size_t`）
- 更灵活，但优化受限

**示例**：
```cpp
// 运行时 Layout：M×N，行主序
int M = get_M();  // 运行时确定
int N = get_N();

auto dynamic_layout = make_layout(
    make_shape(M, N),
    make_stride(N, 1)
);

// 运行时计算
int offset = dynamic_layout(i, j);
```

#### 混合 Layout

CuTe 支持**部分静态、部分动态**：

```cpp
// M 是动态的，N 是静态的
int M = get_M();
auto hybrid_layout = make_layout(
    make_shape(M, Int<16>{}),
    make_stride(Int<16>{}, Int<1>{})
);
```

编译器会尽可能保留静态信息进行优化。

---

## 5. 实际应用场景

### 5.1 线程块 Tiling（CTA Tiling）

#### 场景

将一个大矩阵分配给多个 Thread Block（CTA）处理。

#### 示例

```cpp
// 全局矩阵：1024×1024，行主序
auto global_layout = make_layout(
    make_shape(1024, 1024),
    make_stride(1024, 1)
);

// CTA Tile：128×128
auto cta_tile = make_shape(128, 128);

// 分块
auto cta_tiled = logical_divide(global_layout, cta_tile);
// 结果：((128,128), (8,8)):((1024,1), (128×1024, 128))

// 每个 CTA 获取自己的 tile
int cta_m = blockIdx.x;
int cta_n = blockIdx.y;

auto my_tile = cta_tiled(_, cta_m, cta_n);
// 返回 (128,128) 的 Layout，对应全局矩阵中的一个块
```

---

### 5.2 Warp Tiling（Warp 内分块）

#### 场景

将 CTA 的 tile 进一步分配给不同的 warp。

#### 示例

```cpp
// CTA tile：128×128
auto cta_tile = make_layout(
    make_shape(128, 128),
    make_stride(128, 1)
);

// Warp tile：32×32
auto warp_tile = make_shape(32, 32);

// 分块
auto warp_tiled = logical_divide(cta_tile, warp_tile);
// 结果：((32,32), (4,4)):((128,1), (32×128, 32))

// 每个 warp 获取自己的 tile
int warp_m = warpId / 4;
int warp_n = warpId % 4;

auto my_warp_tile = warp_tiled(_, warp_m, warp_n);
```

---

### 5.3 Thread Tiling（线程内向量化）

#### 场景

每个线程处理多个元素以提高寄存器利用率和指令级并行。

#### 示例

```cpp
// Warp tile：32×32
auto warp_tile = make_layout(
    make_shape(32, 32),
    make_stride(32, 1)
);

// Thread tile：每个线程处理 8×8
auto thread_tile = make_shape(8, 8);

// 分块
auto thread_tiled = logical_divide(warp_tile, thread_tile);
// 结果：((8,8), (4,4)):((32,1), ...)

// 每个线程获取自己的 tile
int thread_m = threadIdx.x / 4;
int thread_n = threadIdx.x % 4;

auto my_thread_tile = thread_tiled(_, thread_m, thread_n);

// 循环处理
for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
        int offset = my_thread_tile(i, j);
        // 处理 data[offset]
    }
}
```

---

### 5.4 转置 (Transpose)

#### 场景

实现高效的矩阵转置，通过 Layout 组合而非显式循环。

#### 实现

```cpp
// 原始矩阵：M×N，行主序
auto src_layout = make_layout(
    make_shape(M, N),
    make_stride(N, 1)
);

// 转置布局：交换 Shape 和 Stride
auto transpose_layout = make_layout(
    make_shape(N, M),
    make_stride(1, N)
);

// Composition：通过转置布局访问原始数据
auto transposed = composition(src_layout, transpose_layout);

// 读取转置后的元素
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
        int offset = transposed(i, j);
        // offset 是原矩阵中 (j, i) 位置的线性索引
    }
}
```

---

### 5.5 Shared Memory Layout（避免 Bank Conflict）

#### 场景

设计 Shared Memory 布局以避免 bank conflict。

#### Bank Conflict 回顾

GPU Shared Memory 分为 32 个 bank，每个 bank 宽度 4 字节。

**关键概念**：
- **Bank 索引** = `(地址 / 4) % 32`
- 如果一个 warp 的 32 个线程同时访问同一个 bank 的不同地址，会发生 bank conflict
- **串行度** = 访问同一 bank 的线程数（最坏情况：32-way conflict）

**示例**：
```
32×32 矩阵，行主序，stride = 32
线程访问同一列：thread[i] 访问 data[i]
- thread 0: addr = 0  → bank 0
- thread 1: addr = 32 → bank 8  (32/4 % 32 = 8)
- thread 2: addr = 64 → bank 16
- thread 3: addr = 96 → bank 24
- thread 4: addr = 128 → bank 0  (128/4 % 32 = 0) ← conflict!
```

#### 解决方案：Padding

```cpp
// 有 bank conflict 的布局：32×32，行主序
auto conflict_layout = make_layout(
    make_shape(32, 32),
    make_stride(32, 1)
);
// 每列 32 个线程访问相同 bank → conflict

// 无 bank conflict 的布局：添加 padding
auto padded_layout = make_layout(
    make_shape(32, 32),
    make_stride(33, 1)  // 步长 33，避免对齐
);
// 每列分散到不同 bank
```

#### CuTe 自动 Padding

```cpp
using namespace cute;

// 使用 swizzle 模式自动处理 bank conflict
auto smem_layout = make_layout(
    make_shape(Int<32>{}, Int<32>{}),
    make_stride(Int<33>{}, Int<1>{})  // 编译时 padding
);
```

---

### 5.6 TMA (Tensor Memory Accelerator) Layout

#### 场景（Hopper）

使用 Hopper 架构的 TMA 硬件加速器进行高效的多维数据传输。

#### TMA 要求

- 对齐：16 字节对齐
- 连续性：最内层维度连续
- 描述符：需要用 Layout 描述传输模式

#### 示例

```cpp
// Global Memory Layout：128×128 tile in row-major
auto gmem_layout = make_layout(
    make_shape(Int<128>{}, Int<128>{}),
    make_stride(Int<128>{}, Int<1>{})
);

// Shared Memory Layout：添加 padding 避免 bank conflict
auto smem_layout = make_layout(
    make_shape(Int<128>{}, Int<128>{}),
    make_stride(Int<136>{}, Int<1>{})  // 128 + 8 padding
);

// TMA Copy：使用 CuTe 的 Copy_Atom
using TMA_Load = Copy_Atom<SM90_TMA_LOAD_2D, half_t>;
auto tma_copy = make_tiled_copy(
    TMA_Load{},
    gmem_layout,
    smem_layout
);

// 执行拷贝
copy(tma_copy, gmem_tensor, smem_tensor);
```

---

## 6. 编程接口和示例

### 6.1 创建 Layout

#### 基本创建

```cpp
#include <cute/layout.hpp>

using namespace cute;

// 方法 1：直接指定 Shape 和 Stride
auto layout1 = make_layout(
    make_shape(8, 16),      // Shape: 8×16
    make_stride(16, 1)      // Stride: 行主序
);

// 方法 2：使用编译时常量
auto layout2 = make_layout(
    make_shape(Int<8>{}, Int<16>{}),
    make_stride(Int<16>{}, Int<1>{})
);

// 方法 3：只指定 Shape（自动推导 stride-1）
auto layout3 = make_layout(make_shape(8, 16));
// 等价于 (8,16):(16,1)
```

#### 嵌套 Layout

```cpp
// 分层 Layout：2×2 个 4×4 的块
auto nested = make_layout(
    make_shape(make_shape(4, 4), make_shape(2, 2)),
    make_stride(make_stride(1, 4), make_stride(16, 64))
);

// 等价记号：((4,4), (2,2)):((1,4), (16,64))
```

---

### 6.2 访问 Layout 属性

```cpp
auto layout = make_layout(make_shape(8, 16), make_stride(16, 1));

// 总大小
int total_size = size(layout);  // 8 * 16 = 128

// Shape
auto shp = shape(layout);       // (8, 16)
int dim0 = size<0>(layout);     // 8
int dim1 = size<1>(layout);     // 16

// Stride
auto str = stride(layout);      // (16, 1)
int stride0 = stride<0>(layout); // 16
int stride1 = stride<1>(layout); // 1

// Rank（维度数）
constexpr int rank = rank(layout);  // 2

// Depth（嵌套深度）
constexpr int depth = depth(layout);  // 1
```

---

### 6.3 应用 Layout（映射坐标）

```cpp
auto layout = make_layout(make_shape(8, 16), make_stride(16, 1));

// 单个坐标
int offset1 = layout(3, 5);  // 3*16 + 5*1 = 53

// 多个坐标（嵌套）
auto nested = make_layout(
    make_shape(make_shape(4, 4), make_shape(2, 2)),
    make_stride(make_stride(1, 4), make_stride(16, 64))
);
int offset2 = nested(make_coord(2, 3), make_coord(1, 0));
// 计算：(2,3) 在第 (1,0) 个块中的位置
```

---

### 6.4 Layout 操作完整示例

```cpp
#include <cute/layout.hpp>
#include <iostream>

using namespace cute;

int main() {
    // 创建 128×64 矩阵，行主序
    auto global_layout = make_layout(
        make_shape(128, 64),
        make_stride(64, 1)
    );

    // CTA Tiling：分成 16×8 个 8×8 的块
    auto cta_tiler = make_shape(8, 8);
    auto cta_tiled = logical_divide(global_layout, cta_tiler);

    // 打印结果
    print("CTA Tiled Layout:\n");
    print(cta_tiled);
    print("\n");

    // 模拟 CTA (2, 3) 访问其 tile
    int cta_m = 2, cta_n = 3;
    auto my_cta_tile = cta_tiled(_, cta_m, cta_n);

    print("My CTA Tile:\n");
    print(my_cta_tile);
    print("\n");

    // 线程 Tiling：每个线程处理 4 个元素
    auto thread_tiler = make_shape(4);
    auto thread_tiled = logical_divide(my_cta_tile, thread_tiler);

    print("Thread Tiled Layout:\n");
    print(thread_tiled);
    print("\n");

    // 线程 5 的数据
    int tid = 5;
    for (int i = 0; i < 4; ++i) {
        int offset = thread_tiled(i, tid);
        std::cout << "Thread " << tid << ", element " << i
                  << " -> global offset " << offset << "\n";
    }

    return 0;
}
```

---

### 6.5 实用工具函数

```cpp
// 判断 Layout 是否连续（stride-1）
bool is_contiguous = is_contiguous(layout);

// 获取 codomain 大小（值域范围）
int codom = cosize(layout);

// 判断两个 Layout 是否兼容（可组合）
bool compatible = compatible(layout_a, layout_b);

// 扁平化（去除嵌套）
auto flat = flatten(nested_layout);

// 获取特定模态
auto mode0 = get<0>(layout);  // 第 0 模态
```

---

## 7. 调试和可视化

### 7.1 打印 Layout

```cpp
#include <cute/layout.hpp>

using namespace cute;

auto layout = make_layout(make_shape(8, 16), make_stride(16, 1));

// 打印完整信息
print(layout);
// 输出：(8,16):(16,1)

// 打印详细信息
print("Layout: ");
print(layout);
print("\n");
print("  Size: ");
print(size(layout));
print("\n");
print("  Shape: ");
print(shape(layout));
print("\n");
print("  Stride: ");
print(stride(layout));
print("\n");
```

### 7.2 验证 Layout 正确性

```cpp
// 验证 coalesce 保持函数不变
auto original = make_layout(make_shape(4, 8), make_stride(1, 4));
auto coalesced = coalesce(original);

for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) {
        assert(original(i, j) == coalesced(i*8 + j));
    }
}

// 验证 composition
auto layout_a = make_layout(make_shape(16), make_stride(2));
auto layout_b = make_layout(make_shape(4, 4), make_stride(4, 1));
auto composed = composition(layout_a, layout_b);

for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
        int idx_b = layout_b(i, j);
        int idx_composed = composed(i, j);
        assert(idx_composed == layout_a(idx_b));
    }
}
```

---

## 8. 高级主题

### 8.1 Swizzle（交织访问）

#### 定义

**Swizzle** 是一个函数对象（functor），它通过**位运算**转换偏移地址，用于：
1. 避免 Shared Memory 的 bank conflict
2. 提高缓存局部性
3. 优化访存模式

**核心思想**：将线性偏移通过 XOR 等位运算重新映射，打散规律的访问模式。

#### Swizzle 函数定义

```cpp
template <int B, int M, int S>
struct Swizzle {
    // 将偏移 offset 通过位运算转换
    CUTE_HOST_DEVICE
    int operator()(int offset) const {
        // XOR-based swizzle
        return offset ^ ((offset >> S) & M) << B);
    }
};
```

**参数说明**：
- **B (Base)**：swizzle 的基础位数（log₂ of swizzle unit size）
- **M (Mask)**：用于 swizzle 的位掩码
- **S (Shift)**：右移位数

#### 数学原理

对于 Swizzle<B, M, S>：

```
swizzled_offset = offset ^ (((offset >> S) & M) << B)
```

**作用**：
1. 将 offset 右移 S 位，提取高位信息
2. 与掩码 M 进行 AND 操作
3. 左移 B 位后与原 offset 进行 XOR
4. 打散原有的规律访问模式

#### 示例：Swizzle<3, 3, 3>

这是 Shared Memory 中常用的 swizzle 模式。

**参数**：
- B = 3：swizzle 单位为 2³ = 8 字节
- M = 3 (0b11)：掩码 2 位
- S = 3：右移 3 位

**计算示例**：
```
offset = 24 (0b11000)

步骤 1: offset >> 3 = 3 (0b11)
步骤 2: 3 & 3 = 3 (0b11)
步骤 3: 3 << 3 = 24 (0b11000)
步骤 4: 24 ^ 24 = 0

swizzled_offset = 0
```

#### CuTe 中的使用

**方法 1：直接使用 Swizzle Stride**

```cpp
using namespace cute;

// 32×128 的 Shared Memory，half_t (2 bytes)
// 使用 Swizzle<3, 3, 3> 避免 bank conflict
auto smem_layout = make_layout(
    make_shape(Int<32>{}, Int<128>{}),
    make_stride(
        Swizzle<3, 3, 3>{},  // 行方向 swizzle
        Int<1>{}             // 列方向连续
    )
);
```

**方法 2：Composition with Swizzle**

```cpp
// 基础布局
auto base_layout = make_layout(
    make_shape(Int<32>{}, Int<128>{}),
    make_stride(Int<128>{}, Int<1>{})
);

// 应用 swizzle
auto swizzle_fn = Swizzle<3, 3, 3>{};
auto swizzled_layout = composition(base_layout, swizzle_fn);
```

#### Swizzle 的效果

**不使用 Swizzle（有 bank conflict）**：
```
线程访问模式（列方向）：
thread 0  -> addr 0   -> bank 0
thread 1  -> addr 128 -> bank 0  ← conflict!
thread 2  -> addr 256 -> bank 0  ← conflict!
...
32-way bank conflict
```

**使用 Swizzle<3, 3, 3>（无 bank conflict）**：
```
线程访问模式（列方向）：
thread 0  -> addr 0   -> swizzled to bank 0
thread 1  -> addr 128 -> swizzled to bank 8
thread 2  -> addr 256 -> swizzled to bank 16
...
分散到不同 bank，无 conflict
```

#### 常用 Swizzle 配置

| 数据类型 | 配置 | 说明 |
|---------|------|------|
| `half` (FP16) | `Swizzle<3, 3, 3>` | 2 字节，8-byte swizzle |
| `float` (FP32) | `Swizzle<2, 2, 3>` | 4 字节，4-byte swizzle |
| `double` (FP64) | `Swizzle<3, 1, 4>` | 8 字节，8-byte swizzle |

#### 验证 Swizzle

```cpp
// 验证 swizzle 消除了 bank conflict
Swizzle<3, 3, 3> sw;

// 测试连续的 32 个地址
for (int i = 0; i < 32; ++i) {
    int addr = i * 128;  // 列方向访问
    int swizzled = sw(addr);
    int bank = (swizzled / 4) % 32;
    printf("thread %d: addr %d -> swizzled %d -> bank %d\n",
           i, addr, swizzled, bank);
}
// 应该看到 bank 分布在 0-31，无重复
```

---

### 8.2 Conditional Layout（条件布局）

根据运行时条件选择不同的 Layout：

```cpp
template <typename T>
auto make_conditional_layout(int M, int N, bool row_major) {
    if (row_major) {
        return make_layout(make_shape(M, N), make_stride(N, 1));
    } else {
        return make_layout(make_shape(M, N), make_stride(1, M));
    }
}
```

---

### 8.3 Layout 作为模板参数

```cpp
template <typename Layout>
__global__ void kernel(float* data, Layout layout) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size(layout)) {
        int offset = layout(tid);
        data[offset] *= 2.0f;
    }
}

// 调用
auto layout = make_layout(make_shape(1024), make_stride(2));
kernel<<<grid, block>>>(data, layout);
```

---

## 9. 总结

### 9.1 核心要点

1. **Layout 是函数**：从逻辑坐标到物理地址的映射
2. **Shape 和 Stride**：定义 Layout 的两个核心组件
3. **代数操作**：通过组合简单操作构建复杂变换
4. **编译时优化**：尽可能使用静态 Layout 以提升性能
5. **层次化思维**：从 Device → CTA → Warp → Thread 逐层分解

### 9.2 操作速查表

| 操作 | 符号 | 作用 | 示例 |
|-----|------|------|------|
| **Coalesce** | `coalesce(A)` | 简化 Layout | `(4,8):(1,4) → (32):(1)` |
| **Composition** | `A ∘ B` | 函数复合 | Reshape, Transpose |
| **Complement** | `B*` | 求补集 | 找到未覆盖的元素 |
| **Division** | `A ÷ B` | 分块 | Tiling |
| **Product** | `A × B` | 重复 | Inverse Tiling |

### 9.3 最佳实践

1. **优先使用编译时常量**（`Int<N>`）以获得最佳性能
2. **逐步分解**：复杂 Layout 通过多步操作构建
3. **验证正确性**：使用断言检查 Layout 属性
4. **打印调试**：使用 `print()` 可视化 Layout
5. **遵循硬件层次**：Layout 设计应反映 GPU 硬件结构

### 9.4 学习路径

1. 理解 Shape 和 Stride 的基本概念
2. 掌握简单的 coalesce 和 composition
3. 学习 Division（Tiling）用于数据分区
4. 应用到实际 GEMM kernel 设计
5. 研究高级技术（Swizzle、TMA）

---

## 10. 参考资源

- [CuTe GitHub Repository](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
- [CuTe Layout Algebra 官方文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
- [CuTe Quickstart](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CUTLASS 3.x 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs)

---

**版本信息**：
- CuTe 版本：CUTLASS 3.x
- 最后更新：2026-01
- 作者：基于 NVIDIA 官方文档编写
