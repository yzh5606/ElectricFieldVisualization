为了实现对偏微分方程解的结果进行可视化，使用者需要传输三个参数：

1. 待求解的偏微分方程
2. 一块给定了初始值的画布，迭代结果将基于这个初始值进行计算
3. 掩码画布，其大小应当与初始值画布的大小一致，用于标定初始值画布中的哪些值为固定值

在 `test1.py` 与 `test2.py` 文件中，变量 `canvas` 即为初始值画布，变量 `mask` 即为掩码画布

在两个测试文件中，变量 `canvas` 与 `mask` 均为大小 100\*100 的画布，初始值画布仅第一列的数据为 100，并且被标记为不可变更，其余值均为 0，且未被标记为不可变更。测试中所使用的偏微分方程如下：

$$\frac{\partial^2 U}{\partial x^2}+\frac{\partial^2 U}{\partial y^2}=0\tag{0.1}$$

第一种思路是最简单，最直接的。我们只需要把一下近似公式带入偏微分方程：

$$\frac{\partial^2U\left(x,y\right)}{\partial x^2}\simeq\frac{U\left(x+\Delta,y\right)+U\left(x-\Delta,y\right)-2U\left(x,y\right)}{\left(\Delta\right)^2}\tag{1.1}$$

$$\frac{\partial^2U\left(x,y\right)}{\partial x\partial y}\simeq\frac{U\left(x+\Delta,y+\Delta\right)+U\left(x-\Delta,y-\Delta\right)-2U(x,y)}{2\left(\Delta\right)^2}\tag{1.2}$$

$$\frac{\partial^2U\left(x,y\right)}{\partial y^2}\simeq\frac{U\left(x,y+\Delta\right)+U\left(x,y-\Delta\right)-2U\left(x,y\right)}{\left(\Delta\right)^2}\tag{1.3}$$

$$\frac{\partial U\left(x,y\right)}{\partial x}=\frac{U\left(x,y+\Delta\right)-U\left(x,y\right)}{\Delta}\tag{1.4}$$

$$\frac{\partial U\left(x,y\right)}{\partial y}=\frac{U\left(x,y+\Delta\right)-U\left(x,y\right)}{\Delta}\tag{1.5}$$

并且将迭代之前的，除 $U\left(x,y\right)$ 之外的数值带入，并进行求解，即可算出迭代后的 $U\left(x,y\right)$ 的值。

然而，由于画布边界位置无法进行迭代，因此只能保持为 0。除此之外，由于 `sympy` 模块计算效率比较低下，造成了计算时间过长的问题。

如果我们直接将以下近似公式代入偏微分方程：

$$\frac{\partial^2U\left(x,y\right)}{\partial x^2}\simeq\frac{U\left(x+\Delta,y\right)+U\left(x-\Delta,y\right)-2U\left(x,y\right)}{\left(\Delta\right)^2}\tag{2.1}$$

$$\frac{\partial^2U\left(x,y\right)}{\partial x\partial y}\simeq\frac{U\left(x+\Delta,y+\Delta\right)-U\left(x-\Delta,y+\Delta\right)-U\left(x+\Delta,y-\Delta\right)+U\left(x-\Delta,y-\Delta\right)}{4\left(\Delta\right)^2}\tag{2.2}$$

$$\frac{\partial^2U\left(x,y\right)}{\partial y^2}\simeq\frac{U\left(x,y+\Delta\right)+U\left(x,y-\Delta\right)-2U\left(x,y\right)}{\left(\Delta\right)^2}\tag{2.3}$$

$$\frac{\partial U\left(x,y\right)}{\partial x}=\frac{U\left(x,y+\Delta\right)-U\left(x,y-\Delta\right)}{2\Delta}\tag{2.4}$$

$$\frac{\partial U\left(x,y\right)}{\partial y}=\frac{U\left(x,y+\Delta\right)-U\left(x,y-\Delta\right)}{2\Delta}\tag{2.5}$$

我们将得到一下结果：

$$
\begin{gather*}
    +\frac{B}{4\Delta^2}U(x-\Delta,y-\Delta)&+(\frac{A}{\Delta^2}-\frac{D}{2\Delta})U(x-\Delta,y)&-\frac{B}{4\Delta^2}U(x-\Delta,y+\Delta)\\
    +(\frac{C}{\Delta^2}-\frac{E}{2\Delta})U(x,y-\Delta)&-(\frac{2A}{\Delta^2}+\frac{2C}{\Delta^2}U(x,y)&+(\frac{C}{\Delta^2}+\frac{E}{2\Delta})U(x,y+\Delta))\\
    -\frac{B}{4\Delta^2}U(x+\Delta,y-\Delta)&+(\frac{A}{\Delta^2}+\frac{D}{2\Delta})U(x+\Delta,y)&+\frac{B}{4\Delta^2}U(x+\Delta,y+\Delta)
\end{gather*}\ =F\tag{3.1}
$$

因此，我们只需要将其中的 8 个变量用迭代前的旧值代入，即可得到另外一个变量的数值，并将其作为迭代后的值中的一个（实际上对于同一个位置的数字通常可以获得 9 个迭代后的值，该如何使用这 9 个数值也是我们该思考的问题之一）。

为了解决画布边界值不准确的问题，我们先将画布扩大一倍，在迭代完成后将其缩小，可以缓解此问题。

由于这种想法在代码实现的过程中没有使用到 `sympy` ，而是使用高效的矩阵工具 `numpy` 进行大量运算，因此，尽管运算复杂度有了提升，但是效率远远高过了 `sympy` 。

随着迭代次数的增加，其结果通常会趋于一个稳定值，我们将结果画布交由 `matplotlib.puplot` 进行图形化处理，即可得到方程近似解的图像。

实际上，这两种解决方案在某些情况下给出的结果并不正确，这也是我们以后如果需要继续改进算法的话所要努力的方向之一。
