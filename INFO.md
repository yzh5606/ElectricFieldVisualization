# 首先，你需要知道的是，写这个代码的人**_其实也只是个什么也不会菜鸟_**

# 偏微分方程

$$A\frac{\partial^2 U}{\partial x^2}+2B\frac{\partial^2 U}{\partial x \partial y}+C\frac{\partial^2 U}{\partial y^2}+D\frac{\partial U}{\partial x}+E\frac{\partial U}{\partial y}=F\tag{0.1}$$

# 近似公式

$$\frac{\partial^2U\left(x,y\right)}{\partial x^2}\simeq\frac{U\left(x+\Delta,y\right)+U\left(x-\Delta,y\right)-2U\left(x,y\right)}{\left(\Delta\right)^2}\tag{1.1}$$

$$\frac{\partial^2U\left(x,y\right)}{\partial x\partial y}\simeq\frac{U\left(x+\Delta,y+\Delta\right)-U\left(x-\Delta,y+\Delta\right)-U\left(x+\Delta,y-\Delta\right)+U\left(x-\Delta,y-\Delta\right)}{4\left(\Delta\right)^2}\tag{1.2}$$

$$\frac{\partial^2U\left(x,y\right)}{\partial y^2}\simeq\frac{U\left(x,y+\Delta\right)+U\left(x,y-\Delta\right)-2U\left(x,y\right)}{\left(\Delta\right)^2}\tag{1.3}$$

$$\frac{\partial U\left(x,y\right)}{\partial x}=\frac{U\left(x,y+\Delta\right)-U\left(x,y-\Delta\right)}{2\Delta}\tag{1.4}$$

$$\frac{\partial U\left(x,y\right)}{\partial y}=\frac{U\left(x,y+\Delta\right)-U\left(x,y-\Delta\right)}{2\Delta}\tag{1.5}$$

# class PDEFunction 的说明

这个类（`class`）的实现原理是个很基本的差分法，但是由于公式 1.2 1.4 1.5 中不包含 $U(x,y)$ 项，因此需要变化一下：

$$\frac{\partial^2U\left(x,y\right)}{\partial x\partial y}\simeq\frac{U\left(x+\Delta,y+\Delta\right)+U\left(x-\Delta,y-\Delta\right)-2U(x,y)}{2\left(\Delta\right)^2}\tag{2.2}$$

_这个公式（2.2）不知道对不对_

$$\frac{\partial U\left(x,y\right)}{\partial x}=\frac{U\left(x,y+\Delta\right)-U\left(x,y\right)}{\Delta}\tag{2.4}$$

$$\frac{\partial U\left(x,y\right)}{\partial y}=\frac{U\left(x,y+\Delta\right)-U\left(x,y\right)}{\Delta}\tag{2.5}$$

将 1.1 2.2 1.3 2.4 2.5 带入偏微分方程 0.1 ，即可得到 $U(x,y)$ 与其周围的 8 个值的关系式 ~~（实际上是 6 个）~~ ，这样，我们可以用那 8 个旧值将 $U(x,y)$ 解出来作为新值，就可以得到一次迭代后的结果。

这玩意运行的很慢，是因为用了`sympy`，它通常用于符号计算。然而实际上我们只需要数值计算，所以完全可以不用`sympy`。所以谁能帮我做一个不用`sympy`的版本？

# class TestPDEFunction 的说明

如果直接把 1.1 ~ 1.5 直接代入 0.1 ，可以获得 $U(x,y)$ 及其周围 8 个值之间的关系，所以我们完全可以用旧值算出这九个值的新值

带入后的结果：

$$
\begin{gather*}
    +\frac{B}{4\Delta^2}U(x-\Delta,y-\Delta)&+(\frac{A}{\Delta^2}-\frac{D}{2\Delta})U(x-\Delta,y)&-\frac{B}{4\Delta^2}U(x-\Delta,y+\Delta)\\
    +(\frac{C}{\Delta^2}-\frac{E}{2\Delta})U(x,y-\Delta)&-(\frac{2A}{\Delta^2}+\frac{2C}{\Delta^2}U(x,y)&+(\frac{C}{\Delta^2}+\frac{E}{2\Delta})U(x,y+\Delta))\\
    -\frac{B}{4\Delta^2}U(x+\Delta,y-\Delta)&+(\frac{A}{\Delta^2}+\frac{D}{2\Delta})U(x+\Delta,y)&+\frac{B}{4\Delta^2}U(x+\Delta,y+\Delta)
\end{gather*}\ =F\tag{3.1}
$$

然而这种方法似乎会出现很多的问题……
