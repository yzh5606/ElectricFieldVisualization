# 于是我们来到了test2.py，这个部分所使用的代码与test1中的不同
# 我们已经将相关代码封装为了TestPDEFunction
# test1.py主要是依靠sympy进行计算，而test2.py依靠numpy，速度有了极大的提升

# 从"./PDESolver/Solver.py"中导入class TestPDEFunction
from PDESolver.Solver import TestPDEFunction

# 导入其他python库
import numpy
import matplotlib.pylab as p

# 下面的代码使用到了lambda
# lambda是“匿名函数”，lambda x,y:x 相当于 def func(x,y): return x
# 它的出现纯属是为了码农们可以少打点字，也就是说，下面的形参要求以函数作为参数传入
func = TestPDEFunction(
    # A，B，C，D，E，F是关于变量x，y的函数
    # 这些字母与公式$A\frac{\partial^2 U}{\partial x^2}+2B\frac{\partial^2 U}{\partial x \partial y}+C\frac{\partial^2 U}{\partial y^2}+D\frac{\partial U}{\partial x}+E\frac{\partial U}{\partial y}=F\tag{0.1}$中的字母完全对应
    A=lambda x, y: 1,
    B=lambda x, y: 0,
    C=lambda x, y: 1,
    D=lambda x, y: 0,
    E=lambda x, y: 0,
    F=lambda x, y: 0,
)

Nmax = 41  # 画布大小
# 我们构建了41*41的画布，并设置它们全为零
canvas = numpy.zeros((Nmax, Nmax), float)
# 我们还构建了41*41的掩码画布，并设置它们全为False
mask = numpy.zeros((Nmax, Nmax), bool)

# 这与test1中所构建的画布相似，取值是第一列为100，其余都为0。
canvas[:, 0:1] = 100.0

# 且标定第一列为固定值
mask[:, 0:1] = True
# 这里还固定了最后一列为固定值
mask[:, -1:] = True
# 首行与首列也为固定值
mask[0:1, :] = True
mask[-1:, :] = True

z = func.solve(canvas, mask, maxIter=10240, expansion=2)
# 此处的solve方法多了expansion参数，它则是将画布进行更细致的分割，以使得最终的结果更加准确

# 以下为画图部分

# 先设定数据
x = []
y = []

for i in range(Nmax):
    x.append(i)
    y.append(i)
X, Y = numpy.meshgrid(x, y)

# 这个部分和test1相同是通过设定x轴，y轴，z轴，颜色，标签等属性
# 并使用matplotlib.pylab将结果以图像的形式呈现出来
fig = p.figure()
ax = fig.add_axes((0, 0, 1, 1), projection="3d")
ax.plot_wireframe(X, Y, z, color="r") # type: ignore
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Potential") # type: ignore
fig.add_axes(ax)
p.show()
