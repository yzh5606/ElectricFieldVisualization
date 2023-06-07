# 从"./PDESolver/Solver.py"中导入class PDEFunction
from PDESolver.Solver import PDEFunction
# 导入其他python库
import matplotlib.pylab as p
import sympy
import numpy

# 定义符号，用于表示各种偏导数
# 此处a为对x的二阶导，c为对y的二阶导
a=sympy.symbols("a")
c=sympy.symbols("c")

eq = a + c - 2
# 这里a代表U对x的二阶偏导，c代表U对y的二阶偏导，因此，它描述了一个方程：
# $\frac{\partial^2 U}{\partial x^2}+\frac{\partial^2 U}{\partial y^2}=0$

solver = PDEFunction(eq,a=a,c=c)
# 利用上面的eq从PDEFunction类构建solver对象

# 此处构建了canvas和mask，canvas指的是初值，mask是用来表示哪个数字是不可变的，其本质为二维数组
canvas = []
# canvas: 设定的初始值
mask = []
# mask: 用于确定哪些值是不可变更的，记录为True

size = 50
# 我们构建了一个50*50的画布

for i in range(size):
    # 这个循环使得canvas的第一列都取值100，其余都取值0。
    # 取值的100和0可以根据需要任意的更改。
    canvas.append([])
    mask.append([])
    canvas[i].append(100)
    mask[i].append(True)
    for j in range(size - 1):
        canvas[i].append(0)
        mask[i].append(False)

result = solver.solve(canvas,mask,maxIter=20)
# 此处调用了solver对象的solve方法，solve方法在solver.py的class PDEFunction中定义
# 为了让数据更加准确，以及时间考虑，我们迭代了20次，算出的结果值赋给了result变量

# 以下为画图部分

# 先设定数据
x = []
y = []
z = numpy.array(result)

for i in range(size):
    x.append(i)
    y.append(i)

X,Y = numpy.meshgrid(x,y)

# 这个部分是通过设定x轴，y轴，z轴，颜色，标签等属性并使用matplotlib.pylab将结果以图像的形式呈现出来
fig = p.figure()
ax = fig.add_axes((0,0,1,1),projection="3d")
ax.plot_wireframe(X, Y, z, color = 'r') # type: ignore
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential') # type: ignore
fig.add_axes(ax)
p.show()