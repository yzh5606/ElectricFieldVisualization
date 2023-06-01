# 从"./PDESolver/Solver.py"中导入class PDEFunction
from PDESolver.Solver import PDEFunction
# 导入其他python库
import matplotlib.pylab as p
import sympy
import numpy

# 定义符号，用于表示各种偏导数
# 此处a为对x的二阶导，c为对y的二阶导
d=sympy.symbols("d")
e=sympy.symbols("e")

eq = d + e - 2

solver = PDEFunction(eq,d=d,e=e)

canvas = []
mask = []

for i in range(20):
    canvas.append([])
    mask.append([])
    canvas[i].append(100)
    mask[i].append(True)
    for j in range(19):
        canvas[i].append(0)
        mask[i].append(False)

result = solver.solve(canvas,mask,maxIter=10)

x = []
y = []
z = numpy.array(result)

for i in range(20):
    x.append(i)
    y.append(i)

X,Y = numpy.meshgrid(x,y)

fig = p.figure() # Create figure 
ax = fig.add_axes((0,0,1,1),projection="3d") # Plot axes 
ax.plot_wireframe(X, Y, z, color = 'r') # Red wireframe  # type: ignore
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential') # type: ignore
fig.add_axes(ax)
p.show() # Show fig