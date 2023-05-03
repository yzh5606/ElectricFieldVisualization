from PDESolver.Solver import PDEFunction
import matplotlib.pylab as p
import sympy
import numpy

a=sympy.symbols("a")
c=sympy.symbols("c")

eq = a + c

solver = PDEFunction(eq,a=a,c=c)

canvas = []
mask = []

for i in range(100):
    canvas.append([])
    mask.append([])
    canvas[i].append(100)
    mask[i].append(True)
    for j in range(99):
        canvas[i].append(0)
        mask[i].append(False)

result = solver.solve(canvas,mask,maxIter=20)

x = []
y = []
z = numpy.array(result)

for i in range(100):
    x.append(i)
    y.append(i)

fig = p.figure() # Create figure 
ax = fig.add_axes((0,0,1,1),projection="3d") # Plot axes 
ax.plot_wireframe(x, y, z, color = 'r') # Red wireframe 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential')
fig.add_axes(ax)
p.show() # Show fig