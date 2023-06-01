import numpy
from PDESolver.Solver import TestPDEFunction
import matplotlib.pylab as p

func = TestPDEFunction(
    A=lambda x, y: 0,
    B=lambda x, y: 0,
    C=lambda x, y: 0,
    D=lambda x, y: 1,
    E=lambda x, y: 1,
    F=lambda x, y: 0,
)

Nmax = 100
canvas = numpy.zeros((Nmax, Nmax), float)
mask = numpy.zeros((Nmax, Nmax), bool)

canvas[:,0] = 100.0  # Line at 100V

# mask[:,0:1]=True
# mask[:,-1:]=True
# mask[0:1,:]=True
# mask[-1:,:]=True
mask[:,0] = True

x = []
y = []
z = func.solve(canvas, mask, maxIter=1000)


for i in range(100):
    x.append(i)
    y.append(i)
X, Y = numpy.meshgrid(x, y)

fig = p.figure()  # Create figure
ax = fig.add_axes((0, 0, 1, 1), projection="3d")  # Plot axes
ax.plot_wireframe(X, Y, z, color="r")  # Red wireframe  # type: ignore
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Potential")  # type: ignore
fig.add_axes(ax)
p.show()  # Show fig
