

import numpy
from srxraylib.plot.gol import plot
import matplotlib.pylab as plt

def quadratic(x,A):
    return (numpy.exp(1j*A*x**2)).real

if __name__ == "__main__":
    N = 10
    x = numpy.linspace(0.0,1.0,N)
    xx = numpy.linspace(-0.01,1.01,5000)
    delta = x[1] - x[0]

    A = 20
    y20 = quadratic(x,A)
    title = "A=%2.1f <? %2.1f"%(A,numpy.pi/N/delta**2)
    plot(xx,quadratic(xx,A),
         x,y20,
         xtitle="x",ytitle="Real[ f(x) ]",title=title,show=False)

    plt.savefig("sample_quadratic_phase_A20.png")
    print("File written to disk: sample_quadratic_phase_A20.png")
    plt.show()

    A = 30
    y30 = quadratic(x,A)
    title = "A=%2.1f <? %2.1f"%(A,numpy.pi/N/delta**2)
    plot(xx,quadratic(xx,A),
         x,y30,
         xtitle="x",ytitle="Real[ f(x) ]",title=title,show=False)
    plt.savefig("sample_quadratic_phase_A30.png")
    print("File written to disk: sample_quadratic_phase_A30.png")
    plt.show()
