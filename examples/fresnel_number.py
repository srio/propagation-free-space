import numpy
import scipy.constants as codata
from srxraylib.plot.gol import plot
import matplotlib.pylab as plt


energy = 10000.0
wavelength =  ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
window_size = 5e-6
aperture_diameter = 1.25e-06
propagation_distance = 75e-6
N = 2048

delta = window_size/N
propagation_distance = numpy.linspace(0,500e-6,100)
#
# info
#
# print("Dz <? N d^2 / lambda; %g < %g " %(propagation_distance, window_size*0.5 * delta/wavelength))
# print("Fresnel number: xmaxD / (lambda Dz) = %g  " %(0.5*window_size /(propagation_distance * wavelength)))

validity2048 = window_size * (window_size/2048) / wavelength
validity4096 = window_size * (window_size/4096) / wavelength
# fresnel = 0.5*window_size /(propagation_distance * wavelength)

plot(1e6*propagation_distance,propagation_distance,
     1e6*propagation_distance,propagation_distance*0 + validity2048,
     1e6*propagation_distance,propagation_distance*0 + validity4096,
     xtitle="propagation distance [um]",
     legend=["propagation distance [m]","limiting condition N=2048 [m]","limiting condition N=4096 [m]"],show=False)

plt.savefig("fresnel_propagator_validity.png")
print("File written to disk: fresnel_propagator_validity.png" )
plt.show()

print("validity 2048: %f um "%(1e6*validity2048))
print("validity 4096: %f um "%(1e6*validity4096))

validity_favre_512 = window_size * (window_size/2048) / wavelength
validity_favre_512 = window_size * (window_size/4096) / wavelength

# plot(propagation_distance,fresnel,title="Fresnel number")