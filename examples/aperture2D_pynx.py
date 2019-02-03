import timeit


import numpy as np
from pynx.wavefront import *
import matplotlib.pylab as plt
import scipy.constants as codata


def get_pynx_wavefront_from_h5file(filename):
    import h5py

    f = h5py.File(filename)

    complex_amplitude = f["wfr/wfr_complex_amplitude_s"].value
    XX = f["wfr/wfr_mesh_X"].value
    YY = f["wfr/wfr_mesh_Y"].value
    energy = f["wfr/wfr_photon_energy"].value
    npixels = int(XX[2])
    pixel_size = -2*XX[1] / npixels # TODO CHECK ODD/EVEN SAMPLING!!!
    wavelength = (codata.h * codata.c / codata.e * 1e9 / energy) * 10 ** (-9)

    f.close()

    w = Wavefront(d=np.zeros((npixels, npixels), dtype=np.complex64),
                  pixel_size=pixel_size, wavelength=wavelength, )

    w.set(complex_amplitude, shift=True)

    return w




energy = 10000.0
wavelength = (codata.h * codata.c / codata.e * 1e9 / energy) * 10 ** (-9)
window_size = 5e-6
aperture_diameter = 1.25e-06
propagation_distance = 75e-6
npixels = 2048*2
pixel_size = window_size / npixels
is_circular = False


if True:  # compute the source
    # Near field propagation of a simple slit
    w = Wavefront(d=np.zeros((npixels, npixels), dtype=np.complex64),
                  pixel_size=pixel_size, wavelength=wavelength)

    a = aperture_diameter / 2 #20e-6 / 2
    x, y = w.get_x_y()


    if is_circular:
        yy = np.outer(y, np.ones_like(x))
        xx = np.outer(np.ones_like(y), x)
        rr2 = xx ** 2 + yy ** 2
        w.set(rr2 < a**2)
    else:
        w.set((abs(y) < a) * (abs(x) < a))

# alternatively, get source from file:

if False:
    if is_circular:
        w = get_pynx_wavefront_from_h5file("aperture_2D_circular.h5")
    else:
        w = get_pynx_wavefront_from_h5file("aperture_2D_rectangular.h5")



w = PropagateNearField(propagation_distance,magnification=1.0,verbose=True) * w


# w = PropagateFRT(propagation_distance) * w
# w = MagnifyNearField(propagation_distance,verbose=True) * w

w = ImshowRGBA(fig_num=1, \
        title="Near field propagation (%d um) of a %3.2fx%3.2f microns aperture"%\
        (1e6*propagation_distance,aperture_diameter*1e6,aperture_diameter*1e6)) * w
plt.show()

complex_amplitude = w.get(shift=True)
xx = np.arange(-npixels // 2, npixels // 2, dtype=np.float32) * pixel_size
intensity = np.abs(complex_amplitude[npixels//2,:])**2
plt.plot(xx*1e6,intensity)
plt.show()

if is_circular:
    filename = "aperture2D_circular_pynx.dat"
else:
    filename = "aperture2D_rectangular_pynx.dat"

f = open(filename,'w')
for i in range(npixels):
    f.write("%g  %g \n"%(xx[i],intensity[i]))
f.close()
print("File written to disk: %s"%filename)

print(energy,npixels,pixel_size)





