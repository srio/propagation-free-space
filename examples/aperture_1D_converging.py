# template for quadratic 1D

import numpy


def quadratic_propagate_wavefront(wavefront1, propagation_distance, radius):

    return wavefront1


def zoom_propagate_wavefront(wavefront1, propagation_distance, magnification_x=1.0):
    wavefront = wavefront1.duplicate()
    shape = wavefront.size()
    delta = wavefront.delta()
    wavenumber = wavefront.get_wavenumber()
    wavelength = wavefront.get_wavelength()

    fft_scale = numpy.fft.fftfreq(shape) / delta

    x = wavefront.get_abscissas()

    x_rescaling = wavefront.get_abscissas() * magnification_x

    r1sq = x ** 2 * (1 - magnification_x)
    r2sq = x_rescaling ** 2 * ((magnification_x - 1) / magnification_x)
    fsq = (fft_scale ** 2 / magnification_x)

    Q1 = wavenumber / 2 / propagation_distance * r1sq
    Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance * fsq)
    Q3 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * r2sq)

    wavefront.add_phase_shift(Q1)

    fft = numpy.fft.fft(wavefront.get_complex_amplitude())
    ifft = numpy.fft.ifft(fft * Q2) * Q3 / numpy.sqrt(magnification_x)

    wf_propagated = GenericWavefront1D.initialize_wavefront_from_arrays(x_rescaling,
                                                                        ifft,
                                                                        wavelength=wavelength)

    return wf_propagated

def sinc(x):
    return numpy.sin(x) / x

def sinc2(x):
    return sinc(x)**2

def run_analytical(w0,detector_array=None):

    F = 0.100000
    D = w0.get_abscissas()[-1] - w0.get_abscissas()[0]
    x = detector_array
    k = w0.get_wavenumber()
    arg =  k * x * D / F / 2
    Ix =  (2 * k * D**2 / numpy.pi / F )**2 * sinc2(arg)

    return x,Ix # returns intensity


if __name__ == "__main__":

    from srxraylib.plot.gol import plot


    npixels= 2048
    D = 100e-6
    radius = -0.1
    photon_energy = 10000.0

    #
    # create input_wavefront
    #
    from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
    w0 = GenericWavefront1D.initialize_wavefront_from_range(x_min=-0.5*D, x_max=0.5*D,
                                                  number_of_points=npixels)
    w0.set_photon_energy(photon_energy)
    w0.set_spherical_wave(radius=radius, complex_amplitude=complex(1, 0))


    print("Validity condition R >? N delta**2 / lambda: %f >? %g "%(0.010, npixels * (D/npixels)**2 / w0.get_wavelength() ))

    #
    # wofry
    #

    # from wofry.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
    # w = FresnelZoom1D.propagate_wavefront(w0, propagation_distance=0.1, magnification_x=0.05)

    w = zoom_propagate_wavefront(w0,propagation_distance=0.1,magnification_x=0.05)


    #
    # analytical
    #
    x,Ix = run_analytical(w0,detector_array=w.get_abscissas())

    plot(w.get_abscissas()*1e6,w.get_intensity()/w.get_intensity().max(),
         1e6*x,Ix/Ix.max(),
         legend=["wofry","analytical"], xtitle="x [um]",
         ylog=False, xrange=[-1,1])

