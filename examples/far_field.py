import numpy
do_plot=True

#
# Note that the tests for the Fraunhofer phase do not make any assert, because a good matching has not yet been found.
#


from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D

from wofry.propagator.test.propagators_test import get_theoretical_diffraction_pattern


# def FF_propagate_wavefront(wavefront, propagation_distance, shift_half_pixel=False):
#     shape = wavefront.size()
#     delta = wavefront.delta()
#     wavelength = wavefront.get_wavelength()
#     wavenumber = wavefront.get_wavenumber()
#     fft_scale = numpy.fft.fftfreq(shape, d=delta)
#     fft_scale = numpy.fft.fftshift(fft_scale)
#     x2 = fft_scale * propagation_distance * wavelength
#
#     if shift_half_pixel:
#         x2 = x2 - 0.5 * numpy.abs(x2[1] - x2[0])
#
#     p1 = numpy.exp(1.0j * wavenumber * propagation_distance)
#     p2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * x2 ** 2)
#     p3 = 1.0j * wavelength * propagation_distance
#
#     fft = numpy.fft.fft(wavefront.get_complex_amplitude())
#     fft = fft * p1 * p2 / p3
#     fft2 = numpy.fft.fftshift(fft)
#
#     wavefront_out = GenericWavefront1D.initialize_wavefront_from_arrays(x2, fft2, wavelength=wavefront.get_wavelength())
#
#     # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
#     wavefront_out.rescale_amplitude(numpy.sqrt(wavefront.get_intensity().sum() /
#                                                wavefront_out.get_intensity().sum()))
#
#     return wavefront_out

def OneD_far_field_propagate_wavefront(wavefront, propagation_distance, fraunhofer=False):
    #
    # check validity
    #
    x = wavefront.get_abscissas()

    half_max_aperture = 0.5 * (x[-1] - x[0])

    if fraunhofer:
        far_field_distance = half_max_aperture ** 2 / wavelength
        if propagation_distance < far_field_distance:
            print(
                "WARNING: Fraunhoffer diffraction valid for distances > > half_max_aperture^2/lambda = %f m (propagating at %4.1f)" %
                (far_field_distance, propagation_distance))
    #
    # compute Fourier transform
    #

    # frequency for axis 1
    npixels = wavefront.size()
    pixelsize = wavefront.delta()
    wavenumber = wavefront.get_wavenumber()


    freq_nyquist = 0.5 / pixelsize
    if numpy.mod(npixels,2) == 0:
        print("EVENNNNN")
        freq_n = numpy.arange(-npixels//2,npixels//2,1) / (npixels//2)
    else:
        print("ODDDDDD")
        freq_n = numpy.arange(-(npixels-1) // 2, (npixels+1) // 2, 1) / ((npixels-1) // 2)

    print(">>>>>>>>",npixels,freq_n.size,freq_n[0],freq_n[-1])
    freq_x = freq_n * freq_nyquist
    fsq = freq_x**2

    x2 = freq_x * propagation_distance * wavelength

    P1 = numpy.exp(1.0j * wavenumber * propagation_distance)
    P2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * fsq)
    P3 = 1.0j * wavelength * propagation_distance

    if fraunhofer:
        exponential = 1.0+0j
    else:
        exponential = numpy.exp(1j * wavenumber / 2 / propagation_distance * x**2)

    F1 = numpy.fft.fft(exponential*wavefront.get_complex_amplitude())  # Take the fourier transform of the image.
    #  Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F1 *= P1
    F1 *= P2
    F1 /= P3
    F2 = numpy.fft.fftshift(F1)

    wavefront_out = GenericWavefront1D.initialize_wavefront_from_arrays(x_array=x2,
                                                                        y_array=F2,
                                                                        wavelength=wavelength)

    # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
    # wavefront_out.rescale_amplitude(numpy.sqrt(wavefront.get_intensity().sum() /
    #                                            wavefront_out.get_intensity().sum()))

    return wavefront_out

if __name__ == "__main__":


    propagation_distance = 3.0
    aperture_type ="square"
    aperture_diameter = 40e-6
    wavefront_length = 800e-6
    wavelength = 1.24e-10
    npoints =1024+1
    normalization = True


    wf = GenericWavefront1D.initialize_wavefront_from_range(x_min=-wavefront_length / 2, x_max=wavefront_length / 2,
                                                            number_of_points=npoints, wavelength=wavelength)

    wf.set_plane_wave_from_complex_amplitude((2.0 + 1.0j))  # an arbitraty value

    wf.clip(-20e-6,20e-6)

    # wf1 = Fraunhofer1D.propagate_wavefront(wf, propagation_distance, shift_half_pixel = True)
    # wf1 = FF_propagate_wavefront(wf, propagation_distance, shift_half_pixel=False)
    wf1 = OneD_far_field_propagate_wavefront(wf, propagation_distance, fraunhofer=True)

    # get the theoretical value
    angle_x = wf1.get_abscissas() / propagation_distance

    intensity_theory = get_theoretical_diffraction_pattern(angle_x, aperture_type=aperture_type,
                                                           aperture_diameter=aperture_diameter,
                                                           wavelength=wavelength, normalization=normalization)


    print(intensity_theory)

    intensity_calculated = wf1.get_intensity()

    if normalization:
        intensity_calculated /= intensity_calculated.max()

    if do_plot:
        from srxraylib.plot.gol import plot
        # plot(
        #      wf1.get_abscissas() * 1e6 , intensity_calculated,
        #      wf1.get_abscissas() * 1e6, intensity_theory,
        #      # xrange=[-150,150],
        #      legend=["numeric","analytical"]
        #      )

        plot(wf1.get_abscissas() * 1e6 / propagation_distance, intensity_calculated,
             angle_x * 1e6, intensity_theory,
             legend=["Numeric", "Theoretical (far field)"],
             legend_position=(0.95, 0.95),
             title="1D diffraction from a %s aperture of %3.1f um at wavelength of %3.1f A" %
                   (aperture_type, aperture_diameter * 1e6, wavelength * 1e10),
             xtitle="X (urad)", ytitle="Intensity", xrange=[-20, 20])




