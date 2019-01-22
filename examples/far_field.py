import numpy
do_plot=True

#
# Note that the tests for the Fraunhofer phase do not make any assert, because a good matching has not yet been found.
#


from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D

# from wofry.propagator.test.propagators2D_test import get_theoretical_diffraction_pattern

from aperture_1D_analytical import fresnel_analytical_rectangle


def fraunhofer_analytical_rectangle(
    fresnel_number=None,propagation_distance=1.140,
    aperture_half=1e-3,wavelength=639e-9,
    detector_array=None,npoints=1000,
    ):


    if fresnel_number is None:
        fresnel_number = aperture_half**2 / (wavelength * propagation_distance)

    print("Fresnel number: ",fresnel_number)


    if detector_array is None:
        if fresnel_number > 1.0:
            window_aperture_ratio = 2.0
        else:
            window_aperture_ratio = 1.0 / fresnel_number
        x = numpy.linspace(-window_aperture_ratio*aperture_half,window_aperture_ratio*aperture_half,npoints)
    else:
        x = detector_array.copy()

    argument_sinc = 2.0 * aperture_half * numpy.pi / wavelength / propagation_distance * x # TODO: check the 2??
    alpha = 2.0 * aperture_half / (wavelength*propagation_distance)**(1.0/2.0) * \
            numpy.exp(1j*numpy.pi/wavelength/propagation_distance * x**2) * \
            numpy.sin(argument_sinc) / argument_sinc

    # TODO note that the global phase (Goldman 4-59) is missing

    return x,alpha


def OneD_far_field_propagate_wavefront(wavefront, propagation_distance, fraunhofer=False):
    #
    # check validity
    #
    x = wavefront.get_abscissas()
    deltax = wavefront.delta()

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
    # TODO: check the phase of this type:
    # P2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * x**2)
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
    F1 /= numpy.sqrt(P3) # this is 1D -> no sqrt for 2D
    F2 = numpy.fft.fftshift(F1)
    F2 *= deltax # why??

    wavefront_out = GenericWavefront1D.initialize_wavefront_from_arrays(x_array=x2,
                                                                        y_array=F2,
                                                                        wavelength=wavelength)

    # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
    # wavefront_out.rescale_amplitude(numpy.sqrt(wavefront.get_intensity().sum() /
    #                                            wavefront_out.get_intensity().sum()))

    return wavefront_out

if __name__ == "__main__":

    import scipy.constants as codata
    from srxraylib.plot.gol import plot
    import matplotlib.pylab as plt

    # wofry tests

    # propagation_distance = 5
    # aperture_type ="square"
    # aperture_diameter = 40e-6
    # wavefront_length = 800e-6
    # wavelength = 1.24e-10
    # npoints =1024
    # normalization = True


    #
    # sajid inputs; distance increased to far field
    #
    propagation_distance = 0.1
    aperture_type ="square"
    wavefront_length = 5e-6 * 10
    aperture_diameter = 1.25e-6 # wavefront_length/4
    energy = 10000.0
    wavelength = ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
    npoints = 2048 * 5
    normalization = False


    amplitude = (5.0 + 3.0j)

    #
    # source
    #
    wf = GenericWavefront1D.initialize_wavefront_from_range(x_min=-wavefront_length / 2, x_max=wavefront_length / 2,
                                                            number_of_points=npoints, wavelength=wavelength)
    wf.set_plane_wave_from_complex_amplitude(amplitude)  # an arbitraty value
    wf.clip(-0.5*aperture_diameter,0.5*aperture_diameter)
    # plot(wf.get_abscissas() * 1e6 , wf.get_intensity())
    deltax = wf.get_abscissas()[1] - wf.get_abscissas()[0]

    #
    # numeric
    #

    # wf1 = Fraunhofer1D.propagate_wavefront(wf, propagation_distance, shift_half_pixel = True)
    # wf1 = FF_propagate_wavefront(wf, propagation_distance, shift_half_pixel=False)
    wf1 = OneD_far_field_propagate_wavefront(wf, propagation_distance, fraunhofer=False)
    angle_x = wf1.get_abscissas() / propagation_distance

    intensity_calculated = wf1.get_intensity()
    phase_calculated = wf1.get_phase(unwrap=False)
    if normalization:
        intensity_calculated /= intensity_calculated.max()

    # plot(wf1.get_abscissas() * 1e6 , intensity_calculated)

    #
    # Fraunhofer
    #

    # using old interface (wofry tests)
    # intensity_theory_fraunhofer = get_theoretical_diffraction_pattern(angle_x, aperture_type=aperture_type,
    #                                                        aperture_diameter=aperture_diameter,
    #                                                        wavelength=wavelength, normalization=normalization)

    x_fraunhofer, alpha = fraunhofer_analytical_rectangle(
                fresnel_number=None,propagation_distance=propagation_distance,
                aperture_half=0.5*aperture_diameter,wavelength=wavelength,
                detector_array=wf1.get_abscissas(),npoints=None,
                )
    intensity_theory_fraunhofer = numpy.abs(amplitude*alpha)**2
    phase_theory_fraunhofer = numpy.unwrap(numpy.angle(alpha))

    if normalization:
        intensity_theory_fraunhofer /= intensity_theory_fraunhofer.max()


    #
    # Fresnel
    #
    x_fresnel, alpha = fresnel_analytical_rectangle(
                fresnel_number=None,propagation_distance=propagation_distance,
                aperture_half=0.5*aperture_diameter,wavelength=wavelength,
                detector_array=wf1.get_abscissas(),npoints=None,
                )
    intensity_theory_fresnel = numpy.abs(amplitude*alpha)**2
    phase_theory_fresnel = numpy.unwrap(numpy.angle(alpha))

    if normalization:
        intensity_theory_fresnel /= intensity_theory_fresnel.max()

    #

    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n")

    fresnel_number = (0.5*aperture_diameter)**2 / (wavelength * propagation_distance)
    print("fresnel numeber: ",fresnel_number)

    half_max_aperture = 0.5 * aperture_diameter # 0.5 * (wf.get_abscissas()[-1] - wf.get_abscissas()[0])
    if propagation_distance > (2*numpy.pi*half_max_aperture**2 /wavelength):
        test = "YESSSSSSSS"
    else:
        test = "NOOOOOOOOO"
    print("Fraunhoffer condition (%s): distances >>? 2 pi half_max_aperture^2/lambda; %f >>?  %f" %
        (test,propagation_distance,(2*numpy.pi*half_max_aperture**2 /wavelength)))


    if propagation_distance > (npoints*deltax**2/wavelength):
        test = "YESSSSSSSS"
    else:
        test = "NOOOOOOOOO"
    print("Validity condition 1 (%s): distances >? N delta**2 / lambda; %f >?  %f " %
        (test,propagation_distance,(npoints*deltax**2/wavelength)))

    if propagation_distance > (half_max_aperture*deltax/wavelength):
        test = "YESSSSSSSS"
    else:
        test = "NOOOOOOOOO"
    print("Validity condition 1-relaxed (%s): distances >? half-aperture delta / lambda; %f >?  %f " %
        (test,propagation_distance,half_max_aperture*deltax/wavelength))

    if propagation_distance < (npoints*deltax**2/wavelength):
        test = "YESSSSSSSS"
    else:
        test = "NOOOOOOOOO"
    print("Validity condition 2 (global phase) (%s): distances <? N delta**2 / lambda; %f <?  %f " %
        (test,propagation_distance,(npoints*deltax**2/wavelength)))

    print("\n\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n")

    if do_plot:
        plot(
             wf1.get_abscissas() * 1e6, intensity_calculated,
             x_fresnel * 1e6, intensity_theory_fresnel,
             xrange=[-200,200],
             ylog=True,
             legend=["numeric","analytical (Fresnel)"], #,"analytical (Fraunhofer)",]
             xtitle="x [um]",ytitle="Intensity [arbitrary units]",
             show=False,
             )

        dumpfile = "far_field_a.png"
        if dumpfile is not None:
            plt.savefig(dumpfile)
            print("File written to disk: %s"%dumpfile)
        plt.show()

        plot(
             wf1.get_abscissas() * 1e6, intensity_calculated,
             x_fresnel * 1e6, intensity_theory_fresnel,
             ylog=True,
             legend=["numeric","analytical (Fresnel)"], #,"analytical (Fraunhofer)",]
             xtitle="x [um]",ytitle="Intensity [arbitrary units",
             show=False,
             )

        dumpfile = "far_field_b.png"
        if dumpfile is not None:
            plt.savefig(dumpfile)
            print("File written to disk: %s"%dumpfile)
        plt.show()



        # plot(wf1.get_abscissas() * 1e6, intensity_theory_fraunhofer,
        #      wf1.get_abscissas() * 1e6, intensity_theory_fresnel,
        #      wf1.get_abscissas() * 1e6, intensity_calculated,
        #      wf.get_abscissas() * 1e6, wf.get_intensity(),
        #      xrange=[-200,200],yrange=[0,1.1*numpy.abs(amplitude)**2],ylog=True,
        #      title="intensity",
        #      legend=["fraunhofer","fresnel","numeric","source"])



        plot(wf1.get_abscissas() * 1e6, phase_theory_fraunhofer-phase_theory_fraunhofer.min(),
             wf1.get_abscissas() * 1e6, phase_theory_fresnel-phase_theory_fresnel.min(),
             wf1.get_abscissas() * 1e6, numpy.unwrap(
                numpy.angle(numpy.exp(1j * wf1.get_wavenumber() * wf1.get_abscissas() ** 2 / 2 / propagation_distance))),
             wf1.get_abscissas() * 1e6, phase_calculated-phase_calculated.min(),
             title="phase",xrange=[-20,20],
             legend=["analytical fraunhofer","analytical fresnel","AD HOC","numeric !! DOES NOT WORK!!!"])






