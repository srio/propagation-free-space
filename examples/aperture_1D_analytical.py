

import numpy
from scipy.special import fresnel, jv
import scipy.constants as codata

from srxraylib.plot.gol import plot



def fresnel_analytical_rectangle(
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

    s_plus  = numpy.sqrt(2.0 * fresnel_number) * ( 1.0 + x / aperture_half)
    s_minus = numpy.sqrt(2.0 * fresnel_number) * ( 1.0 - x / aperture_half)

    fs_plus,fc_plus = fresnel(s_plus)
    fs_minus,fc_minus = fresnel(s_minus)

    Ix = (fc_minus + fc_plus) + 1j*(fs_minus + fs_plus)
    Ix *= 1.0/numpy.sqrt(2.0)

    # TODO note that the global phase (Goldman 4-59) is missing

    return x,Ix

def fresnel_analytical_circle(
    fresnel_number=None,propagation_distance=1.140,
    aperture_half=1e-3,wavelength=639e-9,
    detector_array=None,npoints=100,
    ):

    # see Goldman "Fourier Optics" 4th edition, section 4.5.2

    if fresnel_number is None:
        fresnel_number = aperture_half**2 / (wavelength * propagation_distance)
    else:
        fresnel_number = float(fresnel_number)


    if detector_array is None:
        if fresnel_number > 1.0:
            window_aperture_ratio = 2.0
        else:
            window_aperture_ratio = 1.0 / fresnel_number
        detector_array = numpy.linspace(-window_aperture_ratio*aperture_half,window_aperture_ratio*aperture_half,npoints)

    rprime = detector_array / aperture_half * numpy.sqrt(fresnel_number)

    Irprime = numpy.zeros_like(rprime,dtype=complex)


    for i in range(rprime.size):
        Irprime[i] = _integrate(rprime[i],fresnel_number=fresnel_number)

    print("Fresnel number: ",fresnel_number)

    return detector_array,Irprime


def _integrate(x,fresnel_number=1.0,N=10000):
    eta = numpy.linspace(0.0,numpy.sqrt(fresnel_number),N)
    integrand = eta * numpy.exp(1j*numpy.pi*eta**2) * jv(0,eta*2*numpy.pi*x)
    # return integrand.sum() / eta.size
    return 2 * numpy.pi * numpy.trapz(integrand,eta)

def test__integrate():
    from numpy.testing import assert_almost_equal
    tmp = _integrate(1.0,fresnel_number=10.0,N=10000)
    print("_integrate(1.0,fresnel_number=10.0,N=10000) = ",tmp)
    assert_almost_equal(tmp,-0.0171768-1.19285j,5)


    tmp1 = _integrate(5.0,fresnel_number=5.0,N=10000)
    print("_integrate(5.0,fresnel_number=5.0,N=10000) = ",tmp1)
    assert_almost_equal(tmp1,-0.0192585 - 0.0218896j,5)

if __name__ == "__main__":

    import matplotlib.pylab as plt
    # test__integrate()

    #
    # Input data (near field, as in sajid tests)
    #
    energy = 10000.0
    wavelength = ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
    window_size = 5e-6
    aperture_diameter = window_size/4
    npoints = 2048*2
    propagation_distance = 75e-6


    detector_window = window_size #/20
    is_circular = False

    detector_array = numpy.linspace(-0.5*detector_window,0.5*detector_window,npoints)
    # inputs end

    magnification_x = detector_window / window_size

    if is_circular:
        detector_array = None # numpy.linspace(-5e-6,5e-6,1000)
        r, alpha = fresnel_analytical_circle(propagation_distance=propagation_distance,
                        aperture_half=aperture_diameter/2,
                        wavelength=wavelength,
                        detector_array=detector_array,npoints=npoints,)

        pattern = numpy.abs(alpha)**2

        source = r * 0.0 + 1.0
        source[numpy.where(numpy.abs(r)>(aperture_diameter/2))] = 0.0

        integral_source = source.sum()
        integral_pattern = pattern.sum()
        print(integral_source,integral_pattern)

        plot(r*1e6,pattern,r*1e6,source*integral_pattern/integral_source,xtitle="x [um]",title="CIRCULAR analytical")

    else:
        # detector_array = numpy.linspace(-0.5*detector_window,0.5*detector_window,npoints)
        x, alpha = fresnel_analytical_rectangle(fresnel_number=None, propagation_distance=propagation_distance,
                        aperture_half=aperture_diameter/2,
                        wavelength=wavelength,
                        detector_array=detector_array,npoints=npoints,)

        pattern = numpy.abs(alpha)**2

        source = detector_array * 0.0 + 1.0
        source[numpy.where(numpy.abs(detector_array)>(aperture_diameter/2))] = 0.0

        integral_source = source.sum()
        integral_pattern = pattern.sum()
        print(integral_source,integral_pattern)

        plot(x*1e6,pattern,x*1e6,source*integral_pattern/integral_source,xtitle="x [um]",title="RECTANGULAR analytical")


    #
    # propagation wofry
    #
    from aperture_1D import propagate_with_wofry, plot_intensity

    #
    # Creation of wavefront
    #

    x1 = numpy.linspace(-0.5*window_size,0.5*window_size,npoints)
    wavefront = numpy.ones(npoints,dtype=complex)
    # apply aperture
    wavefront[numpy.where(numpy.abs(x1)>(aperture_diameter/2))] = 0.0

    method = "integral"
    wavefront_propagated, x_propagated = propagate_with_wofry(wavefront,x1,wavelength,propagation_distance,method=method,
                                                              magnification_x=magnification_x)


    # factor = (numpy.abs(wavefront_propagated)**2).mean()
    # wavefront_propagated /= numpy.sqrt(factor)

    plot_intensity(
                        wavefront_propagated, 1e6 * x_propagated,
                        xlabel="x [um]",ylabel="intensity [arbitrary units]",
                        # legend=["Propagated wavefield","Input wavefield",],legend_position=[0.5,0.5],
                        dumpfile=None,title="wofry"
                        )


    #
    # propagation sajid
    #

    from aperture_1D import propagate_with_sajid

    # x = np.linspace(-0.5*window_size,0.5*window_size,npoints)
    # wavefront = np.ones(npoints,dtype=complex)
    # wavefront[np.where(np.abs(x)>(aperture_diameter/2))] = 0.0
    # plot_intensity(wavefront,1e6*x,xlabel="x [um]",ylabel="source intensity [arbitrary units]",title="incident wavefront")

    # method_s = "propTF"
    # method_s = "exact_prop"
    method_s = "exact_prop_numba"
    wavefront_propagated_s, L_propagated_s = propagate_with_sajid(wavefront,x1,wavelength,
                            propagation_distance,method=method_s,
                            magnification_x=magnification_x)
    x_propagated_s = numpy.linspace(-0.5*L_propagated_s,0.5*L_propagated_s,numpy.shape(wavefront_propagated_s)[0])

    # plot_intensity(wavefront_propagated_s,1e6*x_propagated_s,wavefront, 1e6 * x,
    #             xlabel="x [um]",ylabel="propagated intensity [arbitrary units]",
    #             title="XWP(%s)"%method_s)

    plot_intensity(
                        wavefront_propagated_s, 1e6 * x_propagated_s,
                        wavefront, 1e6 * x1,
                        xlabel="x [um]",ylabel="intensity [arbitrary units]",title="sajid",
                        legend=["Propagated wavefield","Input wavefield"],legend_position=[0.5,0.5],
                        dumpfile=None,
                        )


    #

    print("magnification_x=",magnification_x)


    plot(1e6*x,pattern,
         1e6*x_propagated,numpy.abs(wavefront_propagated)**2,
         1e6 * x_propagated_s, numpy.abs(wavefront_propagated_s) ** 2,
         legend=["analytical","WOFRY","XWP"],
         xtitle="x [um]",show=0,xrange=[-0.10,0.10])

    dumpfile = None # "aperture_1D_analytical.png"
    if dumpfile is not None:
        plt.savefig(dumpfile)
        print("File written to disk: %s"%dumpfile)
    plt.show()