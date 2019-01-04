

import numpy
from scipy.special import fresnel, jv
import scipy.constants as codata

from srxraylib.plot.gol import plot

# def fresnel_analytical_rectangle(propagation_distance = 1.140,
#     aperture_half = 1e-3,
#     wavelength = 639e-9,
#     npoints=1000,
#     window_aperture_ratio=2.0,
#     x0 = 0.0,
#     z0 = -1e10, # (source at -infinity
#     ):
#     xM = numpy.linspace(-window_aperture_ratio*aperture_half,window_aperture_ratio*aperture_half,npoints)
#
#     # step 3
#     thetaM = numpy.arctan((x0-xM)/z0)
#     x = xM + propagation_distance * thetaM
#
#     rho = - propagation_distance * z0 / (propagation_distance - z0) / numpy.cos(thetaM)
#     s_plus  = numpy.sqrt( 2.0 / wavelength / rho) * (-xM + aperture_half)
#     s_minus = numpy.sqrt( 2.0 / wavelength / rho) * (-xM - aperture_half)
#
#     fs_plus,fc_plus = fresnel(s_plus)
#     fs_minus,fc_minus = fresnel(s_minus)
#
#     alpha = (1.0-1.0j)/2 * ( (fc_plus - fc_minus) + 1j*(fs_plus - fs_minus) )
#
#     return x,4*alpha # TODO: check this 4!!


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
        x = detector_array

    s_plus  = numpy.sqrt(2.0 * fresnel_number) * ( 1.0 + x / aperture_half)
    s_minus = numpy.sqrt(2.0 * fresnel_number) * ( 1.0 - x / aperture_half)

    fs_plus,fc_plus = fresnel(s_plus)
    fs_minus,fc_minus = fresnel(s_minus)

    Ix = (fc_minus + fc_plus) + 1j*(fs_minus + fs_plus)
    Ix *= 1.0/numpy.sqrt(2.0)

    return x,Ix

def fresnel_analytical_circle(
    fresnel_number=None,propagation_distance=1.140,
    aperture_half=1e-3,wavelength=639e-9,
    detector_array=None,npoints=1000,
    ):



    if fresnel_number is None:
        fresnel_number = aperture_half**2 / (wavelength * propagation_distance)
    else:
        fresnel_number = float(fresnel_number)

    print("Fresnel number: ",fresnel_number)



    if detector_array is None:
        if fresnel_number > 1.0:
            window_aperture_ratio = 2.0
        else:
            window_aperture_ratio = 1.0 / fresnel_number
        x = numpy.linspace(-window_aperture_ratio*aperture_half,window_aperture_ratio*aperture_half,npoints)
    else:
        x = detector_array


    fresnel_number = 10.0
    rprime = numpy.linspace(-4,4,npoints) # x / aperture_half * numpy.sqrt(fresnel_number)
    Irprime = numpy.zeros_like(rprime)

    eta = numpy.linspace(0.0,numpy.sqrt(fresnel_number),1000)
    # print(eta)

    for i in range(rprime.size):
        integrand = eta * numpy.exp(1j*numpy.pi*eta**2) * jv(0,eta*2*numpy.pi*rprime[i])

        print(integrand[0:10],integrand.shape,(jv(0,eta*2*numpy.pi*rprime[i]).shape))
        Irprime[i] = 2 * numpy.pi * integrand.sum() / eta.size


    print("Fresnel number: ",fresnel_number)
    plot(rprime,numpy.abs(Irprime)**2)
    return rprime,Irprime

if __name__ == "__main__":


    # x = numpy.linspace(-10,10,200)
    # sa, ca = fresnel(x)
    # plot(x,sa,x,ca,legend=['sa','ca'])

    # x = numpy.linspace(0,20,200)
    # j0 = jv(0,x)
    # plot(x,j0)


    #
    #
    #
    # x0 = 0.0
    # z0 = -2.507
    # z = 1.140
    # w = 1e-3
    # wavelength = 639e-9

    # xM = numpy.linspace(-2e-3,2e-3,1000)
    #
    # # step 3
    # thetaM = numpy.arctan((x0-xM)/z0)
    # x = xM + z * thetaM
    #
    # rho = - z * z0 / (z - z0) / numpy.cos(thetaM)
    # s_plus  = numpy.sqrt( 2.0 / wavelength / rho) * (-xM + w)
    # s_minus = numpy.sqrt( 2.0 / wavelength / rho) * (-xM - w)
    #
    # fs_plus,fc_plus = fresnel(s_plus)
    # fs_minus,fc_minus = fresnel(s_minus)
    #
    # alpha = (1.0-1.0j)/2 * ( (fc_plus - fc_minus) + 1j*(fs_plus - fs_minus) )

    # x, alpha = fresnel_analytical_rectangle(x0=0.0,z0=-2.507,z=1.140,w=1e-3,wavelength=639e-9)



    #
    # Input data (near field, as in sajid tests)
    #
    energy = 10000.0
    wavelength = ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
    window_size = 5e-6
    aperture_diameter = window_size/4
    npoints = 5000# 2048
    propagation_distance = 75e-6





    is_circular = True

    if is_circular:
        detector_array = None # numpy.linspace(-5e-6,5e-6,1000)
        x, alpha = fresnel_analytical_circle(fresnel_number=1, propagation_distance=propagation_distance,
                        aperture_half=aperture_diameter/2,
                        wavelength=wavelength,
                        detector_array=detector_array,npoints=npoints,)
    else:
        x, alpha = fresnel_analytical_rectangle(fresnel_number=None, propagation_distance=propagation_distance,
                        aperture_half=aperture_diameter/2,
                        wavelength=wavelength,
                        detector_array=None,npoints=npoints,)

        pattern = numpy.abs(alpha)**2

        source = x * 0.0 + 1.0
        source[numpy.where(numpy.abs(x)>(aperture_diameter/2))] = 0.0

        integral_source = source.sum()
        integral_pattern = pattern.sum()
        print(integral_source,integral_pattern)

        plot(x*1e6,pattern,x*1e6,source*integral_pattern/integral_source,xtitle="x [um]")


