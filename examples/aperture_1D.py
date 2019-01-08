

import numpy as np
import matplotlib.pylab as plt
import scipy.constants as codata

# please make specific imports insode the funcctions.


#
# WOFRY FUNCTIONS ######################################################################################################
#
def initialize_wofry_propagators():
    from wofry.propagator.propagator import PropagationManager

    from wofry.propagator.propagators1D.fraunhofer import Fraunhofer1D
    from wofry.propagator.propagators1D.fresnel import Fresnel1D
    from wofry.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
    from wofry.propagator.propagators1D.integral import Integral1D
    from wofry.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
    from wofry.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D


    propagator = PropagationManager.Instance()

    try:
        propagator.add_propagator(Fraunhofer1D())
        propagator.add_propagator(Fresnel1D())
        propagator.add_propagator(FresnelConvolution1D())
        propagator.add_propagator(Integral1D())
        propagator.add_propagator(FresnelZoom1D())
        propagator.add_propagator(FresnelZoomScaling1D())
    except:
        pass

    return propagator

def propagate_with_wofry(wavefront,x,wavelength,propagation_distance,
                         method='fft',aperture_diameter=10e-6,
                         magnification_x=1.0):

    from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
    from wofry.propagator.propagator import PropagationElements
    from wofry.beamline.optical_elements.absorbers.slit import WOSlit1D
    from syned.beamline.shape import Rectangle
    from syned.beamline.element_coordinates import ElementCoordinates
    from syned.beamline.beamline_element import BeamlineElement

    from wofry.propagator.propagators1D.fraunhofer import Fraunhofer1D
    from wofry.propagator.propagators1D.fresnel import Fresnel1D
    from wofry.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
    from wofry.propagator.propagators1D.integral import Integral1D
    from wofry.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
    from wofry.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D

    from wofry.propagator.propagator import PropagationManager, PropagationParameters

    print("\n#")
    print("# 1D aperture propagation ")
    print("#")

    wf = GenericWavefront1D.initialize_wavefront_from_arrays(x,wavefront,wavelength=wavelength)

    propagation_elements = PropagationElements()

    slit = WOSlit1D(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, 0, 0))


    propagation_elements.add_beamline_element(
                    BeamlineElement(optical_element=slit,coordinates=ElementCoordinates(p=0,q=propagation_distance))
                    )

    propagation_parameters = PropagationParameters(wavefront=wf,
                                                   propagation_elements=propagation_elements)

    # initialize propagator

    propagator = initialize_wofry_propagators()

    if method == "fraunhofer":
        handler_name = Fraunhofer1D.HANDLER_NAME
    elif method == "fft":
        handler_name = Fresnel1D.HANDLER_NAME
    elif method == "fft_conv":
        handler_name = FresnelConvolution1D.HANDLER_NAME
    elif method == "integral":
        handler_name = Integral1D.HANDLER_NAME
        propagation_parameters.set_additional_parameters("magnification_x", magnification_x)
    elif method == "zoom":
        handler_name = FresnelZoom1D.HANDLER_NAME
    elif method == "zoom_scaling":
        handler_name = FresnelZoomScaling1D.HANDLER_NAME
    else:
        raise NotImplementedError

    wf1 = propagator.do_propagation(propagation_parameters, handler_name)

    return wf1.get_complex_amplitude(), wf1.get_abscissas()

#
# SAJID FUNCTIONS ######################################################################################################
#

def propagate_with_sajid(wavefront,x,wavelength,propagation_distance,method="propTF",
                         magnification_x=1.0):

    # dependencies found in installing https://github.com/s-sajid-ali/xwp
    # pip install pyfftw
    # pip install tqdm
    # pip install numexpr
    # pip install numba
    # pip install dask
    # pip install cython
    # pip install toolz




    from xwp.spectral_1d import propTF
    from xwp.exact_1d import exact_prop,exact_prop_numba

    wavel = wavelength
    pi = np.pi
    z = propagation_distance
    N = x.size # 2048
    L_in  = x[-1] - x[0] # 5e-6
    in_wave = wavefront # np.zeros(N)


    print('Fresnel Number :', (L_in**2)/(wavel*z))


    if method == 'propTF':
        out_,L_out = propTF(in_wave,L_in/N,L_in,wavel,z)
    elif method == 'prop1FT':
        out_,L_out = prop1FT(in_wave,L_in/N,L_in,wavel,z)
    elif method == 'propFF':
        out_,L_out = propFF(in_wave,L_in/N,L_in,wavel,z)
    elif method == 'propIR':
        out_,L_out = propIR(in_wave,L_in/N,L_in,wavel,z)
    elif method == 'exact_prop_numba':
        out_ = np.zeros((N),dtype='complex128')
        L_out = L_in.copy() * magnification_x
        exact_prop_numba(in_wave,out_,L_in,L_out,wavel,z)
    elif method == 'exact_prop':
        out_ = np.zeros((N),dtype='complex128')
        L_out = L_in.copy()
        exact_prop(in_wave,out_,L_in,L_out,wavel,z)
    else:
        raise NotImplementedError


    return np.abs(out_),L_out



def plot_intensity(wavefront,x,wavefront2=None,x2=None,
                   xlabel="",ylabel="",title="",legend=[None,None],legend_position=None,
                   dumpfile=None):
    # plot intensity
    plt.clf()
    plt.plot(x,np.abs(wavefront)**2,label=legend[0])
    if wavefront2 is not None:
        plt.plot(x2, np.abs(wavefront2) ** 2,label=legend[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    ax = plt.subplot(111)
    if legend[0] is not None:
        ax.legend(bbox_to_anchor=legend_position)

    if dumpfile is not None:
        plt.savefig(dumpfile)
        print("File written to disk: %s"%dumpfile)
    plt.show()

if __name__ == "__main__":

    # units are SI unless specified

    #
    # Input data (far fieeld, as in wofry tests)
    #
    wavelength = 1e-10
    window_size = 1500e-6
    aperture_diameter = 40e-6
    npoints = 1024
    propagation_distance = 30.0


    #
    # Input data (near field, as in sajid tests)
    #
    energy = 10000.0
    wavelength = ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
    window_size = 5e-6
    aperture_diameter = window_size/4
    npoints = 2048
    propagation_distance = 75e-6

    magnification_x = 1.0 # 0.5
    #
    # Creation of wavefront
    #

    x = np.linspace(-0.5*window_size,0.5*window_size,npoints)
    wavefront = np.ones(npoints,dtype=complex) # * 3
    # apply aperture
    wavefront[np.where(np.abs(x)>(aperture_diameter/2))] = 0.0

    # plot_intensity(wavefront,1e6*x,
    #                xlabel="x [um]",ylabel="source intensity [arbitrary units]",title="incident wavefront")

    #
    # propagation wofry
    #
    method = "integral"
    wavefront_propagated, x_propagated = propagate_with_wofry(wavefront,x,wavelength,propagation_distance,
                                method=method,magnification_x=magnification_x)

    plot_intensity(
                        wavefront_propagated, 1e6 * x_propagated,
                        wavefront, 1e6 * x,
                        xlabel="x [um]",ylabel="intensity [arbitrary units]",title="wofry",
                        legend=["Propagated wavefield","Input wavefield"],legend_position=[0.5,0.5],
                        dumpfile="aperture_1D.png",
                        )

    #
    # propagation sajid
    #

    # x = np.linspace(-0.5*window_size,0.5*window_size,npoints)
    # wavefront = np.ones(npoints,dtype=complex)
    # wavefront[np.where(np.abs(x)>(aperture_diameter/2))] = 0.0
    # plot_intensity(wavefront,1e6*x,xlabel="x [um]",ylabel="source intensity [arbitrary units]",title="incident wavefront")

    # method_s = "propTF"
    # method_s = "exact_prop"
    method_s = "exact_prop_numba"
    wavefront_propagated_s, L_propagated_s = propagate_with_sajid(wavefront,x,wavelength,
                            propagation_distance,method=method_s,
                            magnification_x=magnification_x)
    x_propagated_s = np.linspace(-0.5*L_propagated_s,0.5*L_propagated_s,np.shape(wavefront_propagated_s)[0])

    # plot_intensity(wavefront_propagated_s,1e6*x_propagated_s,wavefront, 1e6 * x,
    #             xlabel="x [um]",ylabel="propagated intensity [arbitrary units]",
    #             title="XWP(%s)"%method_s)

    plot_intensity(
                        wavefront_propagated_s, 1e6 * x_propagated_s,
                        wavefront, 1e6 * x,
                        xlabel="x [um]",ylabel="intensity [arbitrary units]",title="sajid",
                        legend=["Propagated wavefield","Input wavefield"],legend_position=[0.5,0.5],
                        dumpfile="aperture_1D.png",
                        )

    #
    #plot comparison
    #
    plot_intensity(wavefront_propagated_s,1e6*x_propagated_s,
                        wavefront_propagated, 1e6 * x_propagated,
                        xlabel="x [um]",ylabel="propagated intensity [arbitrary units]",
                        legend=["XWP(%s)"%method_s,"WOFRY(%s)"%method],legend_position=[0.5,0.5],
                        # dumpfile="aperture_1D.png"
                        )