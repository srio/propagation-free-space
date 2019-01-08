

import numpy as np
import matplotlib.pylab as plt
import scipy.constants as codata

# please make specific imports insode the funcctions.

from aperture_1D import initialize_wofry_propagators, propagate_with_wofry
from aperture_1D import propagate_with_sajid
from aperture_1D import plot_intensity



if __name__ == "__main__":

    # units are SI unless specified

    #
    # Input data (near field, as in sajid tests)
    #
    energy = 10000.0
    wavelength = ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
    window_size = 5e-6
    aperture_diameter = window_size/4
    npoints = 2048//2
    propagation_distance = 75e-6


    #
    # Creation of wavefront
    #

    x = np.linspace(-0.5*window_size,0.5*window_size,npoints)
    wavefront = np.ones(npoints,dtype=complex)
    # apply aperture
    wavefront[np.where(np.abs(x)>(aperture_diameter/2))] = 0.0

    # plot_intensity(wavefront,1e6*x,
    #                xlabel="x [um]",ylabel="source intensity [arbitrary units]",title="incident wavefront")

    #
    # propagation wofry
    #
    method = "integral"
    wavefront_propagated, x_propagated = propagate_with_wofry(wavefront,x,wavelength,
                        propagation_distance,method=method,aperture_diameter=aperture_diameter)

    # plot_intensity(wavefront_propagated,1e6*x_propagated,wavefront, 1e6 * x,
    #             xlabel="x [um]",ylabel="propagated intensity [arbitrary units]",
    #             title="propagated_with_WOFRY_(%s)"%method)


    plot_intensity(
                        wavefront_propagated, 1e6 * x_propagated,
                        wavefront, 1e6 * x,
                        xlabel="x [um]",ylabel="intensity [arbitrary units]",
                        legend=["Propagated wavefield","Input wavefield"],legend_position=[0.5,0.8],
                        dumpfile="aperture_1D_over2.png",
                        )

    #
    # Input data (near field, as in sajid tests)
    #
    energy = 10000.0
    wavelength = ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
    window_size = 5e-6
    aperture_diameter = window_size/4
    npoints = 2048//4
    propagation_distance = 75e-6


    #
    # Creation of wavefront
    #

    x = np.linspace(-0.5*window_size,0.5*window_size,npoints)
    wavefront = np.ones(npoints,dtype=complex)
    # apply aperture
    wavefront[np.where(np.abs(x)>(aperture_diameter/2))] = 0.0

    # plot_intensity(wavefront,1e6*x,
    #                xlabel="x [um]",ylabel="source intensity [arbitrary units]",title="incident wavefront")

    #
    # propagation wofry
    #
    method = "integral"
    wavefront_propagated, x_propagated = propagate_with_wofry(wavefront,x,wavelength,
                        propagation_distance,method=method,aperture_diameter=aperture_diameter)

    # plot_intensity(wavefront_propagated,1e6*x_propagated,wavefront, 1e6 * x,
    #             xlabel="x [um]",ylabel="propagated intensity [arbitrary units]",
    #             title="propagated_with_WOFRY_(%s)"%method)


    plot_intensity(
                        wavefront_propagated, 1e6 * x_propagated,
                        wavefront, 1e6 * x,
                        xlabel="x [um]",ylabel="intensity [arbitrary units]",
                        legend=["Propagated wavefield","Input wavefield"],legend_position=[0.5,0.8],
                        dumpfile="aperture_1D_over4.png",
                        )


    # #
    # # propagation sajid
    # #
    # method_s = "exact_prop_numba"
    # wavefront_propagated_s, L_propagated_s = propagate_with_sajid(wavefront,x,wavelength,
    #                         propagation_distance,method=method_s,
    #                         magnification_x=1.0)
    # x_propagated_s = np.linspace(-0.5*L_propagated_s,0.5*L_propagated_s,np.shape(wavefront_propagated_s)[0])
    #
    # # plot_intensity(wavefront_propagated_s,1e6*x_propagated_s,wavefront, 1e6 * x,
    # #             xlabel="x [um]",ylabel="propagated intensity [arbitrary units]",
    # #             title="XWP(%s)"%method_s)
    #
    # plot_intensity(
    #                     wavefront_propagated_s, 1e6 * x_propagated_s,
    #                     wavefront, 1e6 * x,
    #                     xlabel="x [um]",ylabel="intensity [arbitrary units]",title="sajid",
    #                     legend=["Propagated wavefield","Input wavefield"],legend_position=[0.5,0.5],
    #                     dumpfile="aperture_1D_over4.png",
    #                     )

