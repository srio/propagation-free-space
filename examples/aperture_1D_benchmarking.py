

import numpy as np


from aperture_1D import initialize_wofry_propagators, propagate_with_wofry, propagate_with_sajid, plot_intensity


def propagate_with_any_library(wavefront, x, wavelength, propagation_distance,
                            library="wofry",method="integral", magnification_x="1.0",
                            repeat=1):

    import time

    t0 = time.time()
    if library == "wofry":
        for i in range(repeat):
            wavefront_propagated, x_propagated = propagate_with_wofry(wavefront, x, wavelength, propagation_distance,
                                                                  method=method, magnification_x=magnification_x)
    elif library == "sajid":
        for i in range(repeat):
            wavefront_propagated, L_propagated_s = propagate_with_sajid(wavefront, x, wavelength,
                                                                      propagation_distance, method=method,
                                                                      magnification_x=magnification_x)
        x_propagated = np.linspace(-0.5 * L_propagated_s, 0.5 * L_propagated_s, np.shape(wavefront_propagated)[0])
    else:
        raise Exception(NotImplementedError)

    return wavefront_propagated, x_propagated, time.time()-t0

def plot_bar(x,y,ylog=False,title="",xtitle="",ytitle="",colors=None, filename=None, show=True):
    import matplotlib.pylab as plt

    plt.figure(figsize=(10,4))
    ax = plt.subplot(111)

    tmp = plt.bar(x, y)
    if colors is not None:
        for i in range(len(colors)):
            tmp[i].set_facecolor(colors[i])

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    if ylog:
        ax.set_yscale('log')


    if filename is not None:
        plt.savefig(filename)
        print("File written to disk: %s"%filename)

    if show:
        plt.show()

if __name__ == "__main__":

    import scipy.constants as codata

    # units are SI unless specified

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


    repeat = 1
    #
    # Creation of wavefront
    #

    x = np.linspace(-0.5*window_size,0.5*window_size,npoints)
    wavefront = np.ones(npoints,dtype=complex) #* 3
    # apply aperture
    wavefront[np.where(np.abs(x)>(aperture_diameter/2))] = 0.0

    # plot_intensity(wavefront,1e6*x,
    #                xlabel="x [um]",ylabel="source intensity [arbitrary units]",title="incident wavefront")


    mymethods   = ["integral","zoom", "fft",  "fft_conv", "exact_prop_numba","propTF", "exact_prop"]
    mylibraries = ["wofry",   "wofry","wofry","wofry",    "sajid",           "sajid",  "sajid" ]
    mycolors    = ["r",       "r",    "r",    "r",        "b",               "b",      "b"]

    myresults = np.zeros(len(mymethods))
    for i in range(len(mymethods)):
        wavefront_propagated, x_propagated, ti = propagate_with_any_library(wavefront, x, wavelength, propagation_distance,
                                                                    library=mylibraries[i], method=mymethods[i],
                                                                    magnification_x=magnification_x,
                                                                    repeat=repeat)
        print("library: %s, method: %s, time: %f"%(mylibraries[i],mymethods[i],ti))
        myresults[i] = ti


    plot_bar(mymethods,myresults,ylog=True,ytitle="time [s] for N=%d (%d runs)"%(npoints,repeat),
             colors=mycolors,filename="aperture_1D_benchmarking.png")


