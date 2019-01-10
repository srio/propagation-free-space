def run_wofry():
    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D

    #
    # create input_wavefront
    #
    #
    from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
    input_wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=-0.000050, x_max=0.000050,
                                                                         y_min=-0.000050, y_max=0.000050,
                                                                         number_of_points=(2048, 2048))
    input_wavefront.set_photon_energy(10000)
    input_wavefront.set_spherical_wave(radius=-0.1, complex_amplitude=complex(1, 0))

    #
    # info on current oe
    #
    #
    #    -------WOScreen---------
    #        -------BoundaryShape---------
    #

    #
    # define current oe
    #
    from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen

    optical_element = WOScreen()

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,
                                       coordinates=ElementCoordinates(p=0.000000, q=0.100000,
                                                                      angle_radial=numpy.radians(0.000000),
                                                                      angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),
                                                   propagation_elements=propagation_elements)
    # self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 0.05)
    propagation_parameters.set_additional_parameters('magnification_y', 0.05)

    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,
                                                 handler_name='FRESNEL_ZOOM_XY_2D')

    return input_wavefront,output_wavefront

def run_srw(w0,Rx=-0.1,dRx=0.001,Ry=-0.1,dRy=0.001):
    from orangecontrib.srw.util.srw_objects import SRWData
    from wofrysrw.propagator.wavefront2D.srw_wavefront import SRWWavefront
    wSRW = SRWData(srw_wavefront=SRWWavefront.fromGenericWavefront(w0))
    wsrw = wSRW.get_srw_wavefront()
    wsrw.Rx = Rx
    wsrw.dRx = dRx
    wsrw.Ry = Ry
    wsrw.dRy = dRy
    w1 = run_srw_native(wsrw)
    return w1.toGenericWavefront()

from srwlib import *
from uti_plot import *
def run_srw_native(wfr,do_plot=False):
    ####################################################
    # BEAMLINE

    srw_oe_array = []
    srw_pp_array = []

    drift_after_oe_0 = SRWLOptD(0.1)
    pp_drift_after_oe_0 = [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0.0, 0.0]

    srw_oe_array.append(drift_after_oe_0)
    srw_pp_array.append(pp_drift_after_oe_0)

    ####################################################
    # PROPAGATION

    optBL = SRWLOptC(srw_oe_array, srw_pp_array)
    srwl.PropagElecField(wfr, optBL)

    if do_plot:
        mesh1 = deepcopy(wfr.mesh)
        arI1 = array('f', [0] * mesh1.nx * mesh1.ny)
        srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, mesh1.eStart, 0, 0)
        arI1x = array('f', [0] * mesh1.nx)
        srwl.CalcIntFromElecField(arI1x, wfr, 6, 0, 1, mesh1.eStart, 0, 0)
        arI1y = array('f', [0] * mesh1.ny)
        srwl.CalcIntFromElecField(arI1y, wfr, 6, 0, 2, mesh1.eStart, 0, 0)
        # save ascii file with intensity
        # srwl_uti_save_intens_ascii(arI1, mesh1, <file_path>)
        plotMesh1x = [1000 * mesh1.xStart, 1000 * mesh1.xFin, mesh1.nx]
        plotMesh1y = [1000 * mesh1.yStart, 1000 * mesh1.yFin, mesh1.ny]
        uti_plot2d1d(arI1, plotMesh1x, plotMesh1y,
                     labels=['Horizontal Position [mm]', 'Vertical Position [mm]', 'Intensity After Propagation'])
        uti_plot_show()

    return wfr

if __name__ == "__main__":

    from srxraylib.plot.gol import plot_image, plot
    import matplotlib.pylab as plt

    dumpfile="aperture_2D_converging.png"
    w0, w = run_wofry()


    # plot_image(w.get_intensity(),1e6*w.get_coordinate_x(),1e6*w.get_coordinate_y())
    # plot(1e6*w.get_coordinate_x(),w.get_intensity()[:,w.get_coordinate_y().size//2],
    #      1e6*w.get_coordinate_y(),w.get_intensity()[w.get_coordinate_x().size // 2,:],)

    w1 = run_srw(w0)

    # plot_image(w1.get_intensity(),1e6*w1.get_coordinate_x(),1e6*w1.get_coordinate_y(),title="SRW")
    plot(1e6*w1.get_coordinate_x(),w1.get_intensity()[:,w1.get_coordinate_y().size//2],
         1e6 * w.get_coordinate_x(), w.get_intensity()[:, w.get_coordinate_y().size // 2],
         legend=["SRW","WOFRY"],xtitle="x[um]",ytitle="Intensity [a.u.]",show=False,ylog=False)

    if dumpfile is not None:
        plt.savefig(dumpfile)
        print("File written to disk: %s"%dumpfile)
    plt.show()


    print("Size SRW,WOFRY: ",w1.get_coordinate_x().size,w.get_coordinate_x().size)