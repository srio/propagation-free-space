


def get_generic_wavefront_after_aperture(is_circular=False,npixels=2048):

    #
    # create input_wavefront
    #
    #
    from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
    input_wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=(-2.5e-06),
                                                                         x_max=( 2.5e-06),
                                                                         y_min=(-2.5e-06),
                                                                         y_max=( 2.5e-06),
                                                                         number_of_points=(npixels,npixels))
                                                                         # number_of_points=(RESOLUTION_FACTOR*2048,RESOLUTION_FACTOR*2048))
    input_wavefront.set_photon_energy(10000)
    input_wavefront.set_plane_wave_from_complex_amplitude(complex_amplitude=complex(1,0))
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
    # info on current oe
    #
    #
    #    -------WOSlit---------
    #        -------Rectangle---------
    #        x_left: -6.25e-07 m # x (width) minimum (signed)
    #        x_right: 6.25e-07 m # x (width) maximum (signed)
    #        y_bottom: -6.25e-07 m # y (length) minimum (signed)
    #        y_top: 6.25e-07 m # y (length) maximum (signed)
    #

    #
    # define current oe
    #
    from syned.beamline.shape import Rectangle, Circle

    if is_circular:
        boundary_shape = Circle( 6.25e-07,
                         x_center=0,
                         y_center=0)
    else:
        boundary_shape = Rectangle(x_left  =-6.25e-07,
                                   x_right = 6.25e-07,
                                   y_bottom=-6.25e-07,
                                   y_top   = 6.25e-07
        )

    from wofry.beamline.optical_elements.absorbers.slit import WOSlit
    optical_element = WOSlit(boundary_shape=boundary_shape)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,    coordinates=ElementCoordinates(p=0.000000,    q=0.000000,    angle_radial=numpy.radians(0.000000),    angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),    propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 1.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 1.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront


def run_wofry(input_wavefront):

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
    beamline_element = BeamlineElement(optical_element=optical_element,    coordinates=ElementCoordinates(p=0.000075,    q=0.000000,    angle_radial=numpy.radians(0.000000),    angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),    propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 1.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 1.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront

def run_srw(w0,Rx=None,dRx=None,Ry=None,dRy=None):
    from orangecontrib.srw.util.srw_objects import SRWData
    from wofrysrw.propagator.wavefront2D.srw_wavefront import SRWWavefront
    wSRW = SRWData(srw_wavefront=SRWWavefront.fromGenericWavefront(w0))
    wsrw = wSRW.get_srw_wavefront()
    if  Rx is not None: wsrw.Rx = Rx
    if dRx is not None: wsrw.dRx = dRx
    if  Ry is not None: wsrw.Ry = Ry
    if dRy is not None: wsrw.dRy = dRy
    w1 = run_srw_native(wsrw)
    return w1.toGenericWavefront()

from srwlib import *
from uti_plot import *
def run_srw_native(wfr,do_plot=False):

    ####################################################
    # BEAMLINE

    srw_oe_array = []
    srw_pp_array = []

    oe_0=SRWLOptL(_Fx=0.05, _Fy=0.05, _x=0.0, _y=0.0)

    pp_oe_0 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

    srw_oe_array.append(oe_0)
    srw_pp_array.append(pp_oe_0)

    drift_before_oe_1 = SRWLOptD(7.5e-05)
    pp_drift_before_oe_1 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

    srw_oe_array.append(drift_before_oe_1)
    srw_pp_array.append(pp_drift_before_oe_1)



    ####################################################
    # PROPAGATION

    optBL = SRWLOptC(srw_oe_array, srw_pp_array)
    srwl.PropagElecField(wfr, optBL)

    if do_plot:
        mesh1 = deepcopy(wfr.mesh)
        arI1 = array('f', [0]*mesh1.nx*mesh1.ny)
        srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, mesh1.eStart, 0, 0)
        arI1x = array('f', [0]*mesh1.nx)
        srwl.CalcIntFromElecField(arI1x, wfr, 6, 0, 1, mesh1.eStart, 0, 0)
        arI1y = array('f', [0]*mesh1.ny)
        srwl.CalcIntFromElecField(arI1y, wfr, 6, 0, 2, mesh1.eStart, 0, 0)
        #save ascii file with intensity
        #srwl_uti_save_intens_ascii(arI1, mesh1, <file_path>)
        plotMesh1x = [1000*mesh1.xStart, 1000*mesh1.xFin, mesh1.nx]
        plotMesh1y = [1000*mesh1.yStart, 1000*mesh1.yFin, mesh1.ny]
        uti_plot2d1d(arI1, plotMesh1x, plotMesh1y, labels=['Horizontal Position [mm]', 'Vertical Position [mm]', 'Intensity After Propagation'])
        uti_plot_show()

    return wfr

#
#
#
#
#
#
if __name__ == "__main__":

    import numpy
    from srxraylib.plot.gol import plot_image, plot
    import matplotlib.pylab as plt


    is_circular = True
    save_png = True

    #
    # source
    #

    w0 = get_generic_wavefront_after_aperture(is_circular=is_circular, npixels=2048 * 2)

    if True:
        if is_circular:
            w0.save_h5_file("aperture_2D_circular.h5")
            print("File written to disk: aperture_2D_circular.h5")
        else:
            w0.save_h5_file("aperture_2D_rectangular.h5")
            print("File written to disk: aperture_2D_rectangular.h5")


    #
    # wofry
    #
    if True:



        # plot_image(w0.get_intensity(),1e6*w0.get_coordinate_x(),1e6*w0.get_coordinate_y())


        w = run_wofry(w0)
        #
        plot_image(w.get_intensity(),1e6*w.get_coordinate_x(),1e6*w.get_coordinate_y(),title="WOFRY")

        # plot(1e6*w.get_coordinate_x(),w.get_intensity()[:,w.get_coordinate_y().size//2],
        #      1e6*w.get_coordinate_y(),w.get_intensity()[w.get_coordinate_x().size // 2,:],)


    #
    # srw
    #

    if True:
        w1 = run_srw(w0)
        #
        plot_image(w1.get_intensity(),1e6*w1.get_coordinate_x(),1e6*w1.get_coordinate_y(),
                   xtitle="x [um]",ytitle="y [um]",title="",show=False)

        if save_png:
            if is_circular:
                plt.savefig("aperture_2D_circular_srw.png")
                print("File written to disk: aperture_2D_circular_srw.png")
            else:
                plt.savefig("aperture_2D_rectangular_srw.png")
                print("File written to disk: aperture_2D_rectangular_srw.png")
        plt.show()
        #
        plot(1e6*w1.get_coordinate_x(),w1.get_intensity()[:,w1.get_coordinate_y().size//2],
             xtitle="x [um]",ytitle="Intensity [a.u]",show=False)

        if save_png:
            if is_circular:
                plt.savefig("aperture_2D_circular_profile_srw.png")
                print("File written to disk: aperture_2D_circular_profile_srw.png")
            else:
                plt.savefig("aperture_2D_rectangular_profile_srw.png")
                print("File written to disk: aperture_2D_profile_rectangular_profile_srw.png")
        plt.show()

    #
    # calculate analytical
    #
    import scipy.constants as codata
    from aperture_1D_analytical import fresnel_analytical_circle, fresnel_analytical_rectangle
    # energy = 10000.0
    wavelength = w0.get_wavelength() # ( codata.h*codata.c/codata.e*1e9 /energy)*10**(-9)
    window_size = 5e-6
    aperture_diameter = 1.25e-06
    propagation_distance = 75e-6

    #
    # info
    #
    print("Size: ",w0.get_coordinate_x().size)
    print("Dz <? N d / lambda; %g < %g"%(propagation_distance, w.get_coordinate_x().size * \
          (w.get_coordinate_x()[1]-w.get_coordinate_x()[0])**2/wavelength))
    print("Fresnel number: xmaxD / (lambda Dz) = %f "%(w.get_coordinate_x()[-1] / \
          (propagation_distance * wavelength)))

    # inputs end

    # I reduce the interval
    detector_array = w.get_coordinate_x() # numpy.linspace(w.get_coordinate_x()[0]/20,w.get_coordinate_x()[-1]/20,npoints)

    if is_circular:
        r, alpha = fresnel_analytical_circle(propagation_distance=propagation_distance,
                        aperture_half=aperture_diameter/2,
                        wavelength=wavelength,
                        detector_array=detector_array,npoints=detector_array.size,)

        pattern = numpy.abs(alpha)**2
        x = r
        # source = r * 0.0 + 1.0
        # source[numpy.where(numpy.abs(r)>(aperture_diameter/2))] = 0.0
        #
        # integral_source = source.sum()
        # integral_pattern = pattern.sum()
        # print(integral_source,integral_pattern)
        #
        # plot(r*1e6,pattern,r*1e6,source*integral_pattern/integral_source,xtitle="x [um]")
    else:
        x, alpha = fresnel_analytical_rectangle(fresnel_number=None, propagation_distance=propagation_distance,
                        aperture_half=aperture_diameter/2,
                        wavelength=wavelength,
                        detector_array=detector_array,npoints=detector_array.size,)

        pattern = numpy.abs(alpha)**2

        # source = detector_array * 0.0 + 1.0
        # source[numpy.where(numpy.abs(detector_array)>(aperture_diameter/2))] = 0.0
        #
        # integral_source = source.sum()
        # integral_pattern = pattern.sum()
        # print(integral_source,integral_pattern)
        #
        # plot(x*1e6,pattern,x*1e6,source*integral_pattern/integral_source,xtitle="x [um]",title="analytical")


    #
    # plot all results
    #

    if is_circular:
        factor = 1.0
        yrange=None
    else:
        factor = pattern[pattern.size//2]  # note the factor to account for the y value!!
        yrange=[0.845,1.055]


    if False:
        plot(
             # 1e6*w1.get_coordinate_x(),w1.get_intensity()[:,w1.get_coordinate_y().size//2],
             # 1e6 * w.get_coordinate_x(), w.get_intensity()[:, w.get_coordinate_y().size // 2],
             x*1e6,pattern*factor,
             legend=["analytical"],xtitle="x[um]",ytitle="Intensity [a.u.]",
             # legend=["SRW","WOFRY","analytical"],xtitle="x[um]",ytitle="Intensity [a.u.]",
             xrange=[-0.10,0.10],yrange=yrange,
             show=False,ylog=False)

    if True:
        if is_circular:
            filename = "aperture2D_circular_pynx.dat"
        else:
            filename = "aperture2D_rectangular_pynx.dat"
        pynx_data = numpy.loadtxt(filename)
        print(">>>",pynx_data.shape)

        plot(
             1e6*w1.get_coordinate_x(),w1.get_intensity()[:,w1.get_coordinate_y().size//2],
             1e6*w.get_coordinate_x(),w.get_intensity()[:,w.get_coordinate_y().size//2],
             1e6*pynx_data[:,0],pynx_data[:,1],
             x*1e6,pattern*factor,
             legend=["SRW (Standard)","WOFRY (fresnel)","PYNX (NearField)","analytical"],
             # legend=["WOFRY (fresnel)", "PYNX (NearField)", "analytical"],
             xtitle="x[um]",ytitle="Intensity [a.u.]",
             xrange=[-0.10, 0.10], yrange=yrange, #xrange=[-2.10,2.10],yrange=[0,3],
             show=False,ylog=False)


    if save_png:
        if is_circular:
            plt.savefig("aperture_2D_circular_profile_comparison.png")
            print("File written to disk: aperture_2D_circular_profile_comparison.png.png")
        else:
            plt.savefig("aperture_2D_rectangular_profile_comparison.png")
            print("File written to disk: aperture_2D_rectangular_profile_comparison.png")
    plt.show()

    #
    #
