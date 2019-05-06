import numpy
import scipy.constants as codata
# from scipy import integrate as integrate

from srxraylib.plot.gol import plot
import matplotlib.pylab as plt

# ========================== imports from WOFRY:
def spherical_wave(x, k, R, x0 = 0, A = 1):

    return A * numpy.exp(-1.0j * k *((x - x0) ** 2) / (-2 * R))

def get_amplitude(wfr):

    return numpy.absolute(wfr)

def get_phase(wfr,from_minimum_intensity=0.0,unwrap=0):

    phase = numpy.angle(wfr)
    if (from_minimum_intensity > 0.0):
        intensity = get_intensity(wfr)
        intensity /= intensity.max()
        bad_indices = numpy.where(intensity < from_minimum_intensity )
        phase[bad_indices] = 0.0
    if unwrap:
        phase = numpy.unwrap(phase)
    return phase

def get_intensity(wfr):

    return get_amplitude(wfr)**2

def fresnel(wavefront,k,x,z,R): #RC20190128: implemented as in Wofry!

    wavefront_delta = x[1] - x[0]
    wavefront_size = wavefront.size
    wavefront_wavelength = 2 * numpy.pi / k

    propagation_distance = z

    npixels = wavefront_size
    freq_nyquist = 0.5 / wavefront_delta

    if numpy.mod(npixels, 2) == 0:
        freq_n = numpy.arange(-npixels // 2, npixels // 2, 1) / (npixels // 2)
    else:
        freq_n = numpy.arange(-(npixels - 1) // 2, (npixels + 1) // 2, 1) / ((npixels - 1) // 2)
    fft_scale = freq_n * freq_nyquist

    fft = numpy.fft.fft(wavefront)
    fft *= numpy.exp((-1.0j) * numpy.pi * wavefront_wavelength * propagation_distance * numpy.fft.fftshift(fft_scale)**2)

    return numpy.fft.ifft(fft), x

# ========================== imports from Semianalytical

def semi_analytical_integral(wavefront, k, x, z, R, method = 0):
    E2 = wavefront * 0
    wavefront_delta = x[1] - x[0]
    quadratic_term = numpy.exp((-1j*k*x**2)/(2*R))
    F1 = numpy.multiply(wavefront,quadratic_term)

    if method == 0: # Reimman summation
        for i in range(E2.size):
            x2 = x[i]
            E2[i] = numpy.sum(F1*numpy.exp(1j*k*(R+z*(x - R*x2/(R+z))**2)/(2*R*z))) * wavefront_delta

    global_phase = k*numpy.exp(0.5*(1j*k*x**2)/(R+z))/(1j*2*numpy.pi*z)

    return numpy.multiply(global_phase,E2), x


def fresnel_srw(wavefront,k,x,z):
    wavefront_wavelength = 2 * numpy.pi / k
    wavefront_delta = x[1] - x[0]
    wavefront_size = wavefront.size
    propagation_distance = z

    npixels = wavefront_size
    freq_nyquist = 0.5 / wavefront_delta
    if numpy.mod(npixels, 2) == 0:
        freq_n = numpy.arange(-npixels // 2, npixels // 2, 1) / (npixels // 2)
    else:
        freq_n = numpy.arange(-(npixels - 1) // 2, (npixels + 1) // 2, 1) / ((npixels - 1) // 2)

    theta = freq_n * freq_nyquist * wavefront_wavelength

    fft_E1 = numpy.fft.fft(wavefront)
    K = numpy.exp(-0.5j * k * propagation_distance * numpy.fft.fftshift(theta)**2)
    C = numpy.exp(1j * k * propagation_distance) / (1j * propagation_distance * wavefront_wavelength)
    E2 = C * numpy.fft.ifft(fft_E1 * K)

    return E2, x


def semi_analytical(wavefront,k,x,z,R):
    ''' This semi-analytical implementation is based on change of variables - beta and alpha'''
    quadratic_term =  1.0 # numpy.exp((-1j*k*x**2)/(2*R))
    F1 = numpy.multiply(wavefront,quadratic_term)

    wavefront_wavelength = 2 * numpy.pi / k
    wavefront_delta = x[1] - x[0]
    wavefront_size = wavefront.size
    propagation_distance = z

    npixels = wavefront_size
    freq_nyquist = 0.5 / wavefront_delta
    if numpy.mod(npixels, 2) == 0:
        freq_n = numpy.arange(-npixels // 2, npixels // 2, 1) / (npixels // 2)
    else:
        freq_n = numpy.arange(-(npixels - 1) // 2, (npixels + 1) // 2, 1) / ((npixels - 1) // 2)

    theta = freq_n * freq_nyquist * wavefront_wavelength

    fft_F1 = numpy.fft.fft(F1)
    G = 1j * propagation_distance * wavefront_wavelength * (R / (R + propagation_distance)) * \
        numpy.exp(-0.5j * k * propagation_distance * numpy.fft.fftshift(theta)**2)
    C = numpy.exp(1j * k * propagation_distance) / (1j * propagation_distance * wavefront_wavelength**3)
    F2 = C * numpy.fft.ifft(fft_F1 * G)

    return F2, x

if __name__ == "__main__":
    # wavefront input parameters
    npixels= 2048*4
    D = 100e-6
    radius = -1e-1
    photon_energy = 10000.0
    z = -radius * 0.5
    defocus = 1 - .1e-3 # to avoid singularities when R = -z

    # ========================== Wavefront generation

    wavelength = ( codata.h*codata.c/codata.e*1e9 /photon_energy)*10**(-9)
    x = numpy.linspace(-0.5*D,0.5*D,npixels)

    wfr = spherical_wave(x, 2*numpy.pi/wavelength,radius*1e0)

    # ========================== Wave propagation to z
    #TODO normalise intensity output

    # propagated_wft_fresnel, propagated_x = fresnel(wfr, 2*numpy.pi/wavelength,x, z, radius*defocus)
    propagated_wft_fresnel_srw, propagated_x = fresnel_srw(wfr, 2 * numpy.pi / wavelength, x, z)
    propagated_wft_semi_analytical, propagated_x  = semi_analytical(wfr, 2*numpy.pi/wavelength,x, z, radius*defocus)
    # propagated_wft_integral, propagated_x = semi_analytical_integral(wfr, 2 * numpy.pi / wavelength, x, z, radius *defocus)

    title = 'Semi analytical treatment of the quadratic phase term '
    plot(propagated_x*1E6,get_phase(propagated_wft_semi_analytical, 0, 1),xtitle="x [um]",ytitle="Phase[ f(x) ]",title=title,show=False)
    plot(propagated_x*1E6,get_intensity(propagated_wft_semi_analytical),xtitle="x [um]",ytitle="Intensity[ f(x) ]",title=title,show=False)

    # title = 'Integral formulation of the semi-analytical'
    # plot(propagated_x*1E6,get_phase(propagated_wft_integral, 0, 1),xtitle="x [um]",ytitle="Phase[ f(x) ]",title=title,show=False)
    # plot(propagated_x*1E6,get_intensity(propagated_wft_integral),xtitle="x [um]",ytitle="Intensity[ f(x) ]",title=title,show=False)

    # title = 'fresnel'
    # plot(propagated_x*1E6,get_phase(propagated_wft_fresnel, 0, 1),xtitle="x [um]",ytitle="Phase[ f(x) ]",title=title,show=False)
    # plot(propagated_x*1E6,get_intensity(propagated_wft_fresnel),xtitle="x [um]",ytitle="Intensity[ f(x) ]",title=title,show=False)

    title = 'fresnel_srw'
    plot(propagated_x*1E6,get_phase(propagated_wft_fresnel_srw, 0, 1),xtitle="x [um]",ytitle="Phase[ f(x) ]",title=title,show=False)
    plot(propagated_x*1E6,get_intensity(propagated_wft_fresnel_srw),xtitle="x [um]",ytitle="Intensity[ f(x) ]",title=title,show=False)


    plt.show()








    # N = 10
    # x = numpy.linspace(0.0,1.0,N)
    # xx = numpy.linspace(-0.01,1.01,5000)
    # delta = x[1] - x[0]
    #
    # A = 20
    # y20 = quadratic(x,A)
    # title = "A=%2.1f <? %2.1f"%(A,numpy.pi/N/delta**2)
    # plot(xx,quadratic(xx,A),
    #      x,y20,
    #      xtitle="x",ytitle="Real[ f(x) ]",title=title,show=False)
    #
    # plt.savefig("sample_quadratic_phase_A20.png")
    # print("File written to disk: sample_quadratic_phase_A20.png")
    # plt.show()

    # A = 30
    # y30 = quadratic(x,A)
    # title = "A=%2.1f <? %2.1f"%(A,numpy.pi/N/delta**2)
    # plot(xx,quadratic(xx,A),
    #      x,y30,
    #      xtitle="x",ytitle="Real[ f(x) ]",title=title,show=False)
    # plt.savefig("sample_quadratic_phase_A30.png")
    # print("File written to disk: sample_quadratic_phase_A30.png")
    # plt.show()
