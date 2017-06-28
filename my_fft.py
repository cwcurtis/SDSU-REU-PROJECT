import pyfftw

class my_fft():

    def __init__(self,KT):
        physv = pyfftw.empty_aligned(KT, dtype = 'complex128')
        freqv = pyfftw.empty_aligned(KT, dtype = 'complex128')

        fft_f = pyfftw.FFTW(physv, freqv)
        fft_in = pyfftw.FFTW(freqv, physv, direction='FFTW_BACKWARD')

        self.physv = physv
        self.freqv = freqv
        self.fft = fft_f
        self.ifft = fft_in
        self.length = KT

    def fft(self,fx):
        fk = pyfftw.empty_aligned(self.length, dtype = 'complex128')
        self.physv[:] = fx
        self.fft_f()
        fk[:] = self.freqv
        return fk

    def ifft(self,fk):
        fx = pyfftw.empty_aligned(self.length, dtype = 'complex128')
        self.freqv[:] = fk
        self.fft_in()
        fx[:] = self.physv
        return fx
