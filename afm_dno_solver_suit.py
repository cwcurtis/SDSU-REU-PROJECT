import pyfftw
import numpy as np
import my_fft

def dno_maker(KT, eta, qx, G0, ep, mu, mDk, tnh, Mval, ftrans):

    # KT is number of modes used in pseudo - spectral scheme
    # eta is surface height in physical space
    # q is surface potential in physical space
    # G0 is first term of DNO in physical space
    # ep is epsilon
    # mu is mu
    # Kmesh = pi / Llx * [0:K - K + 1:-1], or is essentially a derivative
    # Mval + 1 is number of terms used in DNO expansion

    phis = np.zeros([KT, Mval + 1],dtype='complex128')

    dnohot = pyfftw.empty_aligned(KT, dtype='complex128')
    dnohot[:] = np.zeros(KT,dtype='complex128')

    phis[:,0] = G0
    epp = 1

    for kk in xrange(2,Mval + 2):

        phic = np.zeros(KT,dtype='complex128')
        Dkp = np.ones(KT,dtype='complex128')
        etap = np.ones(KT,dtype='complex128')

        for ll in xrange(1,kk-1):
            Dkp = mDk*Dkp
            etap = eta*etap/ll
            if np.mod(ll, 2) == 0:
                tvec = Dkp*ftrans.fft(etap*phis[:, kk-ll-1])
            else:
                tvec = Dkp*tnh*ftrans.fft(etap*phis[:, kk-ll-1])
            phic += tvec

        Dkp *= mDk
        etap *= eta/(kk - 1)

        if np.mod(kk, 2) == 0:
            fvec = Dkp*(tnh*ftrans.fft(etap*G0) + 1j/mu*ftrans.fft(etap*qx))
        else:
            fvec = Dkp*(ftrans.fft(etap*G0) + 1j/mu*tnh*ftrans.fft(etap*qx))

        phic = -ftrans.ifft(phic + fvec).real
        phis[:,kk-1] = phic
        epp *= ep
        dnohot[:] += epp*phic

    return dnohot

def nonlinearity(K,eta,q,G0,ep,mu,Kmesh,mDk,tnh,Mval,ftrans):

    KT = 2*K
    # Find the wave numbers to implement the 2/3 de-aliasing throughout
    Kc = int(np.floor(2.*K/3.))
    Kuc = KT-Kc+1
    Kc = Kc+1

    eta[Kc-1:Kuc]=0
    q[Kc-1:Kuc]=0

    etax = pyfftw.empty_aligned(KT, dtype='complex128')
    qx = pyfftw.empty_aligned(KT, dtype='complex128')
    etap = pyfftw.empty_aligned(KT, dtype='complex128')
    dnohot = pyfftw.empty_aligned(KT, dtype='complex128')

    rhs1 = pyfftw.empty_aligned(KT, dtype='complex128')
    rhs2 = pyfftw.empty_aligned(KT, dtype='complex128')
    rhs = pyfftw.empty_aligned(2*KT, dtype='complex128')

    denom = pyfftw.empty_aligned(KT, dtype='complex128')
    numer = pyfftw.empty_aligned(KT, dtype='complex128')

    etax[:] = ftrans.ifft(1j*Kmesh*eta).real
    qx[:] = ftrans.ifft(1j*Kmesh*q).real

    etap[:] = ftrans.ifft(eta).real

    dnohot[:] = dno_maker(KT,etap,qx,G0,ep,mu,mDk,tnh,Mval,ftrans)

    rhs1[:] = ftrans.fft(dnohot)

    numer[:] = (G0+dnohot+ep*etax*qx)**2
    denom[:] = np.ones(KT) + (ep*mu*etax)**2
    rhs2[:] = .5*ep*ftrans.fft(-qx**2 + mu**2*numer/denom)

    rhs1[Kc-1:Kuc] = 0
    rhs2[Kc-1:Kuc] = 0

    rhs[:] = np.concatenate((rhs1,rhs2),0)
    return rhs

def afm_dno_solver(K, ep, mu, Llx, tf, Mval, dt):

    KT = 2 * K
    # Find the wave numbers to implement the 2/3 de-aliasing throughout
    Kc = int(np.floor(2. * K / 3.))
    Kuc = KT - Kc + 1
    Kc = Kc + 1

    ftrans = my_fft.my_fft(KT)

    eta0 = pyfftw.empty_aligned(KT, dtype='complex128')
    q0 = pyfftw.empty_aligned(KT, dtype='complex128')

    eta1 = pyfftw.empty_aligned(KT, dtype='complex128')
    eta2 = pyfftw.empty_aligned(KT, dtype='complex128')

    q1 = pyfftw.empty_aligned(KT, dtype='complex128')
    q2 = pyfftw.empty_aligned(KT, dtype='complex128')

    etan = pyfftw.empty_aligned(KT, dtype='complex128')
    qn = pyfftw.empty_aligned(KT, dtype='complex128')

    G0 = pyfftw.empty_aligned(KT, dtype='complex128')

    etanm1 = pyfftw.empty_aligned(KT, dtype='complex128')
    qnm1 = pyfftw.empty_aligned(KT, dtype='complex128')

    etanp1 = pyfftw.empty_aligned(KT, dtype='complex128')
    qnp1 = pyfftw.empty_aligned(KT, dtype='complex128')

    nlvecn = pyfftw.empty_aligned(KT, dtype='complex128')
    nlvecq = pyfftw.empty_aligned(KT, dtype='complex128')

    nln = pyfftw.empty_aligned(2 * KT, dtype='complex128')
    nlnm1 = pyfftw.empty_aligned(2 * KT, dtype='complex128')
    nlnm2 = pyfftw.empty_aligned(2 * KT, dtype='complex128')
    nlnm3 = pyfftw.empty_aligned(2 * KT, dtype='complex128')

    Xmesh = pyfftw.empty_aligned(KT, dtype='complex128')
    Kmesh = pyfftw.empty_aligned(KT, dtype='complex128')
    tnh = pyfftw.empty_aligned(KT, dtype='complex128')
    mDk = pyfftw.empty_aligned(KT, dtype='complex128')
    L1 = pyfftw.empty_aligned(KT, dtype='complex128')
    Linvd = pyfftw.empty_aligned(KT, dtype='complex128')
    Linv12 = pyfftw.empty_aligned(KT, dtype='complex128')
    Linv21 = pyfftw.empty_aligned(KT, dtype='complex128')

    dx = 2. * Llx / KT

    Xmesh[:] = np.arange(-Llx, Llx, dx)

    Kmesh[:] = np.pi / Llx * np.concatenate((np.arange(0, K + 1), np.arange(-K + 1, 0)), 0)
    nmax = int(np.round(tf / dt))

    mDk[:] = mu * Kmesh
    tnh[:] = np.tanh(mDk)

    L1[:] = Kmesh*np.tanh(mu*Kmesh)/mu

    Linvd[:] = (np.ones(KT,dtype='complex128') + 9. * dt**2 / 16. * L1)**(-1)
    Linv12[:] = 3. * dt / 4. * L1*Linvd
    Linv21[:] = -3. * dt / 4. * Linvd

    eta0[:] = np.cos(np.pi * Xmesh / Llx)
    q0[:] = np.sin(np.pi * Xmesh / Llx)

    etan[:] = ftrans.fft(eta0)
    qn[:] = ftrans.fft(q0)

    etan[Kc-1:Kuc] = 0
    qn[Kc-1:Kuc] = 0
    G0[:] = ftrans.ifft(L1*qn).real

    etanm1[:] = etan
    qnm1[:] = qn

    nln[:] = nonlinearity(K, etan, qn, G0, ep, mu, Kmesh, mDk, tnh, Mval, ftrans)
    nlnm1[:] = nln
    nlnm2[:] = nlnm1
    nlnm3[:] = nlnm2

    for jj in xrange(nmax):

        G0[:] = ftrans.ifft(L1*qn).real
        nln[:] = nonlinearity(K, etan, qn, G0, ep, mu, Kmesh, mDk, tnh, Mval, ftrans)

        nlvecn[:] = 55./24.*nln[0:KT] - 59./24.*nlnm1[0:KT] + 37./24.*nlnm2[0:KT] - 3./8.*nlnm3[0:KT]
        nlvecq[:] = 55./24.*nln[KT:2*KT] - 59./24.*nlnm1[KT:2*KT] + 37./24.*nlnm2[KT:2*KT] - 3./8.*nlnm3[KT:2*KT]

        eta1[:] = Linvd*(etan + etanm1/3. + dt * nlvecn)
        eta2[:] = Linv12*(qn + qnm1/3. + dt * nlvecq)

        q1[:] = Linvd*(qn + qnm1/3. + dt * nlvecq)
        q2[:] = Linv21*(etan + etanm1/3. + dt * nlvecn)

        etanp1[:] = -etanm1/3. + eta1 + eta2
        qnp1[:] = -qnm1/3. + q1 + q2

        etanm1[:] = etan
        etan[:] = etanp1

        qnm1[:] = qn
        qn[:] = qnp1

        nlnm3[:] = nlnm2
        nlnm2[:] = nlnm1
        nlnm1[:] = nln

    return ftrans.ifft(etan).real