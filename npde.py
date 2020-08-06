import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from integrators import ODERK4, SDEEM
from kernels import OperatorKernel

from gpflow import transforms
from param import Param

float_type = tf.float64
jitter0 = 1e-6


class NPODE:
    def __init__(self,Z0,U0,sn0,kern,jitter=jitter0,
                 summ=False,whiten=True,fix_Z=False,fix_U=False,fix_sn=False):
        """ Constructor for the NPODE model
        
        Args:
            Z0: Numpy matrix of initial inducing points of size MxD, M being the
                number of inducing points.
            U0: Numpy matrix of initial inducing vectors of size MxD, M being the
                number of inducing points.
            sn0: Numpy vector of size 1xD for initial signal variance
            kern: Kernel object for GP interpolation
            jitter: Float of jitter level
            whiten: Boolean. Currently we perform the optimization only in the 
                white domain
            summ: Boolean for Tensorflow summary
            fix_Z: Boolean - whether inducing locations are fixed or optimized
            fix_U: Boolean - whether inducing vectors are fixed or optimized
            fix_sn: Boolean - whether noise variance is fixed or optimized
        """
        self.name = 'npode'
        self.whiten = whiten
        self.kern = kern
        self.jitter = jitter
        with tf.name_scope("NPDE"):
            Z = Param(Z0,
                       name = "Z",
                       summ = False,
                       fixed = fix_Z)
            U = Param(U0,
                        name = "U",
                        summ = False,
                        fixed = fix_U)
                            
            sn = Param(np.array(sn0),
                       name = "sn",
                       summ = summ,
                       fixed = fix_sn,
                       transform = transforms.Log1pe())
        self.Z  = Z()
        self.U  = U()
        self.sn = sn()
        self.D  = U.shape[1]
        self.integrator = ODERK4(self)
        self.fix_Z = fix_Z
        self.fix_sn = fix_sn
        self.fix_U = fix_U

    def f(self,X,t=[0]):
        """ Implements GP interpolation to compute the value of the differential
        function at location(s) X.
        Args:
            X: TxD tensor of input locations, T is the number of locations.
        Returns:
            TxD tensor of differential function (GP conditional) computed on 
            input locations
        """
        U = self.U
        Z = self.Z
        kern = self.kern

        N = tf.shape(X)[0]
        M = tf.shape(Z)[0]
        D = tf.shape(Z)[1] # dim of state

        if kern.ktype == "id":
            Kzz = kern.K(Z) + tf.eye(M, dtype=float_type) * self.jitter
        else:
            Kzz = kern.K(Z) + tf.eye(M*D, dtype=float_type) * self.jitter
        Lz = tf.cholesky(Kzz)

        Kzx = kern.K(Z, X)

        A = tf.matrix_triangular_solve(Lz, Kzx, lower=True)

        if not self.whiten:
            A = tf.matrix_triangular_solve(tf.transpose(Lz), A, lower=False)

        f = tf.matmul(A, U, transpose_a=True)

        # transformation for "id - rbf" kernel
        if not kern.ktype == "id" and not kern.ktype == "kr" :
            f = tf.reshape(f,[N,D])

        return f

    def build_prior(self):
        if self.kern.ktype == "id" or self.kern.ktype == "kr":
            if self.whiten:
                mvn = tfd.MultivariateNormalDiag(
                            loc=tf.zeros_like(self.U[:,0]))
            else:
                mvn = tfd.MultivariateNormalFullCovariance(
                            loc=tf.zeros_like(self.U[:,0]),
                            covariance_matrix=self.kern.K(self.Z,self.Z))

            probs = tf.add_n([mvn.log_prob(self.U[:,d]) for d in range(self.kern.ndims)])

        else:
            if self.whiten:
                mvn = tfd.MultivariateNormalDiag(
                            loc=tf.zeros_like(self.U))
            else:
                mvn = tfd.MultivariateNormalFullCovariance(
                            loc=tf.zeros_like(self.U),
                            covariance_matrix=self.kern.K(self.Z,self.Z))
            probs = tf.reduce_sum(mvn.log_prob(tf.squeeze(self.U)))
        return probs

    def forward(self,x0,ts):
        return self.integrator.forward(x0=x0,ts=ts)

    def predict(self,x0,t):
        """ Computes the integral and returns the path
        Args:
            x0: Python/numpy array of initial value
            t: Python/numpy array of time points the integral is evaluated at
            
        Returns:
            ODE solution computed at t, tensor of size [len(t),len(x0)]
        """
        x0 = np.asarray(x0,dtype=np.float64).reshape((1,-1))
        t = [t]
        integrator = ODERK4(self)
        path = integrator.forward(x0,t)
        path = path[0]
        return path

    def Kzz(self):
        kern = self.kern
        Z = self.Z
        M = tf.shape(Z)[0]
        D = tf.shape(Z)[1] # dim of state
        if kern.ktype == "id":
            Kzz = kern.K(Z) + tf.eye(M, dtype=float_type) * self.jitter
        else:
            Kzz = kern.K(Z) + tf.eye(M*D, dtype=float_type) * self.jitter
        return Kzz

    def U(self):
        U = self.U
        if self.whiten:
            Lz = tf.cholesky(self.Kzz())
            U = tf.matmul(Lz,U)
        return U

    def __str__(self):
        rep = 'noise variance:        ' + str(self.sn.eval()) + \
            '\nsignal variance:       ' + str(self.kern.sf.eval()) + \
            '\nlengthscales:          ' + str(self.kern.ell.eval())
        return rep


class NPSDE(NPODE):
    def __init__(self,Z0,U0,sn0,kern,diffus,s=1,jitter=jitter0,
                 summ=False,whiten=True,fix_Z=False,fix_U=False,fix_sn=False):
        """ Constructor for the NPSDE model
        
        Args:
            Z0: Numpy matrix of initial inducing points of size MxD, M being the
                number of inducing points.
            U0: Numpy matrix of initial inducing vectors of size MxD, M being the
                number of inducing points.
            sn0: Numpy vector of size 1xD for initial signal variance
            kern: Kernel object for GP interpolation
            diffus: BrownianMotion object for diffusion GP interpolation
            s: Integer parameterizing how denser the integration points are
            jitter: Float of jitter level
            summ: Boolean for Tensorflow summary
            whiten: Boolean. Currently we perform the optimization only in the 
                white domain
            fix_Z: Boolean - whether inducing locations are fixed or optimized
            fix_U: Boolean - whether inducing vectors are fixed or optimized
            fix_sn: Boolean - whether noise variance is fixed or optimized
        """
        super().__init__(Z0,U0,sn0,kern,jitter=jitter,
                       summ=summ,whiten=whiten,fix_Z=fix_Z,fix_U=fix_U,fix_sn=fix_sn)
        self.name = 'npsde'
        self.diffus = diffus
        self.integrator = SDEEM(self)

    def build_prior(self):
        pf = super().build_prior()
        pg = self.diffus.build_prior()
        return pf + pg
        
    def g(self,ts,Nw=1):
        return self.diffus.g(ts=ts,Nw=Nw)

    def forward(self,x0,ts,Nw=1):
        return self.integrator.forward(x0=x0,ts=ts,Nw=Nw)

    def sample(self,x0,t,Nw):
        """ Draws random samples from a learned SDE system
        Args:
            Nw: Integer number of samples
            x0: Python/numpy array of initial value
            t: Python/numpy array of time points the integral is evaluated at
            
        Returns:
            Tensor of size [Nw,len(t),len(x0)] storing samples
        """
        # returns (Nw, len(t), D)
        x0 = np.asarray(x0,dtype=np.float64).reshape((1,-1))
        t = [t]
        path = self.integrator.forward(x0,t,Nw)
        path = path[0]
        return path

    def __str__(self):
        return super().__str__() + self.diffus.__str__()


class BrownianMotion:
    def __init__(self,sf0,ell0,Z0,U0,whiten=False,summ=False,
                 fix_ell=True,fix_sf=True,fix_Z=True,fix_U=False):
        with tf.name_scope('Brownian'):
            Zg = Param(Z0,
                       name = "Z",
                       summ = False,
                       fixed = fix_Z)
            Ug = Param(U0,
                        name = "U",
                        summ = False,
                        fixed = fix_U)
            self.kern = OperatorKernel(sf0=sf0,
                      ell0=ell0,
                      ktype="id",
                      name='Kernel',
                      summ=summ,
                      fix_ell=fix_ell,
                      fix_sf=fix_sf)
        self.Zg  = Zg()
        self.Ug  = Ug()
        self.jitter = 1e-6
        self.whiten = whiten
        self.fix_Z = fix_Z
        self.fix_U = fix_U

    def g(self,X,t):
        """ generates state dependent brownian motion
        Args:
            X: current states (in rows)
            t: current time (used if diffusion depends on time)
        Returns:
            A tensor of the same shape as X
        """
        Ug = self.Ug
        Zg = self.Zg
        kern = self.kern

        if not kern.ktype == "id":
            raise NotImplementedError()

        M = tf.shape(Zg)[0]
        D = tf.shape(X)[1]

        if kern.ktype == "id":
            Kzz = kern.K(Zg) + tf.eye(M, dtype=float_type) * self.jitter
        else:
            Kzz = kern.K(Zg) + tf.eye(M*D, dtype=float_type) * self.jitter
        Lz = tf.cholesky(Kzz)

        Kzx = kern.K(Zg, X)

        A = tf.matrix_triangular_solve(Lz, Kzx, lower=True)

        if not self.whiten:
            A = tf.matrix_triangular_solve(tf.transpose(Lz), A, lower=False)

        g = tf.matmul(A, Ug, transpose_a=True)
        dw = tf.random_normal(tf.shape(X),dtype=float_type)

        return g*dw

    def __str__(self):
        rep = '\ndiff signal variance:  ' + str(self.kern.sf.eval()) + \
              '\ndiff lengthscales:     ' + str(self.kern.ell.eval())
        return rep

    def build_prior(self):
        if self.whiten:
            mvn = tfd.MultivariateNormalDiag(
                        loc=tf.zeros_like(self.Ug))
        else:
            mvn = tfd.MultivariateNormalFullCovariance(
                        loc=tf.zeros_like(self.Ug),
                        covariance_matrix=self.kern.K(self.Zg,self.Zg))
        return tf.reduce_sum(mvn.log_prob(self.Ug))