import tensorflow as tf
import numpy as np

float_type = tf.float64

from npde import NPODE, NPSDE, BrownianMotion
from kernels import OperatorKernel
from utils import plot_model

def npde_fit(sess,t,Y,model='sde',sf0=1.0,ell0=[2,2],sfg0=1.0,ellg0=[1e5],
             W=6,ktype="id",whiten=True,Nw=50,
             fix_ell=False,fix_sf=False,fix_Z=False,fix_U=False,fix_sn=False,
             fix_ellg=False,fix_sfg=False,fix_Zg=True,fix_Ug=False,
             num_iter=500,print_int=10,eta=5e-3,dec_step=20,dec_rate=0.99,plot_=True):

    print('Model being initialized...')
    def init_U0(Y=None,t=None,kern=None,Z0=None,whiten=None):
        Ug = (Y[1:,:] - Y[:-1,:]) / np.reshape(t[1:]-t[:-1],(-1,1))
        tmp = NPODE(Y[0,:].reshape((1,-1)),t,Y,Z0=Y[:-1,:],U0=Ug,sn0=0,kern=kern,jitter=0.2,whiten=False,
                    fix_Z=True,fix_U=True,fix_sn=True)
        U0 = tmp.f(X=Z0)
        if whiten:
            Lz = tf.cholesky(kern.K(Z0))
            U0 = tf.matrix_triangular_solve(Lz, U0, lower=True)
        U0 = U0.eval()
        return U0

    D = len(ell0)
    Nt = len(Y)
    x0 = np.zeros((Nt,D))
    Ys = np.zeros((0,D))
    for i in range(Nt):
        x0[i,:] = Y[i][0,:]
        Ys = np.vstack((Ys,Y[i]))
    maxs = np.max(Ys,0)
    mins = np.min(Ys,0)
    grids = []
    for i in range(D):
        grids.append(np.linspace(mins[i],maxs[i],W))
    vecs = np.meshgrid(*grids)
    Z0 = np.zeros((0,W**D))
    for i in range(D):
        Z0 = np.vstack((Z0,vecs[i].T.flatten()))
    Z0 = Z0.T

    tmp_kern = OperatorKernel(sf0,ell0,ktype="id",fix_ell=True,fix_sf=True)

    U0 = np.zeros(Z0.shape,dtype=np.float64)
    for i in range(len(Y)):
        U0 += init_U0(Y[i],t[i],tmp_kern,Z0,whiten)
    U0 /= len(Y)

    sn0 = 0.5*np.ones(D)
    Ug0 = np.ones([Z0.shape[0],1])*0.01

    ell0  = np.asarray(ell0,dtype=np.float64)
    ellg0 = np.asarray(ellg0,dtype=np.float64)


    kern = OperatorKernel(sf0=sf0,
                        ell0=ell0,
                        ktype=ktype,
                        fix_ell=fix_ell,
                        fix_sf=fix_sf)

    if model is 'ode':
        npde = NPODE(x0=x0,
                        t=t,
                        Y=Y,
                        Z0=Z0,
                        U0=U0,
                        sn0=sn0,
                        kern=kern,
                        whiten=whiten,
                        fix_Z=fix_Z,
                        fix_U=fix_U,
                        fix_sn=fix_sn)
        with tf.name_scope("cost"):
            X = npde.forward()
            ll = []
            for i in range(len(X)):
                mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(
                                loc=X[i],covariance_matrix=tf.diag(npde.sn))
                ll.append(tf.reduce_sum(mvn.log_prob(Y[i])))
            ll = tf.reduce_logsumexp(ll)
            ode_prior = npde.build_prior()
            cost = -(ll + ode_prior)

    elif model is 'sde':
        diffus = BrownianMotion(sf0=sfg0,
               ell0 = ellg0,
               U0 = Ug0,
               Z0 = Z0,
               whiten=whiten,
               fix_sf=fix_sfg,
               fix_ell=fix_ellg,
               fix_Z=fix_Zg,
               fix_U=fix_Ug)
        npde = NPSDE(x0=x0,
                  t=t,
                  Y=Y,
                  Z0=Z0,
                  U0=U0,
                  sn0=sn0,
                  kern=kern,
                  diffus=diffus,
                  whiten=whiten,
                  fix_Z=fix_Z,
                  fix_U=fix_U,
                  fix_sn=fix_sn)
        with tf.name_scope("cost"):
            Xs = npde.forward(Nw=Nw)
            print(Xs[0].shape)
            ll = 0
            for i in range(len(Y)):
                mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=Y[i],covariance_matrix=tf.diag(npde.sn))
                ll_i = tf.stack([mvn.log_prob(Xs[i][j,:,:]) for j in range(Xs[i].shape[0])]) # Nw x D
                ll_i = tf.reduce_sum(tf.log(tf.reduce_mean(tf.exp(ll_i),axis=0)))
                ll += ll_i
            sde_prior = npde.build_prior()
            cost = -(ll + sde_prior)

    else:
        raise NotImplementedError("model parameter should be either 'ode' or 'sde', not {:s}\n".format(model))


    print('Adam optimizer being initialized...')
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    expdec = tf.train.exponential_decay(eta,global_step,dec_step,dec_rate,staircase=True)

    optimizer = tf.train.AdamOptimizer(expdec).minimize(cost,global_step)

    sess.run(tf.global_variables_initializer())

    print('Optimization starts.')
    print('{:>16s}'.format("iteration")+'{:>16s}'.format("objective"))
    for i in range(1,num_iter+1):
        _cost,_ = sess.run([cost,optimizer])
        if i==1 or i%print_int==0 or i==num_iter:
            print('{:>16d}'.format(i)+'{:>16.3f}'.format(_cost))
    print('Optimization ends.')

    if plot_:
        print('Plotting...')
        plot_model(npde,Nw=50)

    return npde
