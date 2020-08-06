import numpy as np
import tensorflow as tf
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc

float_type = tf.float64

def vdp(x,t=0):
    dx = np.asarray([x[1],
                     (1-x[0]**2)*x[1]-x[0]])
    return dx

def g(x,t,mean_,var_,sigvar_,const_):
    g = ss.multivariate_normal.pdf(x,mean_,var_)*sigvar_ + const_
    return g


def gen_data(model='vdp',Ny=[30],tend=8,x0=np.asarray([2.0,-3.0]),nstd=0.1):
    x0in = x0
    Nt = len(Ny)
    x0 = np.zeros((Nt,2))
    t = np.zeros((Nt,), dtype=np.object)
    Y = np.zeros((Nt,), dtype=np.object)

    mean_ = np.array([-2,1])
    var_ = np.eye(2)*0.5
    if model == 'vdp':
        gtrue = lambda x,t : g(x,t,mean_,var_,0,0)
    elif model == 'vdp-cdiff':
        gtrue = lambda x,t : g(x,t,mean_,var_,0,0.2)
    elif model == 'vdp-sdiff':
        gtrue = lambda x,t : g(x,t,mean_,var_,3.0,0.0)
    else:
        raise NotImplementedError('Only stochastic/deterministic Van der Pol supported')
    diff = lambda x,t: ss.norm.rvs(size=[2,1]) * gtrue(x,t)
    for i in range(Nt):
        tspan = np.linspace(0,tend,Ny[i])
        t[i] = tspan
        X = em_int(vdp, diff, x0in, tspan)
        Y[i] = X + ss.norm.rvs(size=X.shape) * nstd
        x0[i,:] = Y[i][0,:]

    plot_data(t,Y,vdp,gtrue)
    return x0,t,Y,X,2,vdp,gtrue

def em_int(f,g,x0,t):
    """ Euler-Maruyama integration
    
    """
    ts = np.linspace(0,np.max(t),(len(t)-1)*5)
    ts = np.unique(np.sort(np.hstack((ts,t))))
    idx = np.where( np.isin(ts,t) )[0]
    ts = np.reshape(ts,[-1,1])
    dt = ts[1:] - ts[:-1]
    T = len(ts)
    D = len(x0)
    Xs = np.zeros((T,D),dtype=np.float64)
    Xs[0,:] = x0
    for i in range(0,T-1):
        fdt = f(Xs[i,:],ts[i])*dt[i]
        gdt = g(Xs[i,:],ts[i])*np.sqrt(dt[i])
        Xs[i+1,:] = Xs[i,:] + fdt + gdt.flatten()
    X = Xs[idx,:]
    return X


def plot_model(npde,t,Y,Nw=1):
    print(npde)
    Z = npde.Z.eval()
    U = npde.U
    if npde.whiten:
        Lz = tf.cholesky(npde.Kzz())
        U = tf.matmul(Lz,U)
    U = U.eval()
    D = Z.shape[1]
    if D == 2:
        ts = np.linspace(np.min(t[0]),np.max(t[0])*3,len(t[0])*12)
    else:
        ts = np.linspace(np.min(t[0]),np.max(t[0]),len(t[0])*10)

    vdpts = np.linspace(np.min(t[0]),np.max(t[0])*3,len(t[0])*10)
    g = lambda x,t: 0
    true_path = em_int(vdp,g,Y[0][0],vdpts)
    rc('text', usetex=True)

    x0 = np.vstack([Y_[0] for Y_ in Y])
    if npde.name is 'npode':
        X = npde.forward(x0,[ts]*len(Y))
        X = X[0].eval()
        plt.figure(1,figsize=(15,12))
        gs = GridSpec(4, 2)
        ax1 = plt.subplot(gs[0:4,0])
        pathh, = ax1.plot(X[:,0],X[:,1],'-',label='estimated path')
        if npde.kern.ktype == "id":
            ilh = ax1.scatter(Z[:,0],Z[:,1],100, facecolors='none', edgecolors='k',label='inducing locations')
            ivh = ax1.quiver(Z[:,0],Z[:,1],U[:,0],U[:,1],units='height',width=0.006,color='k',label='inducing vectors')
        if Y is not None:
            dh, = ax1.plot(Y[0][:,0],Y[0][:,1],'ro',markersize=4,label='data points')
        ax1.set_xlabel('$x_1$', fontsize=12)
        ax1.set_ylabel('$x_2$', fontsize=12)
        ax1.legend(handles=[pathh,ilh,ivh,dh],loc=2,fontsize='medium')
        ax1.set_title('Vector Field',fontsize=15)

        for d in range(D):
            ax = plt.subplot(gs[d,1])
            trajh, = ax.plot(ts,X[:,d],label='estimated path')
            vdph, = ax.plot(vdpts,true_path[:,d],label='true path')
            if Y is not None:
                dh, = ax.plot(t[0],Y[0][:,d],'ro',markersize=2,label='data points')
            ax.set_xlabel('$t$', fontsize=12)
            ax.set_ylabel('$x_{:d}$'.format(d+1), fontsize=12)
            if d == 0:
                ax.legend(handles=[trajh,vdph,dh],loc=1)
                ax.set_title('Paths over Time',fontsize=15)
        plt.savefig('drift_ode.png', dpi=200)
        plt.show()
    elif npde.name is 'npsde':
        ts = np.linspace(np.min(t[0]),np.max(t[0])*3,len(t[0])*20)
        X = npde.forward(x0,[ts]*len(Y),Nw=Nw)
        X = [x.eval() for x in X] # list of (Nw,len(ts),d)
        print([x.shape for x in X])
        plt.figure(1,figsize=(15,12))
        gs = GridSpec(4, 2)
        ax1 = plt.subplot(gs[0:4,0])
        for j in range(len(X)):
            for i in range(X[j].shape[0]):
                pathh, = ax1.plot(X[j][i,:,0],X[j][i,:,1],'b-',linewidth=0.5,label='samples')
        for j in range(len(X)):
            dh, = ax1.plot(Y[j][:,0],Y[j][:,1],'-ro',markersize=4,linewidth=0.3,label='data points')
        if npde.kern.ktype == "id":
            ilh = ax1.scatter(Z[:,0],Z[:,1],100, facecolors='none', edgecolors='k',label='inducing locations')
            ivh = ax1.quiver(Z[:,0],Z[:,1],U[:,0],U[:,1],units='height',width=0.006,color='k',label='inducing vectors')
        ax1.set_xlabel('$x_1$', fontsize=12)
        ax1.set_ylabel('$x_2$', fontsize=12)
        ax1.legend(handles=[pathh,ilh,ivh,dh],loc=2,fontsize='large')
        ax1.set_title('Vector Field',fontsize=15)
        ts2 = np.linspace(np.min(ts),np.max(ts),len(ts)*3)
        true_vdp = em_int(vdp,lambda x,t:0,Y[0][0,:],ts2)
        for d in range(D):
            ax = plt.subplot(gs[d,1])
            for i in range(X[0].shape[0]):
                pathh, = ax.plot(ts,X[0][i,:,d],'b-',linewidth=0.5,label='samples')
            datah, = ax.plot(t[0],Y[0][:,d],'ro',markersize=3,label='data')
            vdph, = ax.plot(ts2,true_vdp[:,d],'-',color='#33FF00',linewidth=2,label='true vdp')
            ax.set_xlabel('$t$', fontsize=12)
            ax.set_ylabel('$x_{:d}$'.format(d+1), fontsize=12)
            if d==0:
                ax.legend(handles=[pathh,vdph,datah],loc=1)
                ax.set_title('Paths over Time',fontsize=15)
        plt.savefig('drift_sde.png', dpi=200)
        plt.show()

        plt.figure(2,figsize=(8,5))
        W = 50
        xv = np.linspace(-3, 3, W)
        yv = np.linspace(-3, 3, W)
        xvv,yvv = np.meshgrid(xv,yv)
        Zs = np.array([xvv.T.flatten(),yvv.T.flatten()]).T
        Ug = npde.diffus.Ug
        Zg = npde.diffus.Zg
        kern = npde.diffus.kern
        M = tf.shape(Zg)[0]
        Kzz = kern.K(Zg) + tf.eye(M, dtype=float_type) * 1e-6
        Kzx = kern.K(Zg, Zs)
        Lz = tf.cholesky(Kzz)
        A = tf.matrix_triangular_solve(Lz, Kzx, lower=True)
        if not npde.diffus.whiten:
            A = tf.matrix_triangular_solve(tf.transpose(Lz), A, lower=False)
        gs = tf.matmul(A, Ug, transpose_a=True).eval()
        gs = np.abs(np.reshape(gs,[W,W]).T)
        plt.imshow(np.around(gs, decimals=3),origin='lower')
        plt.title('estimated diffusion')
        S = 6
        ax1.set_xlabel('$x_1$', fontsize=12)
        ax1.set_ylabel('$x_2$', fontsize=12)
        plt.xticks(range(0,len(xv),int(W/S)), ['{:.2f}'.format(xv[i]) for i in range(0,len(xv),int(W/S))])
        plt.yticks(range(0,len(yv),int(W/S)), ['{:.2f}'.format(yv[i]) for i in range(0,len(yv),int(W/S))])
        plt.colorbar()
        # plt.savefig('diff.png', dpi=200)
        plt.show()


def plot_data(t,Y,f,gtrue):
    fig = plt.figure(1,figsize=(10,8))
    Nt = len(Y)
    gs = GridSpec(5, 2)
    ax1 = plt.subplot(gs[0:3,0])
    for i in range(Nt):
        ax1.plot(Y[i][:,0],Y[i][:,1],'.-')
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.set_title('trajectories')

    ax2 = plt.subplot(gs[3,0])
    ax2.plot(t[0],Y[0][:,0],'.')
    ax2.set_xlabel('$t$', fontsize=12)
    ax2.set_ylabel('$x_1$', fontsize=12)

    ax3 = plt.subplot(gs[4,0])
    ax3.plot(t[0],Y[0][:,1],'.')
    plt.tight_layout()
    ax3.set_xlabel('$t$', fontsize=12)
    ax3.set_ylabel('$x_2$', fontsize=12)

    W = 100
    xv = np.linspace(-3, 3, W)
    yv = np.linspace(-3, 3, W)
    xvv,yvv = np.meshgrid(xv,yv)
    Zs = np.array([xvv.T.flatten(),yvv.T.flatten()]).T
    diffs = np.zeros(W*W)
    for i in range(W*W):
        diffs[i] = gtrue(Zs[i,:],0)
    if np.sum(np.abs(diffs[0:100])) > 1:
        diffs = np.abs(np.reshape(diffs,[W,W]).T)
        ax4 = plt.subplot(gs[0:3,1])
        im = ax4.imshow(np.around(diffs, decimals=3),origin='lower')
        ax4.set_title('diffusion')
        S = 6
        plt.xticks(range(0,len(xv),int(W/S)), ['{:.2f}'.format(xv[i]) for i in range(0,len(xv),int(W/S))])
        plt.yticks(range(0,len(yv),int(W/S)), ['{:.2f}'.format(yv[i]) for i in range(0,len(yv),int(W/S))])
        fig.colorbar(im)

    plt.tight_layout()
    # plt.show()
