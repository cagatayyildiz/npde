from param import Param

import tensorflow as tf
from gpflow import transforms

float_type = tf.float64
jitter_level = 1e-6

class Kernel:
    def __init__(self,sf0,ell0,name="kernel",learning_rate=0.01,
                 summ=False,fix_sf=False,fix_ell=False):
        with tf.name_scope(name):
            sf = Param(sf0,
                              transform=transforms.Log1pe(),
                              name="sf",
                              learning_rate = learning_rate,
                              summ = summ,
                              fixed = fix_sf)
            ell = Param(ell0,
                              transform=transforms.Log1pe(),
                              name="ell",
                              learning_rate = learning_rate,
                              summ = summ,
                              fixed = fix_ell)
        self.sf = sf()
        self.ell = ell()
        self.fix_sf = fix_sf
        self.fix_ell = fix_ell

    def square_dist(self,X,X2=None):
        X = X / self.ell
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.ell
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

class OperatorKernel(Kernel):
    def __init__(self,sf0,ell0,ktype="id",learning_rate=0.01,
                      summ=False,block=True,name="OperatorKernel",fix_sf=False,
                      fix_ell=False):
        super().__init__(sf0 = sf0,
                         ell0 = ell0,
                         name = name,
                         learning_rate = learning_rate,
                         summ = summ,
                         fix_sf = fix_sf,
                         fix_ell = fix_ell)
        self.ndims = len(ell0)
        self.ktype=ktype
        self.block = block

    def RBF(self,X,X2=None):
        if X2 is None:
            return self.sf**2 * tf.exp(-self.square_dist(X) / 2)
        else:
            return self.sf**2 * tf.exp(-self.square_dist(X, X2) / 2)

    def HessianDivergenceFree(self,X,X2=None):
        D = tf.shape(X)[1]
        N = tf.shape(X)[0]
        M = tf.shape(X2)[0]

        X_expd = tf.expand_dims(X,-1) / self.ell
        X2_expd = tf.transpose(tf.expand_dims(X2,-1),perm=[2,1,0])/ self.ell
        diff = tf.subtract(X_expd,X2_expd)
        diff1 = tf.transpose(tf.expand_dims(diff,-1),perm=[0,2,1,3])
        diff2 = tf.transpose(tf.expand_dims(diff,-1),perm=[0,2,3,1])

        term1 = tf.multiply(diff1,diff2)
        term2 = tf.multiply(
            tf.expand_dims(tf.expand_dims(tf.cast(D,dtype=float_type) - 1.0 - self.square_dist(X, X2),-1),-1),
                    tf.eye(D, batch_shape=[N,M],dtype=float_type))

        H = term1 + term2

        return H

    def HessianCurlFree(self,X,X2=None):
        D = tf.shape(X)[1]
        N = tf.shape(X)[0]
        M = tf.shape(X2)[0]

        X = X / self.ell
        X2 = X2 / self.ell
        X_expd = tf.expand_dims(X,-1)
        X2_expd = tf.transpose(tf.expand_dims(X2,-1),perm=[2,1,0])
        diff = tf.subtract(X_expd,X2_expd)
        diff1 = tf.transpose(tf.expand_dims(diff,-1),perm=[0,2,1,3])
        diff2 = tf.transpose(tf.expand_dims(diff,-1),perm=[0,2,3,1])

        term1 = tf.multiply(diff1,diff2)

        H = tf.eye(D, batch_shape=[N,M],dtype=float_type) - term1

        return H

    def HessianIdentity(self,X,X2=None):
        D = tf.shape(X)[1]
        N = tf.shape(X)[0]
        M = tf.shape(X2)[0]

        H = tf.ones([N,M,D,D],dtype=float_type)

        return H

    def K(self,X,X2=None):
            
        if X2 is None:
            rbf_term = self.RBF(X)
            X2 = X
        else:
            rbf_term = self.RBF(X,X2)
            
        if self.ktype == "id":
            # hes_term = self.HessianIdentity(X,X2)
            return rbf_term

        elif self.ktype == "df":
            hes_term = self.HessianDivergenceFree(X,X2)

        elif self.ktype == "cf":
            hes_term = self.HessianCurlFree(X,X2)
        else:
            raise ValueError("Bad kernel type passed to `ktype`")

        rbf_term = tf.expand_dims(tf.expand_dims(rbf_term,-1),-1)
        K = rbf_term * hes_term / tf.square(self.ell)

        if self.block:
            K = self.tfblock(K)

        return K

    def Ksymm(self,X):
        raise NotImplementedError()

    def Kdiag(self,X):
        raise NotImplementedError()

    def tfblock(self,tensor):
        '''
        input : tensor of shape NxM,DxD
        returns : tensor of shape (ND)x(MD)
        '''
        N = tf.shape(tensor)[0]
        M = tf.shape(tensor)[1]
        D = self.ndims

        stacked_list = []
        for d in range(D):
            t = tf.stack([tf.reshape(tensor[:,:,p,d],[N,M]) for p in range(D)],axis=1)
            t = tf.transpose(tf.reshape(t,[N*D,M]))
            stacked_list.append(t)

        reshaped = tf.stack(stacked_list,axis=1)
        reshaped = tf.transpose(tf.reshape(reshaped,[M*D,N*D]))

        return reshaped

class RBF(Kernel):
    '''
    Taken from GPFlow
    '''
    def __init__(self,sf0,ell0,name="RBFKernel",eta=0.01,summ=False,
                 fix_sf=False,fix_ell=False):                
        super().__init__(sf0,ell0,name=name,learning_rate=eta,summ=summ,
                        fix_sf=fix_sf,fix_ell=fix_ell)

    def K(self,X,X2=None):
        if X2 is None:
            return self.sf**2 * tf.exp(-self.square_dist(X) / 2)
        else:
            return self.sf**2 * tf.exp(-self.square_dist(X, X2) / 2)

    def Ksymm(self,X):
        return self.sf**2 * tf.exp(-self.square_dist(X) / 2)

    def Kdiag(self,X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.sf**2))
