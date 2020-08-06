import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

from abc import ABC, abstractmethod

float_type = tf.float64

class Integrator(ABC):
    """ Base class for integrators
    """
    def __init__(self,model):
        self.model= model

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _step_func(self):
        pass

    @abstractmethod
    def _make_scan_func(self):
        pass


class ODERK4(Integrator):
    """ Runge-Kutta implementation for solving ODEs
    """
    def __init__(self,model):
        super().__init__(model)

    def forward(self,x0,ts):
        Nt = x0.shape[0]
        Xs = np.zeros(Nt,dtype=np.object)
        for i in range(Nt):
            time_grid = ops.convert_to_tensor(ts[i], preferred_dtype=float_type, name='t')
            y0 = ops.convert_to_tensor(x0[i,:].reshape((1,-1)), name='y0')
            time_delta_grid = time_grid[1:] - time_grid[:-1]
            scan_func = self._make_scan_func(self.model.f)
            y_grid = functional_ops.scan(scan_func, (time_grid[:-1],time_delta_grid), y0)
            y_s = array_ops.concat([[y0], y_grid], axis=0)
            Xs[i] = tf.reshape(tf.squeeze(y_s),[len(ts[i]),self.model.D])
        return Xs

    def _step_func(self,f,dt,t,y):
        dt = math_ops.cast(dt, y.dtype)
        k1 = f(y, t)
        k2 = f(y + dt*k1/2, t+dt/2)
        k3 = f(y + dt*k2/2, t+dt/2)
        k4 = f(y + dt*k3, t+dt)
        return math_ops.add_n([k1, 2*k2, 2*k3, k4]) * (dt / 6)

    def _make_scan_func(self,f):
        def scan_func(y, t_dt):
            t, dt = t_dt
            dy = self._step_func(f, dt, t, y)
            dy = math_ops.cast(dy, dtype=y.dtype)
            return y + dy
        return scan_func


class SDEEM(Integrator):
    """ Euler-Maruyama implementation for solving SDEs
    dx = f(x)*dt + g*sqrt(dt)
    """
    def __init__(self,model,s=1):
        super().__init__(model)
        self.s = s

    def forward(self,x0,ts,Nw=1):
        Xs = np.zeros(len(ts),dtype=np.object)
        for i in range(len(ts)):
            t = np.linspace(0,np.max(ts[i]),(len(ts[i])-1)*self.s+1)
            t = np.unique(np.sort(np.hstack((t,ts[i]))))
            idx = np.where( np.isin(t,ts[i]) )[0]
            t = np.reshape(t,[-1,1])
            time_grid = ops.convert_to_tensor(t, preferred_dtype=float_type, name='t')
            time_delta_grid = time_grid[1:] - time_grid[:-1]
            y0 = np.repeat(x0[i,:].reshape((1,-1)),Nw,axis=0)
            y0 = ops.convert_to_tensor(y0, name='y0')
            scan_func = self._make_scan_func(self.model.f,self.model.diffus.g)
            y_grid = functional_ops.scan(scan_func, (time_grid[:-1],time_delta_grid), y0)
            ys = array_ops.concat([[y0], y_grid], axis=0)
            Xs[i] = tf.transpose(tf.gather(ys,idx,axis=0),[1,0,2])
        return Xs

    def _step_func(self,f,g,t,dt,x):
        dt = math_ops.cast(dt, x.dtype)
        return f(x,t)*dt + g(x,t)*tf.sqrt(dt)

    def _make_scan_func(self,f,g):
        def scan_func(y, t_dt):
            t,dt = t_dt
            dy = self._step_func(f,g,t,dt,y)
            dy = math_ops.cast(dy, dtype=y.dtype)
            return y + dy
        return scan_func