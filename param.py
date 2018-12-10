import tensorflow as tf
from gpflow import transforms
float_type = tf.float64

#class Variable(Variable):
#    '''
#    extend tf.Variable to have properties : learning_rate
#    '''
#    pass
#
#    def set_learning_rate(self,value):
#        self._learning_rate = value
#
#    @property
#    def learning_rate(self):
#        if hasattr(self,'_learning_rate'):
#            return self._learning_rate
# 
#        else:
#            return 0.001

class Param:
    '''
    Inheriting from GPFlow
    TODO : add a fixed flag in which case this should return tf.tensor instead of tf.Variable
    '''
    def __init__(self,value,transform = None,fixed=False,name=None,learning_rate=None,summ=False):
        self.value = value
        self.fixed = fixed

        if name is None:
            self.name = "param"
        else:
            self.name = name

        if transform is None:
            self.transform=transforms.Identity()
        else:
            self.transform = transform

        if self.fixed:
            self.tf_opt_var = tf.constant(self.value,name=name,dtype=float_type)
        else:
            # self.tf_opt_var = Variable(self.transform.backward(self.value),name=name,dtype=float_type)
            self.tf_opt_var = tf.Variable(self.transform.backward(self.value),name=name,dtype=float_type)

#        if learning_rate is not None and not self.fixed:
#            self.tf_opt_var.set_learning_rate(learning_rate)

        if summ:
            self.variable_summaries(self.tf_opt_var)

    def __call__(self):
        if self.fixed:
            return self.tf_opt_var
        else:
            return self.transform.forward_tensor(self.tf_opt_var)

    def __set__(self, instance, value):
        self.tf_opt_var.assign(self.transform.backward(value))

    def variable_summaries(self,var):
        """Attach tensorBoard visualization"""
        tf.summary.histogram(self.name, var)

    @property
    def shape(self):
        return self.value.shape
