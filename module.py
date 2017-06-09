import numpy as np


def sgd_momentum(x, dx, config, state):
    """
        This is a very ugly implementation of sgd with momentum 
        just to show an example how to store old grad in state.

        config:
            - momentum
            - learning_rate
        state:
            - old_grad
    """

    # x and dx have complex structure, old dx will be stored in a simpler one
    state.setdefault('old_grad', {})
    # print ('inp', x, dx, config, state)
    i = 0
    for cur_layer_x, cur_layer_dx in zip(x ,dx):
        for cur_x, cur_dx in zip(cur_layer_x ,cur_layer_dx):

            cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            # print ('oldg', cur_old_grad)
            np.add(config['momentum'] * cur_old_grad, config['learning_rate'] * cur_dx, out = cur_old_grad)

            cur_x -= cur_old_grad
            # print ('inp', x, dx, config, state)
            # print ('mm', cur_x)
            i += 1


def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]

    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_idx = indices[start:end]

        yield X[batch_idx], Y[batch_idx]


class Module(object):
    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 

        gradInput = module.backward(input, gradOutput)
    """

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it. 
        """

        # The easiest case:

        # self.output = input
        # return self.output

        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """

        # The easiest case:

        # self.gradInput = gradOutput
        # return self.gradInput

        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []

    def training(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially. 

         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`. 
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []


    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})   


        Just write a little loop. 
        """
        self.output = input.copy()
        for module in self.modules:
            self.output = module.forward(self.output)

        # Your code goes here. ################################################
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input, g_1)   


        !!!

        To ech module you need to provide the input, module saw while forward pass, 
        it is used while computing gradients. 
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass) 
        and NOT `input` to this Sequential module. 

        !!!

        """
        # Your code goes here. ################################################
        print ("Work before backward")

        self.gradInput = gradOutput
        for ind in range(self.modules.__len__() - 1, 0, -1):
            print (ind, self.modules[ind].__repr__())
            self.gradInput = self.modules[ind].backward(self.modules[ind - 1].output, self.gradInput)
        self.modules[0].backward(input, self.gradInput)
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)


class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        self.output = input.dot(self.W.T) + self.b  # Your code goes here.
        # print ('linear out', self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = gradOutput.dot(self.W).reshape(input.shape)  # <gradient of this layer w.r.t. input. You will need gradOutput and self.W>

        assert self.gradInput.shape == input.shape, "wrong shape"
        print ('lb', gradOutput.shape, self.gradInput.shape, self.W.shape)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradW = gradOutput.T.dot(input).reshape(
            self.W.shape)  # <compute gradient of loss w.r.t. weight matrix. You will need gradOutput and input>

        assert self.gradW.shape == self.W.shape

        self.gradb = np.ones_like(self.b) * gradOutput.sum(axis=0)

        assert self.gradb.shape == self.b.shape

        return self.gradW, self.gradb

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        input = np.subtract(input, input.max(axis=1, keepdims=True))
        print (np.sum(input), np.sum(self.output))
        # Your code goes here. ################################################
        self.output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)

        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        # s_e = np.sum(np.exp(input), axis=1, keepdims=True)
        # self.gradInput = (np.exp(input) * np.exp(gradOutput) - np.exp(gradOutput))/s_e
        print('sm', input.shape, gradOutput.shape)
        exp = np.exp(np.subtract(input, input.max(axis=1, keepdims=True)))
        denom = exp.sum(axis=1, keepdims=True)
        e = np.diag(exp.dot(gradOutput.T))
        self.gradInput = - np.diag(e).dot(exp)
        self.gradInput += exp * denom * gradOutput
        self.gradInput /= denom ** 2
        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function 
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result. 

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        # Use this trick to avoid numerical errors
        eps = 1e-15
        input_clamp = np.clip(input, eps, 1 - eps)

        # Your code goes here. ################################################
        self.output = np.sum(np.einsum('ij,ij->i', np.log(input_clamp), target)) / input.shape[0]
        # print (target, input, target.dot(input.T))
        return self.output

    def updateGradInput(self, input, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(input, 1 - 1e-15))
        #print('cri', input, target)
        # Your code goes here. ################################################
        self.gradInput = (target / input_clamp) / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"


def net_image(net, step=1):
    ir = np.arange(-6.0, 6.0, step)
    jr = np.arange(-6.0, 6.0, step)
    pic = np.zeros([ir.shape[0], ir.shape[0]])
    #print (pic.shape)
    for ind, i in enumerate(ir):
        for jnd, j in enumerate(jr):
            pred = net.forward(np.array([[i, j]]))
            # print (pred, i, j)
            pic[ind, jnd] = pred[0][0] * 250
            # if 0.3 < pred[0][0] < 0.6:

            #   x1.append(j)
            #  x2.append(i)
    return pic