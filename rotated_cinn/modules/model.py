# Rotated cINN Model
# ==================
# Defines the model class Rotated_cINN.


# Imports
import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm


# Parameters
ndim_total = 28 * 28


# Main class
class Rotated_cINN(nn.Module):
    '''
    A modified cINN able to generate images from a custom Rotated MNIST dataset.
    As conditional input it uses the domain and the class.
    The former is the rotation angle as a cos, sin pair and the latter is a one-hot representation of the digit.
    '''
    
    def __init__(self, learning_rate):
        super().__init__()

        # Build network
        self.cinn = self.build_inn()

        # Random initialization of parameters
            #? Why this custom initialization? Aren't there standardized torch solutions out there?
        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=learning_rate, weight_decay=1e-5)


    def build_inn(self):

        def subnet(ch_in, ch_out):
            '''the neural network inside the coupling block'''
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        nodes = []   # list of freia Nodes, analagous to the nn.Sequential list of pytorch Modules
        cond = Ff.ConditionNode(12)  # special node providing the external condition for the subnet


        # Input nodes
        nodes.append(Ff.InputNode(1, 28, 28))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))   # just flattens the input image

        # Creating an INN out of 20 coupling blocks
        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
                # In one coupling block only half of the dimensions get transformed. 
                # Fixed permutations between blocks ensures that all the dimensions get transformed similarly often.
                # Here 10 times on average.
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        # Output nodes
        nodes.extend([cond, Ff.OutputNode(nodes[-1])])

        # Full network
        return Ff.ReversibleGraphNet(nodes, verbose=False)



    def forward(self, x, c):
        '''
        turns an image x into a latent code z using the conditional input c
        '''
        
        return self.cinn.forward(x, c)
        

    def reverse(self, z, c):
        '''
        turns a latent code z into an image x using the conditional input c
        '''
        
        return self.cinn(z, c, rev=True)
