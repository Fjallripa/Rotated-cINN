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
    
    def __init__(self, init_identity=False, subnet=None):
        super().__init__()

        self.init_identity = init_identity
        self.subnet = subnet

        # Build network
        self.cinn = self.build_inn()
        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        if not self.init_identity:
            for p in self.trainable_parameters:
                p.data = 0.01 * torch.randn_like(p)
            

    def build_inn(self):
        nodes = []   # list of freia Nodes, analagous to the nn.Sequential list of pytorch Modules
        cond = Ff.ConditionNode(12)  # special node providing the external condition for the subnet


        # Input nodes
        nodes.append(Ff.InputNode(1, 28, 28))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))   # just flattens the input image

        # Creating an INN out of 20 coupling blocks
        subnet = lambda ch_in, ch_out: self._subnet(ch_in, ch_out)
        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))
                # The GLOWCouplingBlock uses two subnets.

        # Output nodes
        nodes.extend([cond, Ff.OutputNode(nodes[-1])])

        # Full network
        return Ff.ReversibleGraphNet(nodes, verbose=False)


    def _subnet(self, ch_in, ch_out):
        '''the neural network inside the coupling block'''
        if self.subnet == None or self.subnet == 'original':
            sub_net = nn.Sequential(nn.Linear(ch_in, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, ch_out))
        elif self.subnet == 'og_broad':
            sub_net = nn.Sequential(nn.Linear(ch_in, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, ch_out))
        elif self.subnet == 'og_two_deeper':
            sub_net = nn.Sequential(nn.Linear(ch_in, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, ch_out))
        else:
            raise ValueError(f"The subnet '{self.subnet}' is not implemented.")
        
        if self.init_identity:
            # Initializing parameters -- last layer initialized to 0.
            nn.init.xavier_normal_(sub_net[0].weight)
            nn.init.zeros_(sub_net[0].bias)
            nn.init.zeros_(sub_net[2].weight)
            nn.init.zeros_(sub_net[2].bias)
            
        return sub_net
    


    def forward(self, x:torch.Tensor, c:torch.Tensor, jac:bool=True) -> tuple[torch.Tensor]:
        '''
        turns an image x into a latent code z using the conditional input c
        '''
        
        return self.cinn.forward(x, c, jac=jac)
        

    def reverse(self, z:torch.Tensor, c:torch.Tensor, jac:bool=True) -> tuple[torch.Tensor]:
        '''
        turns a latent code z into an image x using the conditional input c
        '''
        
        return self.cinn(z, c, rev=True, jac=jac)
