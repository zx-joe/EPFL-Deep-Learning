import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt

class Module (object) :
    def forward( self , x ) :
        raise NotImplementedError
        
    def backward( self , gradwrtoutput ) :
        raise NotImplementedError
        
    def param( self ) :
        return []