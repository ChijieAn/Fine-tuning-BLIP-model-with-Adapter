import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
  
class MyAdapter(nn.Module):
  def __init__(self,input_dim,hidden_dim,output_dim):
    super().__init__()
    self.fc1=nn.Linear(input_dim,hidden_dim)
    self.fc2=nn.Linear(hidden_dim,output_dim)
    self.activation=nn.ReLU()
    self.layer_norm = nn.LayerNorm(input_dim)

    nn.init.normal_(self.fc1.weight, mean=0.0, std=0.0001)
    nn.init.normal_(self.fc2.weight,mean=0.0,std=0.0001)
  def forward(self,x):
    y=self.layer_norm(x)
    y=self.fc1(y)
    #print(y.shape)
    y=self.activation(y)
    #print(y.shape)
    y=self.fc2(y)
    #print(y.shape)
    #print(x.shape)
    return x+y
  
class ConcatenatedModel(nn.Module):
    def __init__(self,net1,net2):
        super(ConcatenatedModel,self).__init__()
        self.net1=net1
        self.net2=net2

    def forward(self,x):
        #in this case, the ResNet18 neural network takes the input diim(batch_size,3,256,256) to 2048
        x=self.net1(x)
        #the projector nn takes input dim 2048 to 128
        x=self.net2(x)

        return x