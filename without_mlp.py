import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
show=ToPILImage()
class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.class_num = 10

        ### initialize the BaseModel ---------------------
        self.embeds=nn.Embedding(10,128).cuda()

        self.Weights_Generator_MLP=nn.Sequential(

            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Tanh()
        )

        self.resnet18 = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:5])
        self.DC=nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)

        self.Class_Score_Predictor_MLP=nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        ## YOUR CODE HERE


        ### ----------------------------------------------

    def forward(self, imgs):

        ### complete the forward path --------------------


        ## YOUR CODE HERE

        out1=self.resnet18(imgs)





        out1=self.DC(out1)
        out1=out1.view(-1,64)
        cls_scores=self.Class_Score_Predictor_MLP(out1)
        #single = self.embeds(imgs)
        #out2=self.Weights_Generator_MLP(single)
        #print(out2.shape)
        ### ----------------------------------------------

        return cls_scores # Dim: [batch_size, 10]



