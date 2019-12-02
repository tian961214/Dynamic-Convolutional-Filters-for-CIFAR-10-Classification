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
        '''
        out1=self.resnet18(imgs)
        out2=self.embeds(labels)
        out2=self.Weights_Generator_MLP(out2)
        print(self.DC.weight.shape)
        self.DC.weight=nn.Parameter(out2.expand(-1,64,-1,-1))

        print(self.DC.weight.shape)

        out1=self.DC(out1)
        out1=out1.view(-1,64)
        cls_scores=self.Class_Score_Predictor_MLP(out1)
        #single = self.embeds(imgs)
        #out2=self.Weights_Generator_MLP(single)
        #print(out2.shape)
        ### ----------------------------------------------

        return cls_scores # Dim: [batch_size, 10]
        '''
        cls_scores = torch.tensor((), dtype=torch.double, device=self.device)
        cls_scores = cls_scores.new_zeros((8, 10))
        for i in range(10):
            v = self.resnet18(imgs)
            w = self.Weights_Generator_MLP(self.embeds.weight.data[i])
            w = F.normalize(w, p=2, dim=0)
            w = w.view_as(self.DC.weight.data)

            self.DC.weight.data = w
            v_hat = self.DC(v)
            v_hat = v_hat / np.sqrt(v_hat.size()[0])

            re_v_hat = v_hat.view(v_hat.size(0), 64)
            #if (re_v_hat.size()[0] is not 64):
            #    re_v_hat.size()[0]=64
            #print('scores: ', self.Class_Score_Predictor_MLP(re_v_hat).size())

            cls_scores[:, [i]] = self.Class_Score_Predictor_MLP(re_v_hat).double()

        return cls_scores

