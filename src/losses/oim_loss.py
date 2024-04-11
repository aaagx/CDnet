# Global imports
import torch
import torch.nn.functional as F
from torch import nn
import sys
sys.path.append("./src")
from losses.protoNorm import PrototypeNorm1d

# Refactored OIM loss with safe float16 computation
class OIMLossSafe(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, use_cq=True):
        super(OIMLossSafe, self).__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None

    def forward(self, inputs, label):
        # Normalize inputs
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
        _loss_oim = F.cross_entropy(projected, label, ignore_index=self.ignore_index, reduction='none')
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()

        # Compute LUT and CQ updates
        with torch.no_grad():
            targets = label
            for x, y in zip(inputs, targets):
                if y < len(self.lut):
                    self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                elif self.cq is not None:
                    self.cq[self.header_cq] = x
                    self.header_cq = (self.header_cq + 1) % self.cq.size(0)

        # Return loss
        return loss_oim

# Refactored OIM loss with safe float16 computation
class LOIMLossSafe(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar,use_cq=True):
        super(LOIMLossSafe, self).__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.norm=PrototypeNorm1d(self.num_features)
        self.sigma1=0.2
        self.sigma2=0.8
        
        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None


    def forward(self, inputs, label,ious):
        # Normalize inputs
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce

        _loss_oim = F.cross_entropy(projected, label, ignore_index=self.ignore_index, reduction='none')
        # #ROIM
        # projected1=torch.argmax(projected,dim=-1)
        # label1=torch.zeros(projected.shape,device=label.device)
        # for i in range(label1.shape[0]):
        #     label1[i][label[i]]=1
        # label1=F.softmax(label1)
        # # print(projected1)
        # # print(label1)
        # _loss_oim1 = F.cross_entropy(label1,projected1,ignore_index=self.ignore_index, reduction='none')
        # loss_oim = (_loss_oim / _loss_oim.size(0)).sum()/self.sigma1+(_loss_oim1/_loss_oim1.size(0)).sum()/self.sigma2+torch.log(torch.tensor(self.sigma1))+torch.log(torch.tensor(self.sigma2))
        
        
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()
        with torch.no_grad():
            targets = label
            if(ious.mean()<0.2):
                for x, y in zip(inputs, targets):
                    if y < len(self.lut):
                        self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                    elif self.cq is not None:
                        self.cq[self.header_cq] = x
                        self.header_cq = (self.header_cq + 1) % self.cq.size(0)
                else:
                # Compute LUT and CQ updates
                    with torch.no_grad():
                        for x, y, s in zip(inputs, targets, ious.view(-1)):
                            if y < len(self.lut):
                                self.lut[y] = F.normalize((1.0 - s) * self.lut[y] + s*x, dim=0)
                            elif self.cq is not None:
                                self.cq[self.header_cq] = x
                                self.header_cq = (self.header_cq + 1) % self.cq.size(0)
            # Return loss
        return loss_oim
    


    # Refactored OIM loss with safe float16 computation
class ArcOIMLossSafe(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar,use_cq=True, margin_angle=0.5, easy_margin=False):
        super(ArcOIMLossSafe, self).__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.norm=PrototypeNorm1d(self.num_features)
        self.sigma1=0.2
        self.sigma2=0.8
        #arcface部分
        self.margin_angle = torch.tensor(margin_angle)  # 角度m
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(self.margin_angle)
        self.sin_m = torch.sin(self.margin_angle)
        self.th = torch.cos(torch.pi - self.margin_angle)
        self.mm = self.sin_m * margin_angle

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None


    def forward(self, inputs, label,ious):
        # Normalize inputs
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # greater_than_one = projected > 1

        # # 计算大于1的元素的个数
        # num_elements_greater_than_one = torch.sum(greater_than_one.int())
        # print(num_elements_greater_than_one.item())  # 使用.item()来获取Python数字
        # greater_than_one = projected < -1

        # # 计算于1的元素的个数
        # num_elements_greater_than_one = torch.sum(greater_than_one.int())
        # print(num_elements_greater_than_one.item())  # 使用.item()来获取Python数字

        # projected=F.normalize(projected, p=2, dim=-1)
        #ArcFace
        cosine = projected
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ)cos(m) - sin(θ)sin(m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)


        # Multiply projections by (inverse) temperature scalar
        output *= self.oim_scalar
        
        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce

        _loss_oim = F.cross_entropy(output, label, ignore_index=self.ignore_index, reduction='none')

        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()
        with torch.no_grad():
            targets = label
            if(ious.mean()<0.2):
                for x, y in zip(inputs, targets):
                    if y < len(self.lut):
                        self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                    elif self.cq is not None:
                        self.cq[self.header_cq] = x
                        self.header_cq = (self.header_cq + 1) % self.cq.size(0)
                else:
                # Compute LUT and CQ updates
                    with torch.no_grad():
                        for x, y, s in zip(inputs, targets, ious.view(-1)):
                            if y < len(self.lut):
                                self.lut[y] = F.normalize((1.0 - s) * self.lut[y] + s*x, dim=0)
                            elif self.cq is not None:
                                self.cq[self.header_cq] = x
                                self.header_cq = (self.header_cq + 1) % self.cq.size(0)
            # Return loss
        return loss_oim
    





    # Refactored OIM loss with safe float16 computation
class circleLOIMLossSafe(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar,use_cq=True,m=0.1):
        super(circleLOIMLossSafe, self).__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.norm=PrototypeNorm1d(self.num_features)
        self.sigma1=0.2
        self.sigma2=0.8
        #circle部分
        self.m=m
        self.o_p=1+self.m
        self.o_n=-self.m
        self.delta_p=1-self.m
        self.delta_n=self.m

        
        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None


    def forward(self, inputs, label,ious):
        # Normalize inputs
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        alpha_p=self.o_p-outputs_labeled[bad_pos_idx, bad_label]
        alpha_n=outputs_labeled-self.o_n
        alpha_p[alpha_p <= 0]=1e-6
        alpha_n[alpha_n <= 0]=1e-6

        #获取所有负对信息
        negative_mask=torch.ones(outputs_labeled.shape,dtype=torch.bool)
        negative_mask[bad_pos_idx,bad_label]=0
        #对负对应用circle loss的方法
        outputs_labeled[negative_mask]-=self.delta_n
        outputs_labeled[negative_mask]*=alpha_n[negative_mask]
        outputs_labeled[:, bad_lut_mask] = -1

        #对正对应用circle loss的方法
        outputs_labeled[bad_pos_idx,bad_label]-self.delta_p
        outputs_labeled[bad_pos_idx,bad_label]*=alpha_p
        zero_mask=outputs_labeled[bad_pos_idx, bad_label]<=0
        outputs_labeled[bad_pos_idx, bad_label][zero_mask] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())

            alpha_n2 = outputs_unlabeled-self.o_n
            alpha_n2 = outputs_unlabeled[ outputs_unlabeled <=0]=1e-6
            outputs_unlabeled*=alpha_n2

            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss


        _loss_oim = F.cross_entropy(projected, label, ignore_index=self.ignore_index, reduction='none')
      
        
        
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()
        with torch.no_grad():
            targets = label
            if(ious.mean()<0.2):
                for x, y in zip(inputs, targets):
                    if y < len(self.lut):
                        self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                    elif self.cq is not None:
                        self.cq[self.header_cq] = x
                        self.header_cq = (self.header_cq + 1) % self.cq.size(0)
                else:
                # Compute LUT and CQ updates
                    with torch.no_grad():
                        for x, y, s in zip(inputs, targets, ious.view(-1)):
                            if y < len(self.lut):
                                self.lut[y] = F.normalize((1.0 - s) * self.lut[y] + s*x, dim=0)
                            elif self.cq is not None:
                                self.cq[self.header_cq] = x
                                self.header_cq = (self.header_cq + 1) % self.cq.size(0)
            # Return loss
        return loss_oim