# Global imports
## general imports
import time
import copy
import collections
import numpy as np
## torch and torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
## typing
from typing import Dict
from torch import Tensor

import sys
sys.path.append("./src")
# Package imports
from losses.ntxent_loss import SafeNTXentLoss
from models.backbone import ArmNextHead,ArmRes5Head,ResnetHead

# Applies "RoI Pooling" to entire feature map extent
class ScenePool(nn.Module):
    def __init__(self, featmap_name, output_size):
        super().__init__()
        # Store params
        self.featmap_name = featmap_name
        self.output_size = output_size

    def forward(self, x: Dict[str, Tensor]):
        f = x[self.featmap_name]
        p = F.adaptive_max_pool2d(f, self.output_size)
        return p


# Learnable scene embedding in R^d
class SceneEmbeddingHead(nn.Module):
    def __init__(self, reid_head, emb_head, pool_size=56):
        super().__init__()
        # Pooling layer
        self.image_roi_pool = ScenePool(featmap_name='feat_res4', output_size=pool_size)
        if(isinstance(reid_head,ArmNextHead) or isinstance(reid_head,ArmRes5Head) or isinstance(reid_head,ResnetHead)):
            #不复制
            print("ArmNextHead without copy")
            self.image_reid_head=reid_head
        else:
            #直接复制
            self.image_reid_head = copy.deepcopy(reid_head)
        self.image_emb_head = copy.deepcopy(emb_head)
        ## We don't use NAE embedding norms for the scene embeddings
        self.image_emb_head.rescaler = None

    def forward(self, x: Dict[str, Tensor]):
        x = self.image_roi_pool(x)
        x = self.image_reid_head(x)
        x = self.image_emb_head(x)[0]
        return x


# BatchNorm1d clone which handles different input shapes
class BatchNorm1d(nn.Module):
    def __init__(self, d):
        super(BatchNorm1d, self).__init__()
        self.norm = nn.BatchNorm1d(d)

    def forward(self, x: Tensor) -> Tensor:
        ## if len(shape) == 2, apply norm directly
        if len(x.size()) == 2:
            return self.norm(x)
        ## if len(shape) == 3, reshape to len(shape) == 2 before and after applying norm
        elif len(x.size()) == 3:
            i, q, d = x.size()
            return self.norm(x.reshape(-1, d)).reshape(i, q, d)
        else:
            raise Exception


# Module to fuse query and scene features
class QueryImageFuser(nn.Module):
    def __init__(self, d, gfn_query_mode=None,
            gfn_activation_mode='se', se_temp=0.2):
        super().__init__()
        self.gfn_query_mode = gfn_query_mode
        self.gfn_activation_mode = gfn_activation_mode
        self.se_temp = se_temp
        if self.gfn_activation_mode != 'identity':
            self.norm = BatchNorm1d(d)

    def se(self, x: Tensor, y: Tensor, p:bool=False) -> Tensor:
        if p:
            return y.unsqueeze(1) * torch.sigmoid(x.unsqueeze(0) / self.se_temp)
        else:
            return y * torch.sigmoid(x / self.se_temp)

    def forward(self, x: Tensor, y: Tensor, p:bool=False) -> Tensor:
        """
        x: query features
        y: scene features
        """
        z = torch.empty(0)
        # Apply fuser function
        ## sigmoidal excitation of scene features using query features
        if self.gfn_activation_mode == 'se':
            z = self.norm(self.se(
                F.normalize(x, p=2.0, dim=-1), 
                F.normalize(y, p=2.0, dim=-1), 
                p=p))
        ## add query and scene features
        elif self.gfn_activation_mode == 'sum':
            if p:
                z = self.norm(x.unsqueeze(0) + y.unsqueeze(1))
            else:
                z = self.norm(x + y)
        ## pass scene features directly through (control)
        elif self.gfn_activation_mode == 'identity':
            if p:
                z = y.unsqueeze(1).repeat(1, x.size(0), 1)
            else:
                z = y

        # Return fused features
        return z


class GalleryFilterNetwork(nn.Module):
    def __init__(self, roi_pool,gfn_head, reid_head, emb_head,
            emb_dim=256, temp=0.1, se_temp=0.2,
            filter_neg=True, gfn_query_mode='batch',
            gfn_activation_mode='se', gfn_scene_pool_size=56,
            pos_num_sample=1, neg_num_sample=1,
            reid_loss=None,
            fuse_batch_size=32, mode='combined', use_image_lut=False, device='cpu'):
        super().__init__()
        self.gfn_query_mode = gfn_query_mode
        self.query_roi_pool = roi_pool
        self.query_reid_head = reid_head
        self.query_emb_head = emb_head
        self.head = SceneEmbeddingHead(gfn_head, emb_head, pool_size=gfn_scene_pool_size)
        self.gfn_mode = mode
        if self.gfn_mode not in  ('image', 'separate'):
            self.fuser = QueryImageFuser(emb_dim,
                gfn_query_mode=gfn_query_mode, gfn_activation_mode=gfn_activation_mode, se_temp=se_temp)
        self.filter_neg = filter_neg
        self.fuse_batch_size = fuse_batch_size
        self.criterion = None
        self.use_image_lut = use_image_lut
        self.image_lut = {}
        self.pos_num_sample = pos_num_sample
        self.neg_num_sample = neg_num_sample
        self.device = device
        self.ntx_temp = temp
        self.gfn_head= gfn_head

        # Set up gfn_query_mode params
        self.reid_loss = reid_loss
        if self.gfn_query_mode == 'oim':
            ## need to use reid_loss here, otherwise pointer to reid_loss.lut gets lost
            self.query_unk = reid_loss.ignore_index


    def get_scene_emb(self, features: Dict[str, Tensor]) -> Tensor:
        scene_emb = self.head(features)
        return scene_emb

    def get_fused_features(self, query_features, query_image_features=None, gallery_image_features=None):
        # For 'separate' mode, we don't need to fuse features
        assert self.gfn_mode != 'separate'
        if self.gfn_mode == 'combined':
            Q, D = query_features.size()
            I = gallery_image_features.size(0)
            fused_features_list = []
            for _query_features in query_features.split(self.fuse_batch_size):
                q = _query_features.size(0)
                rep_query_features = _query_features.unsqueeze(1).repeat(1, I, 1).reshape(q*I, D)
                rep_gallery_image_features = gallery_image_features.unsqueeze(0).repeat(q, 1, 1).reshape(q*I, D)
                _fused_features = self.fuser(rep_query_features, rep_gallery_image_features)
                fused_features_list.append(_fused_features.cpu())
            fused_gallery_features = torch.cat(fused_features_list, dim=0).reshape(Q, I, D)
            #
            fused_query_features = self.fuser(query_features, query_image_features).cpu()
            fused_features = fused_query_features, fused_gallery_features 
        return fused_features

    @torch.jit.export
    def get_scores(self, source_box_feats: Tensor, source_img_feats: Tensor, target_img_feats: Tensor) -> Tensor:
        ## Leave source and target img feats alone
        if self.gfn_mode == 'separate':
            gfn_scores = torch.sigmoid(torch.mm(
                F.normalize(source_box_feats, p=2.0, dim=-1),
                F.normalize(target_img_feats, p=2.0, dim=-1).T,
            ))
        ## Combine source box feats into source img feats and target img feats
        elif self.gfn_mode == 'combined':
            query_features, query_image_features, gallery_image_features = source_box_feats, source_img_feats, target_img_feats
            fused_query_features = self.fuser(query_features, query_image_features)
            #
            gfn_scores_list = []
            for _gallery_image_features in gallery_image_features.split(self.fuse_batch_size):
                _alt_fused_features = self.fuser(query_features, _gallery_image_features, p=True)
                _gfn_scores = torch.sigmoid(torch.einsum('qd,iqd->qi',
                    F.normalize(fused_query_features.float(), p=2.0, dim=-1),
                    F.normalize(_alt_fused_features.float(), p=2.0, dim=-1),
                ))
                gfn_scores_list.append(_gfn_scores.cpu())
            gfn_scores = torch.cat(gfn_scores_list, dim=1)
        ## Use image feats only
        elif self.gfn_mode == 'image':
            gfn_scores = torch.sigmoid(torch.einsum('sd,td->st',
                F.normalize(source_img_feats.float(), p=2.0, dim=-1),
                F.normalize(target_img_feats.float(), p=2.0, dim=-1),
            ))
        else:
            raise Exception
        return gfn_scores

    @torch.jit.ignore
    def forward(self, features, targets, image_shapes):
        # Initialize here for torch.jit
        if self.criterion is None:
            self.criterion = SafeNTXentLoss(temperature=self.ntx_temp)
        # Produce labels
        is_known_dict = {}
        for t in targets:
            for pid, is_known in zip(t['person_id'].tolist(), t['is_known'].tolist()):
                is_known_dict[pid] = is_known
        #取出每张图片带有的ids 然后按图片顺序存为list
        query_pids_list = [t['person_id'] for t in targets]
        #取出是否为已知行人idS的标记并 然后按图片顺序变为list
        known_query_pids_list = [t['person_id'][t['is_known']] for t in targets]
        #生成图片的label
        gfn_image_labels = list(range(len(query_pids_list)))
        image_labels = torch.LongTensor(gfn_image_labels).to(self.device)

        #取出行人的box并标记为list
        gt_boxes = [t["boxes"].to(torch.float) if len(t['boxes']) > 0 else torch.empty(0, 4, dtype=torch.float, device=self.device) for t in targets]
        gt_labels = query_pids_list

        #将每张图片的ids拆出并转为set
        gfn_image_sets = [set(_query_pids.tolist()) for _query_pids in query_pids_list]
        I = len(targets)

        # Boxvar
        #表示person id 在所有已知id中的位置
        gfn_lut_label_list = [t['labels']-1 for t in targets]
        # Boxes
        boxes = gt_boxes
        # Labels
        #取出每张图片的ids并且转化为一个超长的list
        query_pids = sum([qp.tolist() for qp in query_pids_list], [])
        met_num_list = query_pids
        #转化为tensor
        gfn_met_label_tsr = torch.LongTensor(met_num_list).to(self.device)

        # Image indices
        #len(query_pids)为一张图片中的pid的数量 i为图片的label 生成一个list表示该id属于哪个图片
        gfn_image_indices = sum([[i]*len(query_pids) for i, query_pids in enumerate(query_pids_list)], [])
        # Get OIM labels if we are using gfn_query_mode
        ## Make sure to subtract 1 to get correct indices, align with query_unk index
        gfn_lut_label_tsr = torch.cat(gfn_lut_label_list)

        # Prep known tsr
        #将known标记转化为一个list
        gfn_met_known_tsr = torch.BoolTensor([is_known_dict[l] for l in gfn_met_label_tsr.tolist()]).to(self.device)

        # Extract GT query features
        # 训练时直接使用真实框提取特征
        query_features = self.query_roi_pool(features, boxes, image_shapes)

        if(query_features.shape[0]==0):
            Q,D=0,0
        else:
            query_features = self.query_reid_head(query_features)
            query_emb, _ = self.query_emb_head(query_features)
            Q, D = query_emb.size()

        if Q == 0:
            gfn_loss = torch.tensor(0.0).to(self.device)
            gfn_loss_dict = {'gfn_loss': gfn_loss}
            gfn_metric_dict = {}
            return gfn_loss_dict, gfn_metric_dict

        # Extract image features
        scene_emb = self.head(features)
        I = scene_emb.size(0)

        # Additional label prep following (potential)
        gfn_image_idx_tsr = torch.LongTensor(gfn_image_indices).to(self.device)

        # Image LUT
        if self.use_image_lut:
            #
            image_id_list = [target['image_id'].item() for target in targets]
            ## Fetch previous entries
            #找出该batch种所有已知的行人id
            query_label_union = [n for n in list(set.union(*gfn_image_sets)) if is_known_dict[n]]

            #采样数量，为图像数量，若已知的行人id少于图像数量，则采样行人id的数量
            num_sample = min(I, len(query_label_union))
            query_label_union = np.random.choice(query_label_union, size=(num_sample,), replace=False) 
            # chosen_idx should start at length of current list of image ids
            chosen_image_id_set = set()
            chosen_idx = I
            # Randomly shuffle LUT to avoid looking up same PIDs every time
            image_rand_idx = torch.randperm(len(self.image_lut)).tolist()
            image_lut_keys = list(self.image_lut.keys())
            # Storage lists
            lut_image_feat_list = []
            lut_query_image_idx_list = []
            lut_query_feat_list = []
            lut_query_label_list = []   
            lut_query_known_list = []
            lut_lut_label_list = []
            lut_query_label_set_list = []
            pos_query_counter = collections.Counter()
            neg_query_counter = collections.Counter()
            # Get samples from LUT’
            #额外构建一个长期的lut，挖掘额外的正例和负例
            query_image_idx_sets = [set(known_query_pids.tolist()) for known_query_pids in known_query_pids_list]
            for query_idx, query_label in enumerate(query_label_union):
                for rand_idx in image_rand_idx:
                    image_id = image_lut_keys[rand_idx]
                    image_dict = self.image_lut[image_id]
                    image_query_label_set = image_dict['query_label_set']
                    #如果在lut中找到正例
                    if query_label in image_query_label_set:
                        if (image_id not in chosen_image_id_set) and (image_id not in image_id_list): 
                            #将lut中的正例加入本轮的比较
                            lut_image_feat_list.append(image_dict['image_feat'].to(self.device))
                            lut_query_feat_list.append(image_dict['query_feat'].to(self.device)) 
                            lut_query_label_list.append(image_dict['query_label'].to(self.device)) 
                            lut_query_known_list.append(image_dict['query_known'].to(self.device)) 
                            lut_lut_label_list.append(image_dict['lut_label'].to(self.device)) 
                            lut_query_label_set_list.append(image_dict['query_label_set']) 
                            ## This assumes nothing is discarded!!!
                            lut_query_image_idx_list.append(torch.LongTensor([chosen_idx]*len(image_dict['query_feat'])).to(self.device))
                            chosen_idx += 1
                            ## Add id to chosen set
                            chosen_image_id_set.add(image_id)
                            ## Break the loop here to save memory: sample only 1 LUT image per pid
                            pos_query_counter[query_label] += 1
                            if pos_query_counter[query_label] == self.pos_num_sample:
                                break
                # Add another sample: hard negative
                if self.neg_num_sample > 0:
                    query_image_idx_set = None
                    #查看每张图片中的已知行人id
                    for _query_image_idx_set in query_image_idx_sets:
                        #当前采样的query行人id在当前的set中 说明是正例
                        if query_label in _query_image_idx_set:
                            query_image_idx_set = _query_image_idx_set
                            break
                    #额外采样一个负例
                    if query_image_idx_set is None: 
                        print('WARNING: query_image_idx is None')
                    else:
                        for rand_idx in image_rand_idx:
                            image_id = image_lut_keys[rand_idx]
                            image_dict = self.image_lut[image_id]
                            image_query_label_set = image_dict['query_label_set']
                            if query_label not in image_query_label_set:
                                if len(query_image_idx_set.intersection(image_query_label_set)) > 0:
                                    if (image_id not in chosen_image_id_set) and (image_id not in image_id_list): 
                                        #
                                        lut_image_feat_list.append(image_dict['image_feat'].to(self.device))
                                        lut_query_feat_list.append(image_dict['query_feat'].to(self.device)) 
                                        lut_query_label_list.append(image_dict['query_label'].to(self.device)) 
                                        lut_query_known_list.append(image_dict['query_known'].to(self.device)) 
                                        lut_lut_label_list.append(image_dict['lut_label'].to(self.device)) 
                                        lut_query_label_set_list.append(image_dict['query_label_set']) 
                                        ## This assumes nothing is discarded!!!
                                        lut_query_image_idx_list.append(torch.LongTensor([chosen_idx]*len(image_dict['query_feat'])).to(self.device))
                                        chosen_idx += 1
                                        ## Add id to chosen set
                                        chosen_image_id_set.add(image_id)
                                        ## Break the loop here to save memory: sample only 1 LUT image per pid
                                        neg_query_counter[query_label] += 1
                                        if neg_query_counter[query_label] == self.neg_num_sample:
                                            break

            ## Store new entries in LUT
            for image_idx, image_id in enumerate(image_id_list):
                image_mask = gfn_image_idx_tsr == image_idx
                #表示图片中存在已知id
                if image_mask.sum().item() > 0:
                #取出存在已知id的图像的相关信息来构建image_lut
                    self.image_lut[image_id] = {
                        'image_feat': scene_emb[image_idx].unsqueeze(0).detach().cpu(),
                        'query_feat': query_emb[image_mask].detach().cpu(),
                        'query_label': gfn_met_label_tsr[image_mask].cpu(),
                        'query_known': gfn_met_known_tsr[image_mask].cpu(),
                        'lut_label': gfn_lut_label_tsr[image_mask].cpu(),
                        'query_label_set': gfn_image_sets[image_idx],
                    }
            ## Combine previous entries with current batch
            #将额外采样的正例和负例加入比较
            scene_emb = torch.cat([scene_emb, *lut_image_feat_list], dim=0)
            gfn_image_idx_tsr = torch.cat([gfn_image_idx_tsr, *lut_query_image_idx_list], dim=0)
            query_emb = torch.cat([query_emb, *lut_query_feat_list], dim=0)
            gfn_met_label_tsr = torch.cat([gfn_met_label_tsr, *lut_query_label_list], dim=0)
            gfn_met_known_tsr = torch.cat([gfn_met_known_tsr, *lut_query_known_list], dim=0)
            gfn_lut_label_tsr = torch.cat([gfn_lut_label_tsr, *lut_lut_label_list], dim=0)
            gfn_image_sets = gfn_image_sets + lut_query_label_set_list
            ## Assign new number of images, number of queries
            I = scene_emb.size(0)
            Q = query_emb.size(0)
            ## Update image labels
            gfn_image_labels = list(range(I))
            image_labels = torch.LongTensor(gfn_image_labels).to(self.device)

        # Modify GT query features if we are using oim
        if self.gfn_query_mode == 'oim':
            ## Make sure to subtract 1 to get correct indices, align with query_unk index
            #取出所有已知id的mask
            gfn_lut_pid_mask = gfn_lut_label_tsr != self.query_unk
            #query_emb为根据图像中所有的框取出的行人特征
            oim_emb = query_emb
            #将已知id的行人特征替换为re_id中的原型
            oim_emb[gfn_lut_pid_mask] = self.reid_loss.lut[gfn_lut_label_tsr[gfn_lut_pid_mask]].to(self.device)
            oim_nonzero_mask = ~((oim_emb==0).all(dim=1))
            gfn_lut_pid_mask = gfn_lut_pid_mask & oim_nonzero_mask
            query_emb[gfn_lut_pid_mask] = self.reid_loss.lut[gfn_lut_label_tsr[gfn_lut_pid_mask]].to(self.device)
            #取出存在已知id的行人特征
            query_emb = query_emb[gfn_met_known_tsr]
            gfn_met_label_tsr = gfn_met_label_tsr[gfn_met_known_tsr]
            gfn_image_idx_tsr = gfn_image_idx_tsr[gfn_met_known_tsr]
            gfn_met_known_tsr = gfn_met_known_tsr[gfn_met_known_tsr]
            Q = gfn_met_known_tsr.sum().item() 

        # Prep query and image features for GFN: detach during training
        ## Do not combine query features with any image features
        if self.gfn_mode == 'separate':
            gfn_query_features = query_emb
        ## Combine query features with the image against which they are to be compared
        elif self.gfn_mode == 'combined':
            # Get features combined with all possible target images
            gfn_query_features_list = self.fuser(query_emb, scene_emb, p=True)
            # Get features combined with their source image
            # I = scene_emb.size(0)
            # Q = query_emb.size(0)
            identity_mask = torch.zeros(I, Q, dtype=torch.bool).to(self.device)
            identity_mask.scatter_(0, gfn_image_idx_tsr.unsqueeze(0), True)
            #把每个query对应的scene结合的feature取出
            gfn_query_features = gfn_query_features_list[identity_mask]

        # new method for gfn miner
        ## build padded tensor of pids per image
        #构建一个（图片数*一张图片内最多pid数量的张量）
        gfn_max_len = max([len(s) for s in gfn_image_sets])
        gfn_image_pid_tsr = -torch.ones(len(gfn_image_sets), gfn_max_len, dtype=torch.long).to(self.device)
        _gfn_image_pid_mask = torch.zeros(len(gfn_image_sets), gfn_max_len, dtype=torch.bool).to(self.device)

        M = gfn_max_len
        for pid_set_idx, pid_set in enumerate(gfn_image_sets):
            pid_tsr = torch.LongTensor(list(pid_set))
            gfn_image_pid_tsr[pid_set_idx, :pid_tsr.size(0)] = pid_tsr
            _gfn_image_pid_mask[pid_set_idx, :pid_tsr.size(0)] = True
        #放入pid

        #无效部分填充负值
        empty_mask = gfn_image_pid_tsr==-1
        gfn_image_pid_tsr[empty_mask] = (-torch.arange(1, 1+empty_mask.sum().item())).to(self.device)

        ## positive pairs
        # gfn_image_pid_tsr.unsqueeze(2):(图片数,一张图片内最多pid数量的张量,1)
        #gfn_met_label_tsr:(id总数)
        #image_pid_match_mask(图片数,一张图片内最多pid数量的张量,id总数) 变为one-hot编码
        # 再通过any（dim=1）变为（图片数，id总数）表示每张图片内的id在所有id中的的mask 
        image_pid_match_mask = (gfn_image_pid_tsr.unsqueeze(2) == gfn_met_label_tsr).any(dim=1)

        ## mask: for negatives, images must share at least 1 'p' pid in common
        ## 1.找出该batch里所有具有相同id的图片并且标记（这里称为相关图片） 得到 I*I的mask张量
        ## 2.根据张量 和 id属于哪个图片的长向量 运算出 每一张图片有关的id（自己的id+相关下的所有id）得到 I*id总数量的mask张量
        if self.filter_neg:
            # Build mask of which images share at least one pid with another image
            gfn_image_match_mask = (gfn_image_pid_tsr.view(I, 1, M, 1) == gfn_image_pid_tsr.view(1, I, 1, M)).reshape(I, I, -1).any(dim=2)
            # Index mask according to gfn_image_pid_tsr, gfn_met_label_tsr
            #gfn_image_idx_tsr:pid属于哪张图片的list

            image_pid_share_mask = gfn_image_match_mask[:, gfn_image_idx_tsr]
            # image_pid_share_mask : (I,id总数) 
            known_pid_mask = gfn_met_known_tsr.unsqueeze(0).repeat(image_pid_share_mask.size(0), 1)

        ## Build GFN pairs
        #取出不属于当前图片的id的mask
        _image_pid_diff_mask = ~image_pid_match_mask
        if self.filter_neg:
            #                        不在当前图片出现过的id       无关图片的id              不在当前图片出现的id     有关图片的id            已知id
            image_pid_diff_mask = (_image_pid_diff_mask & ~image_pid_share_mask) | (_image_pid_diff_mask & image_pid_share_mask & known_pid_mask)
            # 等于删除了相关图片中的未知id，保证一定是不同的id 这里无关图片的id部分没添加已知id的条件 比较奇怪
        else:
        #                            不在当前图片出现过的id 
            image_pid_diff_mask = _image_pid_diff_mask

        if self.gfn_mode == 'separate':
            # Get pairs
            gfn_a1, gfn_p = torch.where(image_pid_match_mask)
            gfn_a2, gfn_n = torch.where(image_pid_diff_mask)
            #生成坐标
            indices_tuple = (gfn_a1, gfn_p, gfn_a2, gfn_n)

            # Compute GFN Loss
            gfn_loss = self.criterion(scene_emb, image_labels,
                ref_emb=gfn_query_features, ref_labels=gfn_met_label_tsr, indices_tuple=indices_tuple)
                
        elif self.gfn_mode == 'combined':
            gfn_image_features_list = gfn_query_features_list.permute(1, 0, 2)
            match_mask_list, diff_mask_list = [], []
            for query_idx in range(Q):
                _image_mask = torch.zeros_like(image_pid_match_mask)
                _image_mask[:, query_idx] = True
                _image_pid_match_mask = image_pid_match_mask.clone() & _image_mask
                _image_pid_diff_mask = image_pid_diff_mask.clone() & _image_mask
                match_mask_list.append(_image_pid_match_mask)
                diff_mask_list.append(_image_pid_diff_mask)
            if len(match_mask_list) > 0:
                match_mask = torch.stack(match_mask_list, dim=0).reshape(Q, -1)
                diff_mask = torch.stack(diff_mask_list, dim=0).reshape(Q, -1)
            else:
                match_mask = torch.zeros(Q, 0)
                diff_mask = torch.zeros(Q, 0)
            flat_gfn_image_features = gfn_image_features_list.permute(1, 0, 2).reshape(-1, D)
            flat_gfn_image_labels = image_labels.unsqueeze(1).repeat(1, Q).view(-1)
            gfn_a1, gfn_p = torch.where(match_mask)
            gfn_a2, gfn_n = torch.where(diff_mask)
            gfn_same_pairs = (gfn_a1, gfn_p, gfn_a2, gfn_n)
            gfn_loss = self.criterion(gfn_query_features, gfn_met_label_tsr,
                ref_emb=flat_gfn_image_features, ref_labels=flat_gfn_image_labels, indices_tuple=gfn_same_pairs)
        elif self.gfn_mode == 'image':
            # Get pairs
            gfn_a1, gfn_p = torch.where(gfn_image_match_mask)
            gfn_a2, gfn_n = torch.where(~gfn_image_match_mask)
            indices_tuple = (gfn_a1, gfn_p, gfn_a2, gfn_n)

            # Compute GFN Loss
            gfn_loss = self.criterion(scene_emb, image_labels, indices_tuple=indices_tuple)
        else:
            raise NotImplementedError

        # Store loss and additional metrics in dicts
        gfn_loss_dict = {'gfn_loss': gfn_loss}
        gfn_metric_dict = {}

        # Return losses, metrics
        return gfn_loss_dict, gfn_metric_dict
