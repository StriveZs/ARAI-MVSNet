import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import *


def compute_depth(feats, proj_mats, depth_samps, cost_reg, lamb, is_training=False):
    '''

    :param feats: [(B, C, H, W), ] * num_views
    :param proj_mats: [()]
    :param depth_samps: [B, 16, 128, 160]
    :param cost_reg:
    :param lamb:
    :return:
    '''

    proj_mats = torch.unbind(proj_mats, 1)  # 每个图像对应的投影矩阵 [B, 3, 4, 4]
    num_views = len(feats) # 视图数 3
    num_depth = depth_samps.shape[1] # 深度假设平面个数

    assert len(proj_mats) == num_views, "Different number of images and projection matrices"

    ref_feat, src_feats = feats[0], feats[1:] # 参考图像特征，源图像特征
    ref_proj, src_projs = proj_mats[0], proj_mats[1:] # 参考图像投影矩阵，源图像投影矩阵

    ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) # [B, C, D, H/(4/2/1), W/(4/2/1)]
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume # 计算方式等价于MVSNet中的方式

    #todo optimize impl
    for src_fea, src_proj in zip(src_feats, src_projs):
        # 这部分本来是在数据集加载时完成的
        src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
        src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])

        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        # 单应性变换
        warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_samps) # 单应性变换
        # volume 集成 cost volume cost metric公式实现
        if is_training:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        else:
            volume_sum += warped_volume
            volume_sq_sum += warped_volume.pow_(2) #in_place method
        del warped_volume
    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2)) # [B, C, D, W/(4/2/1), H/(4/2/1)]

    prob_volume_pre = cost_reg(volume_variance).squeeze(1)  # 代价体正则化阶段  正则化的cost volume [B, D, W/(4/2/1), H/(4/2/1)]
    prob_volume = F.softmax(prob_volume_pre, dim=1) # 计算softmax得到概率体 [B, D, W/(4/2/1), H/(4/2/1)]
    depth = depth_regression(prob_volume, depth_values=depth_samps) # 回归得到深度图 [B, W/(4/2/1), H/(4/2/1)]
    # 光度一致性
    with torch.no_grad():
        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                            stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                              dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=num_depth - 1)
        prob_conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
    # 这部分对应ATV公式中: variance的计算  sampe_variance为L差值的平方  exp_variance为P
    # t = depth.unsqueeze(1) # [B, 1, W/(4/2/1), H/(4/2/1)]
    # t1 = depth_samps # [B, D, W/(4/2/1), H/(4/2/1)]
    # t2 = t1 - t # [B, D, W/(4/2/1), H/(4/2/1)]  这里的减法 应该是估计深度在D维度上进行复制到和depth_samps相同的维度之后，在执行相减，注意是复制就是重复了D次
    samp_variance = (depth_samps - depth.unsqueeze(1)) ** 2 # [B, D, W/(4/2/1), H/(4/2/1)]  ** 表示乘方操作  深度假设平面 减去 估计深度图的数值  每个位置上差值的平方
    exp_variance = lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5 # [B, W/(4/2/1), H/(4/2/1)]  偏差 * 概率值 再求和 * 超参数

    return {"depth": depth, "confidence": prob_conf, 'variance': exp_variance}

class ARAI_MVSNet(nn.Module):
    def __init__(self, lamb=1.5, stage_configs=[64, 32, 8], base_chs=[8, 8, 8, 8], feat_ext_ch=8):
        super(ARAI_MVSNet, self).__init__()

        self.stage_configs = stage_configs
        self.base_chs = base_chs
        self.lamb = lamb
        self.num_stage = len(stage_configs)
        self.ds_ratio = {"stage1": 8.0,
                         "stage2": 4.0,
                         "stage3": 2.0,
                         "stage4": 1.0
                         }

        self.feature_extraction = ASPFNet(base_channels=feat_ext_ch, num_stage=self.num_stage,)

        self.adaptive_depth_range = SCNet(in_channels=3, base_channels=self.base_chs[1])

        self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature_extraction.out_channels[i],
                                                             base_channels=self.base_chs[i]) for i in range(self.num_stage)])

    def forward(self, imgs, proj_matrices, depth_values, Flow1, Flow2):
        features = []
        for nview_idx in range(imgs.shape[1]):
            img = imgs[:, nview_idx]
            features.append(self.feature_extraction(img))
        ref_img = imgs[:, 0]

        outputs = {}
        depth, cur_depth, exp_var, confidence = None, None, None, None

        if Flow1 and Flow2 == False:
            self.num_stage = 2
        elif Flow1 and Flow2:
            self.num_stage = 4
        else:
            self.num_stage = 1

        for stage_idx in range(self.num_stage): # 分阶段进行
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.ds_ratio["stage{}".format(stage_idx + 1)]
            cur_h = img.shape[2] // int(stage_scale)
            cur_w = img.shape[3] // int(stage_scale)

            if depth is not None:
                cur_depth = depth.detach()
                exp_var = exp_var.detach()

                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                          [cur_h, cur_w], mode='bilinear')
                exp_var = F.interpolate(exp_var.unsqueeze(1), [cur_h, cur_w], mode='bilinear')

                if stage_idx == 1:
                    confidence = F.interpolate(confidence.unsqueeze(1), [cur_h, cur_w], mode='bilinear')

            else:
                cur_depth = depth_values

            # adaptive depth range prediction
            if stage_idx == 1:
                cur_depth = adaptive_depth_range_prediction(adr=self.adaptive_depth_range,
                                                         exp_var=exp_var,
                                                         depth_values=depth_values,
                                                         ref_img=ref_img,
                                                         cur_depth=cur_depth,
                                                         confidence=confidence,
                                                         ndepth=self.stage_configs[stage_idx])
            # adaptive_depth_interval_adjustment
            depth_range_samples = adaptive_depth_interval_adjustment(cur_depth=cur_depth,
                                                            exp_var=exp_var,
                                                            ndepth=self.stage_configs[stage_idx],
                                                            dtype=img[0].dtype,
                                                            device=img[0].device,
                                                            shape=[img.shape[0], cur_h, cur_w])


            outputs_stage = compute_depth(features_stage, proj_matrices_stage,
                                          depth_samps=depth_range_samples,
                                          cost_reg=self.cost_regularization[stage_idx],
                                          lamb=self.lamb,
                                          is_training=self.training)

            depth = outputs_stage['depth']
            exp_var = outputs_stage['variance']
            confidence = outputs_stage['confidence']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage

        return outputs

