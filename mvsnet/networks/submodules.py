import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-12

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

# fixme: ADRP
def adaptive_depth_range_prediction(adr, exp_var, depth_values, ref_img, cur_depth, confidence, ndepth):
    B = cur_depth.shape[0]
    # K = 10.0
    # cur_depth = cur_depth.mul(confidence)
    # depth_min = depth_values[:, 0]  # (B,)
    # depth_max = depth_values[:, -1]
    depth_min = torch.min(cur_depth.view(B, -1), dim=1).values.view(B, -1)
    depth_max = torch.max(cur_depth.view(B, -1), dim=1).values.view(B, -1)
    index_depth_min = torch.argmin(cur_depth.view(B, -1), dim=1, keepdim=True)
    index_depth_max = torch.argmax(cur_depth.view(B, -1), dim=1, keepdim=True)
    min_sigmas, max_sigmas = [], []
    for i in range(B):
        min_sigma = exp_var.view(B, -1)[i][index_depth_min[i]]
        min_sigmas.append(min_sigma)
        max_sigma = exp_var.view(B, -1)[i][index_depth_max[i]]
        max_sigmas.append(max_sigma)
    min_sigmas = torch.cat(min_sigmas, dim=0).view(B, -1)
    max_sigmas = torch.cat(max_sigmas, dim=0).view(B, -1)


    coefficient = adr(ref_img, cur_depth)
    coeffic_abs = torch.abs(coefficient)
    alpha, beta = coefficient[:, 0].view(B, -1), coefficient[:, 1].view(B, -1)
    # coefficient normalization
    coe_range = torch.sum(coeffic_abs, dim=1, keepdim=True)
    # exp_var [B, 1, 256, 320]
    alpha = (alpha / coe_range)
    beta = (beta / coe_range)
    t = alpha * min_sigmas
    new_depth_min = depth_min + alpha * min_sigmas
    new_depth_max = depth_max + beta * max_sigmas
    new_depth_values = torch.cat([new_depth_min, new_depth_max], dim=1)
    # new_interval = (new_depth_max - new_depth_min) / (float(ndepth) - 1.0)

    # new_depth_values = torch.arange(0, ndepth, device=features.device, dtype=features.dtype, requires_grad=False).reshape(1, -1) * new_interval

    return new_depth_values

# fixme: ADIA
def adaptive_depth_interval_adjustment(cur_depth, exp_var, ndepth, device, dtype, shape):
    if cur_depth.dim() == 2:
        #must be the first stage
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1.0)  # (B, )
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) # (B, D, H, W)  生成D个H×W的平面，每个平面的数值对应当前的深度假设平面的数值
    else:
        # cur_depth Stage2: [B, 1, 128, 160] 2X interpolate  [B, 1, 256, 320]  Stage3: [B, 1, 256, 320] 2X interpolate [B, 1, 512, 640]
        # exp_var Stage2: [B, 1, 128, 160] 2X interpolate  [B, 1, 256, 320]  Stage3: [B, 1, 256, 320] 2X interpolate [B, 1, 512, 640]
        low_bound = -torch.min(cur_depth, exp_var) # [B, 1, 256, 320] / [B, 1, 512, 640] 计算
        high_bound = exp_var # [B, 1, 256, 320] / [B, 1, 512, 640]

        # assert exp_var.min() >= 0, exp_var.min()
        assert ndepth > 1
        step = (high_bound - low_bound) / (float(ndepth) - 1) # [B, 1, 512, 640]
        new_samps = []
        mean_samps = []
        mid_value = None
        if ndepth % 2 == 0:
            mid_value = ndepth // 2.0 - 0.5
        else:
            mid_value = ndepth / 2.0
        for i in range(int(ndepth)):
            new_samps.append(cur_depth + low_bound + step * i + eps)
            mean_samps.append(cur_depth + low_bound + step * mid_value + eps)
            # equal_interval_samples = cur_depth + low_bound + step * i + eps

        # depth_range_samples = torch.cat(new_samps, 1)
        equal_interval_samples = torch.cat(new_samps, 1) # [B, 8, 512, 640] 8为depth number
        mean_interval_samples = torch.cat(mean_samps, 1) # [B, 8, 512, 640] 8为depth number

        z_source = torch.abs(equal_interval_samples - mean_interval_samples) / exp_var # [B, 8, 512, 640]
        adaptive_interval_offset = torch.softmax(z_source, dim=1) * step # [B, 8, 512, 640]
        depth_range_samples = equal_interval_samples + adaptive_interval_offset
        # assert depth_range_samples.min() >= 0, depth_range_samples.min()
    return depth_range_samples

def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

class Conv2dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class AtrousConv2dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(AtrousConv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Conv3dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Deconv3dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(Deconv2dBlock, self).__init__()

        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                                   bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2dUnit(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                               bn=bn, relu=relu, bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

#fixme: Atrous Spatial Pyramid Feature Extraction Network
class ASPFNet(nn.Module):
    def __init__(self, base_channels, num_stage=4):
        super(ASPFNet, self).__init__()

        self.base_channels = base_channels
        self.num_stage = num_stage
        # fixme: Encoder
        # stage 1:
        self.conv0 = Conv2dUnit(3, base_channels, 3, 1, padding=1)

        self.conv0_1 = nn.Sequential(
            AtrousConv2dUnit(base_channels, base_channels, 3, 1, dilation=2),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        self.conv0_2 = nn.Sequential(
            AtrousConv2dUnit(base_channels, base_channels, 3, 1, dilation=3),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv0_3 = Conv2dUnit(base_channels * 2, base_channels, 3, 1, padding=1)

        # stage 2:
        self.conv1 = Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2)

        self.conv1_1 = nn.Sequential(
            AtrousConv2dUnit(base_channels * 2, base_channels * 2, 3, 1, dilation=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        self.conv1_2 = nn.Sequential(
            AtrousConv2dUnit(base_channels * 2, base_channels * 2, 3, 1, dilation=3),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv1_3 = Conv2dUnit(base_channels * 4, base_channels * 2, 3, 1, padding=1)

        # stage 3:

        self.conv2 = Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2)

        self.conv2_1 = nn.Sequential(
            AtrousConv2dUnit(base_channels * 4, base_channels * 4, 3, 1, dilation=2),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.conv2_2 = nn.Sequential(
            AtrousConv2dUnit(base_channels * 4, base_channels * 4, 3, 1, dilation=3),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.conv2_3 = Conv2dUnit(base_channels * 8, base_channels * 4, 3, 1, padding=1)

        # stage 4:

        self.conv3 = Conv2dUnit(base_channels * 4, base_channels * 8, 5, stride=2, padding=2)

        self.conv3_1 = nn.Sequential(
            AtrousConv2dUnit(base_channels * 8, base_channels * 8, 3, 1, dilation=2),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )
        self.conv3_2 = nn.Sequential(
            AtrousConv2dUnit(base_channels * 8, base_channels * 8, 3, 1, dilation=3),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )

        self.conv3_3 = Conv2dUnit(base_channels * 16, base_channels * 8, 3, 1, padding=1)

        self.out1 = nn.Conv2d(base_channels * 8, base_channels * 8, 1, bias=False)
        self.out_channels = [8 * base_channels]

        # fixme: Decoder

        self.deconv0 = Deconv2dBlock(base_channels * 8, base_channels * 4, 3)
        self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)
        self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3)

        self.out2 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out3 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
        self.out4 = nn.Conv2d(base_channels, base_channels, 1, bias=False)

        self.out_channels.append(4 * base_channels)
        self.out_channels.append(2 * base_channels)
        self.out_channels.append(base_channels)

    def forward(self, x):
        # stage 1:
        conv0 = self.conv0(x)
        conv0_1 = self.conv0_1(conv0)
        conv0_2 = self.conv0_2(conv0)
        conv0 = self.conv0_3(torch.cat([conv0_1, conv0_2], 1))

        # stage 2:
        conv1 = self.conv1(conv0)
        conv1_1 = self.conv1_1(conv1)
        conv1_2 = self.conv1_2(conv1)
        conv1 = self.conv1_3(torch.cat([conv1_1, conv1_2], 1))

        # stage 3:
        conv2 = self.conv2(conv1)
        conv2_1 = self.conv2_1(conv2)
        conv2_2 = self.conv2_2(conv2)
        conv2 = self.conv2_3(torch.cat([conv2_1, conv2_2], 1))

        # stage 4:
        conv3 = self.conv3(conv2)
        conv3_1 = self.conv3_1(conv3)
        conv3_2 = self.conv3_2(conv3)
        conv3 = self.conv3_3(torch.cat([conv3_1, conv3_2], 1))

        intra_feat = conv3
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out

        intra_feat = self.deconv0(conv2, intra_feat)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = self.deconv1(conv1, intra_feat)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        intra_feat = self.deconv2(conv0, intra_feat)
        out = self.out4(intra_feat)
        outputs["stage4"] = out

        return outputs


# fixme Scalar Calculation Network
class SCNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(SCNet, self).__init__()
        self.i_h = 512
        self.i_w = 640

        self.conv0 = nn.Sequential(
            Conv2dUnit(in_channels+1, base_channels * 4, 3, 1, padding=1),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.conv1 = Conv2dUnit(base_channels * 4, base_channels * 2, 3, 2, padding=1)

        self.conv2 = Conv2dUnit(base_channels * 2, base_channels * 2, 1, 1)

        self.conv3 = Conv2dUnit(base_channels * 2, base_channels * 2, 1, 1)

        self.conv4 = nn.Sequential(
            Conv2dUnit(base_channels * 2, base_channels, 3, 2, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )

        self.maxpool1 = nn.AdaptiveMaxPool2d((self.i_h // 16, self.i_w // 16))
        self.maxpool2 = nn.AdaptiveMaxPool2d((self.i_h // 16, self.i_w // 16))

        self.linear1 = nn.Sequential(
            nn.Linear(base_channels * self.i_h // 16 * self.i_w // 16, 16),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Linear(16, 2)
        # self.linear2 = nn.Sequential(
        #     nn.Linear(16, 2),
        #     nn.ReLU(inplace=True)
        # )


    def forward(self, img, depth): # feat [B, 16, 256, 320]  depth [B, 1, 256, 320]
        b, img_h, img_w = img.shape[0], img.shape[2], img.shape[3]
        d_h, d_w = depth.shape[2], depth.shape[3]
        if img_h != self.i_h and img_w != self.i_w:
            img = F.interpolate(img, (self.i_h, self.i_w), mode='bilinear')
        if d_h != self.i_h and d_w != self.i_w:
            depth = F.interpolate(depth, (self.i_h, self.i_w), mode='bilinear')
        x = torch.cat([img, depth],dim=1) # [B, 17, 256, 320]
        x = self.conv0(x) # [B, 32, 256, 320]
        x1 = self.conv1(x) # [B, 16, 128, 160]
        x = self.conv2(x1) # [B, 16, 128, 160]
        x = self.conv3(x) # [B, 16, 128, 160]
        x = x + x1 # [B, 16, 128, 160]
        x = self.conv4(x) # [B, 8, 64, 80]
        x2 = self.maxpool1(depth) # [B, 1, 32, 40]
        x = self.maxpool2(x) # [B, 1, 32, 40]
        x = torch.mul(x, x2) # [B, 1, 32, 40]
        x = x + x2 # [B, 8, 32, 40]
        x = x.view(b, -1) # [B, 8 * 32 * 40]
        x = self.linear1(x)
        x = self.linear2(x)

        return x




# fixme: 3D UNet
class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, base_channels, padding=1)

        self.conv1 = Conv3dUnit(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3dUnit(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3dUnit(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3dUnit(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3dUnit(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3dUnit(base_channels * 8, base_channels * 8, padding=1)

        self.deconv7 = Deconv3dUnit(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.deconv8 = Deconv3dUnit(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.deconv9 = Deconv3dUnit(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x) # [1, 8, 32, 128, 160] [1, 8, 8, 256, 320]
        conv2 = self.conv2(self.conv1(conv0)) # [1, 16, 16, 64, 80] [1, 16, 4, 128, 160]
        conv4 = self.conv4(self.conv3(conv2)) # [1, 32, 8, 32, 40] [1, 32, 2, 64, 80]
        x = self.conv6(self.conv5(conv4)) # [1, 64, 4, 16, 20] [1, 64, 1, 32, 40]
        t = self.deconv7(x) # [1, 32, 4, 32, 40]
        x = conv4 + t # [1, 32, 4, 32, 40]
        t = self.deconv8(x) # [1, 16, 8, 64, 80]
        x = conv2 + t # [1, 16, 8, 64, 80]
        t = self.deconv9(x) # [1, 8, 16, 128, 160]
        x = conv0 + t # [1, 8, 16, 128, 160]
        x = self.prob(x)
        return x
