import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath
from pointnet2_ops import pointnet2_utils

class FeatureExtractors(torch.nn.Module):
    def __init__(self, device, 
                 rgb_backbone_name = 'vit_base_patch8_224_dino.dino', out_indices = None,
                 group_size = 128, num_group = 1024):
        
        super().__init__()

        self.device = device

        kwargs = {'features_only': True if out_indices else False}

        if out_indices:
            kwargs.update({'out_indices': out_indices})

        layers_keep = 12

        self.rgb_backbone = timm.create_model(model_name = rgb_backbone_name, pretrained = True, **kwargs)
        self.rgb_backbone.blocks = torch.nn.Sequential(*self.rgb_backbone.blocks[:layers_keep]) # Remove Block(s) from 5 to 11.

        self.xyz_backbone = PointTransformer(group_size = group_size, num_group = num_group)
        self.xyz_backbone.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        self.xyz_backbone.blocks.blocks = torch.nn.Sequential(*self.xyz_backbone.blocks.blocks[:layers_keep]) # Remove Block(s) from 5 to 11.


    def forward_rgb_features(self, x):
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        x = self.rgb_backbone.blocks(x) 
        x = self.rgb_backbone.norm(x)

        feat = x[:,1:].permute(0, 2, 1).view(1, -1, 28, 28) # view(1, -1, 14, 14)
        return feat


    def forward(self, rgb, xyz):
        rgb_features = self.forward_rgb_features(rgb)
        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)

        return rgb_features, xyz_features, center, ori_idx, center_idx


def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


class KNN(nn.Module):
    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k

    def forward(self, xyz, centers):
        assert xyz.size(0) == centers.size(0), "Batch size of xyz and centers should be the same"

        B, N_points, _ = xyz.size()
        K = centers.size(1)

        xyz = xyz.unsqueeze(2)  # [B, N, 1, 3]
        centers = centers.unsqueeze(1)  # [B, 1, K, 3]
        distances = torch.norm(xyz - centers, dim=-1)  # [B, N, K]

        _, indices = torch.topk(distances, self.k, dim=1, largest=False, sorted=True)
        return indices


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size)

    def forward(self, xyz):

        batch_size, num_points, _ = xyz.shape
        center, center_idx = fps(xyz.contiguous(), self.num_group)  # B G 3

        # _, idx = self.knn(xyz, center)  # B G M
        idx = self.knn(xyz, center).permute(0,2,1)  # B G M

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx[-1]
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False): # num_state=384 num_node=16
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x): # x [16,384,16]
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Converter(nn.Module):
    def __init__(self, dim_in, dim_temp=384, img_size=384, mids=4):
        super(Converter, self).__init__()

        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_temp = dim_temp

        self.num_n = mids * mids

        self.conv_fc = nn.Conv2d(self.dim_in * 2, self.dim_temp, kernel_size=1)

        # f1
        self.norm_layer_f1 = nn.LayerNorm(dim_in)
        self.conv_f1_Q = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.conv_f1_K = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.ap_f1 = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.gcn_f1 = GCN(num_state=self.dim_temp, num_node=self.num_n)
        self.conv_f1_extend = nn.Conv2d(self.dim_temp, self.dim_in, kernel_size=1, bias=False)

        # f2
        self.norm_layer_f2 = nn.LayerNorm(dim_in)
        self.conv_f2_Q = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.conv_f2_K = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.ap_f2 = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.gcn_f2 = GCN(num_state=self.dim_temp, num_node=self.num_n)
        self.conv_f2_extend = nn.Conv2d(self.dim_temp, self.dim_in, kernel_size=1, bias=False)

    def forward(self, token_pair):
        # tokens list 12x[8,578,768]
        bs, num_token, chs = token_pair[0].shape
        tokens_ls = []
        for index in range(len(token_pair) // 2):
            f1_ = self.norm_layer_f1(token_pair[index * 2][:, 2:, :])  # [8,576,768]
            f2_ = self.norm_layer_f2(token_pair[index * 2 + 1][:, 2:, :])  # [8,576,768]
            f1_ = f1_.permute(0, 2, 1).view(bs, chs, int(self.img_size // 16), int(self.img_size // 16)).contiguous()
            # [8,768,24,24]
            f2_ = f2_.permute(0, 2, 1).view(bs, chs, int(self.img_size // 16), int(self.img_size // 16)).contiguous()
            # [8,768,24,24]
            f1, f2 = f1_, f2_  # [8,768,24,24] / [8,768,24,24]
            fc = self.conv_fc(torch.cat((f1, f2), dim=1))  # [8,384,24,24]
            fc_att = torch.nn.functional.softmax(fc, dim=1)[:, 1, :, :].unsqueeze(1)  # [8,1,24,24]

            # f1 pass
            f1_Q = self.conv_f1_Q(f1).view(bs, self.dim_temp, -1).contiguous()  # [8,384,576] [bs,chs,24*24]
            f1_K = self.conv_f1_K(f1)  # [8,384,24,24]
            f1_masked = f1_K * fc_att  # [8,384,24,24]
            f1_V = self.ap_f1(f1_masked)[:, :, 1:-1, 1:-1].reshape(bs, self.dim_temp, -1)  # [8,384,16]

            f1_proj_reshaped = torch.matmul(f1_V.permute(0, 2, 1), f1_K.reshape(bs, self.dim_temp, -1))  # [8,16,576]
            f1_proj_reshaped = torch.nn.functional.softmax(f1_proj_reshaped, dim=1)  # [8,16,576] Tv

            f1_rproj_reshaped = f1_proj_reshaped  # [8,16,576]
            f1_n_state = torch.matmul(f1_Q, f1_proj_reshaped.permute(0, 2, 1))  # [16,384,16] Ta

            f1_n_rel = self.gcn_f1(f1_n_state)  # [16,384,16]
            f1_state_reshaped = torch.matmul(f1_n_rel, f1_rproj_reshaped)  # [16,384,576]
            f1_state = f1_state_reshaped.view(bs, self.dim_temp, *f1.size()[2:])  # [16,384,24,24]
            f1_out = f1_ + (self.conv_f1_extend(f1_state))  # [16,768,24,24]

            # f2 pass
            f2_Q = self.conv_f2_Q(f2).view(bs, self.dim_temp, -1).contiguous()  # [8,384,576] [bs,chs,24*24]
            f2_K = self.conv_f2_K(f2)  # [8,384,24,24]
            f2_masked = f2_K * fc_att  # [8,384,24,24]
            f2_V = self.ap_f2(f2_masked)[:, :, 1:-1, 1:-1].reshape(bs, self.dim_temp, -1)  # [8,384,16]

            f2_proj_reshaped = torch.matmul(f2_V.permute(0, 2, 1), f2_K.reshape(bs, self.dim_temp, -1))  # [8,16,576]
            f2_proj_reshaped = torch.nn.functional.softmax(f2_proj_reshaped, dim=1)  # [8,16,576]

            f2_rproj_reshaped = f2_proj_reshaped  # [8,16,576]
            f2_n_state = torch.matmul(f2_Q, f2_proj_reshaped.permute(0, 2, 1))  # [16,384,16]

            f2_n_rel = self.gcn_f2(f2_n_state)  # [16,384,16]
            f2_state_reshaped = torch.matmul(f2_n_rel, f2_rproj_reshaped)  # [16,384,576]
            f2_state = f2_state_reshaped.view(bs, self.dim_temp, *f2.size()[2:])  # [16,384,24,24]
            f2_out = f2_ + (self.conv_f2_extend(f2_state))  # [16,768,24,24]

            tokens_ls.extend([f1_out, f2_out])

        return tokens_ls


class UpSampling2x(nn.Module): 
    def __init__(self, in_chs, out_chs):
        super(UpSampling2x, self).__init__()
        temp_chs = out_chs * 4
        self.up_module = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, bias=False),
            nn.BatchNorm2d(temp_chs),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def forward(self, features):
        return self.up_module(features)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupFusion(nn.Module):
    def __init__(self, in_chs, out_chs, start=False):  # 768, 384
        super(GroupFusion, self).__init__()
        temp_chs = in_chs
        if start:
            in_chs = in_chs
        else:
            in_chs *= 2

        self.gf1 = nn.Sequential(nn.Conv2d(in_chs, temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))

        self.gf2 = nn.Sequential(nn.Conv2d((temp_chs + temp_chs), temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))
        self.up2x = UpSampling2x(temp_chs, out_chs)

    def forward(self, f_r, f_l):
        f_r = self.gf1(f_r)  # chs 768
        f12 = self.gf2(torch.cat((f_r, f_l), dim=1))  # chs 768
        return f12, self.up2x(f12)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=scale),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        # fetch_idx = [3, 7, 11] ### ! PD
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            # if i in fetch_idx: ### ! PD
            feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    def __init__(self, group_size = 128, num_group = 1024, encoder_dims = 384):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = group_size
        self.num_group = num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(bert_ckpt_path, map_location=device)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                    incompatible.missing_keys
                )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                    incompatible.unexpected_keys
                )
                
        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts):
        if self.encoder_dims != self.trans_dim:
            B,C,N = pts.shape
            pts = pts.transpose(-1, -2) # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood,  center, ori_idx, center_idx = self.group_divider(pts)
            # # generate mask
            # bool_masked_pos = self._mask_center(center, no_mask = False) # B G
            # encoder the input cloud blocks
            group_input_tokens = self.encoder(neighborhood)  #  B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)
            # prepare cls
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
            # add pos embedding
            pos = self.pos_embed(center)
            # final input
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            return x, center, ori_idx, center_idx 
        else:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)  # B N 3
            # divide the point clo  ud in the same form. This is important

            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)
            group_input_tokens = self.encoder(neighborhood)  # B G N

            pos = self.pos_embed(center)
            # final input
            x = group_input_tokens
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            if len(feature_list) == 12:
                x = torch.cat((feature_list[3],feature_list[7],feature_list[11]), dim=1) 
            elif len(feature_list) == 8:
                x = torch.cat((feature_list[1],feature_list[4],feature_list[7]), dim=1) 
            elif len(feature_list) == 4:
                x = torch.cat((feature_list[1],feature_list[2],feature_list[3]), dim=1) 
            else:
                x = feature_list[-1]
            return x, center, ori_idx, center_idx