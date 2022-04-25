import torch.nn as nn
import torch.nn.functional as F
import torch
import utils
from embedder import get_embedder

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.cal_nerf = Nerf()

        self.bw_latent = nn.Embedding(261, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

    def get_bw_feature(self, pts, ind):
        embedder,dim = get_embedder(10)
        pts = embedder(pts)
        pts = pts.transpose(1, 2)
        latent = self.bw_latent(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_neural_blend_weights(self, pose_pts, smpl_bw, latent_index):
        features = self.get_bw_feature(pose_pts, latent_index)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw

    def pose_points_to_tpose_points(self, pose_pts, batch):
        init_pbw = utils.pts_sample_blend_weights(pose_pts, batch['pbw'],
                                            batch['pbounds'])
        init_pbw = init_pbw[:, :24]

        pbw = self.calculate_neural_blend_weights(
            pose_pts, init_pbw, batch['latent_index'] + 1)

        tpose = utils.pose_points_to_tpose_points(pose_pts, pbw, batch['A'])

        return tpose, pbw

    def calculate_alpha(self, wpts, batch):
        wpts = wpts[None]
        pose_pts = torch.matmul(wpts - batch['Th'], batch['Rh'])
        init_pbw = utils.pts_sample_blend_weights(pose_pts, batch['pbw'],
                                            batch['pbounds'])
        pnorm = init_pbw[:, 24]
        norm_th = 0.1
        pind = pnorm < norm_th
        pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
        pose_pts = pose_pts[pind][None]

        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        init_tbw = utils.pts_sample_blend_weights(tpose, batch['tbw'],
                                            batch['tbounds'])
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch['latent_index'])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)

        alpha = self.tpose_human.calculate_alpha(tpose)
        alpha = alpha[0, 0]

        n_batch, n_point = wpts.shape[:2]
        full_alpha = torch.zeros([n_point]).to(wpts)
        full_alpha[pind[0]] = alpha

        return full_alpha

    def forward(self, wpts, viewdir, dists, batch):
        wpts = wpts[None]
        pose_pts = torch.matmul(wpts - batch['Th'], batch['Rh'])

        with torch.no_grad():
            init_pbw = utils.pts_sample_blend_weights(pose_pts, batch['pbw'],
                                                batch['pbounds'])
            pnorm = init_pbw[:, -1]
            norm_th = 0.05
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind[0]]
            dists = dists[pind[0]]

        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        init_tbw = utils.pts_sample_blend_weights(tpose, batch['tbw'],
                                            batch['tbounds'])
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch['latent_index'])
        ind = batch['latent_index']
        alpha, rgb = self.cal_nerf.calculate_alpha_rgb(tpose, viewdir, ind)

        inside = tpose > batch['tbounds'][:, :1]
        inside = inside * (tpose < batch['tbounds'][:, 1:])
        outside = torch.sum(inside, dim=2) != 3
        alpha = alpha[:, 0]
        alpha[outside] = 0

        alpha_ind = alpha.detach() > 0
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind][None]
        tbw = tbw.transpose(1, 2)[alpha_ind][None]

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * dists)

        rgb = torch.sigmoid(rgb[0])
        alpha = raw2alpha(alpha[0], dists)

        raw = torch.cat((rgb, alpha[None]), dim=0)
        raw = raw.transpose(0, 1)

        n_batch, n_point = wpts.shape[:2]
        raw_full = torch.zeros([n_batch, n_point, 4], dtype=wpts.dtype, device=wpts.device)
        raw_full[pind] = raw

        ret = {'pbw': pbw, 'tbw':tbw, 'raw': raw_full}

        return ret


class Nerf(nn.Module):
    def __init__(self):
        super(Nerf, self).__init__()

        self.nf_latent = nn.Embedding(260, 128)

        self.actvn = nn.ReLU()

        input_ch = 63
        D = 8
        W = 256
        self.skips = [4]
        self.pts_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.alpha_fc = nn.Conv1d(W, 1, 1)

        self.feature_fc = nn.Conv1d(W, W, 1)
        self.latent_fc = nn.Conv1d(384, W, 1)
        self.view_fc = nn.Conv1d(283, W // 2, 1)
        self.rgb_fc = nn.Conv1d(W // 2, 3, 1)
        self.pts_embedder,dim = get_embedder(10)
        self.view_embedder,dim = get_embedder(4)

    def calculate_alpha(self, nf_pts):
        nf_pts = self.pts_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)
        return alpha

    def calculate_alpha_rgb(self, nf_pts, viewdir, ind):
        nf_pts = self.pts_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.nf_latent(ind)
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = self.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        features = torch.cat((features, viewdir), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        return alpha, rgb