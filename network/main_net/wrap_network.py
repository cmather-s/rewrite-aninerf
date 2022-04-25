import torch.nn as nn
from main_net import Network
import torch


class Wrap_network(nn.Module):
    def __init__(self, net):
        super(Wrap_network, self).__init__()

        self.net = Network()

        # self.renderer = make_renderer()

        self.bw_crit = torch.nn.functional.smooth_l1_loss

        self.img2mse = lambda x, y: torch.mean((x - y)**2)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        bw_loss = self.bw_crit(ret['pbw'], ret['tbw'])
        scalar_stats.update({'bw_loss': bw_loss})
        loss += bw_loss

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
        scalar_stats.update({'img_loss0': img_loss0})
        loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
