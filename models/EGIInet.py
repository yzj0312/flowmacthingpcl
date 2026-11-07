import numpy as np
#from deformable_attention import DeformableAttention1D
from torch import nn

from models.flow_matching import (LatentFlowMatching, MissingSetEncoder,
                                  MissingSetDecoder, missing_region)
import torch
from torch.nn import Module
from models.encoders import  transfer_loss_shared_encoder
from config_vipc import cfg

class EGIInet(Module):
    def __init__(self, 
                 embed_dim=cfg.NETWORK.EGIInet.embed_dim,
                 depth=cfg.NETWORK.EGIInet.depth,
                 img_patch_size=cfg.NETWORK.EGIInet.img_patch_size,
                 pc_sample_rate=cfg.NETWORK.EGIInet.pc_sample_rate,
                 pc_sample_scale=cfg.NETWORK.EGIInet.pc_sample_scale,
                 fuse_layer_num=cfg.NETWORK.EGIInet.fuse_layer_num,
                 ):
        super().__init__()
        self.encoder=transfer_loss_shared_encoder(embed_dim=embed_dim,
                                               img_patch_size=img_patch_size,
                                               sample_ratio=pc_sample_rate,
                                               scale=pc_sample_scale,
                                               block_head=cfg.NETWORK.shared_encoder.block_head,
                                               depth=depth,
                                               pc_h_hidden_dim=cfg.NETWORK.shared_encoder.pc_h_hidden_dim,
                                               fuse_layer_num=fuse_layer_num,
                                               )
        flow = cfg.NETWORK.FLOW
        self.target_encoder = MissingSetEncoder(embed_dim, flow.LATENT_TOKENS, flow.HEADS)
        self.flow = LatentFlowMatching(embed_dim, flow.LATENT_TOKENS, flow.HEADS, flow.DEPTH)
        self.decoder = MissingSetDecoder(embed_dim, flow.LATENT_TOKENS, flow.MISSING_POINTS)
        self.missing_points = flow.MISSING_POINTS
        self.train_steps = flow.TRAIN_STEPS
        self.sample_steps = flow.SAMPLE_STEPS

    def forward(self, pc, img, gt=None, return_details=False, noise=None):
        point_tokens, image_tokens, _, style_loss = self.encoder(pc=pc, im=img)
        condition = self.flow.condition(point_tokens, image_tokens)
        steps = self.train_steps if self.training else self.sample_steps
        generated = self.flow.sample(condition, steps, noise=noise)
        missing = self.decoder(generated)
        complete = torch.cat((pc, missing), dim=1)
        details = {'style': style_loss, 'missing': missing}
        if gt is not None:
            target_points = missing_region(pc, gt, self.missing_points)
            target = self.target_encoder(target_points)
            details.update(flow=self.flow.loss(target, condition),
                           target_missing=target_points,
                           target_reconstruction=self.decoder(target))
        if return_details:
            return complete, details
        return complete, style_loss

if __name__ == '__main__':
    import time
    model = EGIInet().cuda()
    pc = torch.rand([4, 2048, 3]).cuda()
    img = torch.rand([4, 3, 224, 224]).cuda()
    s=time.time()
    fine = model(pc, img)
    e=time.time()
    print(e-s)
    print(fine[0].shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")
