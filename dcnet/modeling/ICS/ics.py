"""
Instance-level Camouflage Suppression Module
"""

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .reference_attention import ReferenceTransformer

class ics(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels=256,
        mask_classification=True,
        num_classes: int = 1,
        hidden_dim: int = 256,
        nheads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        mask_dim: int = 256,
        enforce_input_project: bool = False,
    ):

        super().__init__()

        self.mask_classification = mask_classification
        deep_supervision = cfg.MODEL.DCNET.DEEP_SUPERVISION

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = ReferenceTransformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=0,
            num_decoder_layers=cfg.MODEL.DCNET.DEC_LAYERS,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = cfg.MODEL.DCNET.NUM_OBJECT_QUERIES
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        pos = self.pe_layer(x, mask)

        src = x
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)

        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out = {"pred_logits": outputs_class[-1]}
        else:
            out = {}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x