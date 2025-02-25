from typing import List

import torch
import torch.nn as nn


class End2End(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()

        self.strides = [(8, 8), (16, 16), (32, 32)]
        self.offset = 0

        self.model = model.eval()
        self.head = self.rtm_head
        self.topk = self.rtm_topk


    def rtm_topk(self,
                 boxes: torch.Tensor,
                 scores: torch.Tensor,
                 keep_top_k: int = 1000) -> list:

        topk_scores, topk_indices = torch.topk(scores, keep_top_k, dim=1, largest=True)
        topk_boxes = torch.gather(boxes, 1, topk_indices.expand(-1, -1, 4))

        return topk_boxes, topk_scores


    def rtm_head(self,
                 cls_scores: List[torch.Tensor],
                 bbox_preds: List[torch.Tensor]) -> list:

        assert len(cls_scores) == len(bbox_preds)
        device = cls_scores[0].device
        batch_size = bbox_preds[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        mlvl_priors = []
        for level_idx, featmap_size in enumerate(featmap_sizes):
            mlvl_prior = self.mlvl_point_generator(
                featmap_size, level_idx, device=device)
            mlvl_priors.append(mlvl_prior)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        priors = torch.cat(mlvl_priors)

        tl_x = (priors[..., 0] - flatten_bbox_preds[..., 0])
        tl_y = (priors[..., 1] - flatten_bbox_preds[..., 1])
        br_x = (priors[..., 0] + flatten_bbox_preds[..., 2])
        br_y = (priors[..., 1] + flatten_bbox_preds[..., 3])
        bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        scores = flatten_cls_scores

        return bboxes, scores


    def mlvl_point_generator(self,
                             featmap_size,
                             level_idx,
                             dtype=torch.float32,
                             device='cuda'):

        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self.meshgrid(shift_x, shift_y)
        stride_w = shift_xx.new_full((feat_w * feat_h, ), stride_w).to(dtype)
        stride_h = shift_xx.new_full((feat_w * feat_h, ), stride_h).to(dtype)
        shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        all_points = shifts.to(device)
        return all_points


    def meshgrid(self, x, y, row_major=True):
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx


    def forward(self, inputs):
        cls_scores, bbox_preds = self.model(inputs)
        bboxes, scores = self.head(cls_scores, bbox_preds)
        topk_bboxes, topk_scores = self.topk(bboxes, scores)
        return topk_bboxes, topk_scores