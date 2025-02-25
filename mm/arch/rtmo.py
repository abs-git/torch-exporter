from typing import List, Tuple, Union

import torch
import torch.nn as nn

from .rtmo_lib.rtmo_head import DCC


class End2End(nn.Module):
    def __init__(self,
                 model,
                 data_samples,
                 cfg) -> None:
        super().__init__()

        self.model = model.eval()
        self.data_samples = data_samples

        self.flatten_priors = model.head.flatten_priors
        self.flatten_stride = model.head.flatten_stride
        self.bbox_padding = model.head.bbox_padding

        cfg.head.dcc_cfg.update({'num_keypoints':17})
        self.dcc = DCC(**cfg.head.dcc_cfg)
        self.dcc = self.dcc.eval()
        self.dcc.switch_to_deploy(cfg.test_cfg)
        self.dcc.load_state_dict(model.head.dcc.state_dict())


    @torch.no_grad()
    def forward(self, inputs):
        cls_scores, bbox_preds, _, kpt_vis, pose_vecs = self.model(inputs, self.data_samples)
        assert len(cls_scores) == len(bbox_preds)

        flatten_bbox_scores = self.flatten_predictions(cls_scores)
        flatten_bbox_preds = self.flatten_predictions(bbox_preds)
        flatten_pose_vecs = self.flatten_predictions(pose_vecs)
        flatten_kpt_vis = self.flatten_predictions(kpt_vis)

        bboxes = self.decode_bbox(flatten_bbox_preds,
                                    self.flatten_priors.to(inputs.device),
                                    self.flatten_stride.to(inputs.device))
        bboxes_scores = flatten_bbox_scores.sigmoid()

        bbox_cs = torch.cat(self.bbox_xyxy2cs(bboxes[..., :4], self.bbox_padding), dim=-1)
        keypoints = self.dcc.forward_test(flatten_pose_vecs,
                                          bbox_cs,
                                          self.flatten_priors.to(inputs.device))
        keypoints_scores = flatten_kpt_vis.sigmoid().unsqueeze(-1)

        topk_bboxes, topk_bboxes_scores, topk_keypoints, topk_keypoints_scores = self.rtm_topk(
            bboxes, bboxes_scores, keypoints, keypoints_scores
        )

        return topk_bboxes, topk_bboxes_scores, topk_keypoints, topk_keypoints_scores

    def rtm_topk(self,
                 bboxes: torch.Tensor,
                 bboxes_scores: torch.Tensor,
                 keypoints: torch.Tensor,
                 keypoints_scores: torch.Tensor,
                 keep_top_k: int = 1000) -> list:

        topk_bboxes_scores, topk_indices = torch.topk(
            bboxes_scores, keep_top_k, dim=1, largest=True)

        topk_indices = topk_indices.contiguous()
        topk_bboxes = torch.gather(bboxes, 1, topk_indices.expand(
            -1, -1, 4))
        topk_keypoints = torch.gather(keypoints, 1, topk_indices.view(
            -1, topk_indices.shape[1], 1, 1).expand(-1, -1, 17, 2))
        topk_keypoints_scores = torch.gather(keypoints_scores, 1, topk_indices.view(
            -1, topk_indices.shape[1], 1, 1).expand(-1, -1, 17, 1))

        return topk_bboxes, topk_bboxes_scores, topk_keypoints,topk_keypoints_scores


    def flatten_predictions(self, preds: List[torch.Tensor]):
        if len(preds) == 0:
            return None
        preds = [x.permute(0, 2, 3, 1).contiguous().flatten(1, 2) for x in preds]
        return torch.cat(preds, dim=1)


    def decode_bbox(self,
                    pred_bboxes: torch.Tensor,
                    priors: torch.Tensor,
                    stride: Union[torch.Tensor, int]) -> torch.Tensor:
        stride = stride.view(1, stride.size(0), 1)
        priors = priors.view(1, priors.size(0), 2)

        xys = (pred_bboxes[..., :2] * stride) + priors
        whs = pred_bboxes[..., 2:].exp() * stride

        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes


    def bbox_xyxy2cs(self,
                     bbox: torch.Tensor,
                     padding: float = 1.) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        scale = (bbox[..., 2:] - bbox[..., :2]) * padding
        center = (bbox[..., 2:] + bbox[..., :2]) * 0.5

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale

