import torch
from torch import nn
from torch.nn import functional as F

# from detectron2.data import MetadataCatalog
from detectron2.modeling import build_backbone, META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling import pcd, ics

@META_ARCH_REGISTRY.register()
class DCNet(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        self.backbone = build_backbone(cfg)
        self.PCD = pcd(cfg, self.backbone.output_shape())
        self.ICP = ics(cfg)

        matcher = HungarianMatcher(
            cost_class=cfg.MODEL.DCNET.CLASS_WEIGHT,
            cost_mask=cfg.MODEL.DCNET.MASK_WEIGHT,
            cost_dice=cfg.MODEL.DCNET.DICE_WEIGHT,
            num_points=112 ** 2,
        )
        weight_dict = {
            "loss_ce": cfg.MODEL.DCNET.CLASS_WEIGHT, 
            "loss_mask": cfg.MODEL.DCNET.MASK_WEIGHT, 
            "loss_dice": cfg.MODEL.DCNET.DICE_WEIGHT
            }

        if cfg.MODEL.DCNET.DEEP_SUPERVISION:
            dec_layers = cfg.MODEL.DCNET.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.criterion = SetCriterion(
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=["labels", "masks"],
            num_points=112 ** 2,
            oversample_ratio=3,
            importance_sample_ratio=0.75,
        )
        self.size_divisibility = cfg.MODEL.DCNET.SIZE_DIVISIBILITY

        pixel_mean = [123.675, 116.280, 103.530]
        pixel_std = [58.395, 57.120, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.num_queries = cfg.MODEL.DCNET.NUM_OBJECT_QUERIES
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
    
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_a = [x["image_a"].to(self.device) for x in batched_inputs]
        images_a = [(x - self.pixel_mean) / self.pixel_std for x in images_a]

        images = ImageList.from_tensors(images, self.size_divisibility)
        images_a = ImageList.from_tensors(images_a, self.size_divisibility)

        features = self.backbone(images.tensor)
        dc_pixel_features, pixel_embedding = self.PCD(features, images_a.tensor)
        outputs = self.ICP(dc_pixel_features, pixel_embedding)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
                
                # instance segmentation inference
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        num_classes = 1
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
