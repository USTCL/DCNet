from detectron2.config import CfgNode as CN


def add_dcnet_config(cfg):
    """
    Add config for DCNET.
    """

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False

    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1


    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # DCNet model config
    cfg.MODEL.DCNET = CN()

    # loss
    cfg.MODEL.DCNET.DEEP_SUPERVISION = True
    # cfg.MODEL.DCNET.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DCNET.CLASS_WEIGHT = 1.0
    cfg.MODEL.DCNET.DICE_WEIGHT = 1.0
    cfg.MODEL.DCNET.MASK_WEIGHT = 20.0
    cfg.MODEL.DCNET.DEC_LAYERS = 6
    cfg.MODEL.DCNET.NUM_OBJECT_QUERIES = 10
    cfg.MODEL.DCNET.SIZE_DIVISIBILITY = 32


    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn  configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4