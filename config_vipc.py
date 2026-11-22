from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.VIPC_PATH        = '/data16t/yzj/datasets/ShapeNetViPC-extract/ShapeNetViPC-Dataset'#path to dataset
#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.NUM_WORKERS                            = 4
__C.CONST.DATA_perfetch                          = 2
#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = '/data16t/yzj/outputs/latent_completion'#path to save checkpoints and logs
__C.CONST.DEVICE                                 = '0'
#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.LEGACY                              = edict()
__C.NETWORK.LEGACY.embed_dim                    = 256
__C.NETWORK.LEGACY.depth                        = 6
__C.NETWORK.LEGACY.img_patch_size               = 14
__C.NETWORK.LEGACY.pc_sample_rate               = 0.125
__C.NETWORK.LEGACY.pc_sample_scale              = 2
__C.NETWORK.LEGACY.fuse_layer_num               = 2
__C.NETWORK.shared_encoder                       = edict()
__C.NETWORK.shared_encoder.block_head            = 12
__C.NETWORK.shared_encoder.pc_h_hidden_dim       = 192
#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 8
__C.TRAIN.N_EPOCHS                               = 160
__C.TRAIN.SAVE_FREQ                              = 40
__C.TRAIN.LEARNING_RATE                          = 0.001
__C.TRAIN.LR_MILESTONES                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.LR_DECAY_STEP                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.WARMUP_STEPS                           = 1
__C.TRAIN.GAMMA                                  = 0.7
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.CATE                                   = 'plane'
__C.TRAIN.d_size                                 = 1
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.CATE                                    = 'plane'
__C.TEST.BATCH_SIZE                              = 8
# Old reconstruction checkpoints are incompatible with the new encoders.
# Set __C.CONST.WEIGHTS only for a checkpoint from this architecture.

# Independent DINOv2 / PointNet++ condition encoders.
__C.NETWORK.CONDITION = edict()
__C.NETWORK.CONDITION.HEADS = 8
__C.NETWORK.CONDITION.POINT_COUNTS = (512, 128)
__C.NETWORK.CONDITION.POINT_RADII = (0.1, 0.2)  # Dataset coordinate units.
__C.NETWORK.CONDITION.NEIGHBORS = 32
__C.NETWORK.CONDITION.POINT_DEPTH = 2
__C.NETWORK.CONDITION.DINO_FROZEN = True
__C.NETWORK.CONDITION.DINO_REPO = 'facebookresearch/dinov2'
__C.NETWORK.CONDITION.DINO_SOURCE = 'github'  # 'local' for an offline checkout.
__C.NETWORK.CONDITION.DINO_WEIGHTS = None  # Raw official ViT-S/14 state_dict.
