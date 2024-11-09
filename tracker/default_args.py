class Args:
    def __init__(self):
        # General arguments
        self.experiment_name = None
        self.name = None
        self.path = ""
        self.camid = 0
        self.save_result = False
        self.exp_file = None
        self.ckpt = None
        self.device = "gpu"
        self.conf = None
        self.nms = None
        self.tsize = None
        self.fps = 10
        self.fp16 = False
        self.fuse = False
        self.trt = False

        # Tracking arguments
        self.track_high_thresh = 0.0
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.3
        self.track_buffer = 5
        self.match_thresh = 0.1
        self.min_box_area = 10
        self.fuse_score = False

        # CMC arguments
        self.cmc_method = "none"

        # ReID arguments
        self.with_reid = False
        self.fast_reid_config = r"fast_reid/configs/MOT17/sbs_S50.yml"
        self.fast_reid_weights = r"pretrained/mot17_sbs_S50.pth"
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25

        # Additional attributes not defined in argparse directly
        self.ablation = False
        self.mot20 = not self.fuse_score

# Create an instance of Args
args = Args()
