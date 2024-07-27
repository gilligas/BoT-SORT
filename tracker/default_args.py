import argparse

parser = argparse.ArgumentParser("BoT-SORT Demo!")
parser.add_argument("-expn", "--experiment-name", type=str, default=None)
parser.add_argument("-n", "--name", type=str, default=None, help="model name")
parser.add_argument("--path", default="", help="path to images or video")
parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
parser.add_argument("--conf", default=None, type=float, help="test conf")
parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
parser.add_argument("--tsize", default=None, type=int, help="test img size")
parser.add_argument("--fps", default=5, type=int, help="frame rate (fps)")
parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

# tracking args
parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

# CMC
parser.add_argument("--cmc-method", default="none", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

# ReID
parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use reid model")
parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
args = parser.parse_args()
args.ablation = False
args.mot20 = not args.fuse_score