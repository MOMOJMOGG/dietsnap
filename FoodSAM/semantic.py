import sys
import argparse
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from mmcv.utils import DictAction
import argparse
import json
import os
import numpy as np
from typing import Any, Dict, List
import shutil, logging
from FoodSAM.FoodSAM_tools.predict_semantic_mask import semantic_predict
from FoodSAM.FoodSAM_tools.enhance_semantic_masks import enhance_masks
from FoodSAM.FoodSAM_tools.evaluate_foodseg103 import evaluate

# parser = argparse.ArgumentParser(
#     description=(
#         "Runs SAM automatic mask generation and semantic segmentation on an input image or directory of images, "
#         "and then enhance the semantic masks based on SAM output masks"
#     )
# )
args = argparse.Namespace(**{})
args.data_root = 'dataset/FoodSeg103/Images'
args.img_dir   = 'img_dir/test'
args.ann_dir   = 'ann_dir/test'
args.img_path  = None
args.output    = 'output/'
args.SAM_checkpoint = 'FoodSAM/ckpts/sam_vit_h_4b8939.pth'
args.semantic_config = 'FoodSAM/configs/SETR_MLA_768x768_80k_base.py'
args.semantic_checkpoint = 'FoodSAM/ckpts/SETR_MLA/iter_80000.pth'
args.model_type = 'vit_h'
args.device = 'cuda'
args.aug_test = False
args.color_list_path = 'FoodSAM/FoodSAM_tools/color_list.npy'
args.category_txt = 'FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt'
args.num_class = 104
args.area_thr  = 0
args.ratio_thr = 0.5
args.top_k = 80
args.eval  = False
args.options={}
args.eval_options={}

args.points_per_side  = None
args.points_per_batch = None
args.pred_iou_thresh  = None
args.stability_score_thresh = None
args.stability_score_offset = None
args.box_nms_thresh = None
args.crop_n_layers = None
args.crop_nms_thresh = None
args.crop_overlap_ratio = None
args.crop_n_points_downscale_factor = None
args.min_mask_region_area = None


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    os.makedirs(os.path.join(path, "sam_mask"), exist_ok=True)
    masks_array = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        masks_array.append(mask.copy())
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, "sam_mask" ,filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)

    masks_array = np.stack(masks_array, axis=0)
    np.save(os.path.join(path, "sam_mask" ,"masks.npy"), masks_array)
    metadata_path = os.path.join(path, "sam_metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    return


def create_logger(save_folder):
    
    log_file = f"sam_process.log"
    final_log_file = os.path.join(save_folder, log_file)

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    print(f"Create Logger success in {final_log_file}")
    return logger

def main(args: argparse.Namespace, img_path: str) -> None:
    args.img_path = img_path
    
    os.makedirs(args.output, exist_ok=True)
    logger = create_logger(args.output)
    logger.info("running sam!")
    sam = sam_model_registry[args.model_type](checkpoint=args.SAM_checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    
    assert args.data_root or args.img_path
    if args.img_path:
        targets = [args.img_path]
    else:
        img_folder = os.path.join(args.data_root, args.img_dir)
        targets = [
            f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))
        ]
        targets = [os.path.join(img_folder, f) for f in targets]

    for t in targets:
        logger.info(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            logger.error(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image)
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        os.makedirs(save_base, exist_ok=True)
        write_masks_to_folder(masks, save_base)
        shutil.copyfile(t, os.path.join(save_base, "input.jpg"))
    logger.info("sam done!\n")

    
    logger.info("running semantic seg model!")
    semantic_predict(args.data_root, args.img_dir, args.ann_dir, args.semantic_config, args.options, args.aug_test, args.semantic_checkpoint, args.eval_options, args.output, args.color_list_path, args.img_path)
    logger.info("semantic predict done!\n")
    

    logger.info("enhance semantic masks")
    enhance_masks(args.output, args.category_txt, args.color_list_path, num_class=args.num_class, area_thr=args.area_thr, ratio_thr=args.ratio_thr, top_k=args.top_k)
    logger.info("enhance semantic masks done!\n")

    if args.eval and not args.img_path:
        ann_folder = os.path.join(args.data_root, args.ann_dir)
        evaluate(args.output, ann_folder, args.num_class)


    logger.info("The results saved in {}!\n".format(args.output))
    
    return True


if __name__ == "__main__":
    img_path = "test.jpg"
    main(args)
