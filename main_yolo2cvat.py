import os

import click

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device

from lib_utils_yolo2cvat import get_class_index_hashmap, zip_annotation, unzip_task
from predict_redactor import enrich_labels
import shutil


@click.command()
@click.option(
    "--cvat_task",
    help="Path to the exported .zip task file",
    required=True,
    type=str,
)
@click.option(
    "--weights",
    help="Path to the YOLOv5 weights",
    required=True,
    type=str,
)
@click.option(
    "--model_train_imgsz",
    help="YOLOv5 imgsz parameter via training (640 is ok, if don't remember)",
    required=True,
    type=int,
)
@click.option(
    "--classes",
    help="Classes which will be predicted by YOLO",
    default="all",
    required=True,
    type=str,
)
@click.option("--img_format", default="png", help="Format of images", type=str)
@click.option(
    "--prob_thresh", default=0.5, help="Probability threshold for YOLOv5", type=float
)
def main(**kwargs):

    # ------------------ ARG parse ------------------
    cvat_task_pth = kwargs["cvat_task"]
    yolov5_weights_pth = kwargs["weights"]
    wanted_classes = kwargs["classes"]
    img_format = kwargs["img_format"]
    prob_thresh = kwargs["prob_thresh"]
    model_train_imgsz = kwargs["model_train_imgsz"]
    lbl_extention = "txt"

    # --------------- Assertions --------------------
    assert os.path.exists(yolov5_weights_pth), f"{yolov5_weights_pth} does not exist"
    assert os.path.exists(cvat_task_pth), f"{cvat_task_pth} does not exist"

    model = DetectMultiBackend(yolov5_weights_pth, device=select_device())

    model_classes = list(model.names.values())
    print("model_classes:", model_classes)
    if wanted_classes == "all":
        wanted_classes = model_classes
    else:
        assert all(
            [cls in wanted_classes for cls in model_classes]
        ), f"model works with classes {model_classes}, but --classes {wanted_classes} were given"

    # --------------------- main --------------------

    cvat_task_pth = unzip_task(cvat_task_pth)
    class_index_hashmap = get_class_index_hashmap(cvat_task_pth)
    enrich_labels(
        cvat_task_pth,
        img_format,
        wanted_classes,
        class_index_hashmap,
        model,
        prob_thresh,
        model_classes,
        model_train_imgsz,
        lbl_extention=lbl_extention,
    )

    zip_annotation(cvat_task_pth)
    shutil.rmtree(cvat_task_pth)


if __name__ == "__main__":
    main()
