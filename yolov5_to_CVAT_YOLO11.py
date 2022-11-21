import torch
import shutil
from glob import glob
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

OUTPUT = "output"
YOLO_PROBABILITY_THRESHOLD = 0.7

VIZUALIZE = False
RESIZE_IMG_SCALE = 0.5

COLOR_BBOX = (0, 255, 255)
COLOR_FONT = (0, 0, 255)
FONT_FACE = cv2.FONT_HERSHEY_COMPLEX


def filter_predictions_by_confidence(labels: list, bboxes: list) -> list:
    labels_filtered = []
    bboxes_filtered = []

    for lbl, bbx in zip(labels, bboxes):
        if bbx[4] >= YOLO_PROBABILITY_THRESHOLD:
            labels_filtered.append(lbl)
            bboxes_filtered.append(bbx)

    return labels_filtered, bboxes_filtered


def draw_bbox(bbox: list, img: np.array, color: tuple) -> None:
    """Рисует ограничительную рамку детектированного объекта.

    Args:
        bbox (list): координаты рамки.
        img (np.array): кадр видео.
        color (tuple): цвет рамки.
    """
    x1, y1, x2, y2 = (
        int(bbox[0] * img.shape[1]),
        int(bbox[1] * img.shape[0]),
        int(bbox[2] * img.shape[1]),
        int(bbox[3] * img.shape[0]),
    )

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def draw_bbox_label(
    img: np.array,
    text: str,
    pos: tuple,
    fontScale: float,
    fontThickness: float,
    textColor: tuple,
    textBackgroundColor: tuple,
) -> None:
    """Добавляет подпись к ограничительной рамке.

    Args:
        img (np.array): кадр видео.
        text (str): текст подписи.
        pos (tuple): позиция левого верхнего угла
        fontScale (float): масштаб шрифта подписи.
        fontThickness (float): толщина шрифта подписи.
        textColor (tuple): цвет текста подписи.
        textBackgroundColor (tuple): цвет бэкграунда подписи.
    """
    (x, y) = np.array(pos).astype(int)

    text_size, _ = cv2.getTextSize(text, FONT_FACE, fontScale, fontThickness)
    text_w, text_h = text_size

    # прибавляем магические константы, чтобы текст был внутри его бэкграунд-рамки
    text_h += 2
    text_w += 2

    # рисуем бэкграунд-рамку с текстом внутри
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), textBackgroundColor, -1)
    cv2.putText(
        img,
        text,
        (x, int(y + (text_h + fontScale - 1))),
        FONT_FACE,
        fontScale,
        textColor,
        fontThickness,
    )


def resize(img: np.array) -> np.array:
    """Ресайз фреймов, чтобы влазили в экран.

    Args:
        img (np.array): кадр видео.

    Returns:
        np.array: кадр видео после ресайза.
    """
    width = int(img.shape[1] * RESIZE_IMG_SCALE)
    height = int(img.shape[0] * RESIZE_IMG_SCALE)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def reformat_YOLOv5_bbox_output_to_YOLO11(bbox, img_shape):
    img_height, img_width = img_shape[0], img_shape[1]

    x1, y1, x2, y2, p = (
        int(bbox[0] * img_shape[1]),
        int(bbox[1] * img_shape[0]),
        int(bbox[2] * img_shape[1]),
        int(bbox[3] * img_shape[0]),
        bbox[4],
    )

    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

    width, height = abs(x2 - x1), abs(y2 - y1)

    return xc / img_width, yc / img_height, width / img_width, height / img_height, p


def predict(model: torch.nn.Module, img_pth: str) -> tuple:
    img_orig = cv2.imread(img_pth)
    img = [img_orig]
    results = model(img).xyxyn[0].cpu()
    labels, bboxes = results[:, -1].numpy(), results[:, :-1].numpy()

    labels, bboxes = filter_predictions_by_confidence(labels, bboxes)
    if VIZUALIZE:
        img_orig = resize(img_orig)
        for bbx, lbl in zip(bboxes, labels):
            lbl = model.names[int(lbl)]
            draw_bbox(bbx, img_orig, COLOR_BBOX)
            text_pos = (bbx[0] * img_orig.shape[1], bbx[3] * img_orig.shape[0])
            draw_bbox_label(
                img=img_orig,
                text=lbl,
                pos=text_pos,
                fontScale=0.9,
                fontThickness=1,
                textColor=COLOR_FONT,
                textBackgroundColor=COLOR_BBOX,
            )
        cv2.imshow("frame", img_orig)
        cv2.waitKey(0)

    bboxes = [reformat_YOLOv5_bbox_output_to_YOLO11(b, img_orig.shape) for b in bboxes]
    return labels, bboxes


def create_object_data(save_path, model):
    """
    Example:

    classes = 2
    train = data/train.txt
    names = data/obj.names
    backup = backup/
    """

    nclasses = len(model.names)
    with open(save_path, "a") as object_data_file:
        object_data_file.write(f"classes = {nclasses}\n")
        object_data_file.write(f"train = data/train.txt\n")
        object_data_file.write(f"names = data/obj.names\n")
        object_data_file.write(f"backup = backup/")


def create_obj_names(save_path, model):
    """Example:

    helmet
    person
    """

    with open(save_path, "a") as object_names_file:
        for name in model.names.values():
            object_names_file.write(f"{name}\n")


def process_images(images_current_folder, destination_folder, model, img_extention):
    for img_cur_pth in tqdm(glob(f"{images_current_folder}/*.{img_extention}")):
        img_basename = os.path.basename(img_cur_pth)
        text_file_name = img_basename.replace(f".{img_extention}", ".txt")
        classes, bboxes = predict(model, img_cur_pth)

        with open(
            os.path.join(destination_folder, text_file_name), "a"
        ) as detction_bbox_file:
            for cls, bbox in zip(classes, bboxes):
                bbox = " ".join([f"{str(b):.8}" for b in bbox[:-1]])
                cls = int(cls)
                detction_bbox_file.write(f"{cls} {bbox}\n")


def create_train_file(project_root_path, image_path, img_extention):
    """Example:

    data/obj_train_data/Камера 143_07_09_2022 20.50.36.avi_0005.png
    """

    with open(os.path.join(project_root_path, "train.txt"), "a") as train_file:
        for img_path in glob(f"{image_path}/*.{img_extention}"):
            img_basename = os.path.basename(img_path)
            train_file.write(f"data/obj_train_data/{img_basename}\n")


def zip_annotation(root_pth):
    shutil.make_archive(root_pth, "zip", root_pth)


if __name__ == "__main__":
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path="weights/freeze10_batch8_imgsz960_ep100_data1000.pt",
        force_reload=True,
    )

    # ======================== Parameters ===========================

    annotation_archive_name = "annotation1001_1488"
    image_path = "images/1001_1488"
    img_extention = "png"

    # ===============================================================
    root_pth = os.path.join(OUTPUT, annotation_archive_name)

    try:
        shutil.rmtree(root_pth)
    except FileNotFoundError:
        pass

    Path(root_pth).mkdir(parents=True, exist_ok=True)
    create_object_data(os.path.join(root_pth, "obj.data"), model)
    create_obj_names(os.path.join(root_pth, "obj.names"), model)

    images_dst_folder = os.path.join(root_pth, "obj_train_data")
    Path(os.path.join(root_pth, "obj_train_data")).mkdir(parents=True, exist_ok=True)
    process_images(image_path, images_dst_folder, model, img_extention)

    create_train_file(root_pth, image_path, img_extention)

    zip_annotation(root_pth)

    shutil.rmtree(root_pth)
