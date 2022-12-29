import os

from glob import glob
import cv2
import numpy as np

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes, check_img_size
import torch


from tqdm import tqdm


def _get_images_and_labels(task_path, img_extention, lbl_extention):
    img_file_pths = []
    lbl_file_pths = []

    if "." in img_extention:
        img_extention.replace(".", "")

    data_folder = f"{task_path}/obj_train_data"
    if os.path.exists(data_folder):
        img_file_pths.extend(glob(f"{data_folder}/*.{img_extention}"))

    assert len(img_file_pths) > 0, f"No imgs found in {data_folder}"

    lbl_file_pths.extend(
        [p.replace(f".{img_extention}", f".{lbl_extention}") for p in img_file_pths]
    )

    return img_file_pths, lbl_file_pths


def _get_cls_indx_from_line(line):
    return line.split(" ")[0]


def _clean_existed_labels(lbl_file_pth, class_indexes_to_remove):
    new_lines = []
    with open(lbl_file_pth) as f:
        lines = f.readlines()
        for line in lines:
            current_indx = _get_cls_indx_from_line(line)
            if current_indx not in class_indexes_to_remove:
                new_lines.append(line)

    with open(lbl_file_pth, "w") as f:
        for line in new_lines:
            f.write(line)


def _filter_predictions_by_confidence(classes, bboxes, prob_thresh):
    labels_filtered = []
    bboxes_filtered = []
    for lbl, bbx in zip(classes, bboxes):
        if bbx[4] >= prob_thresh:
            labels_filtered.append(lbl)
            bboxes_filtered.append(bbx)

    return labels_filtered, bboxes_filtered


def _reformat_YOLOv5_bbox_output_to_YOLO11(bbox, img_shape):
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

    return (
        round(xc / img_width, 4),
        round(yc / img_height, 4),
        round(width / img_width, 4),
        round(height / img_height, 4),
        round(p, 4),
    )


def _predict_on_frame(
    model,
    img_pth,
    prob_thresh,
    model_classes,
    model_train_imgsz,
    non_max_sup_conf_thresh=0.25,
    non_max_sup_iou_thresh=0.45,
):
    frame = cv2.imread(img_pth)
    model_stride = model.stride

    # Нормализационные коеффициенты для выхода YOLO и перехода к координатам до предобработки изображений
    im0_shape = frame.shape
    norm_scaling_coef = (im0_shape[1], im0_shape[0], im0_shape[1], im0_shape[0], 1)

    # предобработка изображений
    img_size = check_img_size(model_train_imgsz, s=model_stride)
    frame = letterbox(frame, img_size, stride=model_stride)[0]
    frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    frame = np.ascontiguousarray(frame)
    frame = torch.from_numpy(frame).to(model.device)
    frame = frame.half() if model.fp16 else frame.float()
    frame /= 255
    frame = frame[None]

    # predict
    results = model(frame, augment=False, visualize=False)
    results = non_max_suppression(
        results, non_max_sup_conf_thresh, non_max_sup_iou_thresh
    )[0]

    results[:, :4] = scale_boxes(frame.shape[2:], results[:, :4], im0_shape).round()
    labels_int, bboxes = results.cpu()[:, -1].numpy(), results.cpu()[:, :-1].numpy()

    # декодируем имена классов, нормализуем выход ббоксов на 1
    labels = [model_classes[int(lbl)] for lbl in labels_int]
    bboxes = [list(bbox / norm_scaling_coef) for bbox in bboxes]

    classes, bboxes = _filter_predictions_by_confidence(labels, bboxes, prob_thresh)
    bboxes = [_reformat_YOLOv5_bbox_output_to_YOLO11(b, im0_shape) for b in bboxes]

    return classes, bboxes


def _enrich_labels_for_single_image(
    lbl_file_pth,
    img_file_pth,
    class_index_hashmap,
    model,
    model_classes,
    model_train_imgsz,
    prob_thresh,
):

    classes, bboxes = _predict_on_frame(
        model, img_file_pth, prob_thresh, model_classes, model_train_imgsz
    )

    with open(lbl_file_pth, "a") as f:
        for cls, bbox in zip(classes, bboxes):
            cls = class_index_hashmap[cls]
            bbox = " ".join([str(b) for b in bbox[:4]])
            f.write(f"{cls} {bbox}\n")


def enrich_labels(
    task_path,
    img_extention,
    wanted_classes,
    class_index_hashmap,
    model,
    prob_thresh,
    model_classes,
    model_train_imgsz,
    lbl_extention="txt",
):
    img_file_pths, lbl_file_pths = _get_images_and_labels(
        task_path, img_extention, lbl_extention
    )
    N_files = len(img_file_pths)

    print("Enrichment of labels:")
    for img_file_pth, lbl_file_pth in tqdm(
        zip(img_file_pths, lbl_file_pths), total=N_files
    ):
        class_indexes_to_remove = [class_index_hashmap[cls] for cls in wanted_classes]
        _clean_existed_labels(lbl_file_pth, class_indexes_to_remove)

        _enrich_labels_for_single_image(
            lbl_file_pth,
            img_file_pth,
            class_index_hashmap,
            model,
            model_classes,
            model_train_imgsz,
            prob_thresh,
        )

        os.remove(img_file_pth)
