import os

from glob import glob
import cv2

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


def _predict_on_frame(model, img_pth, prob_thresh, model_classes):
    img_orig = cv2.imread(img_pth)
    img = [img_orig]
    results = model(img).xyxyn[0].cpu()
    classes, bboxes = results[:, -1].numpy(), results[:, :-1].numpy()
    classes = [model_classes[int(cls)] for cls in classes]

    classes, bboxes = _filter_predictions_by_confidence(classes, bboxes, prob_thresh)
    bboxes = [_reformat_YOLOv5_bbox_output_to_YOLO11(b, img_orig.shape) for b in bboxes]

    classes = classes * len(bboxes)

    return classes, bboxes


def _enrich_labels_for_single_image(
    lbl_file_pth, img_file_pth, class_index_hashmap, model, model_classes, prob_thresh
):

    classes, bboxes = _predict_on_frame(model, img_file_pth, prob_thresh, model_classes)

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
            prob_thresh,
        )

        os.remove(img_file_pth)
