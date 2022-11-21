# YOLO2CVAT
Can infer YOLOv5 model to CVAT YOLO1.1 data format

# Installation

`git clone https://github.com/ankhafizov/YOLO2CVAT.git`

`cd YOLO2CVAT`

`pip install -e .` 

# Usage

1. Create the Task (https://blog.roboflow.com/cvat/) in CVAT and upload photos
2. Go to Tasks --> Actions --> Export Task Dataset. The content of exported .zip must be the likes of the following:

```
task_name.zip/
├─ obj_train_data/
│  ├─ 1.png
│  ├─ 1.txt
│  ├─ 2.png
│  ├─ 2.txt
├─ obj.data
├─ obj.names
├─ train.txt
```
3. Execute bash command:

```
yolo2cvat --cvat_task ./task_name.zip --weights path/to/weights.pt --labels "B|C|D"
```

Then, YOLOv5 CNN will process all images in `task_name/obj_train_data` folder. If it produce, let's say, predictions for A, B, C, D classes, only classes B, C and D will be used as an output prediction (because of `--labels "B|C|D"`). If A class was labeled manually before the tsk export, all its instances will be kept. Contrary, if there are manually labeled, e.g. B or C instances, all of them would be _removed_.

4. The output of the step 4 will be task_name_annotated.zip archive. Go to CVAT web and Tasks --> Actions --> Upload annotation. Upload this output and select YOLO1.1 format.

# Help:

- --cvat_task - Path to the exported .zip task file
- --weights - Path to the YOLOv5 weights
- --labels - Path to converted dataset folder {root}{dataset name}

help:

```
cvat2yolo --help
```
