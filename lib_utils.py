import shutil


def get_class_index_hashmap(task_path):
    with open(f"{task_path}/obj.names", "r+") as stream:
        classes = stream.read().splitlines()

    return {cls: str(indx) for indx, cls in enumerate(classes)}


def zip_annotation(root_pth):
    print("Creating zip annotation_file")
    shutil.make_archive(f"{root_pth}_annotated", "zip", root_pth)
    print("Done")


def unzip_task(task_pth):
    print(f"Unpacking {task_pth}")
    new_task_pth = task_pth.replace(".zip", "")
    shutil.unpack_archive(task_pth, new_task_pth)
    print("Done")
    return new_task_pth
