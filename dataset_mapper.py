import os
import cv2
from detectron2.structures import BoxMode

def get_yolo_dicts(img_dir, label_dir, class_mapping):
    """
        Reads images and their YOLO .txt labels and converts them to the list of dicts that
        Detectron2 expects, with boxes in absolute XYXY pixels. Images without a label file are kept
        with no annotations.

    Args:
        img_dir (str): Directory with the images.
        label_dir (str): Directory with the YOLO .txt labels.
        class_mapping (dict[str, int]): YOLO class ID (as text) to Detectron2 category ID, e.g. {'0': 0}.
            Lines with IDs outside the mapping are skipped.

    Returns:
        list[dict]: One record per image, in Detectron2's standard dataset format.
    """
    dataset_dicts = []

    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"[!] Erro: Diretório {img_dir} ou {label_dir} não encontrado.")
        return dataset_dicts

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, img_file in enumerate(img_files):
        record = {}

        filename = os.path.join(img_dir, img_file)
        # Altura e largura para converter as boxes em pixels
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        objs = []

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = parts[0]
                if class_id not in class_mapping:
                    continue

                # YOLO: [x_center, y_center, width, height] normalizados
                x_center, y_center, w_norm, h_norm = map(float, parts[1:5])

                # Converte para pixels
                x_c = x_center * width
                y_c = y_center * height
                w = w_norm * width
                h = h_norm * height

                x_min = x_c - (w / 2)
                y_min = y_c - (h / 2)
                x_max = x_c + (w / 2)
                y_max = y_c + (h / 2)

                obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_mapping[class_id],
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
