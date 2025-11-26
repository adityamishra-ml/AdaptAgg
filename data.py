import os
import json
from pathlib import Path

# The 7 classes you want, correctly mapped to indices 0-6
CLASSES = ["bike", "car", "truck", "bus", "motor", "rider", "person"]

BASE_DIR = "clients"

def convert_coco_to_yolo(json_file, images_dir, labels_dir):
    """
    Reads a COCO JSON annotation file and converts it to YOLO .txt format.
    It will only include classes defined in the global CLASSES list and will
    map their original COCO IDs to the new indices (0-6).
    """
    with open(json_file, "r") as f:
        coco = json.load(f)

    # --- THIS IS THE KEY LOGIC ---
    # Create a mapping from the original COCO category ID to the new YOLO index (0-6)
    # e.g., if coco has {"id": 13, "name": "car"}, this creates a map {13: 1}
    # because "car" is at index 1 in our CLASSES list.
    cat2id = {cat["id"]: CLASSES.index(cat["name"]) for cat in coco["categories"] if cat["name"] in CLASSES}

    images = {img["id"]: img for img in coco["images"]}
    os.makedirs(labels_dir, exist_ok=True)

    for ann in coco["annotations"]:
        # Get the new YOLO class index (0-6) from our mapping
        cls_id = cat2id.get(ann["category_id"])
        
        # If the class is not one of the 7 we care about, skip it
        if cls_id is None:
            continue

        img_info = images[ann["image_id"]]
        file_name = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]
        x, y, bw, bh = ann["bbox"]
        
        # Normalize coordinates for YOLO format
        x_c = (x + bw / 2) / w
        y_c = (y + bh / 2) / h
        bw /= w
        bh /= h

        label_path = os.path.join(labels_dir, Path(file_name).stem + ".txt")
        with open(label_path, "a") as f:
            f.write(f"{cls_id} {x_c} {y_c} {bw} {bh}\n")

def prepare_client(client_dir):
    print(f"Processing {client_dir}...")

    train_dir = os.path.join(client_dir, "train_images")
    val_dir = os.path.join(client_dir, "val_images")
    train_json = os.path.join(client_dir, "train.json")
    val_json = os.path.join(client_dir, "val.json")

    # Create YOLO folder structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(client_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(client_dir, "labels", split), exist_ok=True)

    # Move images (assuming they are already in train_images/ and val_images/)
    os.system(f"mv {train_dir}/* {client_dir}/images/train/")
    os.system(f"mv {val_dir}/* {client_dir}/images/val/")

    # Convert annotations
    convert_coco_to_yolo(train_json, f"{client_dir}/images/train", f"{client_dir}/labels/train")
    convert_coco_to_yolo(val_json, f"{client_dir}/images/val", f"{client_dir}/labels/val")

    # Write the data.yaml file for YOLO training
    yaml_path = os.path.join(client_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {client_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

if __name__ == "__main__":
    # Assuming you have 6 clients, named client_0 to client_5
    for cid in range(6):
        client_dir = os.path.join(BASE_DIR, f"client_{cid}")
        if os.path.exists(client_dir):
            prepare_client(client_dir)
        else:
            print(f"Warning: Directory not found for {client_dir}, skipping.")