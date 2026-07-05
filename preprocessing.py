import json
import os

def coco_to_yolo(json_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Map category IDs to YOLO's strict 0-indexed format
    category_mapping = {}
    for i, category in enumerate(data['categories']):
        category_mapping[category['id']] = i

    # Print the class names so you can easily copy them to dataset.yaml
    print(f"\n--- Class Map for {os.path.basename(json_path)} ---")
    for i, category in enumerate(data['categories']):
        print(f"  {i}: {category['name']}")

    # 2. Map image IDs to their actual filenames and dimensions
    images = {}
    for img in data['images']:
        images[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # 3. Convert every annotation
    print(f"Converting {len(data['annotations'])} annotations...")
    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]

        # Get target image dimensions
        img_info = images[image_id]
        img_w = img_info['width']
        img_h = img_info['height']

        # Calculate YOLO coordinates (normalized center-x, center-y, width, height)
        x_min, y_min, box_w, box_h = bbox
        x_center = (x_min + box_w / 2.0) / img_w
        y_center = (y_min + box_h / 2.0) / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h

        # Prevent coordinates from spilling slightly outside image bounds (common dataset bug)
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        yolo_class = category_mapping[category_id]

        # Determine the text file name
        base_name = os.path.basename(img_info['file_name'])
        txt_name = os.path.splitext(base_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        # Append to the text file
        with open(txt_path, 'a') as txt_file:
            txt_file.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"Success. Saved YOLO labels to {output_dir}")

# Define your exact paths
train_json = r"D:\Projects\EdgeObjectDetector\instances_train2019.json"
train_out = r"D:\Projects\EdgeObjectDetector\yolo_format\labels\train"

val_json = r"D:\Projects\EdgeObjectDetector\instances_val2019.json"
val_out = r"D:\Projects\EdgeObjectDetector\yolo_format\labels\val"

# Run the conversions
print("Starting Train conversion...")
coco_to_yolo(train_json, train_out)

print("\nStarting Val conversion...")
coco_to_yolo(val_json, val_out)