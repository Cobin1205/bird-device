from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# bad images to remove (no bird found)
bad = []

model = YOLO('yolov8s.pt')
root = Path('model/CUB_200_2011/images')

dest = Path('model/CUB_200_2011/cropped_images')

for dir in root.iterdir():
    for img_path in dir.iterdir():

        if img_path.is_file():

            try:
                results = model(str(img_path))
                image = Image.open(img_path)

                best_box = None
                best_area = 0

                for result in results:
                    boxes = result.boxes

                    for box in boxes:
                        cls = int(box.cls[0])

                        # 14 = bird (COCO dataset)
                        if cls == 14:

                            x1, y1, x2, y2 = box.xyxy[0]

                            area = (x2 - x1) * (y2 - y1)

                            if area > best_area:
                                best_area = area
                                best_box = (x1, y1, x2, y2)

                # -------------------------
                # If no bird found
                # -------------------------
                if best_box is None:
                    bad.append(img_path)
                    print(f'No bird found: {img_path}')
                    continue

                # -------------------------
                # Crop biggest bird box
                # -------------------------
                x1, y1, x2, y2 = best_box

                width, height = image.size

                box_w = x2 - x1
                box_h = y2 - y1

                # padding (fixed so it doesn't distort mid-calculation)
                pad_x = box_w * 0.10
                pad_y = box_h * 0.10

                x1 = max(0, int(x1 - pad_x))
                y1 = max(0, int(y1 - pad_y))
                x2 = min(width, int(x2 + pad_x))
                y2 = min(height, int(y2 + pad_y))

                cropped = image.crop((x1, y1, x2, y2))

                save_path = dest / dir.name / img_path.name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                cropped.save(save_path)

                print(f'Cropped {save_path}')

            except Exception as e:
                print(f'Ran into a problem with {img_path}. {e}')