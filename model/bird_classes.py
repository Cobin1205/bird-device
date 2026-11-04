import os

class ClassObject:
    def get_class_list():
        dataset_path = "model/CUB_200_2011/cropped_images"
        classes = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
        return classes