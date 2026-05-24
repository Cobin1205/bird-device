from pathlib import Path
from PIL import Image
import imagehash
import os

hashes = {}

root = Path("model/CUB_200_2011/images")

toRemove = []

for img_path in root.rglob("*.jpg"):

    try:
        img = Image.open(img_path)

        h = imagehash.phash(img)

        if h in hashes:
            print(f"DUPLICATE:")
            print(img_path)
            print(hashes[h])
            toRemove.append(hashes[h])

        else:
            hashes[h] = img_path

    except Exception as e:
        print(e)

print("\nRemoving... ")
count = 0
for path in toRemove:
    try:
        os.remove(path)
        count += 1
    except Exception as e:
        print(f"Failed to remove file: {path}")
        print(e)
        
print(f"{count} duplicate images removed")