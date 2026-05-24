import requests
from pathlib import Path
import time
import os
from dotenv import load_dotenv
import sys

load_dotenv()
ACCESS_KEY = os.getenv("ACCESS_KEY")

# This is for making a new directory for a new bird
'''
search = 'Tufted-Titmouse'
searchClean = search.replace('-', '').replace(' ', '')
newDir = Path('model/CUB_200_2011/images') / searchClean
newDir.mkdir(exist_ok=True, parents=True)
'''

# Increment this number manually every time we run the code
# It just adds this number before the image counter to prevent
# Overwriting files with the same name
iterationNumber = 2

#LAST BIRD WAS EASTERN BLUEBIRD <-------------

# New implementation for adding images to existing folders
for path in Path('model/CUB_200_2011/images').iterdir():
    if path.is_dir() and path.name > "Eastern_Bluebird":
        search = path.name.replace('_', ' ')
        newDir = Path('model/CUB_200_2011/images/' + path.name)

        image_counter = 0

        # Loop through page 1 and page 2 to get 60 images total (30 per page)
        for page in [1, 2]:
            # per_page=30 gives us the maximum number of records allowed in one request
            url = f"https://api.unsplash.com/search/photos?query={search}&per_page=30&page={page}"
            headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                for photo in data['results']:
                    # Using 'regular' quality - clean resolution for model training
                    imageUrl = photo['urls']['regular']
                    
                    try:
                        # This download does NOT count against your 50 req/hour limit
                        imageBin = requests.get(imageUrl).content
                        
                        filePath = newDir / f"{path.name}{iterationNumber}{image_counter}.jpg"
                        with open(filePath, "wb") as newImg:
                            newImg.write(imageBin)
                        
                        print(f"Saved: {filePath.name}")
                        image_counter += 1
                        
                    except Exception as e:
                        print(f"Failed to download image {image_counter}: {e}")
                        
            elif response.status_code == 403:
                print("Rate limit hit! Wait an hour or apply for Production Mode.")
                break
            else:
                print(f"API Error on page {page}: {response.status_code}")

        print(f"Downloaded {image_counter} images of {search}")
