import io
import time

import cv2
import keyboard
import requests
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from ultralytics import YOLO
import numpy as np

verbose = True

def vprint(input):
    if verbose:
        print(input)

def capture_image(cam):
    try:
        ret, frame = cam.read()
        if not ret:
            raise RuntimeError()
        pillow_frame = Image.fromarray(frame)
        pillow_frame.convert('RGB')
        pillow_frame.save('rpi/image.jpg')
        return pillow_frame
    except Exception as e:
        print('Failed to capture image')
        return None
        

# Input: file path to image containing birds
# Output: list of cropped bird images
def get_birds(image_path: str, transform):
    model = YOLO("yolo11n.pt")
    results = model(image_path, verbose=False)[0]
    BIRD_CLASS_ID = 14 #bird
    bird_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == BIRD_CLASS_ID:
            coords = box.xyxy[0].tolist()
            bird_boxes.append(coords)
    vprint(f'Num of Birds: {len(bird_boxes)}')
    if not bird_boxes:
        vprint("No birds detected in the image.")
        return [], []
    with Image.open(image_path) as img:
        birds = []
        for box in bird_boxes:
            vprint(f'bird boxes: {box}')
            final_img = img.crop((box[0], box[1], box[2], box[3]))
            final_img = final_img.convert("RGB")
            final_img = transform(final_img)
            final_img = final_img.unsqueeze(0)
            birds.append(final_img)
    return birds, bird_boxes

def predict(model, tensors, classes):
    with torch.no_grad():
        ret = []
        for tensor in tensors:
            output = model(tensor)
            predicted_index = torch.argmax(output, dim=1).item()
            predicted_species = classes[predicted_index]
            ret.append(predicted_species)
            vprint(f'species predicted: {predicted_species}')
        return ret

def main():
    print("=====================================")
    print('Starting Bird Device Application')
    print("=====================================")

    # Input frame
    # Ask, is there bird?
    # if there are birds, find bird boxes and select tight frame
    # Give tight bird image to id model
    # Get bird name
    # Send to server: bird name, image, time taken

    # =================================
    # Persisting Variables
    # =================================

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print('Failed to initialize camera')

    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    model = torch.jit.load("bird_model.pt")
    model.eval()
    classes = ['American_Crow', 'American_Goldfinch', 'American_Robin', 'Baltimore_Oriole', 'Barn_Swallow', 'Black_Capped_Chickadee', 
               'Blue_Grosbeak', 'Blue_Jay', 'Bobolink', 'Brewer_Blackbird', 'Brown_Creeper', 'Brown_Thrasher', 'Brownheaded_Cowbird', 
               'Cardinal', 'Carolina_Wren', 'Cedar_Waxwing', 'Chipping_Sparrow', 'Common_Grackle', 'Common_Raven', 'Common_Yellowthroat', 
               'Dark_eyed_Junco', 'Downy_Woodpecker', 'Eastern_Bluebird', 'Eastern_Towhee', 'European_Starling', 'Field_Sparrow', 
               'Fox_Sparrow', 'Gray_Catbird', 'Hairy_Woodpecker', 'House_Finch', 'House_Sparrow', 'House_Wren', 'Indigo_Bunting', 
               'Lincoln_Sparrow', 'Magnolia_Warbler', 'Mockingbird', 'Mourning_Dove', 'Northern_Flicker', 'Orchard_Oriole', 'Pileated_Woodpecker', 
               'Pine_Warbler', 'Purple_Finch', 'Red_bellied_Woodpecker', 'Red_headed_Woodpecker', 'Red_winged_Blackbird', 
               'Rose_breasted_Grosbeak', 'Ruby_throated_Hummingbird', 'Rusty_Blackbird', 'Savannah_Sparrow', 'Scarlet_Tanager', 
               'Song_Sparrow', 'Spotted_Catbird', 'Tree_Sparrow', 'Tree_Swallow', 'Tufted_Titmouse', 'White_breasted_Nuthatch', 
               'White_crowned_Sparrow', 'White_throated_Sparrow', 'Yellow_Warbler', 'Yellow_headed_Blackbird']
    
    webhook_url = 'https://webhook.site/a32c1a03-0dd5-4ef2-a876-baae164519dd'

    response = requests.post(webhook_url, {'idk': 'test'})

    num_birds = 0

    # =================================
    # Program Loop
    # =================================

    while(True):
        # Capture image as 'image.jpg'
        ret = capture_image(cam)
        if not ret:
            break

        # get list of birds in image in the form of tensors
        bird_tensors, boxes = get_birds("rpi/image.jpg", transform=t)
        new_num_birds = len(bird_tensors)
        if new_num_birds > num_birds:
            # ====== get the whole image, as well as detected birds, wrap it in object, send to endpoint ======
            # Make prediction on each bird tensor
            predictions = predict(model=model, tensors=bird_tensors, classes=classes)

            post_img = Image.open('rpi/image.jpg')
            draw = ImageDraw.Draw(post_img)
            for i in range(len(boxes)):
                x_min = boxes[i][0]
                y_min = boxes[i][1]
                x_max = boxes[i][2]
                y_max = boxes[i][3]
                draw.line([x_min, y_min, x_min, y_max], fill='red', width=3) # left
                draw.line([x_min, y_max, x_max, y_max], fill='red', width=3) # top 
                draw.line([x_max, y_max, x_max, y_min], fill='red', width=3) # right
                draw.line([x_min, y_min, x_max, y_min], fill='red', width=3) # bottom
                draw.text((x_min, y_min), predictions[i], fill=(255,255,255), stroke_width=2, stroke_fill='red')

            post_img.show(post_img)

            # TODO: Maybe pass the boxes lists to here
            # And then use some pillow stuff to draw the boxes and labels here

            payload = {
                'bird_names': predictions,
                'image_array': np.asarray(post_img)
            }

            response = requests.post(webhook_url, payload)

            print(response)
            

        time.sleep(0.1)

        if keyboard.is_pressed('esc'):
            print('Exiting Program Loop')
            break

    cam.release()
    print('Program Stopped')


if __name__ == '__main__':
    main()