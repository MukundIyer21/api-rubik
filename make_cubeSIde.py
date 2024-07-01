import heapq
import cv2
import heapq
import base64
from PIL import Image
import io
def prediction_function(model1,img="",results =[],base64_image="",class_map ={0:"blue",1:"green",2:"orange",3:"red",4:"white",5:"yellow"}):
    def preprocess_image(image_path, target_size=(1280,736)):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, target_size)
        return resized_image
    if base64_image!="":
        image_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(image_bytes))
    # predict and get all the results
    if results ==[]:
        if type(img)=="str":
            results = model1.predict(preprocess_image(img),conf=0.75,max_det=9)
        else:
            results = model1.predict(img,conf=0.75,max_det=9)
    for result in results:
        boxes = result.boxes
        # print(boxes.cls)
        # print(boxes.xywh)
        # result.show()

    # converting the tensors recieved to a matrix
    cordinates = list(boxes.xywh)
    cordinates = [i.tolist() for i in cordinates]
    preds = boxes.cls.tolist()

    # making a hashmap with the posistions as key and the colorClass as value
    predMap = {}
    for cord,colorID in zip(cordinates,preds):
        predMap[tuple(cord)] = class_map[colorID]

    # using a heap for faster popping from matrix 
    heap = [(sublist[1], sublist) for sublist in cordinates]
    heapq.heapify(heap)
    rubiks_face =[]
    final_colors =[]
    for i in range(3):
        row = []
        for j in range(3):
            row.append(heapq.heappop(heap)[1])
        rubiks_face.append(sorted(row, key=lambda x: x[0]))

    # mapping the classNames to the actual colors
    for i in rubiks_face:
        final_colors.append([predMap[tuple(val)] for val in i])
    return final_colors