import csv

import cv2
from keras.models import load_model
# from keras.preprocessing import image
import numpy as np
import os

# image folder
from tensorflow.python.keras.models import model_from_json

folder_path = "F:\\Uni\\MA\\coding\\vball_orig\\ball-net\\train\\b"
# folder_path = "F:\\Uni\\MA\\resources\\train\\b"
model_base_path = "../ball-net/model/"
# dimensions of images
img_width, img_height = 320, 240
size = 32
dim = 3

# load the trained model
# json_file = open(model_base_path + "/model_austria.json", 'r')
json_file = open(model_base_path + "/model_1.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights(model_base_path + "/model_austria.h5")
loaded_model.load_weights(model_base_path + "/model_1.h5")

# load all images into a list
images = []
image_list = os.listdir(folder_path)
for img in image_list:
    img_path = os.path.join(folder_path, img)
    # img = image.load_img(img, target_size=(img_width, img_height))
    # img = image.img_to_array(img)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (size, size))
    image = np.reshape(image, [1, size, size, dim])
    # img = np.expand_dims(img, axis=0)
    images.append(image)

# stack up images list to pass for prediction
image_stack = np.vstack(images)
classes = loaded_model.predict(image_stack, batch_size=10)
print(classes)
sum_of_correct_detections = 0
results = list()
for index, class_entry in enumerate(classes):
    # result_item = [ball_prob, non_ball_prob, image_list[index]]
    result_item = (class_entry[0], class_entry[1], image_list[index])
    # np.append(class_entry, image_list[index])
    # if ball_prob >= .95:
    if class_entry[0] >= .95:
        sum_of_correct_detections += 1

    # results.append(class_entry)
    results.append(result_item)



# sum_of_correct_detections = sum(map(lambda x : x[0] >= .95, classes))
accuracy = sum_of_correct_detections / len(results)
print(f'accuracy: {accuracy}')

with open('results.csv', 'w', newline='') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['ball_prob', 'non_ball_prob', 'file'])
    for row in results:
        csv_out.writerow(row)
