# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:48:17 2021

@author: Necro
"""


import tensorflow as tf
from PIL import Image
from numpy import asarray
import numpy as np
import os  

"""
There were several models already trained and provided 

"""
#load image, resize to 384x384,convert to numpyarray
#set the directory to get prediction
#dirpath= "C:\\Users\\Moina\\Desktop\\imgclass_test\\img"
dirpath= "dummy_img_dir"  
dir_list = os.listdir(dirpath)
path = dirpath + "\\" +dir_list[-1]
img = Image.open(path) 
#set the width and height according to the model specs,see the model image size map
new_width  = 384
new_height = 384
#for v2xl_21k
#new_width  = 480
#new_height = 480
img = img.resize((new_width, new_height), Image.ANTIALIAS)
#img.save('1.jpg')
numpydata = asarray(img)
#expand the dimensions according to the required tflite model input  
npd = np.expand_dims(numpydata, axis=0)
print("The data type of the image is",npd.dtype)
print("The shape of the image is",npd.shape)
#npd = np.array(npd,dtype=np.uint8)
#load the tflite model
TFLITE_FILE_PATH = 'model1.tflite'
tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])
tflite_interpreter.set_tensor(input_details[0]['index'], npd)
tflite_interpreter.invoke()

tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions.shape)

#possible food classes, same as set in foodclasses101 dict below
print("pred map: ",tflite_model_predictions)

prediction_classes = np.argmax(tflite_model_predictions, axis=1)
print(type(prediction_classes))


    
foodclasses101= {0: 'apple_pie', 1: 'baby_back_ribs', 2: 'baklava', 3: 'beef_carpaccio', 4: 'beef_tartare', 5: 'beet_salad', 6: 'beignets', 7: 'bibimbap', 8: 'bread_pudding', 9: 'breakfast_burrito', 10: 'bruschetta', 11: 'caesar_salad', 12: 'cannoli', 13: 'caprese_salad', 14: 'carrot_cake', 15: 'ceviche', 16: 'cheesecake', 17: 'cheese_plate', 18: 'chicken_curry', 19: 'chicken_quesadilla', 20: 'chicken_wings', 21: 'chocolate_cake', 22: 'chocolate_mousse', 23: 'churros', 24: 'clam_chowder', 25: 'club_sandwich', 26: 'crab_cakes', 27: 'creme_brulee', 28: 'croque_madame', 29: 'cup_cakes', 30: 'deviled_eggs', 31: 'donuts', 32: 'dumplings', 33: 'edamame', 34: 'eggs_benedict', 35: 'escargots', 36: 'falafel', 37: 'filet_mignon', 38: 'fish_and_chips', 39: 'foie_gras', 40: 'french_fries', 41: 'french_onion_soup', 42: 'french_toast', 43: 'fried_calamari', 44: 'fried_rice', 45: 'frozen_yogurt', 46: 'garlic_bread', 47: 'gnocchi', 48: 'greek_salad', 49: 'grilled_cheese_sandwich', 50: 'grilled_salmon', 51: 'guacamole', 52: 'gyoza', 53: 'hamburger', 54: 'hot_and_sour_soup', 55: 'hot_dog', 56: 'huevos_rancheros', 57: 'hummus', 58: 'ice_cream', 59: 'lasagna', 60: 'lobster_bisque', 61: 'lobster_roll_sandwich', 62: 'macaroni_and_cheese', 63: 'macarons', 64: 'miso_soup', 65: 'mussels', 66: 'nachos', 67: 'omelette', 68: 'onion_rings', 69: 'oysters', 70: 'pad_thai', 71: 'paella', 72: 'pancakes', 73: 'panna_cotta', 74: 'peking_duck', 75: 'pho', 76: 'pizza', 77: 'pork_chop', 78: 'poutine', 79: 'prime_rib', 80: 'pulled_pork_sandwich', 81: 'ramen', 82: 'ravioli', 83: 'red_velvet_cake', 84: 'risotto', 85: 'samosa', 86: 'sashimi', 87: 'scallops', 88: 'seaweed_salad', 89: 'shrimp_and_grits', 90: 'spaghetti_bolognese', 91: 'spaghetti_carbonara', 92: 'spring_rolls', 93: 'steak', 94: 'strawberry_shortcake', 95: 'sushi', 96: 'tacos', 97: 'takoyaki', 98: 'tiramisu', 99: 'tuna_tartare', 100: 'waffles'}
print(foodclasses101[prediction_classes[0]])

