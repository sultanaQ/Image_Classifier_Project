# import libraries 
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json


parser = argparse.ArgumentParser(description="class predction --part 2--")

parser.add_argument('image_path')
parser.add_argument('model')
parser.add_argument('--top_k', default=5)
parser.add_argument('--category_names',default='label_map.json')  
args = parser.parse_args()


image_path = args.image_path
top_k = int(args.top_k)


#mapping 
with open(args.category_names, 'r') as f:
    class_names = json.load(f)

    
#load
model = tf.keras.models.load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer} )


#preprocessing
image = Image.open(image_path)
image = np.asarray(image)
image = tf.convert_to_tensor(image,tf.float32)
image = tf.image.resize(image,(224,224))
image /= 255
image = image.numpy()
image = np.expand_dims(image,axis=0)


#prediction
prediction = model.predict(image)
top_values, top_indices = tf.math.top_k(prediction, top_k)
print("\n---------Top propabilities: ",top_values.numpy()[0])
top_probabilities = top_indices.cpu().numpy()[0]
top_classes = [class_names[str(int(value)+1)] for value in top_probabilities]
print('---------Top classes: ', top_classes)
probs= top_values.numpy()[0]


#class and probabilities
print(f"---------Top ({top_k}) flowers names from image ({image_path}).\n")
print(top_classes)

print("\n---------Probability):")
print(probs)
print(top_probabilities)



