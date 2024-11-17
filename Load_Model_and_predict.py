import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_fruit(image_path):
    model_path = "/Users/talhabinomar/Downloads/Power BI/MODEL/fruit_classification_model_2.keras" #load saved model keras file
    model = load_model(model_path)
    with open("/Users/talhabinomar/Downloads/Power BI/MODEL/class_indices.json", 'r') as f: #Loading class indexes labels
        class_indices = json.load(f)

    class_names = {v: k for k, v in class_indices.items()}
    img = load_img(image_path, target_size=(150, 150))  # Resizing to 150*150
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # rescaling image
    img_array = np.expand_dims(img_array, axis=0) 
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    print(f"Predicted class is : ",predicted_class)
    print(f"Confidence is : ",confidence)

    #confidence probability of all the classes
    print("\nAll class probabilities:")
    for i, prob in enumerate(predictions[0]):
        print(f"{class_names[i]}: {prob*100:.2f}%")

image_p = "/Users/talhabinomar/Downloads/Power BI/MODEL/banan3.jpg"  # replace this path with your image path
predict_fruit(image_p)
