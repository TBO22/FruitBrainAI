<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Fruit Classification with CNN</h1>

<p>This repository contains an image classification model developed using Convolutional Neural Networks (CNN) for classifying fruits into 6 different categories: fresh apples, fresh bananas, fresh oranges, rotten apples, rotten bananas, and rotten oranges. The model uses a deep learning approach to classify images of fruits into these categories with high accuracy.</p>

<h2>Model Overview</h2>
<p>This project employs a CNN architecture to perform classification of fruit images. The model is designed to classify images into six classes:</p>
<ul>
    <li>Fresh Apples</li>
    <li>Fresh Bananas</li>
    <li>Fresh Oranges</li>
    <li>Rotten Apples</li>
    <li>Rotten Bananas</li>
    <li>Rotten Oranges</li>
</ul>
<h2>Dataset</h2>

<p>This model uses the "Fruits Fresh and Rotten for Classification" dataset available on Kaggle. The dataset consists of images of both fresh and rotten fruits, categorized into six classes: fresh apples, fresh bananas, fresh oranges, rotten apples, rotten bananas, and rotten oranges.</p>
<p>The dataset was created and shared by <strong>Sriram R</strong> on Kaggle. You can access the dataset here: <a href="https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification/data" target="_blank">Fruits Fresh and Rotten for Classification Dataset</a>.</p>
<p>I would like to extend my gratitude to the dataset creator for providing this valuable resource that made this project possible.</p>

<h3>Model Architecture</h3>
<p>The model is a Convolutional Neural Network (CNN) built from scratch using the Keras Sequential API. Below are the details of the architecture:</p>

<ul>
    <li><strong>Input Layer:</strong> 
        <ul>
            <li>Input shape: <code>(img_height, img_width, 3)</code> (where the image height and width are specified, and 3 corresponds to the RGB color channels).</li>
        </ul>
    </li>
    <li><strong>Convolutional Layer 1:</strong>
        <ul>
            <li>Number of filters: 32</li>
            <li>Filter size: <code>(3, 3)</code></li>
            <li>Activation function: <strong>ReLU (Rectified Linear Unit)</strong></li>
            <li>Padding: Valid (no padding)</li>
            <li>Stride: 1 (default)</li>
        </ul>
    </li>
    <li><strong>Max-Pooling Layer 1:</strong>
        <ul>
            <li>Pool size: <code>(2, 2)</code></li>
            <li>Stride: 2 (default)</li>
        </ul>
    </li>
    <li><strong>Convolutional Layer 2:</strong>
        <ul>
            <li>Number of filters: 64</li>
            <li>Filter size: <code>(3, 3)</code></li>
            <li>Activation function: <strong>ReLU</strong></li>
        </ul>
    </li>
    <li><strong>Max-Pooling Layer 2:</strong>
        <ul>
            <li>Pool size: <code>(2, 2)</code></li>
            <li>Stride: 2 (default)</li>
        </ul>
    </li>
    <li><strong>Convolutional Layer 3:</strong>
        <ul>
            <li>Number of filters: 128</li>
            <li>Filter size: <code>(3, 3)</code></li>
            <li>Activation function: <strong>ReLU</strong></li>
        </ul>
    </li>
    <li><strong>Max-Pooling Layer 3:</strong>
        <ul>
            <li>Pool size: <code>(2, 2)</code></li>
            <li>Stride: 2 (default)</li>
        </ul>
    </li>
    <li><strong>Flatten Layer:</strong>
        <ul>
            <li>Flattens the 3D output into a 1D array to feed into the fully connected layers.</li>
        </ul>
    </li>
    <li><strong>Fully Connected (Dense) Layer:</strong>
        <ul>
            <li>Number of neurons: 256</li>
            <li>Activation function: <strong>ReLU</strong></li>
        </ul>
    </li>
    <li><strong>Dropout Layer:</strong>
        <ul>
            <li>Dropout rate: 0.5 (50% of the neurons are randomly dropped during training to prevent overfitting).</li>
        </ul>
    </li>
    <li><strong>Output Layer:</strong>
        <ul>
            <li>Number of neurons: 6 (corresponding to the 6 fruit classes).</li>
            <li>Activation function: <strong>Softmax</strong> (to output probabilities for each class in multi-class classification).</li>
        </ul>
    </li>
</ul>

<p>The architecture begins with three convolutional layers, each followed by max-pooling layers. The convolutional layers extract features from the input images, while the pooling layers reduce the spatial dimensions to avoid overfitting. After flattening the output, it is passed through a fully connected layer with 256 neurons, followed by a dropout layer for regularization. The output layer, with 6 neurons, uses the softmax activation to predict the class probabilities for the six fruit categories.</p>


<hr>

<h2>Model Performance</h2>

<h3>Classification Results:</h3>

<table border="1">
    <thead>
        <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Fresh Apples</td>
            <td>0.94</td>
            <td>0.95</td>
            <td>0.94</td>
            <td>395</td>
        </tr>
        <tr>
            <td>Fresh Banana</td>
            <td>0.99</td>
            <td>0.96</td>
            <td>0.98</td>
            <td>381</td>
        </tr>
        <tr>
            <td>Fresh Oranges</td>
            <td>0.97</td>
            <td>0.96</td>
            <td>0.96</td>
            <td>388</td>
        </tr>
        <tr>
            <td>Rotten Apples</td>
            <td>0.91</td>
            <td>0.91</td>
            <td>0.91</td>
            <td>601</td>
        </tr>
        <tr>
            <td>Rotten Banana</td>
            <td>0.99</td>
            <td>0.97</td>
            <td>0.98</td>
            <td>530</td>
        </tr>
        <tr>
            <td>Rotten Oranges</td>
            <td>0.89</td>
            <td>0.95</td>
            <td>0.92</td>
            <td>403</td>
        </tr>
    </tbody>
</table>

<h3>Overall Metrics:</h3>
<ul>
    <li><strong>Accuracy</strong>: 93%</li>
    <li><strong>Macro Average</strong>:
        <ul>
            <li>Precision: 0.95</li>
            <li>Recall: 0.95</li>
            <li>F1-Score: 0.95</li>
        </ul>
    </li>
    <li><strong>Weighted Average</strong>:
        <ul>
            <li>Precision: 0.95</li>
            <li>Recall: 0.95</li>
            <li>F1-Score: 0.95</li>
        </ul>
    </li>
</ul>

<h3>Training and Validation Accuracy:</h3>
<ul>
    <li><strong>Training Accuracy</strong>: 90.54%</li>
    <li><strong>Test Accuracy</strong>: 94.8%</li>
    <li><strong>Validation Accuracy</strong>: 93.6%</li>
</ul>

<hr>

<h2>How the Model Works</h2>

<h3>Data Preprocessing:</h3>
<p>The dataset consists of images categorized into the six fruit classes. The images were preprocessed to:</p>
<ul>
    <li>Resize all images to a uniform size (e.g., 150*150 pixels).</li>
    <li>Normalize pixel values to the range [0, 1] to help with faster convergence during training.</li>
    <li>Augment the dataset with transformations like rotation, flipping, and zooming to improve generalization and prevent overfitting.</li>
</ul>

<h3>Model Training:</h3>
<p>The model was trained on a split dataset:</p>
<ul>
    <li><strong>Training Set</strong>: 80% of the dataset</li>
    <li><strong>Validation Set</strong>: 10% of the dataset</li>
    <li><strong>Test Set</strong>: 10% of the dataset</li>
</ul>
<p>We used the following parameters:</p>
<ul>
    <li><strong>Epochs</strong>: 15</li>
    <li><strong>Batch Size</strong>: 32</li>
    <li><strong>Optimizer</strong>: Adam optimizer with a learning rate of 0.0001.</li>
    <li><strong>Loss Function</strong>: Categorical cross-entropy.</li>
</ul>

<h3>Model Evaluation:</h3>
<p>The model was evaluated using accuracy, precision, recall, and F1-score to determine its performance on both the training and test datasets. The results above show the model’s effectiveness in distinguishing between fresh and rotten fruits.</p>

<hr>
<h3 align="center">Confusion Matrix</h3>
<p align="center">
  <img src="confusion_matrix.png" alt="Confusion Matrix Image" width="400">
</p>

<h2>How to Use the Model</h2>

<h3>Prerequisites:</h3>
<pre>
1. Python 3.x installed.
2. TensorFlow installed.
3. Numpy installed.
</pre>

<h3>Loading the Model:</h3>
<p>You can load the trained model using the following code:</p>
<pre>
from keras.models import load_model

model = load_model('fruit_classification_model.h5')
</pre>

<h3>Making Predictions:</h3>
<p>Once the model is loaded, you can classify a new image using the following code:</p>
<pre>
import numpy as np
from keras.preprocessing import image

# Load the image to predict
img_path = 'path_to_image.jpg'  # Replace with the path to the image
img = image.load_img(img_path, target_size=(150, 150))

# Convert the image to a numpy array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize the image

# Predict the class of the image
prediction = model.predict(img_array)
classes = ['Fresh Apples', 'Fresh Bananas', 'Fresh Oranges', 'Rotten Apples', 'Rotten Bananas', 'Rotten Oranges']
predicted_class = classes[np.argmax(prediction)]

print(f'The predicted class is: {predicted_class}')
</pre>

<h3>Example Input and Output:</h3>
<ul>
    <li><strong>Input</strong>: An image of a fresh apple.</li>
    <li><strong>Output</strong>: The predicted class is: Fresh Apples.</li>
</ul>

<hr>

<h2>Model Evaluation and Performance</h2>

<p>The model demonstrates good performance, with an accuracy of 94.8% on the test set. The precision and recall scores for the classes are generally high, particularly for <strong>rotten bananas</strong>, which has an impressive precision of 0.99 and recall of 0.97.</p>

<h3>Confusion Matrix:</h3>
<p>The model shows some difficulty with <strong>rotten apples</strong>, having a lower recall (0.91), which might be due to image quality or similarity with other fruit types. However, the <strong>rotten bananas</strong> and <strong>other</strong> classes have very high precision and recall, which is a good indicator that the model can differentiate between fresh and rotten fruits quite well.</p>
<hr>

<h2>Conclusion</h2>

<p>This fruit classification model provides a reliable solution for distinguishing between fresh and rotten fruits, using deep learning techniques with Convolutional Neural Networks (CNN). The model achieved a high test accuracy of 94.8% and can be easily used for classifying images of fruits into one of six categories.</p>

<h3>Future Improvements:</h3>
<ul>
    <li><strong>Increase dataset size</strong> for better model generalization.</li>
    <li><strong>Enhance model architecture</strong> by experimenting with more CNN layers or more advanced models like ResNet or Inception.</li>
    <li><strong>Implement real-time fruit classification</strong> using camera feeds for practical applications.</li>
</ul>

</body>
</html>
