# Plant Disease Image Classification(CNN) Project.

# Description
This project is focused on the classification of plant diseases using images, leveraging the power of Convolutional Neural Networks (CNNs). The aim is to identify various diseases affecting plant leaves, thereby assisting in the early intervention and management of plant health. By utilizing CNNs, the system efficiently categorizes images of plant leaves into different classes based on the type of disease present. A key feature of the project is its high accuracy in detecting and classifying diseases. The model is trained and validated on publicly available datasets, ensuring robustness and reliability in real-world applications. Furthermore, real-time predictions and a user-friendly interface are made possible through the deployment of the model in a web application, enabling users such as farmers and agricultural professionals to upload leaf images and receive instant diagnostic feedback. This project not only aids in disease detection but also contributes to improving agricultural practices by minimizing crop losses and enhancing crop yield through timely disease management.  


## Overview

This project aims to leverage Convolutional Neural Networks (CNNs) to classify plant diseases from images of plant leaves. The goal is to assist farmers and agricultural professionals in early detection and management of plant diseases, ultimately reducing crop losses and improving crop yields


### Features

- Accurate Disease Detection: Utilizes CNNs to classify images of plant leaves into various disease categories.
- Real-time Predictions: Provides quick and accurate diagnoses through a user-friendly interface.
- Scalable Deployment: Can be deployed on cloud platforms for large-scale use

### Dataset 

- Description: 
 
  The dataset used in this project consists of over 20,000 images, each in the .jpg format. These images are categorized into 15 distinct classes, with each class representing a different type of plant disease, including a category for healthy, disease-free leaves. The dataset includes a diverse range of plant species and disease manifestations to ensure robust model training and accurate classification.

- **Key Characteristics**:
  
  i] **Image format** :  All images are in .jpg format, which ensures compatibility with most image processing libraries and tools.

  ii] **Number of Images** : 20,000+ images, providing a comprehensive dataset for training and validation.

  iii] **Classes** : The dataset is organized into 15 classes. These classes include various plant diseases, with each category containing images related to a specific disease affecting to leaves of the plant.


- **Classes Incuded**:

      1.  Pepper_bell_Bacterial_spot
      2.  Pepper_bell_healthy
      3.  Potato_Early_blight
      4.  Potato_healthy
      5.  Potato_Late_blight
      6.  Tomato_Target_Spot
      7.  Tomato_Tomato_mosaic_virus
      8.  Tomato_Tomato_YellowLeaf_Curl_Virus
      9.  Tomato_Bacterial_spot
      10. Tomato_Early_blight
      11. Tomato_healthy
      12. Tomato_Late_blight
      13. Tomato_Leaf_Mold
      14. Tomato_Septoria_leaf_spot
      15. Tomato_Spider_mites_Two_spotted_spider_mite


- **Data Usage** :

  The dataset is used to train and validate a CNN model. Each image is preprocessed to a consistent size for input into the neural network. Augmentation techniques are applied to increase the diversity of the training set, improve model robustness, and prevent overfitting. The dataset is split into training, validation, and test subsets to evaluate the model’s performance comprehensively.

- **Source** : 

     Plant Village : 
        
      https://www.kaggle.com/datasets/arjuntejaswi/plant-village


### Model Architecture

   1. **Input Shape** : 
   Our models input shape is defined as (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS). This means that the model expects input images to be of a specific size (we have taken our Image size as 128x128 pixels) and have a specific number of color channels (as our dataset contain color images we have taken channel as 3 as it represented as RGB image). The batch size determines the number of images that will be processed simultaneously during training and inference (Our Batch size is 16).
       
      input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
   2. **Number of Classes** : 
   We have created variable named as 'num_classes' which is set to 15, indicating that the model is designed to classify the input images into 15 different classes, likely representing different plant diseases or a healthy class.
       
      num_classes = 15

   3. **Moderl Architecture** : 
   - Resize and Rescale: The first layer is a custom layer called resize_and_rescale, which likely performs image preprocessing tasks such as resizing the input images to the required size (i.e. 128x128 pixels) and normalizing the pixel values(by dividing each image by 255).

         model = Sequential([
         resize_and_rescale,
         

   - Data Augmentation: The second layer is a data augmentation layer, which applies various transformations (e.g., rotation, flipping) to the input images during training to increase the diversity of the training data and improve the model's generalization.

          data_augmentation,

   - Convolutional Layers: The model consists of 3 sets of convolutional layers, each followed by a max-pooling layer. The convolutional layers extract relevant features from the input images, while the max-pooling layers reduce the spatial dimensions of the feature maps, helping to make the model more robust to spatial variations.

          # Input layer and first Hidden layer
          layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape),
          layers.MaxPooling2D((2, 2)),
          # Second Hidden layer
          layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'),
          layers.MaxPooling2D((2, 2)),
          # Third Hidden layer
          layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'),
          layers.MaxPooling2D((2, 2)),

   - Flatten Layers: 
     After the convolutional and max-pooling layers, the feature maps are flattened into a 1D vector, which is then passed to the fully connected layers.

          # Fourth Hidden layer
          layers.Flatten(),

  - Fully Connected Layers: The model has two fully connected (dense) layers. The first dense layer has 64 units and uses the 'ReLU' activation function, while the second dense layer has num_classes units (i.e. 15) and uses the 'softmax' activation function to produce the final classification probabilities.

         # Fifth Hidden layer
         layers.Dense(64, activation = 'relu'),
         # Output layer
         layers.Dense(num_classes, activation = 'softmax')
         ])


 4. **Model Building** : 
 The model.build() function is called with the specified input shape, which initializes the model's weights and prepares it for training.
         
    model.build(input_shape = input_shape)


5. **Model Compilation** : 
      
       model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

The model.compile() function is used to configure the training process of the model. The combination of the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric is a common and effective choice for training a multi-class image classification model like the one presented in the given architecture. 

The model.compile() function sets up the training process by:
 
    1. Specifying the optimization algorithm (Adam) to be used for updating the model's weights during training.

    2. Defining the loss function (sparse categorical cross-entropy) to be minimized during the training process.

    3. Selecting the performance metric (accuracy) to be monitored and reported during training and validation.

  
  - Optimizer: The optimizer is set to 'adam', which is a popular optimization algorithm used in deep learning. The Adam optimizer is an adaptive learning rate optimization algorithm that combines the benefits of momentum and RMSProp, making it well-suited for a wide range of machine learning problems, including image classification.

  - Loss Function: The loss function is set to 'sparse_categorical_crossentropy'. This loss function is used for multi-class classification problems, where the target labels are integers representing the class indices. The "sparse" in the name indicates that the target labels are not one-hot encoded, but rather represented as integer values. This is a common choice for classification tasks with a large number of classes, as it can be more efficient than using one-hot encoding.

  - Metrics: The 'accuracy' metric is specified, which will be used to evaluate the model's performance during training and validation. The accuracy metric calculates the fraction of correctly classified samples, providing a straightforward way to assess the model's classification performance.



### Installation 
### Prerequisites

- Python 3.8 or above

- Tensorflow / keras 2.0

- Other libraries: Numpy, Matplotlib, pillow, Streamlit.

### Setup

1. Clone the repository:

       https://github.com/KrishnaSalve/Plant-Disease-Classification-Prediction.git

2. Navigate to the project directory:

       cd plant-disease-classification


3. Install required packages:

       pip install -r requirements.txt


### Usage 

1. Run the Streamlit Application:

       streamlit run app.py

      This will start the web application and make it accessible at http://localhost:5000/.


2. Upload an Image:

- Open your web browser and navigate to http://localhost:5000/.

- You will see a web interface with an option to upload an image.

- Click on the "Choose File" button and select an image of a plant leaf.

- It will automatically predict the leaf image after uploading the image and provided the suitable result.

3. View the Prediction:

- The application will process the uploaded image and display the predicted plant disease or healthy status.

- The prediction result will be shown on the web page, along with the confidence score.

4. Interpret the Results:

- The application will classify the uploaded image into one of the 15 classes, which represent different plant diseases or a healthy class.

- Refer to the "Classes Included" section in the README for a list of the available classes and their descriptions.


### Results
The plant disease image classification model has been evaluated on a comprehensive dataset, using the following performance metrics: 

- Accuracy: The portion of correctly classified samples in the training dataset which is 0.97 % 
- Validation Accuracy: The proportion of correctly classified samples in the validation dataset, which provides an estimate of the model's performance on unseen data. In our case val_accuracy is 0.97 % which is quite similar to our accuracy which means our model is performing good on training as well as validation datasets.

- Loss: The value of the loss function, which quantifies the difference between the model's predictions and the true labels during training. In our case which is 0.07.
- Validation loss: The value of the loss function on the validation dataset, indicating the model's generalization performance, which is 0.06.

**Qualitative Results**

The models performance on the test dataset is as follows:

|Metric |	Value|
|-|-|
|Accuracy|	97.6%
|Validation Accuracy|	96.3%
|Loss|	0.06
|Validation Loss|	0.07


**Visualization**

- Training and Validation Curves of Accuracies over Epochs

  ![Screenshot_20240822_110311](https://github.com/user-attachments/assets/77eaa148-64d7-4c1e-af4d-a0712e4da423)


- Training and Validation Curves of Losses over Epochs.

  ![Screenshot_20240822_110330](https://github.com/user-attachments/assets/9113ea0b-9535-4c7c-98f6-c756ccdf7127)


The plots shows the model's learning progress, with the training and validation metrics converging, indicating a well-trained and stable model.


### Contributing
We welcome contributions from the community to help improve and expand the plant disease image classification project. If you're interested in contributing, please follow these guidelines:

**Report Issues** : 

If you encounter any bugs, errors, or have suggestions for improvements, please open an issue on the project's GitHub repository. When reporting an issue, provide a clear and detailed description of the problem, including any relevant error messages or screenshots.

**Submit Bug Fixes or Enhancements** : 

If you've identified a bug or have an idea for an enhancement, feel free to submit a pull request. Before starting work, please check the existing issues to ensure that your proposed change hasn't already been addressed.
When submitting a pull request, make sure to:

    1. Fork the repository and create a new branch for your changes.
    2. Clearly describe the problem you're solving and the proposed solution in the pull request description.
    3. Follow the project's coding style and conventions.
    4. Include relevant tests (if applicable) to ensure the stability of your changes.
    5. Update the documentation, including the README file, if necessary.






**Improve Documentation**

If you notice any issues or have suggestions for improving the project's documentation, such as the README file, please submit a pull request with the necessary changes.

**Provide Feedback**

We value your feedback and suggestions for improving the plant disease image classification project. Feel free to share your thoughts, ideas, or use cases by opening an issue or reaching out to the project maintainers.

**Code of Conduct**

When contributing to this project, please adhere to the project's Code of Conduct to ensure a welcoming and inclusive environment for all participants.

Thank you for your interest in contributing to the plant disease image classification project. We appreciate your support and look forward to collaborating with you.


### Contact

If you have any questions, feedback, or would like to get in touch with the project maintainers, you can reach us through the following channels:

- **Project Maintainer**

Name : Krishna Salve 

Email : krishnasalve97@gmail.com

Linkedin : Krishna Salve

GitHub : KrishnaSalve



- **Project Repository**

     https://github.com/KrishnaSalve/Plant-Disease-Classification-Prediction




