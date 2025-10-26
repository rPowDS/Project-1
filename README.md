Analyzing Fashion-MNIST with Machine Learning
W200 Project 2 | Trevor Lang, Ryan Powers, Carmen Liang

This project develops and compares two machine learning models for classifying images from the Fashion-MNIST dataset: a Random Forest (RF) classifier and a Convolutional Neural Network (CNN).

📖 Project Overview

The goal of this project was to address the following question:

How does the distribution of images across different classes in the Fashion-MNIST dataset affect the performance of Random Forest and Convolutional Neural Network (CNN) algorithms?

We implemented, trained, and evaluated both models to compare their effectiveness in a real-world image classification task.

💾 Dataset

We used the Fashion-MNIST dataset, a popular benchmark for machine learning algorithms.

Contents: 70,000 grayscale images of 10 different categories of fashion products.

Image Size: Each image is 28×28 pixels.

Labels: The 10 classes are:

0: T-shirt/top

1: Trouser

2: Pullover

3: Dress

4: Coat

5: Sandal

6: Shirt

7: Sneaker

8: Bag

9: Ankle boot

🤖 Models Implemented

Random Forest (RF): An ensemble learning method that builds multiple decision trees. Implemented using scikit-learn.

Convolutional Neural Network (CNN): A deep learning algorithm specifically designed for computer vision and learning features from images. Implemented using TensorFlow and Keras.

🛠️ How to Run This Project

To replicate our analysis and run the models, follow these steps:

Clone the repository:

Bash
git clone https://github.com/UC-Berkeley-I-School/Project2-Powers-Lang-Liang.git
cd Project2-Powers-Lang-Liang
Install the required libraries. We recommend using a virtual environment.

Bash
pip install -r requirements.txt
Run the Jupyter Notebooks:

For the Random Forest model, open and run Randomforest_Project2-3.ipynb.

For the CNN model, open and run fashion-mnist_tensorflow_cnn-2.ipynb.

📊 Results and Conclusion

Our analysis showed a clear difference in performance between the two models.

Random Forest: Achieved an accuracy of approximately 82%. The model performed well but struggled to distinguish between similar classes like "Shirt" and "T-shirt/top".

Convolutional Neural Network (CNN): Achieved a final test accuracy of over 92%. The CNN was more effective at learning the spatial features in the images, leading to higher accuracy and better performance on this complex dataset.

Conclusion: The CNN model handled the image classification task significantly better than the Random Forest model. This suggests that for complex image data like Fashion-MNIST, deep learning architectures like CNNs are a more appropriate and powerful choice.
