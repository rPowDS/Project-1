<p align="center">
  <img src="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png" width="700">
</p>

# Analyzing Fashion-MNIST with Machine Learning
### W200 Project 2 | Trevor Lang, Ryan Powers, Carmen Liang
> Comparing the performance of Random Forest and Convolutional Neural Network (CNN) models for classifying 70,000 fashion product images.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-F89939.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Project-Complete-green.svg" alt="Project Complete">
</p>

---

## 🚀 At-a-Glance

This repository explores two machine learning approaches for the Fashion-MNIST dataset. Here's what you'll find:

* **Project Report:** [**Read the full PDF Report**](Copy%20of%20DS200%20Project2%20Report.docx.pdf)
* **Project Slides:** [**View the Presentation Slides**](DS200%20Analyzing%20Fashion-MNIST%20dataset.pdf)
* **Notebook 1:** [**Random Forest Classifier**](Randomforest_Project2-3.ipynb)
* **Notebook 2:** [**CNN Classifier**](fashion-mnist_tensorflow_cnn-2.ipynb)

---

## 📖 Project Overview

The goal of this project was to address the following research question:

> *How does the distribution of images across different classes in the Fashion-MNIST dataset affect the performance of Random Forest and Convolutional Neural Network (CNN) algorithms?*

We implemented, trained, and evaluated both models from scratch to compare their effectiveness in a real-world image classification task.

## 💾 The Dataset

We used the **Fashion-MNIST dataset**, a popular "drop-in replacement" for the original MNIST dataset, which is more complex and provides a better benchmark for modern ML algorithms.

* **Contents:** 70,000 grayscale images (60k training, 10k test)
* **Image Size:** $28 \times 28$ pixels
* **Classes:** 10 categories of fashion products
    * `0`: T-shirt/top
    * `1`: Trouser
    * `2`: Pullover
    * `3`: Dress
    * `4`: Coat
    * `5`: Sandal
    * `6`: Shirt
    * `7`: Sneaker
    * `8`: Bag
    * `9`: Ankle boot

## 🤖 Models & Methodology

We implemented two distinct models to compare a classical machine learning approach with a deep learning approach.

1.  **Random Forest (RF):** An ensemble learning method that builds multiple decision trees. We used this to see how a strong, non-neural-network model would perform on raw pixel data.
    * *Implementation:* `scikit-learn`

2.  **Convolutional Neural Network (CNN):** A deep learning architecture specifically designed for computer vision. CNNs are built to automatically and adaptively learn spatial hierarchies of features from images.
    * *Implementation:* `TensorFlow` and `Keras`

## 📊 Results: RF vs. CNN

Our analysis showed a clear performance difference. The CNN, which is designed for spatial data like images, significantly outperformed the Random Forest model.

### Performance Comparison

| Model | Final Accuracy | Key Observation |
| :--- | :---: | :--- |
| **Random Forest** | ~82% | Struggled to distinguish between similar classes (e.g., "Shirt" vs. "T-shirt/top"). |
| **CNN** | **~92%** | Effectively learned image features, leading to high accuracy and better generalization. |

### Visualizing the Results

* **(Pro-Tip):** This is where you should add the best images from your reports! I'll put placeholders here for you. You can add your images by dragging and dropping them into an issue on your GitHub repo to get a URL, or just adding them to the repo and linking to them directly.*

| Random Forest Prediction Plot | CNN Prediction Plot |
| :---: | :---: |
| ![RF Prediction Plot](path/to/your/RF_Prediction_Plot.png) | ![CNN Prediction Plot](path/to/your/CNN_Prediction_Plot.png) |
| *The RF model making several errors on similar items. Red text indicates a misclassification.* | *The CNN model correctly classifies a wider variety of items.* |

### Conclusion

The CNN model handled the image classification task significantly better than the Random Forest model. This suggests that for complex, high-dimensional image data like Fashion-MNIST, **deep learning architectures like CNNs are a more appropriate and powerful choice** as they can learn relevant features automatically.

## 🛠️ How to Run This Project

To replicate our analysis and run the models:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/UC-Berkeley-I-School/Project2-Powers-Lang-Liang.git](https://github.com/UC-Berkeley-I-School/Project2-Powers-Lang-Liang.git)
    cd Project2-Powers-Lang-Liang
    ```

2.  **Install the required libraries.** We recommend using a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebooks:**
    * For the Random Forest model: `Randomforest_Project2-3.ipynb`
    * For the CNN model: `fashion-mnist_tensorflow_cnn-2.ipynb`
