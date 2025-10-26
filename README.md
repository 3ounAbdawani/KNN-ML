# KNN.ipynb: K-Nearest Neighbors Classifier Implementation (From Scratch)

## 📋 Project Description
This Jupyter notebook, `KNN.ipynb`, provides a complete, **from-scratch implementation** of the **K-Nearest Neighbors (KNN)** classification algorithm. It is designed to illustrate the core mechanics of KNN, including distance calculation, finding nearest neighbors, and voting for the final prediction.

The notebook demonstrates the following steps:
1.  **Custom Class Implementation**: Creating a Python class named `KNNClassifier` that includes `fit` and `predict` methods.
2.  **Dataset Preparation**: Loading the **Iris dataset** and splitting it into training and testing sets.
3.  **Model Training and Prediction**: Instantiating and using the custom KNN classifier.
4.  **Evaluation**: Implementing a **custom function** to generate a confusion matrix.
5.  **Visualization**: Displaying the results using a plotted **Confusion Matrix**.

---

## 🛠️ Custom `KNNClassifier` Implementation

The heart of this notebook is the `KNNClassifier` class.

### Key Methods:
* **`__init__(self, k=3)`**: Initializes the classifier with the number of neighbors, $k$, defaulting to **3**.
* **`fit(self, X, y)`**: Simply stores the training data (`X_train` and `y_train`) as KNN is a lazy learning algorithm.
* **`predict(self, X_test)`**:
    * Calculates the **Euclidean distance** between each test point and all training points. The distance is calculated as $\sqrt{\sum (x - x_{\text{train}})^2}$.
    * Finds the indices of the $k$ nearest neighbors using `np.argsort`.
    * Determines the most common label among the $k$ neighbors using the `collections.Counter` module.

---

## 📊 Dataset and Setup
### Dataset
The project uses the well-known **Iris dataset**, which is loaded via `sklearn.datasets.load_iris`.

### Data Split
The data is split into training and testing sets with a `test_size` of **0.2** (20% for testing) and uses a `random_state` of 42 for reproducibility.

### Evaluation
The model's performance on the test set is evaluated by:
1.  **Custom Confusion Matrix**: A function is defined within the notebook to calculate the confusion matrix manually.
2.  **Visualization**: The resulting confusion matrix is visualized using `matplotlib.pyplot.imshow`.

---

## 💻 Dependencies
To run this notebook, you will need a standard Python environment with the following libraries:

```bash
pip install numpy matplotlib scikit-learn
