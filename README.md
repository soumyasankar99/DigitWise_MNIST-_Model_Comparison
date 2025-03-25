## 🧠 DigitWise MNIST Comparison – Classic ML vs Deep Learning (with W&B Integration)

Welcome to **DigitWise MNIST Comparison**, a one-stop Jupyter notebook project where we explore, train, compare, and **visually evaluate** multiple machine learning and deep learning models on the MNIST dataset.

What makes this notebook special? It’s not just about accuracy - it’s about **experiment tracking**, **metric logging**, and **visual storytelling** using **[Weights & Biases (wandb)](https://wandb.ai)**.

---

### 📌 Objective

To **compare classical ML models** (like Logistic Regression, KNN, Random Forest) with **a Deep Learning CNN model** on the MNIST digit classification task using:
- **Model performance metrics**
- **Confusion matrices**
- **ROC curves**
- **Real-time experiment tracking with WandB**

The ultimate goal is to **evaluate models both quantitatively and visually** in a reproducible and centralized environment.

---

### 🔧 Tools & Libraries Used

- Python (Jupyter Notebook)
- Scikit-learn (Logistic Regression, KNN, Random Forest)
- TensorFlow & Keras (Deep Learning CNN)
- Matplotlib & Seaborn (Visualization)
- Weights & Biases (Logging & Experiment Tracking)

---

### 🧭 Workflow Overview

This notebook follows a well-structured, multi-stage pipeline:

#### 1. 📥 **Dataset Loading & Preprocessing**
- Used the MNIST dataset via `tensorflow.keras.datasets`.
- Scaled images to [0, 1] and flattened for classical models.
- One-hot encoded labels for DL model.

#### 2. 🧪 **Model Training**

| Model                | Framework    | Notes                        |
|---------------------|--------------|------------------------------|
| Logistic Regression | scikit-learn | Simple linear model          |
| KNN (k=3,5)          | scikit-learn | Distance-based classifier    |
| Random Forest        | scikit-learn | Ensemble of decision trees   |
| CNN                 | Keras        | Deep model with conv layers  |

Each classical model is trained using `fit()` with default or tuned parameters.

The CNN model uses:
- 2D Convolution layers
- MaxPooling
- Dense layers with softmax
- ReLU activations
- Adam optimizer

#### 3. 🔍 **Evaluation**

For **each model**, we evaluated:
- Accuracy
- Confusion Matrix
- ROC Curve (multi-class using One-vs-Rest strategy)
- Classification Report (precision, recall, F1)

#### 4. 📊 **WandB Integration**

This is the heart of the project.

We used **Weights & Biases** to:
- Initialize experiment tracking for the entire notebook
- Log **hyperparameters** like batch size, epochs, optimizer
- Log **accuracy** of each classical model
- Use a **custom Keras callback** to log DL model metrics after each epoch
- Log **confusion matrices** and **ROC curves** for each model
- Optionally upload model weights/artifacts

##### ✅ Custom Logging Examples:
- `wandb.log()` for manual logging of metrics
- `tf.keras.callbacks.Callback` subclass to log deep learning metrics
- Custom function to log confusion matrix as W&B image
- ROC curve visualizations with `wandb.Image()` support

---

### 📌 Example Logs on WandB

- **Run Metadata**: Optimizer, Epochs, Model Name, etc.
- **Metric Graphs**: Accuracy, Loss per epoch
- **Visuals**: Confusion Matrices, ROC curves
- **Artifacts** (optional): Trained model weights/files

---

### 🗂️ Structure (All in One Notebook)

```
digitwise_mnist_comparison.ipynb
├── Dataset preprocessing
├── Classical model training
├── CNN model training
├── Evaluation metrics
├── WandB tracking & visualization
```

---

### 📈 Sample Results (You may customize this)

| Model                | Accuracy  |
|---------------------|-----------|
| Logistic Regression | 91.3%     |
| KNN (k=5)           | 96.8%     |
| Random Forest       | 96.2%     |
| CNN (2 conv layers) | 98.7%     |

---

### 📎 How to Run This Notebook

1. Clone the repo and open the notebook.
2. Make sure you have the following installed:
```bash
pip install tensorflow scikit-learn matplotlib seaborn wandb
```
3. Log in to WandB (create account if needed):
```bash
wandb login
```
4. Run the notebook cells one by one to:
   - Load data
   - Train models
   - Log and visualize everything on WandB

---

### 📦 Future Work

- Tune hyperparameters of CNN using W&B Sweeps
- Add more models (SVM, XGBoost)
- Apply PCA or TSNE for data visualization
- Convert to modular Python scripts

---

### 👨‍💻 Author

Crafted with ❤️ by Soumya Sankar
Open for collaboration, PRs, and feedback!

---

### 📄 License

MIT License – use it, fork it, share it!

Just say the word 🚀

----
