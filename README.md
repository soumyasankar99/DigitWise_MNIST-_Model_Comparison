<img width="928" alt="wandb-7" src="https://github.com/user-attachments/assets/a19732e7-8dd9-42d3-a8a0-ff3edc7962df" />


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

<img width="960" alt="wandb" src="https://github.com/user-attachments/assets/1e60693f-5625-48ee-8dc1-d5540ba0a6ba" />

#### 3. 🔍 **Evaluation**

For **each model**, we evaluated:
- Accuracy
- Confusion Matrix
- ROC Curve (multi-class using One-vs-Rest strategy)
- Classification Report (precision, recall, F1)

<img width="960" alt="wandb-1" src="https://github.com/user-attachments/assets/6d9edeb2-4111-41aa-9f7a-16c7ef8d70b1" /> 


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

![media_images_Random Forest Confusion Matrix_6_a38f25627dbeba8aa223](https://github.com/user-attachments/assets/18bf0167-3531-40a7-be7e-f1e8ecf8c0cf)



![media_images_Random Forest ROC Curve_2_69831bebeee6acf98f38](https://github.com/user-attachments/assets/ee448c73-2948-4420-a447-6409fccf5df2)



---

### 📌 Example Logs on WandB

- **Run Metadata**: Optimizer, Epochs, Model Name, etc.
- **Metric Graphs**: Accuracy, Loss per epoch
- **Visuals**: Confusion Matrices, ROC curves
- **Artifacts** (optional): Trained model weights/files

<img width="960" alt="wandb-2" src="https://github.com/user-attachments/assets/1c1e3ed0-b1e1-4b8c-a0bf-a9c010f88d93" />


<img width="960" alt="wandb-4" src="https://github.com/user-attachments/assets/26d210a1-af83-4159-bd4a-c4e67579a047" />

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

### 📈 Sample Results 

| Model                | Accuracy  |
|---------------------|-----------|
| Logistic Regression | 91.3%     |
| KNN (k=5)           | 96.8%     |
| Random Forest       | 96.2%     |
| CNN (2 conv layers) | 98.7%     |


![download](https://github.com/user-attachments/assets/fce3ec84-7a2d-4f07-a060-9734c8b8f417)

<img width="960" alt="wandb-5" src="https://github.com/user-attachments/assets/6c515603-0875-4852-8b22-67560f9ccb92" />


<img width="960" alt="wandb-6" src="https://github.com/user-attachments/assets/abf5bee4-b6c7-410e-8f40-31ac5a3f6b6f" />


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
