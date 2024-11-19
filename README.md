# SCT_ML_3  

## Project Title:  
**Cat vs Dog Image Classification with SVM**

---

## Project Description:  
This project focuses on classifying images of cats and dogs using a Support Vector Machine (SVM). The goal was to create a simple yet effective image classification model, preprocess the dataset for consistency, and evaluate the performance using metrics such as accuracy, precision, recall, and F1-score. This project demonstrates the application of classical machine learning techniques for image classification.

---

## Dataset:  
The dataset used for this project is the **Kaggle Cats vs Dogs Dataset**. It contains 25,000 labeled images of cats and dogs.  
You can find and download the dataset here: [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/competitions/dogs-vs-cats)

---

## Requirements:  
The following libraries and tools are required to run this project:  
- Python 3.7 or above  
- NumPy  
- OpenCV  
- scikit-learn  
- matplotlib  
- seaborn  

To install the dependencies, use:  
```bash
pip install -r requirements.txt
```

---

## Steps to Reproduce:  
Follow these steps to run the project locally:  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/SCT_ML_3.git
   cd SCT_ML_3
   ```

2. Install the required Python libraries:  
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Kaggle dataset:  
   - Go to [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/competitions/dogs-vs-cats).  
   - Download the dataset and extract it to a folder (e.g., `train/`).

4. Update the dataset path in the `Cats_Dogs_Classification.ipynb` file:  
   Modify the `train_path` variable to point to your dataset folder.  

5. Run the Jupyter Notebook:  
   ```bash
   jupyter notebook Cats_Dogs_Classification.ipynb
   ```

6. Follow the steps in the notebook to preprocess the dataset, train the SVM model, and evaluate its performance.

---

## Results:  

### **Accuracy**:  
The model achieved **57.20% accuracy** on the test dataset.

### **Classification Report:**  
```
              precision    recall  f1-score   support

           0       0.55      0.63      0.59       971
           1       0.60      0.52      0.55      1029

    accuracy                           0.57      2000
   macro avg       0.57      0.57      0.57      2000
weighted avg       0.58      0.57      0.57      2000
```

### **Confusion Matrix:**  
![Confusion Matrix](results/confusion_matrix.png)

The confusion matrix shows the model's performance on correctly classifying cats and dogs.

---

## Conclusion:  
This project demonstrates the use of Support Vector Machines (SVMs) for image classification tasks. Although the achieved accuracy is modest, it showcases the challenges of classical machine learning techniques for image data and highlights the importance of feature extraction and dataset preprocessing.

Future improvements may include feature engineering, hyperparameter tuning, or using deep learning models like CNNs for better performance.
