
# 🌍 Urban Land Cover Classification Project

This repository contains a machine learning project focused on classifying urban land cover types using high-resolution aerial imagery. The project utilizes a dataset from the UCI Machine Learning Repository, employing multiple classification algorithms to analyze and predict land cover categories. The aim is to inform sustainable urban development and improve climate adaptation strategies (https://archive.ics.uci.edu/dataset/295/urban+land+cover).

## 📋 Project Overview

In this project, we aim to classify urban land cover into nine distinct categories:

- **Categories:** Concrete, Car, Asphalt, Building, Tree, Grass, Shadow, Pool, Soil

Accurate classification of urban land cover can help city planners understand the distribution of different surfaces, such as green spaces, asphalt, and buildings. This is crucial for mitigating the urban heat island effect, enhancing disaster preparedness, and improving overall quality of life.

### 🎯 Problem Statement

How can we accurately classify urban land cover types using aerial imagery data to support sustainable urban development and climate adaptation efforts?

### 🌍 Motivation

Urban areas face challenges like heat absorption, reduced biodiversity, and increased flooding risks due to impermeable surfaces. Understanding land cover distribution can guide the implementation of reflective materials, green spaces, and sustainable urban planning strategies, especially in rapidly urbanizing cities like Singapore.

## 📈 Dataset

The dataset used for this project is the **Urban Land Cover** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/295/urban+land+cover). It consists of training and testing data with labeled classes representing different urban land cover types.

### Features

- **Input Variables:** Spectral, spatial, and texture features extracted from aerial images.
- **Target Variable:** Land cover category (e.g., asphalt, tree, building).

## 🛠️ Tools and Technologies

- **Python 3.10**
- **Jupyter Notebook**
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`

## 🚀 Getting Started

### Prerequisites

To run this project, you need Python installed along with the required libraries. Install the dependencies using:

```bash
pip install -r requirements.txt
```

Remember to change the directory to your data location!

### Running the Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook data_with_everything.ipynb
```

Follow the cells in the notebook to explore data analysis, visualization, and model building.

## 🔍 Project Structure

```plaintext
├── code.ipynb                       # Main analysis notebook
├── README.md                        # Project documentation
├── data/                            # Directory containing raw and cleaned data
└── figures/                         # Directory for visual outputs
```

## 🧹 Data Cleaning

The dataset was thoroughly cleaned and preprocessed, including:

- Handling missing values
- Normalizing and scaling features
- Label encoding the target variable

The cleaned data improved the performance of the classifiers, reducing noise and outliers.

## 📊 Exploratory Data Analysis (EDA)

The EDA provided insights into the distribution of land cover classes and key features:

- **Basic Statistics:** Summary statistics of each feature
- **Visualization:** Before and after plots of data cleaning, class distribution histograms

Key findings:
- The dataset contains an imbalance in class distribution, with fewer samples for certain categories like `pool` and `soil`.
- Visualizations revealed distinct patterns between classes, aiding feature selection.
- Most imporant variables for determing class are: 'Mean_R',  'Mean_NIR', 'Bright' and 'NDVI' (ordered) 

## 🤖 Machine Learning Techniques

We implemented three classification algorithms to solve the land cover classification problem:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **K-Nearest Neighbors (K-NN)**

### Random Forest Classifier

The Random Forest Classifier was chosen as the primary model due to its robustness and ability to handle imbalanced datasets. It combines multiple decision trees and averages their predictions, reducing overfitting and improving accuracy.

- **Classes:**  
  `'asphalt': 0, 'building': 1, 'car': 2, 'concrete': 3, 'grass': 4, 'pool': 5, 'shadow': 6, 'soil': 7, 'tree': 8`

### Evaluation Metrics

- **Precision:** Accuracy of positive predictions
- **Recall:** Proportion of actual positives correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of samples per class

### Model Performance

The Random Forest model achieved strong results with an **overall weighted average F1-score of 0.85**, indicating good classification performance. The model performed well on classes like `car`, `pool`, and `shadow`, but had lower accuracy for the `soil` class due to imbalanced data.

## 📊 Results and Visualizations

The project includes various visualizations:

- **Confusion Matrix:** Illustrates the classification performance
- **Precision-Recall Curve:** Shows the trade-off between precision and recall
- **Feature Importance Plot:** Highlights the most significant features used by the Random Forest model

## 📑 Key Insights

- The Random Forest Classifier outperformed other models in terms of accuracy and generalization.
- Imbalanced classes, especially `soil`, affected the model’s performance, indicating a need for more data or resampling techniques.
- Urban land cover classification provides actionable insights for sustainable city planning, particularly in managing green spaces and mitigating the urban heat island effect.

## 💡 Future Work

- Explore additional data augmentation techniques to balance the dataset.
- Integrate more advanced machine learning models, such as Gradient Boosting or Neural Networks.
- Investigate the impact of seasonal variations on urban land cover classification.

## 🤝 Acknowledgments

- **Team Members:** Ida Andersen (N2400447C), Katja Hold (N2400840K), Akanksha Mathur (U2323265C)
- Special thanks to our course instructors for guidance on machine learning techniques.
