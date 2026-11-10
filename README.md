# WhereAmI – Intelligent Indoor Localization Using Wi-Fi Fingerprints

An intelligent indoor localization system that uses Wi-Fi Received Signal Strength Indicator (RSSI) measurements and Machine Learning algorithms to estimate the indoor location of a device.

WhereAmI explores how Wi-Fi fingerprinting can be used to identify a building and floor, estimate spatial coordinates, and analyze the reliability of indoor positioning predictions.

---

## Overview

GPS provides reliable positioning in outdoor environments but often struggles indoors due to signal attenuation, multipath propagation, and obstacles such as walls and ceilings.

**WhereAmI** addresses this challenge using Wi-Fi fingerprinting. Instead of relying on GPS, the system analyzes RSSI measurements collected from nearby Wireless Access Points (WAPs) to learn location-specific signal patterns.

The project evaluates multiple Machine Learning algorithms for indoor localization and investigates their performance under different signal conditions.

## Objectives

- Understand and analyze Wi-Fi RSSI fingerprint data.
- Perform preprocessing and feature engineering on high-dimensional wireless signal measurements.
- Classify indoor locations based on building and floor information.
- Estimate indoor spatial coordinates using regression models.
- Compare different Machine Learning algorithms.
- Analyze model robustness under missing and noisy Wi-Fi signals.
- Explore clustering patterns in Wi-Fi fingerprint data.
- Investigate prediction confidence and uncertainty estimation.

## Dataset

**UJIIndoorLoc – Indoor Localization Database**

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
- Training samples: 19,937
- Validation samples: 1,111
- Total features: 529
- Wi-Fi RSSI features: 520
- Metadata columns: 9
- Evaluated building-floor classes: 13

Each observation represents a Wi-Fi fingerprint collected at a particular indoor location.

The dataset contains RSSI measurements from 520 Wireless Access Points (WAPs), along with location-related metadata.

**Important:** A value of `100` in the RSSI features indicates that the corresponding WAP was not detected. It does not represent an actual signal strength of 100 dBm.

## Methodology

The project follows the workflow below:

1. **Dataset Acquisition:** Load the UJIIndoorLoc training and validation datasets.
2. **Exploratory Data Analysis:** Examine class distributions, access point availability, RSSI distributions, and feature correlations.
3. **Data Preprocessing:** Select the 520 RSSI features, convert missing-signal indicators (`100`) into missing values, and perform median imputation.
4. **Feature Scaling:** Standardize RSSI features for distance-based algorithms and PCA.
5. **Classification:** Train and evaluate multiple algorithms to predict building-floor classes.
6. **Dimensionality Reduction:** Apply Principal Component Analysis (PCA) to investigate the effect of reduced feature dimensions.
7. **Coordinate Regression:** Predict indoor spatial coordinates using regression algorithms.
8. **Uncertainty Analysis:** Examine prediction confidence and split conformal prediction.
9. **Robustness Analysis:** Evaluate model performance under simulated missing and noisy RSSI measurements.
10. **Clustering:** Apply K-Means to explore patterns in the reduced feature space.

All preprocessing transformations are fitted using training data and then applied to the validation data.

## Machine Learning Algorithms

The following algorithms were implemented and evaluated:

| Algorithm | Task |
|---|---|
| K-Nearest Neighbors (KNN) | Classification |
| PCA + KNN | Classification |
| Gaussian Naive Bayes | Classification |
| Random Forest Classifier | Classification |
| KNN Regressor | Coordinate Regression |
| Random Forest Regressor | Coordinate Regression |
| K-Means | Unsupervised Clustering |

## Experimental Results

### 1. Classification Performance

The classification models were evaluated using Accuracy and Balanced Accuracy.

| Model | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Most Frequent Baseline | 3.60% | 7.69% |
| KNN (k = 5) | 77.86% | 79.41% |
| PCA + KNN (60 components) | 79.12% | 80.46% |
| Gaussian Naive Bayes | 37.89% | 46.74% |
| **Random Forest** | **91.00%** | **89.89%** |

**Best Performing Model: Random Forest Classifier**

The Random Forest classifier achieved the highest classification accuracy of **91.00%** and a balanced accuracy of **89.89%**.

The close values indicate that the model performs relatively consistently across the evaluated building-floor classes.

The Random Forest classifier was configured with:

- Number of estimators: 250
- Maximum features: `sqrt`
- Minimum samples per leaf: 1
- Class weighting: `balanced_subsample`
- Random state: 42

### 2. Coordinate Regression Performance

Regression models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| KNN Regressor | 10.9521 | 22.2244 | 0.943819 |
| **Random Forest Regressor** | **10.2505** | **19.9886** | **0.954986** |

**Best Performing Model: Random Forest Regressor**

The Random Forest Regressor achieved an R² score of approximately 0.955, indicating that it explains a high proportion of the variance in the target coordinates.

*Note: The coordinate errors are expressed in the dataset's coordinate units. They should not be interpreted as metres without verifying the coordinate system and its scale.*

### 3. Robustness Analysis

The robustness of the KNN classifier was evaluated by artificially introducing missing RSSI measurements and Gaussian noise.

**A. Effect of Missing Wi-Fi Measurements**

| Artificially Masked Signals | Accuracy |
|---|---:|
| 0% | 77.86% |
| 10% | 75.25% |
| 20% | 69.04% |
| 30% | 64.54% |
| 40% | 59.05% |
| 50% | 52.48% |

**B. Effect of Gaussian Noise**

| Gaussian Noise Standard Deviation | Accuracy |
|---|---:|
| 0 dB | 77.86% |
| 2 dB | 77.14% |
| 5 dB | 74.62% |
| 8 dB | 72.10% |
| 12 dB | 67.33% |

The results demonstrate that classification performance decreases as signal information is removed or corrupted. However, the model retains some predictive capability under the tested perturbations.

*These experiments use synthetic signal perturbations and do not represent measurements from actual real-world signal failures.*

### 4. Uncertainty Estimation

Prediction confidence was investigated using KNN class-vote proportions and split conformal prediction.

- Mean maximum KNN class-vote confidence: 87.70%.
- KNN classification accuracy: 77.86%.
- Nominal conformal coverage: 90%.
- Observed validation coverage: 62.56%.
- Mean prediction set size: 0.68.
- Conformal threshold: 0.1846.

The observed conformal coverage was below the nominal 90% target. Therefore, the experiment does not demonstrate the intended coverage on the evaluated validation data. The calibration procedure, implementation, and data-distribution assumptions require further investigation.

The maximum KNN vote proportion is treated as a confidence proxy, not a calibrated probability.

### 5. K-Means Clustering

K-Means clustering was performed on a reproducible subset of 5,000 samples using the first 20 PCA components.

| Number of Clusters (K) | Silhouette Score |
|---|---:|
| 3 | 0.7312 |
| 5 | 0.3458 |
| 8 | 0.3413 |
| 10 | 0.2993 |

The highest silhouette score was obtained for K = 3.

These clusters represent patterns in the reduced feature space and should not be interpreted as actual building or floor labels.

## Technologies and Libraries

- **Programming Language:** Python
- **Environment:** Jupyter Notebook / Google Colab
- **Data Processing:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib

## Project Structure

A suggested repository structure is:

```text
WhereAmI-Indoor-Localization/
│
├── notebooks/
│   └── WhereAmI_PRML_Project.ipynb
│
├── results/
│   ├── figures/
│   └── evaluation_results/
│
├── models/
│
├── README.md
├── requirements.txt
└── .gitignore
```

The folders for models and results can contain the exported artifacts and plots generated by the notebook.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/WhereAmI-Indoor-Localization.git
cd WhereAmI-Indoor-Localization
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the main libraries directly:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib jupyter
```

### 4. Download the dataset

Download the UJIIndoorLoc dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).

Place the training and validation files in the dataset location expected by the notebook, or update the dataset paths accordingly.

### 5. Run the notebook

```bash
jupyter notebook
```

Open the WhereAmI project notebook and execute the cells sequentially.

Google Colab can also be used to run the notebook.

## Key Findings

- Wi-Fi RSSI fingerprinting can be used for indoor building-floor classification.
- Random Forest outperformed the other evaluated classification models.
- PCA-based dimensionality reduction retained useful information for KNN classification.
- Random Forest regression achieved an R² score of approximately 0.955 for coordinate estimation.
- Missing and noisy Wi-Fi signals negatively affected classification accuracy.
- K-Means revealed exploratory structure in the reduced feature space.
- The conformal prediction experiment did not achieve its nominal coverage target, highlighting the importance of reliable uncertainty estimation.

## Limitations and Future Improvements

Although the experimental results demonstrate the potential of Wi-Fi fingerprint-based indoor localization, several improvements are possible:

- Evaluate localization using additional datasets and independent environments.
- Investigate alternative feature-selection and dimensionality-reduction techniques.
- Explore advanced models such as XGBoost and neural networks.
- Improve robustness against missing access points and changing signal conditions.
- Investigate calibration methods to improve prediction confidence.
- Audit the conformal prediction implementation and calibration assumptions.
- Develop a real-time indoor localization interface using live Wi-Fi measurements.
- Explore deployment on mobile or embedded devices.

## Conclusion

WhereAmI demonstrates the application of Machine Learning techniques to indoor localization using Wi-Fi RSSI fingerprints.

By evaluating classification, regression, dimensionality reduction, clustering, robustness, and uncertainty estimation, the project provides an experimental study of different approaches to indoor positioning.

The Random Forest classifier achieved **91.00% accuracy**, while the Random Forest regressor achieved an **R² score of 0.954986** on the evaluated validation data.

The project highlights the potential of Wi-Fi fingerprinting as an alternative approach to indoor positioning where conventional GPS-based localization may be unreliable.

---

## Author

**Mahidhar Reddy Ailuri**  
B.Tech – Electrical, Electronics and Communication Engineering  
Indian Institute of Technology Dharwad

---

## Acknowledgements

- UJIIndoorLoc dataset contributors.
- UCI Machine Learning Repository.
- Faculty and mentors associated with the Pattern Recognition and Machine Learning (PRML) course.

---

*Developed as part of the Pattern Recognition and Machine Learning (PRML) course project.*