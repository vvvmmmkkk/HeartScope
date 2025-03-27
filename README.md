# ECG-Arrhythmia-Classifier

### Classifying Arrhythmias from ECG Signals 📈

This project, HeartScope, is a machine learning-based ECG arrhythmia prediction system designed to analyze electrocardiogram (ECG) data and classify heartbeats into different categories. The goal is to provide an early warning system for cardiac abnormalities and assist healthcare professionals in detecting arrhythmias.

The main goal of this project is to help healthcare professionals detect and classify arrhythmias much more accurately, which helps in improving the patient care with AI.
It allows the user to input the ECG data as a file or by manually entering the ECG values.

## 📝 **Problem Description**

Arrhythmias are abnormal heart rhythms that can vary in severity, from benign to life-threatening. Detecting them early is essential for timely medical intervention and improved patient care.

### **Objective**

This project aims to develop a machine learning model that classifies arrhythmias based on ECG signal features.

### 📊 **Dataset**

This project uses an ECG dataset from Kaggle, which is based on the **MIT-BIH Arrhythmia Dataset** from PhysioNet. The dataset contains essential features derived from two-lead ECG signals(lead II and lead V5), which are used to train the arrhythmia classification model.

- **Number of records**: 460846
- **Number of features**: 33

For more detailed information about the dataset, including explanations of the columns, please refer to the [data folder](./data/README.md).

Ready to see how AI can help detect arrhythmias and save lives? Let’s get started! ✨



## 🔧 Tools & Techniques

To bring this project to life, I used:

- **Containerization:** Docker and Docker Compose
- **Web Application Framework (Local Deployment):** Flask (for local web deployment)
- **Web Application Framework (Cloud Deployment):** Streamlit (for cloud-based web deployment)

## ✨ Setup

### **Local Setup**

#### **Clone the Repository**:

```bash
git clone https://github.com/vvvmmmkkk/HeartScope.git
cd HeartScope
```


## Exploratory Data Analysis and Modeling

The exploratory data analysis and modeling are done in the [notebooks directory](notebooks/). The exploratory data analysis and model building are done in the `notebook.ipynb` notebook.

The notebook directory also contains the model called `model.pkl`, where the model from the `notebook.ipynb` is stored.

It also contains the training script (which contains the script for training the model with the best AUC) which you can run by running `python train.py` in the terminal

```bash
python train.py
```

## Get Going

Ready to dive into your project? Here’s a quick guide to get you started.

### 📁 **Deployment**

### **Local Deployment**

- **Tools Used**: Flask for building your web app and Docker for containerizing it.
- **Where to Find It**: Head over to the [deployment/local_deployment](deployment/local_deployment) folder.

The README in that folder covers everything you need to get your app running locally.

It’s got the details for setting up Flask and Docker, so you can test things out on your own machine.

### **Cloud Deployment**

- **Tools Used**: Streamlit community cloud for hosting your app and Streamlit for the web interface.
- **Where to Find It**: Navigate to the [deployment/web_deployment](deployment/web_deployment) folder.

The README in that folder guides you through deploying your app using Streamlit. It’s perfect for getting your app live on the cloud.


