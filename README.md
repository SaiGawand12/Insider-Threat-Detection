# 🔐 Insider Threat Detection (CERT Dataset)

A **Streamlit-based dashboard** for detecting insider threats using machine learning (Isolation Forest + Autoencoder) on CERT-like logon and file access data. The dashboard can also generate **dummy datasets** for demonstration purposes.

---

## **Table of Contents**

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Machine Learning Models](#machine-learning-models)
* [Dashboard Components](#dashboard-components)
* [Demo](#demo)
* [License](#license)

---

## **Overview**

Insider threats occur when employees or users intentionally or unintentionally compromise organizational data. This project detects anomalous behavior in **user logon and file access logs** using:

* **Isolation Forest** → detects global anomalies
* **Autoencoder** → detects subtle behavioral anomalies

The dashboard provides interactive visualization and user-level anomaly monitoring.

---

## **Features**

* Upload your own `logon.csv` and `file.csv`
* Use **dummy dataset generator** if CSVs are not available
* Detect anomalies using:

  * Isolation Forest
  * Autoencoder (Deep Learning)
* Interactive visualization:

  * Scatter plot of anomaly scores
  * Line chart per user
  * Data table of daily anomalies
* Highlight top anomalous users and dates for investigation

---

## **Dataset**

The dashboard expects CSV files with the following columns:

### **Logon CSV (`logon.csv`)**

| Column   | Type   | Description            |
| -------- | ------ | ---------------------- |
| id       | int    | Unique record ID       |
| date     | string | Logon/logoff timestamp |
| user     | string | Username               |
| pc       | string | PC used                |
| activity | string | Logon / Logoff         |

### **File CSV (`file.csv`)**

| Column   | Type   | Description                   |
| -------- | ------ | ----------------------------- |
| id       | int    | Unique record ID              |
| date     | string | File access timestamp         |
| user     | string | Username                      |
| pc       | string | PC used                       |
| filename | string | Accessed file                 |
| activity | string | Activity type (`File Access`) |
| size     | int    | File size in bytes            |

> A **dummy dataset generator** is included for testing without real CERT data.

---

## **Project Structure**

```text
insider-threat-detection/
│
├── data/                         # Optional: store CSVs here
│   ├── logon.csv
│   └── file.csv
│
├── models/                       # Trained autoencoder model
│   └── autoencoder.pth
│
├── insider_threat_detection.ipynb # Jupyter notebook for training
├── insider_threat_detection.py    # Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

**Explanation:**

* `data/` → place your CSV datasets here (optional)
* `models/` → stores the trained PyTorch autoencoder model
* `insider_threat_detection.ipynb` → notebook to preprocess, train, and save the autoencoder
* `insider_threat_detection.py` → Streamlit app for interactive analysis
* `requirements.txt` → required Python packages

---

## **Installation**

1. Clone this repository:

```bash
git clone https://github.com/SaiGawand12/Insider-Threat-Detection.git
cd Insider-Threat-Detection
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the autoencoder model using the Jupyter notebook:

```bash
jupyter notebook insider_threat_detection.ipynb
```

This will save `models/autoencoder.pth`.

---

## **Usage**

Run the Streamlit app:

```bash
streamlit run insider_threat_detection.py
```

* Upload your CSVs or use the **dummy dataset**
* Explore the anomaly table, scatter plot, and user-level line charts
* Identify unusual activity and potential insider threats

---

## **Machine Learning Models**

### **1. Isolation Forest**

* Detects **outliers** in user activity
* Output column: `anomaly_iso` (`1`=normal, `-1`=anomaly)

### **2. Autoencoder**

* Deep learning model trained to **reconstruct normal behavior**
* High reconstruction error → flagged as anomaly
* Output columns:

  * `score` → reconstruction error
  * `anomaly_auto` → `1`=anomaly, `0`=normal

---

## **Dashboard Components**

1. **Anomaly Detection Table** → top anomalous days and users
2. **Scatter Plot** → visualization of reconstruction errors
3. **Line Chart per User** → trend analysis for selected user
4. **Dummy Dataset Generator** → automatic dataset if no CSVs provided

---

## **Demo**

*(Replace these with actual screenshots from your Streamlit app)*

![Dashboard Screenshot](docs/screenshot_dashboard.png)
![User Trend Screenshot](docs/screenshot_user.png)

---

## **Requirements**

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
torch
streamlit
```

---

## **License**

MIT License © 2025
Use freely for research, learning, or internal security projects.
