# 🛰️ IndiCrimeTopology  
### Unveiling Cybercrime Patterns in India: A Clustering-Based Geographic and Visual Analytical Approach 

An ML-powered geospatial exploration of India’s cybercrime landscape using **K-Means** and **Spectral Clustering**. This project transforms raw cybercrime data into risk stratification maps, hotspot detection models, and dominant motive insights — enabling data-driven cybersecurity strategy and regional policy planning.

---

## 📌 Project Overview

With rapid digital transformation and increasing internet penetration in India, cybercrime incidents have grown significantly. However, regional patterns, evolving hotspots, and motive distributions remain underexplored.

This project performs a **pan-India cybercrime pattern analysis (2017–2020)** using unsupervised machine learning techniques to:

- 📊 Categorize States/UTs into risk levels  
- 🔥 Detect cybercrime hotspots  
- 🧠 Identify dominant cybercrime motives  
- ⚖️ Compare clustering algorithm performance  

---

## 🎯 Objectives

- Analyze temporal growth of cybercrime (2017–2020)
- Perform geospatial risk segmentation
- Identify persistent and intermittent cybercrime hotspots
- Determine dominant cybercrime motives per State/UT
- Compare K-Means and Spectral Clustering performance

---

## 🧠 Machine Learning Techniques Used

### 1️⃣ K-Means Clustering
- Distance-based clustering
- Used for:
  - Risk level categorization (K=4)
  - Hotspot detection (K=2)
  - Motive grouping (K=3)

### 2️⃣ Spectral Clustering
- Graph-based clustering using similarity matrix and Laplacian
- Captures subtle nonlinear regional relationships
- Demonstrated superior overall performance

---

## 📂 Dataset

- Source: Public cybercrime dataset (2017–2020)
- Coverage: 36 Indian States and Union Territories
- Features:
  - 18 cybercrime motive categories
  - Population
  - Total cases
  - Latitude & Longitude
  - Normalized Crime Rate (NCR)

### 📐 Normalized Crime Rate Formula

- NCR = (Total Cybercrimes / Population) × 100,000

---

## 📊 Analytical Cases

### 🔹 Case 1: Risk Level Categorization

**Features Used:**  
- Latitude  
- Longitude  
- Normalized Crime Rate  

**Clusters:** K = 4  

- 🟢 Low Risk  
- 🟠 Moderate Risk  
- 🔴 High Risk  
- 🔴 Very High Risk  

📍 Visualized using interactive **Folium maps**

---

### 🔹 Case 2: Cybercrime Hotspot Identification

**Features Used:**  
- Latitude  
- Longitude  
- Total Crime Count  

**Clusters:** K = 2  

- 🔵 Non-Hotspot  
- 🔴 Hotspot  

**Key Observations:**
- Karnataka & Uttar Pradesh → Consistent hotspots (2017–2020)
- Maharashtra → Hotspot only in 2017
- Spectral Clustering detected more subtle hotspots than K-Means

---

### 🔹 Case 3: Dominant Cybercrime Motive Prediction

- For each State/UT, the most prevalent motive is determined using:

  Dominant Motive = argmax(Motive Counts)

**Key Findings:**
- Fraud dominates in 26/36 States by 2020
- Sexual exploitation increased significantly by 2020
- Regional variation observed in smaller UTs and NE states

---

## 📈 Performance Comparison

| Method    | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| K-Means    | 0.60     | 1.00      | 0.20   | 0.33     |
| Spectral   | 0.92     | 0.86      | 0.99   | 0.92     |

### 🔎 Interpretation

- **Spectral Clustering** → Better overall detection & recall
- **K-Means** → Higher precision (more conservative hotspot detection)

Spectral is preferable for comprehensive detection; K-Means is preferable when precision is critical.

---

## 📊 Exploratory Insights

- Cybercrime cases more than doubled (2017 → 2020)
- Major spike observed between 2018–2019
- Uttar Pradesh, Karnataka, Maharashtra → High contributors
- Telangana → Emerging cybercrime hotspot
- Northeast states → Consistently low/moderate risk

---

## 🗺️ Visualization Tools

- Folium (Interactive Geospatial Maps)
- Matplotlib
- Seaborn
- Line Charts, Bar Plots

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Folium  

---

## 📌 Key Contributions

✔ Pan-India cybercrime clustering analysis  
✔ Risk stratification mapping  
✔ Hotspot comparison study  
✔ Policy-driven data interpretation  

---

## 🚀 Future Scope

- Incorporate post-2020 cybercrime surge
- Real-time monitoring dashboard
- Predictive modeling (Time-Series / LSTM)
- Integration with law enforcement GIS systems
- AI-driven early warning systems

---

## 🧑‍💻 Author

**Swarnasmita Roy**  
B.Tech – Robotics & Artificial Intelligence  
Focus: Machine Learning, Geospatial Analytics, Cyber Intelligence  

---

## ⭐ Why This Project Matters

This project demonstrates how unsupervised machine learning and geospatial intelligence can:

- Reveal hidden regional cybercrime structures  
- Support evidence-based policymaking  
- Improve cybersecurity resource allocation  
- Enable smarter digital infrastructure planning  

---

If you find this project interesting, feel free to ⭐ star the repository!
