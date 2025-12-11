# 🍷🔬 VINEYARD VOYAGER 🔬🍷

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=8B0000&center=true&vCenter=true&width=800&lines=Discovering+Wine+Families+Through+Unsupervised+ML;Hierarchical+Clustering+%2B+PCA+Visualization;Chemical+Properties+%E2%86%92+Natural+Wine+Groups)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://vineyard-voyager-project.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/vineyard-voyager?style=for-the-badge)](https://github.com/mayank-goyal09/vineyard-voyager/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mayank-goyal09/vineyard-voyager?style=for-the-badge)](https://github.com/mayank-goyal09/vineyard-voyager/network)

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="800">
</div>

### 🍇 **Uncover hidden wine patterns** using **Hierarchical Clustering ML** 🤖
### 🧬 Chemical Properties × AI = **Natural Wine Families** 🔮

---

## 🌟 **WHAT IS THIS?** 🌟

<table>
<tr>
<td>

### 🔮 **The Magic**

This **unsupervised ML project** discovers **natural wine families** using **Hierarchical Clustering** on chemical property data. Unlike supervised learning, the model finds hidden patterns without being told what to look for!

**Think of it as:**
- 🧬 Brain = Hierarchical Clustering Algorithm
- 🍷 Input = Chemical Properties (11+ features)
- 🎨 Output = Natural Wine Groups/Clusters

</td>
<td>

### ⚡ **Key Features**

✅ Unsupervised clustering  
✅ Hierarchical dendrogram visualization  
✅ PCA-based 2D/3D plots  
✅ Interactive Streamlit dashboard  
✅ Red & White wine analysis  
✅ Cluster profiling & insights  

</td>
</tr>
</table>

---

## 🛠️ **TECH STACK** 🛠️

![Tech Stack](https://skillicons.dev/icons?i=python,github,vscode,git)

| **Category** | **Technologies** |
|--------------|------------------|
| 🐍 **Language** | Python 3.8+ |
| 📊 **Data Science** | Pandas, NumPy, Scikit-learn |
| 🎨 **Frontend** | Streamlit |
| 📈 **Visualization** | Matplotlib, Seaborn, Plotly |
| 🧪 **ML Technique** | Hierarchical Clustering, PCA |
| 📦 **Data Source** | UCI Wine Quality Dataset |

---

## 📂 **PROJECT STRUCTURE** 📂

```
🍷 vineyard-voyager/
│
├── 📁 app.py                    # Streamlit web application
├── 📁 main.ipynb                # Model training & EDA notebook
├── 📦 requirements.txt          # Dependencies
├── 📊 winequality-red.csv       # Red wine dataset
├── 📊 winequality-white.csv     # White wine dataset
├── 📄 winequality.names         # Dataset documentation
├── 🚫 .gitignore                # Git ignore rules
└── 📖 README.md                 # You are here!
```

### 📋 **File Descriptions**

| **File** | **Purpose** |
|----------|-------------|
| `app.py` | Interactive Streamlit dashboard with clustering visualization |
| `main.ipynb` | Jupyter notebook with EDA, preprocessing, and model training |
| `winequality-red.csv` | Red wine physicochemical properties (1599 samples) |
| `winequality-white.csv` | White wine physicochemical properties (4898 samples) |
| `requirements.txt` | Python dependencies (pandas, scikit-learn, streamlit, etc.) |

---

## 🚀 **QUICK START** 🚀

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-705f7be0b224.gif" width="600">
</div>

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/vineyard-voyager.git
cd vineyard-voyager
```

### **Step 2: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 3: Run the App** 🎯

```bash
streamlit run app.py
```

### **Step 4: Open in Browser** 🌐

The app will automatically open at: **`http://localhost:8501`**

---

## 🎮 **HOW TO USE** 🎮

<table>
<tr>
<td>

### 🔹 **Explore Mode**

1. Open the live app
2. Select wine type (Red/White)
3. Choose number of clusters
4. View dendrogram & PCA plots
5. Analyze cluster profiles

</td>
<td>

### 🔹 **Developer Mode** 🤓

1. Open `main.ipynb`
2. Run EDA cells
3. Train clustering model
4. Visualize dendrograms
5. Export cluster assignments

</td>
</tr>
</table>

---

## 🧪 **HOW IT WORKS** 🧪

### **Pipeline Breakdown:**

```
1️⃣ Data Loading → Load red/white wine datasets
2️⃣ Preprocessing → Standardize chemical features
3️⃣ Clustering → Apply Hierarchical Clustering
4️⃣ Dimensionality Reduction → PCA for visualization
5️⃣ Visualization → Dendrograms, scatter plots, cluster profiles
6️⃣ Deployment → Interactive Streamlit dashboard
```

### **Chemical Features Used:**

The model analyzes **11 physicochemical properties**:

| Feature | Description |
|---------|-------------|
| Fixed Acidity | Non-volatile acids (tartaric acid) |
| Volatile Acidity | Acetic acid amount (vinegar taste) |
| Citric Acid | Freshness and flavor |
| Residual Sugar | Sugar remaining after fermentation |
| Chlorides | Salt amount |
| Free SO₂ | Prevents microbial growth |
| Total SO₂ | Total sulfur dioxide |
| Density | Wine density (g/cm³) |
| pH | Acidity level (0-14 scale) |
| Sulphates | Wine preservative |
| Alcohol | Alcohol content (% volume) |

---

## 📊 **MODEL INSIGHTS** 📊

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="500">
</div>

### **Clustering Performance**

| **Metric** | **Value** |
|------------|-----------|
| 🎯 **Optimal Clusters** | 3-5 (via dendrogram) |
| 📊 **Linkage Method** | Ward's Method |
| 🎨 **Variance Explained (PCA)** | 70%+ |
| 🔍 **Silhouette Score** | 0.45+ |

*Evaluated on standardized wine quality datasets*

### **What Makes Clusters Different?**

Each wine cluster represents a unique "wine family" with distinct:
- **Alcohol Content** 🍷
- **Acidity Levels** 🍋
- **Sweetness Profile** 🍬
- **Chemical Balance** ⚗️

---

## 💡 **FEATURES** 💡

### ✨ **What Makes This Special?**

```python
# Feature List
features = {
    "Unsupervised Learning": "🧬 No labels needed - finds patterns naturally",
    "Interactive Dendrograms": "🌳 Tree-based cluster visualization",
    "PCA Visualization": "🎨 2D/3D scatter plots",
    "Dual Wine Analysis": "🍷 Red & White wine comparison",
    "Cluster Profiling": "📊 Statistical summaries per cluster",
    "Production Ready": "🚀 Deployed on Streamlit Cloud"
}
```

---

## 📚 **SKILLS DEMONSTRATED** 📚

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="500">
</div>

- ✅ **Unsupervised ML**: Hierarchical Clustering, Dendrogram Analysis
- ✅ **Dimensionality Reduction**: PCA for visualization
- ✅ **Data Preprocessing**: StandardScaler, feature engineering
- ✅ **Exploratory Data Analysis**: Correlation analysis, distributions
- ✅ **Web Development**: Streamlit interactive dashboard
- ✅ **Python Libraries**: Pandas, NumPy, Scikit-learn, Plotly
- ✅ **Git & GitHub**: Version control, project documentation

---

## 🔮 **FUTURE ENHANCEMENTS** 🔮

- [ ] Add DBSCAN and K-Means comparison
- [ ] Implement t-SNE visualization
- [ ] Create wine recommendation system
- [ ] Add quality score prediction layer
- [ ] Include more wine datasets (variety, region)
- [ ] Build REST API endpoints
- [ ] Add export functionality (cluster reports)

---

## 🤝 **CONTRIBUTING** 🤝

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="500">
</div>

Contributions are **always welcome**! 🎉

1. 🍴 Fork the Project
2. 🌱 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🎁 Open a Pull Request

---

## 📝 **LICENSE** 📝

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 **CONNECT WITH ME** 👨‍💻

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank_Goyal-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Email](https://img.shields.io/badge/Email-itsmaygal09%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsmaygal09@gmail.com)

**Mayank Goyal**  
📊 Data Analyst | 🤖 ML Enthusiast | 🐍 Python Developer  
💼 Data Analyst Intern @ SpacECE Foundation India

---

## ⭐ **SHOW YOUR SUPPORT** ⭐

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="500">
</div>

Give a ⭐️ if this project helped you understand unsupervised learning!

### 🍷 **Built with Passion & ❤️ by Mayank Goyal** 🍷

**"Uncovering wine secrets, one cluster at a time!"** 🍇

---

<div align="center">
  <a href="https://github.com/mayank-goyal09">
    <img src="https://github.com/mayank-goyal09.png" width="100" style="border-radius:50%">
  </a>
</div>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer">
</div>