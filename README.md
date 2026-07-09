# 🌾 Karnataka APMC Commodity Arrival — Sales Prediction Model
### Task 03 | Sales Prediction | Data Science Internship

> A Sales Prediction Model built using Python regression techniques
> that forecasts monthly commodity arrival (sales volume) at Karnataka
> APMC agricultural markets using real government data from 2012.

---

## 📌 Project Overview

Karnataka has 148 APMC (Agricultural Produce Market Committee) markets
across 29 districts where farmers bring their commodities every month.
Predicting how much of a commodity will arrive at a market helps
farmers, traders and market administrators plan supply and demand better.

This project uses **Linear Regression, Decision Tree and Random Forest**
to predict the monthly arrival quantity of 156 commodities across
Karnataka markets — trained on 21,000+ real government transaction records.

---

## 🎯 What We Predict

> **Target Variable → Arrival (Quantity in Quintals)**
> Given a District, Market, Commodity and Month —
> predict how much of that commodity will arrive at that market.

Example:
| District | Market | Commodity | Month | Predicted Arrival |
|----------|--------|-----------|-------|------------------|
| Hassan | HASSAN | Maize | March | 1,713 Quintals |
| Mysore | MYSORE | Paddy | August | 3,204 Quintals |
| Belgaum | BELGAUM | Groundnut | November | 2,226 Quintals |

---

## 🏆 Model Results

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | Best R² | - | - |
| Decision Tree | - | - | - |
| Random Forest | Best MAE | - | - |

> Results will populate after running step3_train.py on your machine

---

## 🛠️ Tech Stack

- **Python** — Core programming language
- **Pandas & NumPy** — Data cleaning, preprocessing, feature engineering
- **Scikit-learn** — Linear Regression, Decision Tree, Random Forest,
  OneHotEncoder, Pipeline, train/test split
- **Matplotlib & Seaborn** — EDA charts and model result visualizations
- **Streamlit** — Interactive live prediction dashboard
- **Joblib** — Model saving and loading
- **xlrd** — Reading .xls government dataset files

---
---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/karnataka-sales-prediction.git
cd karnataka-sales-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to: **https://data.gov.in**
- Search: **"Commodity Market Arrivals Karnataka 2012"**
- Download the `.xls` file
- Place it inside `data/` folder as `CommMktArrivals2012.xls`

---

## 🚀 How to Run — Step by Step

```bash
# Step 1 — Clean raw data, handle missing values, encode month
python step1_preprocess.py

# Step 2 — Generate EDA charts (4 charts saved to outputs/)
python step2_eda.py

# Step 3 — Train Linear Regression, Decision Tree, Random Forest
#           Evaluate with R², MAE, RMSE — saves best model
python step3_train.py

# Step 4 — Make custom predictions using saved model
python step4_predict.py

# Step 5 — Launch interactive Streamlit dashboard
streamlit run step5_dashboard.py
```

Then open your browser at **http://localhost:8501**

---

## 📊 What Each Step Does

### Step 1 — Preprocessing
- Loads raw `.xls` file (21,422 rows, 10 columns)
- Strips whitespace from all text columns
- Drops useless columns: Address, Telephone, Year
- Removes zero-arrival rows (187 rows removed)
- Caps extreme outliers using **IQR method** (3×IQR rule)
- Encodes Month → Month_Num (Jan=1, Dec=12)
- Saves → `data/cleaned_data.csv`

### Step 2 — EDA
- Top 15 commodities by total arrival
- Monthly arrival trend across all 12 months
- Top 10 districts by volume
- Arrival distribution (raw vs log scale)

### Step 3 — Model Training
- Applies **log transform** on target (Arrival is heavily skewed)
- One-hot encodes: District, Taluk, Market, Commodity, Unit
- Splits **80% train / 20% test**
- Trains 3 models inside **scikit-learn Pipelines**
- Evaluates with R², MAE, RMSE
- Saves best model to `outputs/best_model.pkl`

### Step 4 — Predictions
- Loads saved best model
- Runs 8 example predictions across Karnataka markets
- Interactive mode: type your own inputs and get a prediction

### Step 5 — Dashboard
- Pick District → Taluk → Market → Commodity → Month
- Click Predict to get live arrival forecast
- View monthly trend chart for selected commodity
- Browse all EDA and model charts in separate tabs

---

## 📂 Dataset

- **Source:** Karnataka Department of Agriculture & Farmers Welfare
- **Portal:** data.gov.in (Government of India Open Data)
- **File:** CommMktArrivals2012.xls
- **Size:** 21,422 rows × 10 columns
- **Coverage:** 148 markets, 29 districts, 156 commodities, 12 months

### Columns
| Column | Description |
|--------|-------------|
| District Name | One of 29 Karnataka districts |
| Taluk Name | Sub-district level location |
| Market Name | APMC market name (148 unique) |
| Address | Market address (dropped) |
| Telephone | Contact number (dropped) |
| Commodity | Type of commodity (156 unique) |
| Year | Always 2012 (dropped) |
| Month | Jan to Dec |
| Arrival | **Target** — quantity arrived (Quintals/Numbers) |
| Unit | Quintal / Numbers / Thousands |

---

## 📊 Key Concepts Used

| Concept | Purpose |
|---------|---------|
| Log Transform | Makes skewed Arrival data more normal for better modeling |
| One-Hot Encoding | Converts text categories (District, Commodity) to numbers |
| IQR Outlier Capping | Prevents extreme values from breaking the model |
| Scikit-learn Pipeline | Keeps preprocessing + model together, prevents data leakage |
| Train/Test Split | Tests model on unseen data for real performance measurement |
| Feature Importance | Shows which columns influenced predictions the most |

---

## 📸 Dashboard Preview

The Streamlit dashboard has 3 tabs:
- 🔮 **Live Prediction** — Select market + commodity + month → get predicted arrival
- 📈 **EDA & Model Charts** — All visualizations in one place
- 📋 **Raw Data Browser** — Filter by district or commodity

---

## 💡 Real World Impact

> Predicting commodity arrivals at Karnataka mandis helps:
> - **Farmers** — Know the best market and month to sell their crop
> - **Traders** — Plan inventory and pricing in advance
> - **Market Administrators** — Allocate storage and logistics efficiently
> - **Government** — Monitor agricultural supply chain across districts

---

## 👨‍💻 Author

Built as part of a **Data Science & Machine Learning Internship**
**Task 03:** Sales Prediction Model
**Dataset:** Karnataka APMC Commodity Market Arrivals 2012
**Source:** Department of Agriculture & Farmers Welfare, Government of Karnataka

---

## 📦 Requirements

## 📁 Project Structure# Karnataka-APMC-Commodity-Arrival-Sales-Prediction-Model
Karnataka has 148 APMC (Agricultural Produce Market Committee) markets across 29 districts where farmers bring their commodities every month. Predicting how much of a commodity will arrive at a market helps farmers, traders and market administrators plan supply and demand better.
