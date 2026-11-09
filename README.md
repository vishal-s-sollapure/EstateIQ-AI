# EstateIQ AI 🏢
### AI-Powered Real Estate Valuation & Investment Intelligence Platform

EstateIQ AI is an end-to-end Machine Learning and Geospatial Intelligence platform engineered to predict residential property prices across Bengaluru, India. It combines advanced regression models, geospatial feature extraction via OpenStreetMap (OSM), and Explainable AI (SHAP) to deliver transparent, data-driven real estate valuations and investment metrics.

---

## 📌 Project Architecture & Highlights

- **Predictive Modeling**: Multi-algorithm price regression (Random Forest, LightGBM, XGBoost, CatBoost) fine-tuned for the Bengaluru real estate market.
- **Geospatial Intelligence**: OpenStreetMap (OSM) proximity metrics analyzing distances to metro stations, tech parks, arterial highways, top schools, and healthcare facilities.
- **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations) integration to visualize feature contributions for each property valuation.
- **Interactive Web Interface**: Streamlit-based web dashboard offering intuitive property inputs, price estimation, and geospatial maps.

---

## 📁 Project Directory Structure

```text
EstateIQ-AI/
├── data/
│   ├── raw/                # Original, immutable Bengaluru real estate datasets
│   └── processed/          # Cleaned, encoded, and feature-engineered datasets
├── notebooks/              # Jupyter notebooks for EDA, prototyping, and experimentation
├── src/                    # Modular production source code
│   ├── __init__.py
│   ├── data/               # Ingestion and data cleaning scripts
│   ├── features/           # Feature engineering and OSM geospatial extractors
│   ├── models/             # Model training, hyperparameter tuning, and evaluation
│   └── utils/              # Helper functions, config loaders, and logging utilities
├── models/                 # Serialized model artifacts (.joblib, .pkl) and encoders
├── reports/                # Evaluation metrics, performance plots, and SHAP summaries
├── screenshots/            # UI captures and visuals for documentation & portfolio
├── docs/                   # Architecture notes, data dictionary, and design docs
├── app.py                  # Streamlit web application entry point
├── train.py                # End-to-end model training pipeline script
├── requirements.txt        # Python package dependencies
├── .gitignore              # Files and folders ignored by Git
└── README.md               # Project documentation and guide
```

---

## 🚀 Quickstart Guide (Windows PowerShell)

### 1. Clone or Open the Workspace
```powershell
cd c:\Users\VISHAL\OneDrive\Desktop\EstateIQ-AI
```

### 2. Activate Virtual Environment
A virtual environment (`venv`) is already present in this directory. Activate it in PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
> *Note: If PowerShell shows an `Execution_Policies` restriction, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in your current PowerShell window.*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Model Training (Placeholder)
```powershell
python train.py
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

---

## 🗺️ Roadmap & Development Milestones

1. **Phase 1: Project Setup & EDA**
   - Clean folder structure setup (Completed)
   - Exploratory Data Analysis on Bengaluru dataset (handling ranges in sqft, missing values, outliers)
2. **Phase 2: Geospatial Engineering**
   - Geocoding Bengaluru locations
   - OSM POI extraction (transit, tech corridors, amenities)
3. **Phase 3: Model Building & Explainability**
   - Baseline models vs. Ensembles (XGBoost / LightGBM)
   - Model interpretability with SHAP summary & waterfall plots
4. **Phase 4: Dashboard & Deployment**
   - Interactive Streamlit app with Folium maps
   - Final report and portfolio documentation

---

## 👨‍💻 Author
- Second-Year CSE (AI & ML) Student
- Designed for academic rigor, real-world utility, and clean software engineering standards.

