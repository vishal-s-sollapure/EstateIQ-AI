# EstateIQ AI — Phase 6.5: Production Deployment Guide

**Document ID:** DEP-6.5-001  
**Project:** EstateIQ AI — Bengaluru Real Estate Valuation & Investment Intelligence Platform  
**Target Platform:** Streamlit Community Cloud (with Render & Docker alternatives)  
**Repository:** `https://github.com/vishal-s-sollapure/EstateIQ-AI.git`  
**Branch:** `main`  
**Application Entry Point:** `app.py`  
**Status:** **Deployment Ready** (100% Pre-Flight Verified)

---

## 1. Deployment Readiness Checklist

Every pre-flight requirement has been audited, reconciled, and committed to `main`:

| Component | Status | Verification Detail |
| :--- | :---: | :--- |
| **Source Code** | ✅ Verified | Modular services (`src/services/`) and views (`src/ui/`) with explicit `__init__.py` files. |
| **Dependencies** | ✅ Pinned | [`requirements.txt`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/requirements.txt) includes `streamlit`, `xgboost`, `scikit-learn`, `pydeck`, `pyarrow`, `geopy`, `requests`, `pandas`, `numpy`. |
| **Server Config** | ✅ Configured | [`.streamlit/config.toml`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/.streamlit/config.toml) with headless mode, security flags, and theme. |
| **Champion Model** | ✅ Committed | [`models/estateiq_advanced_model.joblib`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/models/estateiq_advanced_model.joblib) (677 KB) whitelisted in `.gitignore` and pushed to GitHub. |
| **Geospatial Assets** | ✅ Committed | All 5 POI and locality CSVs in `data/geospatial/` tracked. |
| **Visual Figures** | ✅ Committed | All 9 high-res SHAP and benchmark diagnostic charts in `reports/figures/` tracked. |
| **Process Config** | ✅ Configured | [`Procfile`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/Procfile) for PaaS deployments (Render/Heroku/Railway). |
| **Automated Tests** | ✅ Passing | 27 of 27 unit and end-to-end regression tests pass in 4.6 seconds. |

---

## 2. Option A: Streamlit Community Cloud (Recommended Primary Target)

Streamlit Community Cloud is 100% free, connects directly to your GitHub repository, provides free automatic SSL (`https://`), and features automatic CI/CD redeployment whenever code is pushed to `main`.

### Step-by-Step Instructions:

1. **Sign in to Streamlit Cloud:**
   - Navigate to **[share.streamlit.io](https://share.streamlit.io/)**.
   - Click **"Sign in with GitHub"** and authenticate using your GitHub account (`vishal-s-sollapure`).

2. **Create New Application:**
   - In your Streamlit workspace, click the **"Create app"** (or **"New app"**) button in the top-right corner.
   - Choose **"I already have an app"**.

3. **Configure Deployment Settings:**
   - **Repository:** `vishal-s-sollapure/EstateIQ-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (Custom Subdomain):** Choose a custom subdomain, for example:
     - `estateiq-ai` $\to$ Resulting URL: `https://estateiq-ai.streamlit.app`
     - *(If already taken, try `estateiq-bengaluru` or `estateiq-valuation`)*.

4. **Advanced Settings (Optional):**
   - Click **"Advanced settings"**.
   - Under **Python version**, select **3.12** (or **3.11**).
   - Under **Secrets**, leave empty (EstateIQ AI runs 100% air-gapped during inference and requires zero external API keys).

5. **Deploy:**
   - Click **"Deploy!"** 🚀.
   - Watch the build logs. Streamlit Cloud will:
     1. Clone the repository.
     2. Install packages from `requirements.txt`.
     3. Apply settings from `.streamlit/config.toml`.
     4. Launch the application.
   - **Expected Build Time:** ~60 to 90 seconds.

6. **Post-Deployment Verification:**
   - Verify the 4 presets in Tab 1 (Hoodi 2BHK, Whitefield 3BHK, Indiranagar 4BHK, E-City 2BHK).
   - Verify PyDeck map rendering and amenity distance badges.
   - Verify Tab 2 (SHAP Waterfall and Beeswarm).
   - Verify Tab 3 (Tournament leaderboard).
   - Verify Tab 4 (Locality directory filtering).

---

## 3. Option B: Render Web Service (Alternative PaaS)

If you prefer deploying on Render:

1. Sign in to **[dashboard.render.com](https://dashboard.render.com/)**.
2. Click **New +** $\to$ **Web Service**.
3. Connect your GitHub repository `vishal-s-sollapure/EstateIQ-AI`.
4. Configure service settings:
   - **Name:** `estateiq-ai`
   - **Region:** Singapore / Oregon / Frankfurt
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Instance Type:** Free (512 MB RAM)
5. Click **Create Web Service**.

---

## 4. Automatic CI/CD Pipeline

Once deployed on Streamlit Community Cloud:
- Any change committed and pushed to `main` on GitHub triggers an **instant automatic cloud rebuild**.
- Zero manual deployment steps are required for future updates or bug fixes!
