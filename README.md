# 🔋 MoO3 Interface Intelligence Platform

A complete AI-powered battery material discovery dashboard built with Streamlit.

## Features
- Works with **any** raw Excel (.xlsx) or CSV dataset — not just MoO3 data
- Auto-cleans messy Excel files (mixed text+numbers, sparse rows, merged cells)
- Interactive analytics: bar, pie, scatter, violin, correlation heatmaps
- Trains Random Forest, Gradient Boosting, or Ridge Regression models
- Feature importance visualisation and residual analysis
- Capacity prediction with 90% confidence intervals
- Top-material recommendations with radar charts
- CSV export of cleaned data and top materials

## Quick Setup (VS Code)

### Step 1 — Install Python 3.11+
Download from https://www.python.org/downloads/ and install.
Make sure to check "Add Python to PATH" during installation.

### Step 2 — Create a virtual environment
Open a terminal in VS Code (Ctrl+` or Terminal → New Terminal), then:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app opens automatically at http://localhost:8501

## Usage
1. The browser opens automatically. If not, go to http://localhost:8501
2. Click "Browse files" in the sidebar and upload `Literature_survey.xlsx`
   (or any .xlsx / .csv battery material dataset)
3. Navigate the pages using the sidebar radio buttons

## File Structure
```
moo3_app/
├── app.py            ← Main application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` inside the venv |
| `No numeric columns found` | Your dataset may need different column names. Check the Dataset tab for detected columns. |
| Port already in use | Run `streamlit run app.py --server.port 8502` |
| Excel load error | Make sure openpyxl is installed: `pip install openpyxl` |
| Blank page in browser | Hard refresh with Ctrl+Shift+R |

## Notes on the Dataset
The app auto-detects columns containing keywords like "capacity", "size", "cycle",
"morphology", etc., and extracts numbers from messy strings like "722 mAh/g after 100 cycles".
It works best when at least one column has numeric-extractable capacity values.
