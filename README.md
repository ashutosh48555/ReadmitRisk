# 🏥 ReadmitRisk AI - Hospital Readmission Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ML](https://img.shields.io/badge/ML-Ensemble%20Model-orange)](https://scikit-learn.org/)

> **An intelligent patient-facing application that predicts 30-day hospital readmission risk using advanced machine learning and AI-powered insights**

**Developer:** Ashutosh Kumar Singh  
**Dataset:** UCI Diabetes 130-US Hospitals (101,767 patient records)  
**Model:** Voting Ensemble (Random Forest + XGBoost + Logistic Regression)  
**Performance:** 69.3% AUC-ROC

📖 **[Read Comprehensive Documentation →](PROJECT_OVERVIEW.md)**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM (8GB recommended)
- Google Gemini API keys (for AI insights)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/readmitrisk-ai.git
cd readmitrisk-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
mkdir .streamlit
echo 'GEMINI_API_KEYS = ["your-key-1", "your-key-2"]' > .streamlit/secrets.toml
echo 'GEMINI_MODEL = "gemini-2.5-flash"' >> .streamlit/secrets.toml

# 5. Run application
streamlit run app.py
```

**Access at:** http://localhost:8502

---

## ✨ Key Features

### 🎯 Core Functionality
- **Risk Prediction:** 30-day hospital readmission probability (0-100%)
- **AI Insights:** Personalized recommendations powered by Google Gemini 2.5 Flash
- **Interactive Dashboard:** Risk gauge, feature importance, and visual analytics
- **PDF Reports:** Downloadable comprehensive patient risk assessments
- **Mobile-Responsive:** Auto-collapsing sidebar and optimized layouts

### 🤖 Machine Learning
- **17 Algorithms Trained:** Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM, Random Forest, XGBoost, and more
- **Ensemble Model:** Voting classifier combining top 3 models
- **69.3% AUC-ROC:** Strong performance on highly imbalanced dataset
- **Feature Importance:** Transparent explanations of risk factors

### 💡 AI-Powered Insights
- **Multi-Key Rotation:** 5 Gemini API keys with automatic failover
- **Contextual Recommendations:** Tailored advice based on patient profile
- **Educational Content:** Explains medical concepts in patient-friendly language
- **Dynamic Q&A:** Interactive follow-up questions

## 📚 Documentation

**[📖 Complete Project Overview →](PROJECT_OVERVIEW.md)**

Comprehensive 3,000+ line technical documentation including:
- Detailed explanations of all 17 ML algorithms
- Complete methodology and mathematical foundations
- Educational context for students and teachers
- Model development pipeline
- Performance analysis and comparisons
- Future enhancement roadmap

---

## 🛠️ Technology Stack

**Machine Learning:** Python, Scikit-learn, XGBoost, Pandas, NumPy  
**Web Framework:** Streamlit, Plotly, ReportLab  
**AI Integration:** Google Gemini 2.5 Flash (Multi-key rotation)  
**Development:** Jupyter Notebook, Git

---

## 📁 Project Structure

```plaintext
ML/
├── app.py                           # Main Streamlit app
├── ReadmitRisk_Main_Notebook.ipynb  # Complete ML pipeline
├── PROJECT_OVERVIEW.md              # Comprehensive documentation
├── README.md                        # This quick start guide
├── requirements.txt                 # Dependencies
├── .streamlit/secrets.toml          # API keys (not in repo)
├── dataset/
│   └── diabetic_data.csv            # 101,767 patient records
└── models/
    ├── best_readmission_model.pkl   # Voting Ensemble
    ├── scaler.pkl                   # Feature scaler
    └── *.pkl                        # Other model components
```

---

## 📊 Model Performance

| Model | AUC-ROC | F1-Score | Training Time |
|-------|---------|----------|---------------|
| **Voting Ensemble** | **69.3%** | **30.1%** | 45s |
| XGBoost | 69.1% | 29.3% | 18s |
| Random Forest | 68.9% | 28.3% | 22s |
| Logistic Regression | 67.8% | 26.5% | 3s |

**Note:** Performance reflects realistic results on highly imbalanced healthcare data (11% readmission rate). See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for detailed analysis.

---

## 📖 Usage

```bash
# 1. Start the app
streamlit run app.py

# 2. Open browser at http://localhost:8502

# 3. Enter patient data:
   - Demographics (age, gender, race)
   - Hospital stay details
   - Medications and diagnoses
   
# 4. Get results:
   - Risk score (0-100%)
   - AI-powered recommendations
   - Feature importance analysis
   - Download PDF report
```

---

## 🎓 Educational Value

This project demonstrates mastery of:
- **Data Science:** Preprocessing, EDA, Feature Engineering
- **Machine Learning:** 17 algorithms across all categories
- **Model Evaluation:** Cross-validation, Multiple metrics
- **Ensemble Methods:** Voting, Bagging, Boosting
- **AI Integration:** LLM-powered insights
- **Software Engineering:** Production-ready application
- **Domain Knowledge:** Healthcare analytics

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for complete learning outcomes.

---

## 🔮 Future Enhancements

- SHAP explanations for predictions
- Deep learning models (LSTM, Transformers)
- Real-time monitoring dashboard
- EHR system integration
- Mobile app (iOS/Android)
- Clinical validation studies

---

## 📧 Contact

**Developer:** Ashutosh Kumar Singh  
**Repository:** [GitHub]  
**Documentation:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)



---

## ⚠️ Disclaimer

**IMPORTANT:** This tool is for educational and informational purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

The predictions are based on historical data and statistical models. Individual patient circumstances may vary significantly. Healthcare decisions should always be made in consultation with qualified medical professionals.

---

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** - Diabetes dataset
- **Google** - Gemini AI API
- **Streamlit** - Web framework
- **Scikit-learn & XGBoost** - ML libraries
- **Healthcare Community** - Domain inspiration

---

**⭐ Star this project if you found it helpful!**

---

## 🎯 Future Enhancements

- [ ] Multi-disease support (heart failure, pneumonia, etc.)
- [ ] Real-time EHR integration
- [ ] Mobile app (iOS/Android)
- [ ] Multilingual support
- [ ] Telemedicine integration
- [ ] Advanced visualizations (3D risk maps)
- [ ] Patient tracking dashboard
- [ ] Clinical trial matching

---

**Built with ❤️ by Ashutosh Kumar Singh**  
*Empowering patients through AI and predictive analytics*

---

**Version:** 2.0  
**Last Updated:** December 2025  
**Status:** Production Ready ✅
