# Fake Job Description Detection

## Project Overview
This project applies NLP techniques and Machine Learning to detect fraudulent job postings.
The goal is to solve the class imbalance problem and identify linguistic patterns in scam ads.

## 🏆 Key Results
- **Best Model:** Linear SVM (Calibrated)
- **Performance:** F1-Score **0.92**, Recall **0.89**.
- **Key Insight:** Scammers often use "copy-paste" tactics from legit companies, but subtle cues like "urgency" and specific phrasing ('ll vs will) reveal them.

## 🛠 Tech Stack
- **Python:** Pandas, NumPy, Scikit-learn
- **NLP:** TF-IDF (1-2 ngrams), Regex Feature Engineering
- **Models:** Logistic Regression, Random Forest, SVM

## 📂 Structure
- `Notebooks/`: Contains the EDA, Training, and Error Analysis notebooks.
- `Images/`: Confusion Matrix and Threshold Analysis plots.
