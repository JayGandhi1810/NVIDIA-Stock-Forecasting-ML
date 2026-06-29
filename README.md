Data-Driven Finance: A Comparative Performance Analysis of Machine Learning and Traditional Models in the Equity Market
MSc Dissertation Project
Author: Jay Gandhi.
Programme: MSc Data Analytics
Institution: Berlin School of Business and Innovation (BSBI)
Academic Year: 2025–2026
________________________________________
📖 Project Overview
This repository contains the complete implementation, dataset, and supporting materials developed for my MSc dissertation entitled:
“Data-Driven Finance: A Comparative Performance Analysis of Machine Learning and Traditional Models in the Equity Market.”
The study investigates whether modern Machine Learning techniques outperform traditional econometric models when forecasting stock market movements. Historical daily stock data for NVIDIA Corporation were collected from Yahoo Finance and analysed using Python.
The project combines financial econometrics with machine learning by implementing:

•	Autoregressive Integrated Moving Average (ARIMA)
•	Generalised Autoregressive Conditional Heteroskedasticity (GARCH)
•	Random Forest
•	Extreme Gradient Boosting (XGBoost)
•	Long Short-Term Memory (LSTM)
The comparative analysis evaluates each model using statistical forecasting errors and classification performance metrics.
________________________________________
🎯 Research Aim
To compare the predictive performance of traditional econometric models and machine learning techniques for forecasting equity market movements using NVIDIA historical stock data.
________________________________________
✅ Research Objectives
•	Collect historical NVIDIA stock market data from Yahoo Finance.
•	Perform Exploratory Data Analysis (EDA).
•	Engineer financial and technical features.
•	Develop traditional econometric forecasting models.
•	Develop machine learning forecasting models.
•	Compare model performance using standard evaluation metrics.
•	Identify the most effective forecasting approach.
________________________________________
📊 Dataset
Source: Yahoo Finance
Company: NVIDIA Corporation (Ticker: NVDA)
Period: January 2015 – December 2025
Frequency: Daily
The dataset includes:
•	Date
•	Open
•	High
•	Low
•	Close
•	Adjusted Close
•	Volume
Additional engineered features include:
•	Daily Return
•	5-Day Moving Average (MA5)
•	20-Day Moving Average (MA20)
•	Rolling Volatility
•	Relative Strength Index (RSI)
•	Moving Average Convergence Divergence (MACD)
________________________________________
⚙️ Technologies Used
•	Python 3.11
•	Visual Studio Code
•	NumPy
•	pandas
•	Matplotlib
•	Seaborn
•	scikit-learn
•	statsmodels
•	arch
•	XGBoost
•	TensorFlow / Keras
•	Yahoo Finance API (yfinance)
________________________________________
📁 Repository Structure
NVIDIA-Stock-Forecasting-ML
│
├── data/
│   └── NVDA_Historical_Data.csv
│
├── figures/
│   └── Figures used in Chapter 4
│
├── src/
│   ├── download_data.py
│   ├── exploratory_data_analysis.py
│   ├── feature_engineering.py
│   ├── arima_model.py
│   ├── garch_model.py
│   ├── random_forest_model.py
│   ├── xgboost_model.py
│   └── lstm_model.py
│
├── requirements.txt
├── README.md
└── LICENSE
________________________________________
🤖 Forecasting Models
Traditional Econometric Models
•	ARIMA
•	GARCH
Machine Learning Models
•	Random Forest
•	XGBoost
•	Long Short-Term Memory (LSTM)
________________________________________
📈 Model Performance Summary
Model	Performance Metric	Result
ARIMA	Mean Absolute Error (MAE)	53.365
ARIMA	Root Mean Square Error (RMSE)	60.4894
GARCH	Volatility Persistence (α + β)	0.9005
Random Forest	Accuracy	49.64%
Random Forest	Precision	56.31%
Random Forest	Recall	26.36%
Random Forest	F1 Score	35.91%
XGBoost	Accuracy	50.12%
XGBoost	Precision	55.17%
XGBoost	Recall	36.36%
XGBoost	F1 Score	43.84%
LSTM	Accuracy	46.59%
________________________________________
🔍 Key Findings
•	ARIMA successfully modelled long-term price trends but produced relatively high forecasting errors.
•	GARCH effectively captured volatility clustering within NVIDIA stock returns.
•	Random Forest demonstrated moderate predictive capability.
•	XGBoost achieved the strongest overall performance among the machine learning models.
•	LSTM showed limited predictive capability under the selected feature set and forecasting horizon.
Overall, the findings indicate that XGBoost provided the most effective balance between predictive accuracy and model robustness for this study.
________________________________________
▶️ Running the Project
1.	Clone the repository:
git clone https://github.com/JayGandhi1810/NVIDIA-Stock-Forecasting-ML.git
2.	Install the required packages:
pip install -r requirements.txt
3.	Execute the scripts in the following order:
•	download_data.py
•	exploratory_data_analysis.py
•	feature_engineering.py
•	arima_model.py
•	garch_model.py
•	random_forest_model.py
•	xgboost_model.py
•	lstm_model.py
________________________________________
📚 Academic Citation
If you use this repository for academic purposes, please cite the associated MSc dissertation.
________________________________________
📄 Licence
This repository is provided for educational and academic research purposes.
