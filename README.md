# Machine Learning Projects — UET Peshawar (5th Semester)

Two end-to-end data science projects built using Python, covering exploratory data analysis and supervised machine learning.

---

## Project 1: NYC Taxi Fare Prediction

**Objective:** Predict taxi fare amounts in New York City using real-world trip data.

### What I Built
A complete machine learning pipeline on the NYC Yellow Taxi Trip Dataset, from raw data to a tuned predictive model.

### Pipeline
| Step | Details |
|------|---------|
| Data Cleaning | Removed invalid trips, filtered wrong coordinates, negative fares |
| Feature Engineering | Haversine trip distance, time-based features (hour, day, month) |
| Train-Test Split | Time-aware split — trained on earlier dates, tested on later |
| EDA | 3+ visualizations showing fare trends and trip patterns |
| Models | Linear Regression (baseline) + Random Forest Regressor |
| Evaluation | RMSE and RMSLE — measures error in dollars across all fare ranges |
| Tuning | GridSearchCV and RandomizedSearchCV on best model |
| Interpretation | Feature importances, model limitations, future improvements |

### Tech Stack
`Python` `pandas` `numpy` `scikit-learn` `matplotlib` `Jupyter Notebook`

### Key Learnings
- Trip distance and time of day are the strongest predictors of fare
- Random Forest significantly outperformed Linear Regression
- Time-aware splitting prevents data leakage in time-series data

📓 [View Notebook](./nyc-taxi-fare-prediction.ipynb)

---

## Project 2: Exploratory Data Analysis — Retail Sales

**Objective:** Explore and analyze the Superstore Sales Dataset to uncover business insights.

**Dataset:** [Kaggle Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

### What I Explored
- Sales and profit breakdown by category, region, and segment
- Top 5 cities and customers by revenue
- Discount vs profit relationship
- Monthly and yearly sales trends
- Correlation between sales, profit, discount, and quantity

### Visualizations
Bar charts · Pie charts · Histograms · Scatter plots · Boxplots · Heatmaps · Time-series line charts

### Tech Stack
`Python` `pandas` `numpy` `matplotlib` `Jupyter Notebook`

### Key Insights
- Technology is the most profitable product category
- Higher discounts consistently reduce profit margins
- Sales show clear seasonal patterns across months

📓 [View Notebook](./retail-sales-eda.ipynb)

---

## Setup

```bash
# Clone the repo
git clone https://github.com/hooriaaltaf/ML-5th-Sem.git
cd ML-5th-Sem

# Install dependencies
pip install pandas numpy matplotlib scikit-learn jupyter

# Launch Jupyter
jupyter notebook
```

---

## Author

**Hooria Altaf** — CS Student @ UET Peshawar  
[GitHub](https://github.com/hooriaaltaf) · [LinkedIn](https://www.linkedin.com/in/hooria-altaf-09b882212)
