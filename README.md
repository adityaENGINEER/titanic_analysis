# 🚢 Titanic Dataset — Exploratory Data Analysis

A beginner-friendly data analysis project using the classic Titanic dataset.
Covers data loading, cleaning, feature engineering, statistical analysis, and visualization — all in a single Python script.

---

## 📊 What This Project Does

- Loads the Titanic dataset directly from a public URL (no manual download needed)
- Cleans missing values in `Age`, `Embarked`, and drops `Cabin`
- Engineers a new `AgeGroup` feature from passenger ages
- Computes key statistics using **NumPy**
- Performs group-level analysis using **Pandas**
- Generates a 6-panel visualization using **Matplotlib**

---

## 📈 Visualizations Included

| Plot | Description |
|------|-------------|
| Overall Survival Count | Bar chart of survivors vs non-survivors |
| Survival Rate by Gender | Comparison between male and female survival rates |
| Survival Rate by Passenger Class | 1st, 2nd, and 3rd class survival rates |
| Age Distribution by Survival | Overlapping histogram of ages |
| Fare Distribution by Class | Box plot showing fare spread per class |
| Survival by Embarkation Port | Pie chart for Cherbourg, Queenstown, Southampton |

---

## 🛠️ Tech Stack

- Python 3.x
- [NumPy](https://numpy.org/) — statistical computations
- [Pandas](https://pandas.pydata.org/) — data manipulation and grouping
- [Matplotlib](https://matplotlib.org/) — data visualization

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/titanic-eda.git
cd titanic-eda
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python titanic_analysis.py
```

The chart will be saved as `titanic_analysis.png` in the project folder.

---

## 📁 Project Structure

```
titanic-eda/
├── titanic_analysis.py     # Main analysis script
├── titanic_analysis.png    # Output chart (generated on run)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📌 Key Findings

- Overall survival rate: **38.4%**
- Women had a significantly higher survival rate than men
- 1st class passengers survived at nearly double the rate of 3rd class
- Children had a higher survival rate compared to adults

---

## 📦 Dataset

The Titanic dataset is loaded directly from:
```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```
No manual download required.

---

## 🙋 Author

Made by **[Your Name]** — feel free to fork, use, and improve!
