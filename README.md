# Smart Anti-Corruption System

A Python-based application designed to detect suspicious financial and administrative activities in government data using **Artificial Intelligence (AI)** and **Blockchain concepts**.  
The system leverages machine learning and predictive analytics to uncover patterns that may indicate corruption, enhancing transparency, accountability, and decision-making in the public sector.

---

## **System Overview**

The system is a software application that:

- Uses **AI** (machine learning, anomaly detection, predictive analytics) to detect unusual or suspicious patterns.
- Processes **big data** to analyze financial and administrative government records.
- Applies **Blockchain concepts** to ensure data integrity and protect against tampering.

### **Key Advantages**

- **Big Data Analysis:** Accurately analyzes large volumes of government data.  
- **Transparency & Accountability:** Enables authorities to monitor financial and administrative transactions.  
- **Early Corruption Detection:** AI predicts illegal or suspicious activities before they impact operations.  
- **Innovative Technologies:** Combines AI with blockchain to improve reliability and trust in data.

### **System Goals**

- **Improve Oversight:** Monitor transactions to reduce corruption opportunities.  
- **Enhance Public Trust:** Ensure government systems are transparent and accountable.  
- **Support Decision-Making:** Provide accurate reports and actionable insights.  

### **Applications in the Palestinian Public Sector**

- Monitor financial and administrative operations in ministries and government institutions.  
- Protect public funds from misuse or fraud.  
- Support internal auditing through alerts and smart analyses.  
- Predict risks and enable proactive measures.

---

## **System Components**

| Component | Description | Technology |
|-----------|------------|-----------|
| **Database** | Stores financial and administrative data securely | SQL / NoSQL / Data Lakes |
| **Data Processing Layer** | Cleans and prepares data for analysis | Python (pandas, numpy) |
| **AI Analysis Layer** | Detects suspicious patterns and predicts corruption | Machine Learning: Isolation Forest, XGBoost, Neural Networks |
| **Network & Text Analysis Layer** | Analyzes relationships between employees and suppliers, contracts, and reports | NetworkX, NLP |
| **Dashboard & Reporting** | Displays analyses, risk alerts, and reports | Dash, Plotly, Matplotlib |
| **Blockchain Layer** | Ensures data integrity and prevents manipulation | Blockchain, Smart Contracts |

---

## **System Flowchart**

[Collect Financial & Administrative Data]
│
▼
[Clean & Prepare Data]
│
▼
[Feature Engineering]
│
▼
[Data Analysis & Anomaly Detection]
│
▼
[Network & Relationship Analysis]
│
▼
[Corruption Prediction & Alerts]
│
▼
[Results & Reports / Dashboard]
│
▼
[Continuous System Improvement]


---

## **Getting Started**

### **Requirements**

- Python 3.8+  
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `networkx`
  - `scikit-learn`

Install dependencies:

```bash
pip install -r requirements.txt
### Demo Code

Save the following as demo.py:

"""
Smart Anti-Corruption System - Demo
Author: Your Name
Description: Detect suspicious financial and administrative activities using AI and network analysis.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.ensemble import IsolationForest

# 1️⃣ Create Sample Data
np.random.seed(42)
data = {
    'employee_id': range(1, 21),
    'department': np.random.choice(['Finance', 'HR', 'Procurement'], 20),
    'salary': np.random.randint(3000, 7000, 20),
    'expenses': np.random.randint(100, 5000, 20),
    'num_transactions': np.random.randint(1, 50, 20),
    'supplier_id': np.random.randint(1, 10, 20)
}
df = pd.DataFrame(data)

# Add anomalies to simulate corruption
df.loc[5, 'expenses'] = 15000
df.loc[12, 'num_transactions'] = 80

# 2️⃣ Clean Data
# Example: fill missing values or remove duplicates
# df.dropna(inplace=True)
# df.drop_duplicates(inplace=True)

# 3️⃣ Feature Engineering
features = df[['salary', 'expenses', 'num_transactions']]

# 4️⃣ Anomaly Detection
model = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = model.fit_predict(features)
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Suspicious'})

# 5️⃣ Network Analysis
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(f"Emp{row['employee_id']}", f"Supplier{row['supplier_id']}")

plt.figure(figsize=(10,6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
plt.title("Employee-Supplier Network")
plt.show()

# 6️⃣ Display Results
print("=== Anti-Corruption Detection Results ===")
print(df[['employee_id', 'department', 'salary', 'expenses', 'num_transactions', 'anomaly']])

# Scatter plot for expenses vs transactions
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='expenses', y='num_transactions', hue='anomaly', palette=['green','red'], s=100)
plt.title("Suspicious Activity Detection - Expenses vs Transactions")
plt.xlabel("Expenses")
plt.ylabel("Number of Transactions")
plt.show()

###Running the Demo
python demo.py

This will produce:

A table of suspicious activities.

A scatter plot for expenses vs number of transactions.

A network graph showing employee-supplier relationships.
### Repository Structure
smart-anti-corruption-system/
│
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── demo.py             # Demo Python script
└── data/               # Optional CSV data folder
