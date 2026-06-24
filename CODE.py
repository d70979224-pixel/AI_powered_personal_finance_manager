import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# PHASE 1: DATA ACQUISITION & EXPLORATION

# 1. Loading the clean CSV files
expenses_df = pd.read_csv("Expenses_clean.csv")
income_df = pd.read_csv("Income_clean.csv")

# 2. Combining the files into one single master dataframe
df = pd.concat([expenses_df, income_df], ignore_index=True)

# Filtering out categories that have fewer than 2 examples so stratify works perfectly
category_counts = df['category'].value_counts()
valid_categories = category_counts[category_counts >= 2].index
df = df[df['category'].isin(valid_categories)].reset_index(drop=True)

# 3. EDA graph for the report (Clean layout)
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df, 
    y='category', 
    order=df['category'].value_counts().index, 
    hue='category', 
    palette='viridis', 
    legend=False
)
plt.title('Distribution of Transaction Categories')
plt.xlabel('Number of Transactions')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('category_distribution.png')
plt.show()

# 4. Spliting the data according to syllabus rules (80/10/10 ratio)
X = df['tags'].astype(str)
y = df['category'].astype(str)

# First split: 80% Train, 20% Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Second split: Divide the remaining 20% into 50% validation and 50% test WITHOUT second stratification.
# This avoids the single-member remainder bug entirely while keeping the exact 80/10/10 split size!
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Data Splits Created:-")
print(f"Training features: {X_train.shape[0]} rows")
print(f"Validation features: {X_val.shape[0]} rows")
print(f"Testing features: {X_test.shape[0]} rows\n")

# BASELINE MODEL IMPLEMENTATION

# 1. Vectorize text features using basic TF-IDF tokenization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 2. Initialize and train a simple Logistic Regression baseline classifier
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_vec, y_train)

# 3. Evaluating the model on validation data to show the baseline score
y_pred = baseline_model.predict(X_val_vec)
baseline_accuracy = accuracy_score(y_val, y_pred)

print("BASELINE MODEL METRICS")
print(f"Baseline Accuracy Score: {baseline_accuracy * 100:.2f}%\n")
print("Detailed Classification Report:-")
print(classification_report(y_val, y_pred, zero_division=0))

# TEST PREDICTION 


sample_transaction_text = ["tag_1"]
sample_vec = vectorizer.transform(sample_transaction_text)
sample_prediction = baseline_model.predict(sample_vec)

print("Sample Model Testing:-")
print(f"Input Text: {sample_transaction_text}")
print(f"Predicted Budget Category: {sample_prediction[0]}")