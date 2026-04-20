# =============================================
# Spam Email Classifier - Model Implementation
# =============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_excel("Spam_Email_Dataset.xlsx")
print("Dataset loaded:", df.shape)

# ---------------------------
# 2. Data Preprocessing
# ---------------------------
# Drop columns that are not useful for training
df = df.drop(columns=["email_id", "subject", "email_text", "sender_email", "sender_domain"])

# Check for nulls (none expected but good practice)
df = df.dropna()

print("After preprocessing:", df.shape)
print("Spam count:", df["label"].value_counts().to_dict())

# ---------------------------
# 3. Split Features & Label
# ---------------------------
X = df.drop(columns=["label"])
y = df["label"]

# ---------------------------
# 4. Train/Test Split (80/20)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------
# 5. Train the Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

# ---------------------------
# 6. Evaluate the Model
# ---------------------------
y_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"\n=== Results ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

# ---------------------------
# 7. Confusion Matrix (Plot)
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Spam", "Spam"],
            yticklabels=["Not Spam", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix saved!")

# ---------------------------
# 8. Test with 3 Sample Inputs (for Demo)
# ---------------------------
print("\n=== Sample Predictions ===")
samples = X_test.iloc[:3]
preds   = model.predict(samples)
for i, p in enumerate(preds):
    label = "SPAM 🚨" if p == 1 else "NOT Spam ✅"
    print(f"Email {i+1}: {label}")