from utils.datasets import * 

X, y, fs = use_dataset("Telco-Customer-Churn")

prep = fs()

X_prep = prep.fit_transform(X) 

print(X_prep)
