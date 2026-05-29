from utils.datasets import * 

X, y, fs = use_dataset("titanic")

X_prep = fs.fit_transform(X) 

print(X_prep)
