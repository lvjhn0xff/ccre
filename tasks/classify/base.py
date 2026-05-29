import json

from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score

from utils.datasets import *

class CrossFoldExperiment: 
    def __init__(
        self, 
        dataset = "wdbc",
        dataset_version = 1,
        splits = 5,
        repeats = 10,
        model = None,
        seed = 42,
        preprocessor_args = {
            "use_onehot" : True, 
            "use_scaling" : True, 
            "use_power" : True
        }
    ):
        self.dataset = dataset 
        self.dataset_version = dataset_version
        self.splits = splits
        self.repeats = repeats
        self.seed = seed
        self.model = model
        self.preprocessor_args = preprocessor_args

    def info(self): 
        return { 
            "dataset" : self.dataset, 
            "dataset_version" : self.dataset_version, 
            "folds" : self.splits, 
            "model" : str(self.model()), 
            "preprocessor_args" : self.preprocessor_args
        }

    def display_info(self): 
        info = self.info() 
        
        print("EXPERIMENT SET-UP")
        print("============================================================")
    
        for key in info: 
            print(f"{key} : {json.dumps(info[key], indent=4)}")
            
    def execute(self): 

        # Load dataset.
        print("# Loading dataset...")
        X, y, make_preprocessor = \
            use_dataset(
                self.dataset, 
                version=self.dataset_version, 
                p_threshold=0.05
            ) 

        # Create stratified k-fold object. 
        print("# Creating StratifiedKFold...")

        rskf = RepeatedStratifiedKFold(
            n_splits=self.splits, 
            n_repeats=self.repeats,
            random_state=self.seed
        )

        # Make spits. 
        print("# Creating splits...")
        fold_no = 1
        for train_index, test_index in rskf.split(X, y): 
            print(f"\tFold No. : {fold_no}")
            
            # Get train and test splits.
            print(f"\t\tGetting train and test splits...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index] 

            print("\t\tX_train :", X_train.shape) 
            print("\t\tX_test  :", X_test.shape)

            # Describing pipeline. 
            print("\t\tDescribing pipeline...")
            
            steps = [
                make_preprocessor(**self.preprocessor_args),
                SMOTE(random_state=self.seed), 
                self.model()
            ]

            # Create pipeline.
            print("\t\tCreating pipeline...")
            pipeline = make_pipeline(*steps)

            # Fitting classifier. 
            print("\t\tTraining classifier.")
            pipeline.fit(X_train, y_train)

            # Get predictions. 
            print(f"\t\tGetting predictions.") 
            y_train_predict = pipeline.predict(X_train) 
            y_test_predict = pipeline.predict(X_test) 

            y_train_predict_proba = None 
            y_test_predict_proba = None 

            if hasattr(pipeline, "predict_proba"):
                y_train_predict_proba = pipeline.predict_proba(X_train) 
                y_test_predict_proba = pipeline.predict_proba(X_test) 

            # Evaluate predictions. 
            train_bas = balanced_accuracy_score(y_train, y_train_predict) 
            test_bas  = balanced_accuracy_score(y_test, y_test_predict)
            print(f"\t\t\tBalanced Accuracy Score")
            print(f"\t\t\t\tTrain : {train_bas}")    
            print(f"\t\t\t\tTest  : {test_bas}")    

            fold_no += 1