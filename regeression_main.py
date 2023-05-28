import train
import test
import utils
from sklearn.model_selection import train_test_split
import dataloaders
import datasets

if __name__ == "__main__":
    dataset_path = utils.get_datasetPath()
    df = dataloaders.house_loader(dataset_path)
    # preprocessing
    X_ , target = datasets.preprocess(df, utils.get_inputType())
    X_train_org, X_test, y_train_org, y_test = train_test_split(X_, target, test_size=0.15, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org, test_size=0.15, shuffle=True, random_state=42)
    model = train.regression_train(X_train, X_val, y_train, y_val)
    print("test process is started")
    test.regression_test(model, X_test, y_test) 