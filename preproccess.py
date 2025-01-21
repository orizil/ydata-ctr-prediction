import argparse
import pandas as pd
import sklearn

# test adding comment

def preproccess(csv_path):
    print(f'location:{csv_path}')
    df = pd.read_csv(csv_path)
    print(df.columns)

    # TODO: drop duplicates
    df_no_duplicates = df.drop_duplicates(subset=['session_id'], inplace=False)

    # TODO: split to X and y
    X, y = df.iloc[:,:-1], df.iloc[:,-1]

    # TODO: split train and test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=42)
    print(X_train.shape)
    print(y_train.shape)

    # TODO: save train csv, test csv


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--csv-raw-path', type=str)
    args = args.parse_args()
    print(args)

    preproccess(args.csv_raw_path)
