""" We will use 10-fold cross-validation. In each fold,
we will split into training and validation set to determine the best threshold for classification."""
import numpy as np
from sklearn.model_selection import GroupKFold
import pandas as pd
from aif360.datasets import StandardDataset

DATA_DIR = '/media/bigdata/10. Stages/3. Afgerond/2020-08 Jesse Kuiper/'

def get_favourable1(DoseDiazepamPost):
    return DoseDiazepamPost==0

def train_test_split(dataframe: pd.DataFrame, training_set_fraction: float, group_labels: np.ndarray, shuffle: bool):
    raise NotImplementedError()

Dataset1 = pd.read_csv(DATA_DIR + "Dataset14Days.csv", sep=';')
groups = Dataset1[['PseudoID']].values
Dataset1.drop(columns=['PseudoID'], inplace=True)

gkf = GroupKFold(n_splits=10)
for dev, test in gkf.split(Dataset1, groups=groups):
    df_dev = Dataset1.iloc[dev]
    df_test = Dataset1.iloc[test]
    groups_dev = groups[dev]
    df_train, df_val = train_test_split(df_dev, training_set_fraction=0.625, group_labels=groups_dev, shuffle=True)
    dataset_train, dataset_val, dataset_test = [StandardDataset(el,
                                                                'DoseDiazepamPost',
                                                                get_favourable1,
                                                                ['Geslacht'],
                                                                [['Man']])
                                                for el in (df_train, df_val, df_test)]
    # Todo: now we can train all the models and compute all the metrics
