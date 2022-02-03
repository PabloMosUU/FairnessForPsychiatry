""" We will use 10-fold cross-validation. In each fold,
we will split into training and validation set to determine the best threshold for classification."""
import aif360.datasets
import numpy as np
import sklearn.pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from collections import defaultdict

np.random.seed(1)
DATA_DIR = '/media/bigdata/10. Stages/3. Afgerond/2020-08 Jesse Kuiper/'
THRESHOLDS = np.linspace(0.1, 0.8, 50)
MODEL_NAMES = ['logreg', 'rf', 'logregreweight', 'rfreweight', 'prejudiceremover']

def is_favorable(DoseDiazepamPost):
    return DoseDiazepamPost==0

def train_test_split(dataframe: pd.DataFrame, training_set_fraction: float, group_labels: np.ndarray, shuffle: bool):
    if not shuffle:
        raise ValueError('This method is only implemented with shuffling')
    assert len(group_labels) == len(dataframe), 'groups and data have different lengths'
    gss = GroupShuffleSplit(n_splits=1, train_size=training_set_fraction)
    for train_ids, test_ids in gss.split(dataframe, groups=group_labels):
        return dataframe.iloc[train_ids], dataframe.iloc[test_ids]
    raise Exception('There should have been one split, but none was reached')

def train_model(name: str,
                dataset: aif360.datasets.StandardDataset,
                protected_attribute_name: str,
                unprivileged_groups: list,
                privileged_groups: list) -> sklearn.pipeline.Pipeline:
    # Dataset manipulations
    if name.endswith('reweight'):
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_transf_train = RW.fit_transform(dataset)
        dataset = dataset_transf_train
    # Model initialization and fitting
    if name == 'prejudiceremover':
        model = PrejudiceRemover(sensitive_attr=protected_attribute_name, eta=25.0)
        return model.fit(dataset)
    else:
        if name.startswith('logreg'):
            fitter = LogisticRegression(solver='liblinear', random_state=1)
            fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
        elif name.startswith('rf'):
            fitter = RandomForestClassifier(n_estimators=500, min_samples_leaf=25)
            fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
        else:
            raise ValueError('Unknown model', name)
        model = make_pipeline(StandardScaler(),
                              fitter)
        return model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

def compute_metrics(model: sklearn.pipeline.Pipeline,
                    dataset: aif360.datasets.StandardDataset,
                    thresh_arr: np.ndarray,
                    unprivileged_groups: list,
                    privileged_groups: list) -> dict:
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        print('aif360 inprocessing algorithm')
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # noinspection PyUnresolvedReferences
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            dataset,
            dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        # calculate the F1 score

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        metric_arrs['F1_score'].append(metric.true_positive_rate() /
                                       (metric.true_positive_rate() + (0.5 * (
                                                   metric.false_positive_rate() + metric.false_negative_rate()))))

        metric_arrs['FP Diff'].append(metric.false_positive_rate_difference())

        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        # metric_arrs['theil_ind'].append(metric.theil_index())

    return metric_arrs

def get_best_threshold(some_metrics: dict) -> np.ndarray:
    return np.argmax(some_metrics['bal_acc'])

def split_by_privilege(dataset: aif360.datasets.StandardDataset,
                       protected_attribute_ix: int,
                       protected_attribute_name: str) -> tuple:
    unprivileged = [{protected_attribute_name: v} for v in
                    dataset.unprivileged_protected_attributes[protected_attribute_ix]]
    privileged = [{protected_attribute_name: v} for v in
                  dataset.privileged_protected_attributes[protected_attribute_ix]]
    return unprivileged, privileged

def save_metrics(fold_metrics: dict) -> None:
    print(fold_metrics)

def train_val_test_model(model_name: str,
                         dataset_train: aif360.datasets.StandardDataset,
                         dataset_val: aif360.datasets.StandardDataset,
                         dataset_test: aif360.datasets.StandardDataset,
                         unprivileged_groups: list,
                         privileged_groups: list,
                         male_name: str
                         ) -> dict:
    if model_name == 'prejudiceremover':
        pr_orig_scaler = StandardScaler()
        dataset_train, dataset_val, dataset_test = [el.copy() for el in (dataset_train, dataset_val, dataset_test)]
        for dataset in (dataset_train, dataset_val, dataset_test):
            dataset.features = pr_orig_scaler.fit_transform(dataset.features)
    trained_model = train_model(model_name, dataset_train, male_name, unprivileged_groups, privileged_groups)
    validation_metrics = compute_metrics(trained_model, dataset_val, THRESHOLDS, unprivileged_groups, privileged_groups)
    best_threshold = get_best_threshold(validation_metrics)
    # Todo: retrain model on the entire dev set
    test_metrics = compute_metrics(trained_model,
                                   dataset_test,
                                   np.array([THRESHOLDS[best_threshold]]),
                                   unprivileged_groups,
                                   privileged_groups)
    return test_metrics

def train_all_models(model_names: list,
                     dataset_train: aif360.datasets.StandardDataset,
                     dataset_val: aif360.datasets.StandardDataset,
                     dataset_test: aif360.datasets.StandardDataset) -> list:
    sens_ind = 0  # TODO: magic variable
    protected_attribute_name = dataset_train.protected_attribute_names[sens_ind]
    unprivileged_groups, privileged_groups = split_by_privilege(dataset_train, sens_ind, protected_attribute_name)
    model_metrics = []
    for model_name in model_names:
        test_metrics = train_val_test_model(model_name,
                                            dataset_train,
                                            dataset_val,
                                            dataset_test,
                                            unprivileged_groups,
                                            privileged_groups,
                                            protected_attribute_name)
        model_metrics.append(test_metrics)
    return model_metrics

def cross_validation(data: pd.DataFrame, group_labels: list, n_splits: int, train_dev_frac: float) -> list:
    assert len(data) == len(group_labels), 'dataframe and groups must have the same length'
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for dev, test in gkf.split(data, groups=group_labels):
        df_dev = data.iloc[dev]
        df_test = data.iloc[test]
        groups_dev = group_labels[dev]
        df_train, df_val = train_test_split(df_dev,
                                            training_set_fraction=train_dev_frac,
                                            group_labels=groups_dev,
                                            shuffle=True)
        dataset_train, dataset_val, dataset_test, dataset_dev = [StandardDataset(el,
                                                                                 'DoseDiazepamPost',
                                                                                 is_favorable,
                                                                                 ['Geslacht'],
                                                                                 [['Man']])
                                                                 for el in (df_train, df_val, df_test, df_dev)]
        model_metrics = train_all_models(MODEL_NAMES, dataset_train, dataset_val, dataset_test)
        fold_metrics.append(model_metrics)
    return fold_metrics

def simple_split(data: pd.DataFrame) -> dict:
    DW = StandardDataset(
        data,
        "DoseDiazepamPost",
        is_favorable,
        ["Geslacht"],
        [["Man"]])
    # noinspection PyTypeChecker
    (dataset_train,
     dataset_val,
     dataset_test) = DW.split([0.5, 0.8], shuffle=True)
    model_metrics = train_all_models(MODEL_NAMES, dataset_train, dataset_val, dataset_test)
    assert len(model_metrics) == len(MODEL_NAMES), "length of retrieved metrics does not much number of models"
    return {MODEL_NAMES[i]: metrics for i, metrics in enumerate(model_metrics)}

if __name__ == '__main__':
    Dataset1 = pd.read_csv(DATA_DIR + "Dataset14Days.csv", sep=';')
    patient_ids = Dataset1[['PseudoID']].values
    Dataset1.drop(columns=['PseudoID'], inplace=True)
    single_split_metrics = simple_split(Dataset1)
    save_metrics(single_split_metrics)

'''
    fold_metrics = cross_validation(Dataset1, groups, 10, 0.625)
    save_metrics(fold_metrics)
'''
