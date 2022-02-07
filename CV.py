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
MODEL_NAMES = ['logreg', 'rf', 'logregreweight', 'rfreweight', 'prejudiceremover']
MODEL_THRESHOLDS = {'logreg': np.linspace(0.1, 0.9, 50),
                    'rf': np.linspace(0.1, 0.8, 50),
                    'logregreweight': np.linspace(0.01, 0.8, 50),
                    'rfreweight': np.linspace(0.01, 0.8, 50),
                    'prejudiceremover': np.linspace(0.01, 0.8, 50)}

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

def train_val_test_model(model_name: str,
                         dataset_train: aif360.datasets.StandardDataset,
                         dataset_val: aif360.datasets.StandardDataset,
                         dataset_test: aif360.datasets.StandardDataset,
                         unprivileged_groups: list,
                         privileged_groups: list,
                         male_name: str,
                         fixed_threshold=None
                         ) -> dict:
    if model_name == 'prejudiceremover':
        pr_orig_scaler = StandardScaler()
        dataset_train, dataset_val, dataset_test = [el.copy() for el in (dataset_train, dataset_val, dataset_test)]
        dataset_train.features = pr_orig_scaler.fit_transform(dataset_train.features)
        if not fixed_threshold:
            dataset_val.features = pr_orig_scaler.transform(dataset_val.features)
        dataset_test.features = pr_orig_scaler.transform(dataset_test.features)
    trained_model = train_model(model_name, dataset_train, male_name, unprivileged_groups, privileged_groups)
    if fixed_threshold:
        test_thresholds = np.array([fixed_threshold])
    else:
        validation_metrics = compute_metrics(trained_model,
                                             dataset_val,
                                             MODEL_THRESHOLDS[model_name],
                                             unprivileged_groups,
                                             privileged_groups)
        best_threshold = get_best_threshold(validation_metrics)
        test_thresholds = np.array([MODEL_THRESHOLDS[model_name][best_threshold]])
    # Todo: retrain model on the entire dev set
    test_metrics = compute_metrics(trained_model,
                                   dataset_test,
                                   test_thresholds,
                                   unprivileged_groups,
                                   privileged_groups)
    return test_metrics

def train_all_models(model_names: list,
                     dataset_train: aif360.datasets.StandardDataset,
                     dataset_val: aif360.datasets.StandardDataset,
                     dataset_test: aif360.datasets.StandardDataset,
                     fixed_threshold=None
                     ) -> pd.DataFrame:
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
                                            protected_attribute_name,
                                            fixed_threshold)
        model_metrics.append(test_metrics)
    assert len(model_metrics) == len(MODEL_NAMES), "length of retrieved metrics does not much number of models"
    return get_dataframe({MODEL_NAMES[i]: metrics for i, metrics in enumerate(model_metrics)})


def cross_validation(data: pd.DataFrame,
                     group_labels: list,
                     n_splits: int,
                     train_dev_frac: float,
                     fixed_threshold=None) -> list:
    assert len(data) == len(group_labels), 'dataframe and groups must have the same length'
    assert not fixed_threshold or train_dev_frac == 1
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for dev, test in gkf.split(data, groups=group_labels):
        df_dev, df_test = [data.iloc[el] for el in (dev, test)]
        groups_dev = group_labels[dev]
        if fixed_threshold:
            df_train = df_dev.sample(frac=1, replace=False, random_state=42)
            df_val = pd.DataFrame(columns=df_train.columns)
        else:
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
        model_metrics_df = train_all_models(MODEL_NAMES, dataset_train, dataset_val, dataset_test, fixed_threshold)
        model_metrics_df['fold'] = len(fold_metrics)
        fold_metrics.append(model_metrics_df)
    return fold_metrics

def simple_split(data: pd.DataFrame) -> pd.DataFrame:
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
    return train_all_models(MODEL_NAMES, dataset_train, dataset_val, dataset_test)

def get_dataframe(model_metrics: dict) -> pd.DataFrame:
    # table summary of the results
    pd.set_option('display.multi_sparse', False)
    names = list(model_metrics.keys())
    results = [model_metrics[name] for name in names]
    debias = pd.Series(['Reweighting'
                        if name.endswith('reweight')
                        else ('Prejudice Remover' if name == 'prejudiceremover' else '')
                        for name in names],
                       name='Bias Mitigator')
    clf = pd.Series(['Logistic Regression'
                     if name.startswith('logreg') or name == 'prejudiceremover'
                     else ('Random Forest' if name.startswith('rf') else '')
                     for name in names],
                    name='Classifier')
    results_df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index([debias, clf])
    return results_df

def get_latex(df: pd.DataFrame) -> str:
    df.rename(columns={'bal_acc': 'Bal acc', 'F1_score': 'F1 score', 'disp_imp': 'Disp imp',
                               'avg_odds_diff': 'Avg odds diff', 'stat_par_diff': 'Stat par diff',
                               'eq_opp_diff': 'Eq opp diff'}, inplace=True)
    df.drop(columns=['FP Diff'], inplace=True)
    df.reset_index(inplace=True)
    df.fillna('', inplace=True)
    df['Classifier'] = df['Classifier'].map({'Logistic Regression': 'Log. Reg.',
                                                             'Random Forest': 'Rnd. For.'})
    df.loc[df['Bias Mitigator'] == 'Prejudice Remover', 'Classifier'] = 'Log. Reg.'
    cols = df.columns
    lines = [' & '.join(cols) + ' \\\\', '\\hline', '\\hline']
    for i, elll in enumerate(
            [' & '.join(['{:.3f}'.format(ell) if type(ell) != str else ell for ell in row]) + ' \\\\' for row in
             df[[el for el in cols]].values]):
        if i in (2, 4):
            lines.append('\\hline')
        lines.append(elll)
    return '\n'.join(lines)


if __name__ == '__main__':
    Dataset1 = pd.read_csv(DATA_DIR + "Dataset14Days.csv", sep=';')
    patient_ids = Dataset1[['PseudoID']].values
    Dataset1.drop(columns=['PseudoID'], inplace=True)

    """
    single_split_metrics = simple_split(Dataset1)
    out_df = single_split_metrics
    out_df['fold'] = 0
    out_df.to_csv('single_split.csv', sep=';')
    print(get_latex(single_split_metrics))
    """

    fold_dfs = cross_validation(Dataset1, patient_ids, 5, 1, fixed_threshold=0.5)
    for ix, fold_df in enumerate(fold_dfs):
        fold_df.to_csv('CV/CV5_fixed_threshold/fold' + str(ix) + '.csv', sep=';')
    full_df = pd.concat(fold_dfs)
    full_df.to_csv('CV/CV5_fixed_threshold/cross_validation.csv', sep=';')
