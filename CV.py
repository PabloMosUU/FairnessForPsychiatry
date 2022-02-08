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
import matplotlib.pyplot as plt

np.random.seed(1)
DATA_DIR = '/media/bigdata/10. Stages/3. Afgerond/2020-08 Jesse Kuiper/'
MODEL_NAMES = ['logreg', 'rf', 'logregreweight', 'rfreweight', 'prejudiceremover']
MODEL_THRESHOLDS = {'logreg': np.linspace(0.1, 0.9, 50),
                    'rf': np.linspace(0.1, 0.8, 50),
                    'logregreweight': np.linspace(0.01, 0.8, 50),
                    'rfreweight': np.linspace(0.01, 0.8, 50),
                    'prejudiceremover': np.linspace(0.01, 0.8, 50)}
OUTPUT_DIR = 'CV/CV5_retrain_repro/'
N_SPLITS = 5

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
    """
    Initialize and train a model
    :param name: the name of the model
    :param dataset: training dataset
    :param protected_attribute_name: name of the protected attribute
    :param unprivileged_groups: definition of unprivileged groups
    :param privileged_groups: definition of privileged groups
    :return: the trained model
    """
    # Model initialization and fitting
    if name == 'prejudiceremover':
        model = PrejudiceRemover(sensitive_attr=protected_attribute_name, eta=25.0)
    else:
        if name.startswith('logreg'):
            fitter = LogisticRegression(solver='liblinear', random_state=1)
        elif name.startswith('rf'):
            fitter = RandomForestClassifier(n_estimators=500, min_samples_leaf=25)
        else:
            raise ValueError('Unknown model', name)
        model = make_pipeline(StandardScaler(), fitter)
    return fit_model(model, dataset, name, unprivileged_groups, privileged_groups)


def get_fit_parameters(name: str, dataset: StandardDataset) -> dict:
    if name.startswith('logreg'):
        return {'logisticregression__sample_weight': dataset.instance_weights}
    elif name.startswith('rf'):
        return {'randomforestclassifier__sample_weight': dataset.instance_weights}
    else:
        raise ValueError('Unknown model', name)


def fit_model(model, dataset: StandardDataset, name: str, unprivileged_groups: list, privileged_groups: list):
    """
    Fit a model
    :param model: the initialized model
    :param dataset: training data
    :param name: the name of the model
    :param unprivileged_groups: definition of unprivileged groups
    :param privileged_groups: definition of privileged groups
    :return: a trained model
    """
    # Dataset manipulations
    if name.endswith('reweight'):
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_transf_train = RW.fit_transform(dataset)
        dataset = dataset_transf_train
    if type(model) == PrejudiceRemover:
        return model.fit(dataset)
    else:
        fit_params = get_fit_parameters(name, dataset)
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


def remove_prejudice(dataset: StandardDataset, fit: bool, scaler: StandardScaler) -> StandardDataset:
    scaled_dataset = dataset.copy()
    if fit:
        scaled_dataset.features = scaler.fit_transform(scaled_dataset.features)
    else:
        scaled_dataset.features = scaler.transform(scaled_dataset.features)
    return scaled_dataset

def train_val_test_model(model_name: str,
                         dataset_train: aif360.datasets.StandardDataset,
                         dataset_val: aif360.datasets.StandardDataset,
                         dataset_test: aif360.datasets.StandardDataset,
                         unprivileged_groups: list,
                         privileged_groups: list,
                         protected_attribute: str,
                         fold_number: int,
                         fixed_threshold=None,
                         dataset_dev=None
                         ) -> dict:
    """
    Train, validate and test a single model
    :param model_name: the name of the model
    :param dataset_train: training dataset
    :param dataset_val: validation dataset
    :param dataset_test: test dataset
    :param unprivileged_groups: definition of unprivileged groups
    :param privileged_groups: definition of privileged groups
    :param protected_attribute: name of the protected attribute
    :param fold_number: for the purpose of saving plots
    :param fixed_threshold: if set, do not perform validation to determine the best threshold
    :param dataset_dev: if set, use this dataset for retraining after validation
    :return: metrics on the test set
    """
    assert not fixed_threshold or not dataset_dev, 'Cannot set a fixed threshold and also retrain'
    if model_name == 'prejudiceremover':
        pr_orig_scaler = StandardScaler()
        dataset_train = remove_prejudice(dataset_train, True, pr_orig_scaler)
        if not fixed_threshold:
            dataset_val = remove_prejudice(dataset_val, False, pr_orig_scaler)
        dataset_test = remove_prejudice(dataset_test, False, pr_orig_scaler)
        if dataset_dev:
            dataset_dev = remove_prejudice(dataset_dev, True, pr_orig_scaler)
    trained_model = train_model(model_name,
                                dataset_train,
                                protected_attribute,
                                unprivileged_groups,
                                privileged_groups)
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
        if dataset_dev:
            trained_model = fit_model(trained_model, dataset_dev, model_name, unprivileged_groups, privileged_groups)
        plot_fairness_metrics(MODEL_THRESHOLDS[model_name], validation_metrics, model_name, fold_number)
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
                     fold_number: int,
                     fixed_threshold=None,
                     dataset_dev=None
                     ) -> pd.DataFrame:
    """
    Train all models given by model_names
    :param model_names: the list of models to be trained
    :param dataset_train: training dataset
    :param dataset_val: validation dataset
    :param dataset_test: test dataset
    :param fold_number: for the purpose of saving plots
    :param fixed_threshold: if set, do not perform validation to pick the best threshold
    :param dataset_dev: if set, use this dataset for retraining after validation
    :return: a dataframe with test metrics for all models
    """
    assert not fixed_threshold or not dataset_dev, 'Cannot fix threshold and also retrain'
    sens_ind = 0  # TODO: magic variable (here and elsewhere)
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
                                            fold_number=fold_number,
                                            fixed_threshold=fixed_threshold,
                                            dataset_dev=dataset_dev)
        model_metrics.append(test_metrics)
    assert len(model_metrics) == len(MODEL_NAMES), "length of retrieved metrics does not much number of models"
    return get_dataframe({MODEL_NAMES[i]: metrics for i, metrics in enumerate(model_metrics)})


def cross_validation(data: pd.DataFrame,
                     group_labels: list,
                     n_splits: int,
                     train_dev_frac: float,
                     fixed_threshold=None,
                     retrain=False) -> list:
    """
    Perform cross-validation on the data to determine performance and bias metrics
    :param data: the entire dataset
    :param group_labels: patient IDs corresponding to the rows of the data
    :param n_splits: how many folds of cross-validation to do
    :param train_dev_frac: what fraction of the development set to use for training
    :param fixed_threshold: if set, no validation is done on the development set to find the optimal threshold
    :param retrain: if True, retrain model on entire dev set after validation (requires no fixed_threshold)
    :return: a list of metrics corresponding to each fold
    """
    assert len(data) == len(group_labels), 'dataframe and groups must have the same length'
    assert 1 >= train_dev_frac > 0, 'Invalid train_dev_frac; must be >0 and <=1'
    assert (train_dev_frac < 1 and not fixed_threshold) or (fixed_threshold and not retrain and train_dev_frac == 1), \
        "Either validate to pick threshold or fixed threshold and do not retrain"
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
        model_metrics_df = train_all_models(MODEL_NAMES,
                                            dataset_train,
                                            dataset_val,
                                            dataset_test,
                                            fold_number=len(fold_metrics),
                                            fixed_threshold=fixed_threshold,
                                            dataset_dev=dataset_dev)
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
    return train_all_models(MODEL_NAMES, dataset_train, dataset_val, dataset_test, fold_number=0)

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
    df = df.copy()
    disparate_impact_name = 'DI'
    average_odds_difference_name = 'AOD'
    statistical_parity_difference_name = 'SPD'
    equal_opportunity_difference_name = 'EOD'
    f1_score_name = 'F1'
    balanced_accuracy_name = r'Acc$_{\rm bal}$'
    df.rename(columns={'bal_acc': balanced_accuracy_name,
                       'F1_score': f1_score_name,
                       'disp_imp': disparate_impact_name,
                       'avg_odds_diff': average_odds_difference_name,
                       'stat_par_diff': statistical_parity_difference_name,
                       'eq_opp_diff': equal_opportunity_difference_name},
              inplace=True)
    df.drop(columns=['FP Diff'], inplace=True, errors='ignore')
    df.reset_index(inplace=True)
    df.fillna('', inplace=True)
    logistic_regression_name = 'LR'
    random_forest_name = 'RF'
    df['Classifier'] = df['Classifier'].map({'Logistic Regression': logistic_regression_name,
                                             'Random Forest': random_forest_name})
    df.loc[df['Bias Mitigator'] == 'Prejudice Remover', 'Classifier'] = logistic_regression_name
    prejudice_remover_name = 'PR'
    reweighting_name = 'RW'
    df.loc[df['Bias Mitigator'] == 'Prejudice Remover', 'Bias Mitigator'] = prejudice_remover_name
    df.loc[df['Bias Mitigator'] == 'Reweighing', 'Bias Mitigator'] = reweighting_name
    df.loc[df['Bias Mitigator'] == 'Reweighting', 'Bias Mitigator'] = reweighting_name
    df.rename(columns={'Classifier': 'Clf.', 'Bias Mitigator': 'Mit.'}, inplace=True)
    cols = ['Classifier',
            'Bias Mitigator',
            balanced_accuracy_name,
            f1_score_name,
            disparate_impact_name,
            average_odds_difference_name,
            statistical_parity_difference_name,
            equal_opportunity_difference_name]
    lines = [' & '.join(cols) + ' \\\\', '\\hline', '\\hline']
    for i, elll in enumerate(
            [' & '.join(['{:.3f}'.format(ell) if type(ell) != str else ell for ell in row]) + ' \\\\' for row in
             df[[el for el in cols]].values]):
        if i in (2, 4):
            lines.append('\\hline')
        lines.append(elll)
    return '\n'.join(lines)


def get_latex_performance(df: pd.DataFrame, delta=False) -> str:
    df = df.copy()
    f1_score_name = 'F1'
    balanced_accuracy_name = r'Acc$_{\rm bal}$'
    if delta:
        f1_score_name, balanced_accuracy_name = [r'$\Delta$' + el for el in (f1_score_name, balanced_accuracy_name)]
    df.rename(columns={'bal_acc': balanced_accuracy_name,
                       'F1_score': f1_score_name},
              inplace=True)
    df.reset_index(inplace=True)
    df.fillna('', inplace=True)
    logistic_regression_name = 'LR'
    random_forest_name = 'RF'
    df['Classifier'] = df['Classifier'].map({'Logistic Regression': logistic_regression_name,
                                             'Random Forest': random_forest_name})
    df.loc[df['Bias Mitigator'] == 'Prejudice Remover', 'Classifier'] = logistic_regression_name
    prejudice_remover_name = 'PR'
    reweighting_name = 'RW'
    df.loc[df['Bias Mitigator'] == 'Prejudice Remover', 'Bias Mitigator'] = prejudice_remover_name
    df.loc[df['Bias Mitigator'] == 'Reweighing', 'Bias Mitigator'] = reweighting_name
    df.loc[df['Bias Mitigator'] == 'Reweighting', 'Bias Mitigator'] = reweighting_name
    classifier_name = 'Clf.'
    mitigator_name = 'Mit.'
    df.rename(columns={'Classifier': classifier_name, 'Bias Mitigator': mitigator_name}, inplace=True)
    cols = [classifier_name, mitigator_name,
            balanced_accuracy_name,
            f1_score_name]
    lines = [r"\begin{tabular}{|l|l|c|c|}", r"\hline",
             r"\multicolumn{2}{|c|}{Model} & \multicolumn{2}{c|}{Performance} \\", r"\hline"]
    lines += [' & '.join(cols) + ' \\\\', '\\hline', '\\hline']
    for i, elll in enumerate(
            [' & '.join(['{:.3f}'.format(ell) if type(ell) != str else format_with_uncertainty(ell, delta)
                         for ell in row]) + ' \\\\' for row in
             df[[el for el in cols]].values]):
        if i in (2, 4):
            lines.append('\\hline')
        lines.append(elll)
    lines += [r"\hline", r"\end{tabular}"]
    return '\n'.join(lines)

def format_with_uncertainty(value: str, delta: bool) -> str:
    if delta and '+/-' in value:
        mean, stdev = [float(el) for el in value.split('+/-')]
        if abs(mean) > 2 * stdev:
            value = r'\textbf{' + value + '}'
    return value.replace('+/-', r'$\pm$')

def get_latex_fairness(df: pd.DataFrame, delta=False) -> str:
    df = df.copy()
    disparate_impact_name = 'DI'
    average_odds_difference_name = 'AOD'
    statistical_parity_difference_name = 'SPD'
    equal_opportunity_difference_name = 'EOD'
    if delta:
        disparate_impact_name, \
            average_odds_difference_name, \
            statistical_parity_difference_name, \
            equal_opportunity_difference_name = [r'$\Delta$' + el for el in (disparate_impact_name,
                                                                             average_odds_difference_name,
                                                                             statistical_parity_difference_name,
                                                                             equal_opportunity_difference_name)]
    df.rename(columns={'disp_imp': disparate_impact_name,
                       'avg_odds_diff': average_odds_difference_name,
                       'stat_par_diff': statistical_parity_difference_name,
                       'eq_opp_diff': equal_opportunity_difference_name},
              inplace=True)
    df.reset_index(inplace=True)
    df.fillna('', inplace=True)
    logistic_regression_name = 'LR'
    random_forest_name = 'RF'
    df['Classifier'] = df['Classifier'].map({'Logistic Regression': logistic_regression_name,
                                             'Random Forest': random_forest_name})
    df.loc[df['Bias Mitigator'] == 'Prejudice Remover', 'Classifier'] = logistic_regression_name
    prejudice_remover_name = 'PR'
    reweighting_name = 'RW'
    df.loc[df['Bias Mitigator'] == 'Prejudice Remover', 'Bias Mitigator'] = prejudice_remover_name
    df.loc[df['Bias Mitigator'] == 'Reweighing', 'Bias Mitigator'] = reweighting_name
    df.loc[df['Bias Mitigator'] == 'Reweighting', 'Bias Mitigator'] = reweighting_name
    classifier_name = 'Clf.'
    mitigator_name = 'Mit.'
    df.rename(columns={'Classifier': classifier_name, 'Bias Mitigator': mitigator_name}, inplace=True)
    cols = [classifier_name,
            mitigator_name,
            disparate_impact_name,
            average_odds_difference_name,
            statistical_parity_difference_name,
            equal_opportunity_difference_name]
    lines = [r"\begin{tabular}{|l|l|c|c|c|c|}", r"\hline",
             r"\multicolumn{2}{|c|}{Model} & \multicolumn{4}{c|}{Fairness} \\", r"\hline"]
    lines += [' & '.join(cols) + ' \\\\', '\\hline', '\\hline']
    for i, elll in enumerate(
            [' & '.join(['{:.3f}'.format(ell) if type(ell) != str else format_with_uncertainty(ell, delta)
                         for ell in row]) + ' \\\\' for row in
             df[[el for el in cols]].values]):
        if i in (2, 4):
            lines.append('\\hline')
        lines.append(elll)
    lines += [r"\hline", r"\end{tabular}"]
    return '\n'.join(lines)


def plot(x, x_name, y_left, y_left_name, y_right, y_right_name, filename=None):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 1)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 1)
    else:
        ax2.set_ylim(-0.25, 1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    if filename:
        fig.savefig(filename)
    plt.close('all')


def plot_fairness_metrics(thresh_arr: np.ndarray, val_metrics: dict, model_name: str, fold: int) -> None:
    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1 / disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)',
         filename=OUTPUT_DIR + f'{model_name}_DI_fold{fold}.eps')
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.',
         filename=OUTPUT_DIR + f'{model_name}_AOD_fold{fold}.eps')


if __name__ == '__main__':
    Dataset1 = pd.read_csv(DATA_DIR + "Dataset14Days.csv", sep=';')
    patient_ids = Dataset1[['PseudoID']].values
    Dataset1.drop(columns=['PseudoID'], inplace=True)
    fold_dfs = cross_validation(Dataset1, patient_ids, n_splits=N_SPLITS, train_dev_frac=0.625, retrain=True)
    for ix, fold_df in enumerate(fold_dfs):
        fold_df.to_csv('CV/CV5_retrain_repro/fold' + str(ix) + '.csv', sep=';')
    full_df = pd.concat(fold_dfs)
    full_df.to_csv(OUTPUT_DIR + 'cross_validation.csv', sep=';')

    """
    single_split_metrics = simple_split(Dataset1)
    out_df = single_split_metrics
    out_df['fold'] = 0
    out_df.to_csv('single_split.csv', sep=';')
    print(get_latex(single_split_metrics))
    """
