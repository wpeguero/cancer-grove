"""Useful Functions for Statistical Analysis."""

import torch
from torch.utils import data
import polars as pl
import scipy

def calculate_confusion_matrix(fin_predictions:pl.DataFrame):
    """Calculate the confusion matrix using pandas.

    Calculates the confusion matrix using a csv file that
    contains both the predictions and actual labels. This
    function then creates a crosstab of the data to develop
    the confusion matrix.

    Parameters
    ----------
    fin_predictions : Pandas DataFrame
        DataFrame containing the prediction and actual
        labels.

    Returns
    -------
    Polars DataFrame
        Cross tab containing the confusion matrix of the
        predictions compared to the actual labels.

    Dictionary
        Contains the basic metrics obtained from the
        confusion matrix. The metrics are the following:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    """
    ct = fin_predictions.pivot(values="predictions", index="classification", columns="classification", aggregate_function='count')
    # Set the initial values
    tp = ct.values[1][1]
    tn = ct.values[0][0]
    fn = ct.values[0][1]
    fp = ct.values[1][0]
    # Calculate the metrics
    metrics = dict()
    metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn) # Ability of model to get the correct predictions
    metrics['Precision'] = tp / (tp + fp) # Ability of model to label actual positives as positives (think retrospectively)
    metrics['Recall'] = tp / (tp + fn) # Ability of model to correctly identify positives
    metrics['F1 Score'] = (2 * metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
    return ct, metrics

def calculate_image_t_test(dataset:data.Dataset, labels:dict, ci:int=0.95): #two sample t-test of unequal variance; Requires Correction of Degrees of Freedom
    """Calculate the t-test for comparing two samples of data with unequal variance."""
    assert ci > 0.0 and ci < 1.0, "the parameter `ci` must fall between 0 and 1.\nCurrent value: {}".format(ci)
    means = dict()
    standard_deviations = dict()
    variances = dict()
    counts = dict()
    for k,v in labels.items():
        count = 0
        means.update({k:torch.mean(torch.Tensor([torch.mean(X) for X,y in dataset if int(y.argmax().item()) == v]))})
        standard_deviations.update({k:torch.std(torch.Tensor([torch.mean(X) for X,y in dataset if int(y.argmax().item()) == v]))})
        variances.update({k:torch.var(torch.Tensor([torch.mean(X) for X,y in dataset if int(y.argmax().item()) == v]))})
        for _,y in dataset:
            if y.argmax().item() == v:
                count+=1
        counts.update({k:count})
    base_stats = {'means':means, 'stds':standard_deviations, 'variances':variances, 'counts':counts}
    m1 = means['cat']
    m2 = means['loaf']
    std1 = standard_deviations['cat']
    std2 = standard_deviations['loaf']
    var1 = variances['cat']
    var2 = variances['loaf']
    n1 = counts['cat']
    n2 = counts['loaf']
    t_value = (m1-m2)/ torch.sqrt((var1/n1) + (var2/n2))
    df_numerator = torch.square((torch.square(var1)/n1) + (torch.square(var2)/n2))
    df_denominator = (torch.square(torch.square(var1)) / ((n1**2)*(n1-1))) + (torch.square(torch.square(var2)) / ((n2**2)*(n2-1)))
    df = df_numerator / df_denominator
    #t = scipy.stats.t.ppf(ci, df)
    print('t value: ', t_value)
    print('degrees of freedom: {}'.format(df))
    return base_stats
