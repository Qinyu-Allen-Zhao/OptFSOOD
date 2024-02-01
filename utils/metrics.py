import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

recall_level_default = 0.95


def get_measures_from_pred(ood_predictions, id_predictions, recall_level=recall_level_default):
    print(f"ID samples: {len(id_predictions)}")
    print(f"OOD samples: {len(ood_predictions)}")
    scores = np.concatenate([id_predictions, ood_predictions])
    labels = np.zeros(len(scores))
    labels[:len(id_predictions)] += 1

    auroc, aupr, fpr = get_measures(labels, scores, recall_level)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))

    return auroc, aupr, fpr, "{:.2f} & {:.2f}".format(100 * fpr, 100 * auroc)


def get_measures(labels, scores, recall_level=recall_level_default):
    fpr, thresh = fpr_recall(scores, labels, recall_level)
    auroc, aupr_in, aupr_out = auc(scores, labels)

    return auroc, aupr_in, fpr


# fpr_recall
def fpr_recall(conf, label, tpr):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(label, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


# auc
def auc(conf, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, conf)

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(label, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1 - label, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def cal_auroc_from_conf(id_conf, ood_conf):
    y_pred = np.concatenate([id_conf, ood_conf])
    y_test = np.zeros_like(y_pred)
    y_test[:len(id_conf)] += 1
    auroc = roc_auc_score(y_test, y_pred)
    return auroc
