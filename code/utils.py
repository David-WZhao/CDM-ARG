from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

def arr2hot(arr, N):
    res = [0] * N
    for e in arr:
        res[e - 1] = 1
    return res

def evaluate(pred, trues, classes):
    count = pred.shape[0]
    preds = []
    lebels = [i for i in range(classes)]
    for i in range(count):
        preds.append(np.argmax(pred[i]))
    acc = accuracy_score(trues, preds)
    # micro-precision
    micro_p = precision_score(trues, preds, labels=lebels, average='micro')
    # micro-recall
    micro_r = recall_score(trues, preds, labels=lebels, average='micro')
    # micro f1-score
    micro_f1 = f1_score(trues, preds, labels=lebels, average='micro')

    # macro-precision
    macro_p = precision_score(trues, preds, average='macro')
    # macro-recall
    macro_r = recall_score(trues, preds, average='macro')
    # macro f1-score
    macro_f1 = f1_score(trues, preds, average='macro')

    # Calculate AUC and AUCPR
    # For multi-class classification, use one-vs-rest (OVR) approach
    try:
        # Binarize the labels for multi-class ROC AUC and PR AUC calculation
        trues_binarized = label_binarize(trues, classes=lebels)

        # For binary classification, label_binarize returns shape (n_samples, 1)
        # We need to handle this case separately
        if classes == 2:
            # For binary classification, use the probability of the positive class
            auc = roc_auc_score(trues, pred[:, 1])
            aucpr = average_precision_score(trues, pred[:, 1])
        else:
            # For multi-class, use OVR (one-vs-rest) strategy with macro averaging
            # Check if all classes are present in the test set
            unique_classes = np.unique(trues)
            if len(unique_classes) < classes:
                auc = roc_auc_score(trues_binarized, pred, multi_class='ovr', average='weighted')
                aucpr = average_precision_score(trues_binarized, pred, average='weighted')
            else:
                auc = roc_auc_score(trues_binarized, pred, multi_class='ovr', average='macro')
                aucpr = average_precision_score(trues_binarized, pred, average='macro')
    except Exception as e:
        print(f"Warning: Could not calculate AUC/AUCPR: {e}")
        auc = np.nan  # Use NaN instead of 0.0 to indicate calculation failed
        aucpr = np.nan

    return acc, macro_p, macro_r, macro_f1, auc, aucpr


# def evaluate(pred, y):
#     bs = pred.shape[0]
#     # auc = roc_auc_score(y, pred, multi_class='ovo')
#     auc = 0.5
#     # rmse = np.sqrt(np.mean((y - pred) ** 2))
#     rmse = 0.1
#     # pred[pred >= 0.5] = 1
#     # pred[pred < 0.5] = 0
#     TP, FP, TN, FN = 0, 0, 0, 0
#     for i in range(bs):
#         maxP = np.argmax(pred[i])
#         if maxP == y[i]:
#             if maxP == 1:
#                 TP += 1
#             else:
#                 TN += 1
#         elif maxP == 1:
#             FP += 1
#         else:
#             FN += 1
#     print('total predict num: {}, correct predict: {}, wrong predict: {}'.format(TP + FP + TN + FN, TP + TN, FP + FN))
#     print('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP, TN, FP, FN))
#     acc = (TP + TN) / (TP + FP + TN + FN)
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1 = 2 * precision * recall / (precision + recall)
#     print('acc: {}, auc: {}, precision: {}, recall: {}, f1: {}, rmse: {}'.format(acc, auc, precision, recall, f1, rmse))
#     return acc, auc, precision, recall, f1, rmse