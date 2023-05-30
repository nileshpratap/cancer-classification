#%%
# MCC
# sensitivity
# specificity
# precision
# accuracy
# false_discovery_rate
# false_positive_rate
# positive_predictive_value
# negative_predictive_value
# dice - MSIM based
# weighted_specificity

# %%
from utilities.confusion_matrix import calc_ConfusionMatrix
import numpy as np
import nibabel as nib

# create above function using tp, tn, fp, fn as inputs

def calc_MCC_CM(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator

def calc_Sensitivity_CM(tp, fn):
    return tp / (tp + fn)

def calc_Specificity_CM(tn, fp):
    return tn / (tn + fp)

def calc_Precision_CM(tp, fp):
    return tp / (tp + fp)

def calc_Accuracy_CM(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def calc_False_Discovery_Rate_CM(tp, fp):
    return fp / (tp + fp)

def calc_False_Positive_Rate_CM(fp, tn):
    return fp / (fp + tn)

def calc_Positive_Predictive_Value_CM(tp, fp):
    return tp / (tp + fp)

def calc_Negative_Predictive_Value_CM(tn, fn):
    return tn / (tn + fn)

def calc_Weighted_Specificity_CM(tp, tn, fp, fn):
    alpha = 0.1
    if (fp + tn) != 0:
        wspec = (alpha * tn) / ((1 - alpha) * fp + alpha * tn)
    else:
        wspec = 0.0
    # Return weighted specificity
    return wspec

def calc_Dice_CM(truth, pred, c=1):
    # Obtain sets with associated class
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    # Calculate Dice
    if (pd.sum() + gt.sum()) != 0:
        dice = 2 * np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
    else:
        dice = 0.0
    # Return computed Dice
    return dice

def calc_mismDice_CM(truth, pred):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred)
    alpha = 0.1
    # Identify metric wing
    p = tp + fn
    # Compute & return normal dice if p > 0
    if p > 0:
        return calc_Dice_CM(truth, pred, c=1)
    # Compute & return weighted specificity if p = 0
    else:
        return (tp * alpha) / (alpha * tn + (1 - alpha) * fp)

def calc_balanced_accuracy(sens, spec):
    return (sens + spec) / 2

#tversky scores
def calc_tversky_score(tp, tn, fp, fn):
    alpha = 0.7
    return tp / (tp + (alpha * fn) + ((1 - alpha) * fp))


#function to calculate all above metrics by calling above functions
def calc_all_metrics_CM(truth, pred):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred)
    # Compute & return all metrics
    return calc_MCC_CM(tp, tn, fp, fn), calc_Sensitivity_CM(tp, fn), calc_Specificity_CM(tn, fp), \
           calc_Precision_CM(tp, fp), calc_Accuracy_CM(tp, tn, fp, fn), calc_False_Discovery_Rate_CM(tp, fp), \
           calc_False_Positive_Rate_CM(fp, tn), calc_Positive_Predictive_Value_CM(tp, fp), \
           calc_Negative_Predictive_Value_CM(tn, fn), calc_mismDice_CM(truth, pred), \
           calc_Weighted_Specificity_CM(tp, tn, fp, fn), calc_tversky_score(tp, tn, fp, fn), \
           calc_balanced_accuracy(calc_Sensitivity_CM(tp, fn), calc_Specificity_CM(tn, fp)), \
            calc_Dice_CM(truth, pred, c=1)


#%% funcrion to calculate all the sttas and add them in dict & return dict
def calc_stats(gt_path, seg_path, dict):

    print(gt_path)
    print(seg_path)

    # if gt_path exists
    if gt_path.exists():
        print("GT path exists")
    if seg_path.exists():
        print("Seg path exists")

    for i in seg_path.glob("*.nii.gz"):
        print(i.name)
        segmentation = nib.load(str(i)).get_fdata()
        ground_truth = nib.load(str(gt_path / i.name)).get_fdata()
        print(segmentation.shape)
        print(ground_truth.shape)

        # flatten data
        pred_data = segmentation.flatten()
        label_data = ground_truth.flatten()

        # calculate all metrics using function calc_all_metrics_CM
        stats = calc_all_metrics_CM(label_data, pred_data)
        dict[i.name] = {'mcc': stats[0], 'sens': stats[1], 'spec': stats[2], 'prec': stats[3], 'acc': stats[4],
                                'FDR': stats[5], 'FPR': stats[6], 'PPV': stats[7], 'NPV': stats[8], 'mism': stats[9],
                                'wspec': stats[10], 'tversky': stats[11], 'balanced_acc': stats[12], 'dice': stats[13]}

    return dict

#%% other way to calculate stats
def calc_stats2(data_path, gt_name, pred_name, dict):

    print(data_path)

    # if data_path exists
    if data_path.exists():
        print("data_path path exists")

    for i in data_path.glob("*"):
        # print(i.name)
        file_name = i.name.replace("-1w", "")
        print(file_name)

        seg_file_name = i / pred_name
        gt_file_name = i / gt_name

        segmentation = nib.load(str(seg_file_name)).get_fdata()
        ground_truth = nib.load(str(gt_file_name)).get_fdata()
        print(segmentation.shape)
        print(ground_truth.shape)

        # flatten data
        pred_data = segmentation.flatten()
        label_data = ground_truth.flatten()

        # calculate all metrics using function calc_all_metrics_CM
        stats = calc_all_metrics_CM(label_data, pred_data)
        dict[i.name] = {'mcc': stats[0], 'sens': stats[1], 'spec': stats[2], 'prec': stats[3], 'acc': stats[4],
                        'FDR': stats[5], 'FPR': stats[6], 'PPV': stats[7], 'NPV': stats[8], 'dice': stats[9],
                        'wspec': stats[10], 'tversky': stats[11], 'balanced_acc': stats[12]}

    return dict

#%% function to calculate stats b/w 1w gt and 24h mask or vice versa.
def calc_stats3(gt_path, gt_name, pred_path, pred_name, dict):
    for i in gt_path.glob("*"):
        # print(i.name)
        file_name = i.name.replace("-1w", "")
        # print(file_name)

        # if path exists, then continue
        new_filename = file_name + "-24h"
        if (pred_path / new_filename).exists():
            print(str(pred_path / file_name) + "  path exists")
            seg_file_name = pred_path / new_filename / pred_name
            gt_file_name = i / gt_name

            segmentation = nib.load(str(seg_file_name)).get_fdata()
            ground_truth = nib.load(str(gt_file_name)).get_fdata()
            print(segmentation.shape)
            print(ground_truth.shape)

            # flatten data
            pred_data = segmentation.flatten()
            label_data = ground_truth.flatten()

            # calculate all metrics using function calc_all_metrics_CM
            stats = calc_all_metrics_CM(label_data, pred_data)
            dict[i.name] = {'mcc': stats[0], 'sens': stats[1], 'spec': stats[2], 'prec': stats[3], 'acc': stats[4],
                            'FDR': stats[5], 'FPR': stats[6], 'PPV': stats[7], 'NPV': stats[8], 'dice': stats[9],
                            'wspec': stats[10], 'tversky': stats[11], 'balanced_acc': stats[12]}

    return dict