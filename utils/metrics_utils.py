import numpy as np
import sklearn
from scipy.ndimage.measurements import label
from bisect import bisect


class GroundTruthComponent:

    def __init__(self, anomaly_scores):

        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()
        self.index = 0
        self.last_threshold = None

    def compute_overlap(self, threshold):
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        while (self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold):
            self.index += 1

        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            """WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    correction = 0.
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)

            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):

    assert len(anomaly_maps) == len(ground_truth_maps)

    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)

    structure = np.ones((3, 3), dtype=int)

    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):

        labeled, n_components = label(gt_map, structure)

        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)

    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

    fprs = [1.0]
    pros = [1.0]
    thr = [0.0]

    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)
        thr.append(threshold)

    fprs = fprs[::-1]
    pros = pros[::-1]
    thr = thr[::-1]

    return fprs, pros, thr


def calculate_au_pro(gts, predictions, integration_limit = [0.3, 0.1, 0.05, 0.01], num_thresholds = 99):
    pro_curve = compute_pro(anomaly_maps = predictions, ground_truth_maps = gts, num_thresholds = num_thresholds)

    au_pros = []

    for int_lim in integration_limit:
        au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max = int_lim)
        au_pro /= int_lim

        au_pros.append(au_pro)

    return au_pros, pro_curve


def calculate_au_prc(gts, predictions):
    
    fpr, tpr, _ = sklearn.metrics.roc_curve(gts, predictions)
    au_prc = sklearn.metrics.auc(fpr, tpr)

    return au_prc