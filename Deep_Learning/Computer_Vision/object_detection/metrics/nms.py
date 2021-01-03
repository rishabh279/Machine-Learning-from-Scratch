from Deep_Learning.Computer_Vision.object_detection.metrics.iou import intersection_over_union
import torch


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Parameters:
    -----------
    bboxes: tensor
        List of lists containing all boxes with each bbox as [class_pred, prob_score, x1, y1, x2, y2]
    iou_threshold: float
        Threshold where predicted bboxes is correct
    threshold: float
        Threshold to remove predicted bboxes
    box_format: str

    Return:
    -------
    list: bboxes after performing NMS
    """

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]
        bboxes_nms.append(bboxes)

    return bboxes_nms