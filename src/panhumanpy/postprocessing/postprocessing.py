"""Module with postprocessing functions to refine annotations mannualy."""

import numpy as np
import pandas as pd


def categorize_annotate_fine(value):
    if isinstance(value, list):
        return "Further"
    elif value == "False":
        return "No-match"
    elif isinstance(value, str):
        return "Match"
    return "Unknown"


def refine_annotations(encoder_map, fine_annot, softmax, predictions, ann_level):
    predictions = predictions.copy()
    level_dicts = {f"level_{i}": obj for i, obj in enumerate(encoder_map)}
    annot_dict = dict(
        zip(fine_annot["Orig_Label"], fine_annot[f"Annotate_{ann_level}"])
    )
    softmax = {f"arr_{i}": np.array(row) for i, row in enumerate(softmax)}

    for fine_label in fine_annot[f"Annotate_{ann_level}"]:
        annot_dict[fine_label] = fine_label

    def map_to_annotate_fine(label):
        label, flag = label.split("_")
        if flag == "True":
            fine_label = annot_dict.get(label)
            if pd.isna(fine_label):
                matching_fine_labels = fine_annot[
                    fine_annot[f"Orig_Label"].str.startswith(label, na=False)
                ][f"Annotate_{ann_level}"].tolist()
                matching_fine_labels = list(set(matching_fine_labels))
                return matching_fine_labels
            if fine_label == label:
                return fine_label
            elif fine_label in label:
                return fine_label
        return "False"

    predictions[f"annotate_{ann_level}"] = predictions[
        "abs_cell_type_label_with_flag"
    ].apply(map_to_annotate_fine)

    def compute_annotate_fine_prob(row):
        label, flag = row["abs_cell_type_label_with_flag"].split("_")
        if flag == "True":
            fine_label = annot_dict.get(label)
            if pd.isna(fine_label):
                return None
            if fine_label == label:
                return row["final_level_prob"]
            elif fine_label in label:
                num_levels = fine_label.count("|") + 1
                prob_col = f"prob_level_{num_levels}"
                return row.get(prob_col, None)
        return None

    predictions[f"annotate_{ann_level}_prob"] = predictions.apply(
        compute_annotate_fine_prob, axis=1
    )
    predictions["result_type"] = predictions[f"annotate_{ann_level}"].apply(
        categorize_annotate_fine
    )

    int_level_dicts = {
        int(key.split("_")[1]): value for key, value in level_dicts.items()
    }
    precomputed_softmax = {}

    for level, level_dict in int_level_dicts.items():
        arr_key = f"arr_{level}"
        if arr_key in softmax:
            softmax_array = softmax[arr_key]
            precomputed_softmax[level] = {
                key: softmax_array[:, idx]
                for key, idx in level_dict.items()
                if idx < softmax_array.shape[1]
            }

    predictions["best_prediction"] = None
    predictions = predictions.reset_index(names="index")

    for idx, row in predictions[predictions["result_type"] == "Further"].iterrows():
        annotations = row[f"annotate_{ann_level}"]
        max_value = -np.inf
        best_prediction = None

        for pred in annotations:
            levels = len(pred.split("|")) - 1

            if pred in precomputed_softmax[levels]:
                value = precomputed_softmax[levels][pred][idx]
                if value > max_value:
                    max_value = value
                    best_prediction = pred

        predictions.at[idx, "best_prediction"] = best_prediction
        predictions.at[idx, f"annotate_{ann_level}_prob"] = max_value

    predictions[f"annotate_{ann_level}_final"] = np.where(
        predictions["result_type"] == "Further",
        predictions["best_prediction"],
        predictions[f"annotate_{ann_level}"],
    )

    predictions[f"annotate_{ann_level}"] = predictions[
        f"annotate_{ann_level}_final"
    ].apply(lambda x: x.split("|")[-1] if isinstance(x, str) else x)
    results = predictions[
        [
            "abs_cell_type_label",
            f"annotate_{ann_level}",
            f"annotate_{ann_level}_prob",
            "result_type",
            "index",
        ]
    ].copy()
    results.set_index("index", inplace=True)
    predictions.drop(
        ["best_prediction", f"annotate_{ann_level}_final", "result_type"],
        axis=1,
        inplace=True,
    )

    return results
