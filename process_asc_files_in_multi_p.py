from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from sys import platform as _platform
from functools import partial
import multiprocessing
import os
from tqdm.auto import tqdm
from multi_proc_funcs import DIST_MODELS_FOLDER, process_trial_choice, set_up_models
import sys
import pandas as pd


def get_cpu_count():
    if os.sys.platform in ("linux", "linux2", "darwin"):
        return os.cpu_count()
    elif os.sys.platform == "win32":
        return multiprocessing.cpu_count()
    else:
        return 1


def process_asc_files_in_multi_proc(
    algo_choice,
    choice_handle_short_and_close_fix,
    discard_fixations_without_sfix,
    discard_far_out_of_text_fix,
    x_thres_in_chars,
    y_thresh_in_heights,
    short_fix_threshold,
    merge_distance_threshold,
    discard_long_fix,
    discard_long_fix_threshold,
    discard_blinks,
    measures_to_calculate_multi_asc,
    include_coords_multi_asc,
    sent_measures_to_calculate_multi_asc,
    trials_by_ids,
    classic_algos_cfg,
    models_dict,
    fix_cols_to_add_multi_asc,
):
    funcc = partial(
        process_trial_choice,
        algo_choice=algo_choice,
        choice_handle_short_and_close_fix=choice_handle_short_and_close_fix,
        for_multi=True,
        discard_fixations_without_sfix=discard_fixations_without_sfix,
        discard_far_out_of_text_fix=discard_far_out_of_text_fix,
        x_thres_in_chars=x_thres_in_chars,
        y_thresh_in_heights=y_thresh_in_heights,
        short_fix_threshold=short_fix_threshold,
        merge_distance_threshold=merge_distance_threshold,
        discard_long_fix=discard_long_fix,
        discard_long_fix_threshold=discard_long_fix_threshold,
        discard_blinks=discard_blinks,
        measures_to_calculate_multi_asc=measures_to_calculate_multi_asc,
        include_coords_multi_asc=include_coords_multi_asc,
        sent_measures_to_calculate_multi_asc=sent_measures_to_calculate_multi_asc,
        classic_algos_cfg=classic_algos_cfg,
        models_dict=models_dict,
        fix_cols_to_add=fix_cols_to_add_multi_asc,
    )
    workers = min(len(trials_by_ids), 32, get_cpu_count() - 1)
    with multiprocessing.Pool(workers) as pool:
        out = pool.map(funcc, trials_by_ids.values())
    return out


def make_json_compatible(obj):
    if isinstance(obj, dict):
        return {k: make_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_compatible(v) for v in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj


def main():
    try:
        input_data = sys.stdin.buffer.read()

        (
            algo_choice,
            choice_handle_short_and_close_fix,
            discard_fixations_without_sfix,
            discard_far_out_of_text_fix,
            x_thres_in_chars,
            y_thresh_in_heights,
            short_fix_threshold,
            merge_distance_threshold,
            discard_long_fix,
            discard_long_fix_threshold,
            discard_blinks,
            measures_to_calculate_multi_asc,
            include_coords_multi_asc,
            sent_measures_to_calculate_multi_asc,
            trials_by_ids,
            classic_algos_cfg,
            models_dict,
            fix_cols_to_add_multi_asc,
        ) = json.loads(input_data)
        if (
            "DIST" in algo_choice
            or "Wisdom_of_Crowds_with_DIST" in algo_choice
            or "DIST-Ensemble" in algo_choice
            or "Wisdom_of_Crowds_with_DIST_Ensemble" in algo_choice
        ):
            del models_dict  # Needed to stop pickling from failing for multiproc
            models_dict = set_up_models(DIST_MODELS_FOLDER)
        else:
            models_dict = {}
        out = process_asc_files_in_multi_proc(
            algo_choice,
            choice_handle_short_and_close_fix,
            discard_fixations_without_sfix,
            discard_far_out_of_text_fix,
            x_thres_in_chars,
            y_thresh_in_heights,
            short_fix_threshold,
            merge_distance_threshold,
            discard_long_fix,
            discard_long_fix_threshold,
            discard_blinks,
            measures_to_calculate_multi_asc,
            include_coords_multi_asc,
            sent_measures_to_calculate_multi_asc,
            trials_by_ids,
            classic_algos_cfg,
            models_dict,
            fix_cols_to_add_multi_asc,
        )
        out2 = []
        for dffix, trial in out:
            dffix = dffix.to_dict("records")
            trial = make_json_compatible(trial)
            out2.append((dffix, trial))
        json_data_out = json.dumps(out2)
        sys.stdout.flush()
        print(json_data_out)
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
