import pickle
from io import StringIO
import re
import zipfile
import os
import plotly.graph_objects as go
from io import StringIO
import numpy as np
import pandas as pd
from PIL import Image
import json
from matplotlib import pyplot as plt
import pathlib as pl
import matplotlib as mpl
from streamlit.runtime.uploaded_file_manager import UploadedFile
from tqdm.auto import tqdm
import time
import requests
from icecream import ic
from matplotlib import font_manager
from multi_proc_funcs import (
    COLORS,
    PLOTS_FOLDER,
    RESULTS_FOLDER,
    add_boxes_to_ax,
    add_text_to_ax,
    matplotlib_plot_df,
    save_trial_to_json,
    sigmoid,
)
import emreading_funcs as emf

ic.configureOutput(includeContext=True)
TEMP_FIGURE_STIMULUS_PATH = PLOTS_FOLDER / "temp_matplotlib_plot_stimulus.png"
all_fonts = [x.name for x in font_manager.fontManager.ttflist]
mpl.use("agg")

DIST_MODELS_FOLDER = pl.Path("models")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PLOTS_FOLDER = pl.Path("plots")

names_dict = {
    "SSACC": {"Descr": "Start of Saccade", "Pattern": "SSACC <eye > <stime>"},
    "ESACC": {
        "Descr": "End of Saccade",
        "Pattern": "ESACC <eye > <stime> <etime > <dur> <sxp > <syp> <exp > <eyp> <ampl > <pv >",
    },
    "SFIX": {"Descr": "Start of Fixation", "Pattern": "SFIX <eye > <stime>"},
    "EFIX": {"Descr": "End of Fixation", "Pattern": "EFIX <eye > <stime> <etime > <dur> <axp > <ayp> <aps >"},
    "SBLINK": {"Descr": "Start of Blink", "Pattern": "SBLINK <eye > <stime>"},
    "EBLINK": {"Descr": "End of Blink", "Pattern": "EBLINK <eye > <stime> <etime > <dur>"},
    "DISPLAY ON": {"Descr": "Actual start of Trial", "Pattern": "DISPLAY ON"},
}
metadata_strs = ["DISPLAY COORDS", "GAZE_COORDS", "FRAMERATE"]


POPEYE_FIXATION_COLS_DICT = {
    "start": "start_time",
    "stop": "end_time",
    "xs": "x",
    "ys": "y",
}
EMREADING_COLS_DROPLIST = ["hasText", "char_trial"]
EMREADING_COLS_DICT = {
    "sub": "subject",
    "item": "item",
    "condition": "condition",
    "SFIX": "start_time",
    "EFIX": "end_time",
    "xPos": "x",
    "yPos": "y",
    "fix_number": "fixation_number",
    "fix_dur": "duration",
    "wordID": "on_word_EM",
    "outOfBnds": "out_of_bounds",
    "outsideText": "out_of_text_area",
}


def download_url(url, target_filename):
    max_retries = 4
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url)
            if r.status_code != 200:
                ic(f"Download failed due to unsuccessful response from server: {r.status_code}")
                return -1
            open(target_filename, "wb").write(r.content)
            return 0

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
                ic(f"Download failed due to an error; will try again in {attempt*2} seconds:", e)
            else:
                ic(f"Failed after all attempts ({url}). Error details:\n{e}")
                return -1


def asc_to_trial_ids(
    asc_file, close_gap_between_words, paragraph_trials_only, ias_files, trial_start_keyword, end_trial_at_keyword
):
    asc_encoding = ["ISO-8859-15", "UTF-8"][0]
    trials_dict, lines = file_to_trials_and_lines(
        asc_file,
        asc_encoding,
        close_gap_between_words=close_gap_between_words,
        paragraph_trials_only=paragraph_trials_only,
        uploaded_ias_files=ias_files,
        trial_start_keyword=trial_start_keyword,
        end_trial_at_keyword=end_trial_at_keyword,
    )

    enum = (
        trials_dict["paragraph_trials"]
        if paragraph_trials_only and "paragraph_trials" in trials_dict.keys()
        else range(trials_dict["max_trial_idx"])
    )
    trials_by_ids = {trials_dict[idx]["trial_id"]: trials_dict[idx] for idx in enum}
    return trials_by_ids, lines, trials_dict


def get_trials_list(
    asc_file, close_gap_between_words, paragraph_trials_only, ias_files, trial_start_keyword, end_trial_at_keyword
):
    if hasattr(asc_file, "name"):
        savename = pl.Path(asc_file.name).stem
    else:
        savename = pl.Path(asc_file).stem

    trials_by_ids, lines, trials_dict = asc_to_trial_ids(
        asc_file,
        close_gap_between_words=close_gap_between_words,
        paragraph_trials_only=paragraph_trials_only,
        ias_files=ias_files,
        trial_start_keyword=trial_start_keyword,
        end_trial_at_keyword=end_trial_at_keyword,
    )
    trial_keys = list(trials_by_ids.keys())
    savename = RESULTS_FOLDER / f"{savename}_metadata_overview.json"

    offload_list = [
        "gaze_df",
        "dffix",
        "chars_df",
        "saccade_df",
        "x_char_unique",
        "line_heights",
        "chars_list",
        "words_list",
        "dffix_sacdf_popEye",
        "fixdf_popEye",
        "saccade_df",
        "sacdf_popEye",
        "combined_df",
        "events_df",
    ]
    trials_dict_cut_down = {}
    for k_outer, v_outer in trials_dict.items():
        if isinstance(v_outer, dict):
            trials_dict_cut_down[k_outer] = {}
            for prop, val in v_outer.items():
                if prop not in offload_list:
                    trials_dict_cut_down[k_outer][prop] = val
        else:
            trials_dict_cut_down[k_outer] = v_outer
    save_trial_to_json(trials_dict_cut_down, savename=savename)
    return trial_keys, trials_by_ids, lines, asc_file, trials_dict


def calc_xdiff_ydiff(line_xcoords_no_pad, line_ycoords_no_pad, line_heights, allow_multiple_values=False):
    x_diffs = np.unique(np.diff(line_xcoords_no_pad))
    if len(x_diffs) == 1:
        x_diff = x_diffs[0]
    elif not allow_multiple_values:
        x_diff = np.min(x_diffs)
    else:
        x_diff = x_diffs

    if np.unique(line_ycoords_no_pad).shape[0] == 1:
        return x_diff, line_heights[0]
    y_diffs = np.unique(np.diff(line_ycoords_no_pad))
    if len(y_diffs) == 1:
        y_diff = y_diffs[0]
    elif len(y_diffs) == 0:
        y_diff = 0
    elif not allow_multiple_values:
        y_diff = np.min(y_diffs)
    else:
        y_diff = y_diffs
    return np.round(x_diff, decimals=2), np.round(y_diff, decimals=2)


def add_words(chars_list):
    chars_list_reconstructed = []
    words_list = []
    sentence_list = []
    sentence_start_idx = 0
    sentence_num = 0
    word_start_idx = 0
    chars_df = pd.DataFrame(chars_list)
    chars_df["char_width"] = chars_df.char_xmax - chars_df.char_xmin
    word_dict = None
    on_line_num = -1
    line_change_on_next_char = False
    num_chars = len(chars_list)
    for idx, char_dict in enumerate(chars_list):
        # check if line change will happen after current char
        on_line_num = char_dict["assigned_line"]
        if idx < num_chars - 1:
            line_change_on_next_char = on_line_num != chars_list[idx + 1]["assigned_line"]
        else:
            line_change_on_next_char = False
        chars_list_reconstructed.append(char_dict)
        if char_dict["char"] in [" "] or len(chars_list_reconstructed) == len(chars_list) or line_change_on_next_char:
            word_xmin = chars_list_reconstructed[word_start_idx]["char_xmin"]
            if chars_list_reconstructed[-1]["char"] == " " and len(chars_list_reconstructed) != 1:
                word_xmax = chars_list_reconstructed[-2]["char_xmax"]

                word = "".join(
                    [
                        chars_list_reconstructed[idx]["char"]
                        for idx in range(word_start_idx, len(chars_list_reconstructed) - 1)
                    ]
                )
            elif len(chars_list_reconstructed) == 1:
                word_xmax = chars_list_reconstructed[-1]["char_xmax"]
                word = " "
            else:
                word = "".join(
                    [
                        chars_list_reconstructed[idx]["char"]
                        for idx in range(word_start_idx, len(chars_list_reconstructed))
                    ]
                )
                word_xmax = chars_list_reconstructed[-1]["char_xmax"]
            word_ymin = chars_list_reconstructed[word_start_idx]["char_ymin"]
            word_ymax = chars_list_reconstructed[word_start_idx]["char_ymax"]
            word_x_center = round((word_xmax - word_xmin) / 2 + word_xmin, ndigits=2)
            word_y_center = round((word_ymax - word_ymin) / 2 + word_ymin, ndigits=2)
            word_length = len(word)
            assigned_line = chars_list_reconstructed[word_start_idx]["assigned_line"]
            word_dict = dict(
                word_number=len(words_list),
                word=word,
                word_length=word_length,
                word_xmin=word_xmin,
                word_xmax=word_xmax,
                word_ymin=word_ymin,
                word_ymax=word_ymax,
                word_x_center=word_x_center,
                word_y_center=word_y_center,
                assigned_line=assigned_line,
            )
            if len(word) > 0 and word != " ":
                words_list.append(word_dict)
            for cidx, char_dict in enumerate(chars_list_reconstructed[word_start_idx:]):
                if char_dict["char"] == " ":
                    char_dict["in_word_number"] = len(words_list)
                    char_dict["in_word"] = " "
                    char_dict["num_letters_from_start_of_word"] = 0
                else:
                    char_dict["in_word_number"] = len(words_list) - 1
                    char_dict["in_word"] = word
                    char_dict["num_letters_from_start_of_word"] = cidx

            word_start_idx = idx + 1

        if chars_list_reconstructed[-1]["char"] in [".", "!", "?"] or idx == (len(chars_list) - 1):
            if idx != sentence_start_idx:
                chars_df_temp = pd.DataFrame(chars_list_reconstructed[sentence_start_idx:])
                line_texts = []
                for sidx, subdf in chars_df_temp.groupby("assigned_line"):
                    line_text = "_".join(subdf.char.values)
                    line_text = line_text.replace("_ _", " ")
                    line_text = line_text.replace("_", "")
                    line_texts.append(line_text.strip())
                sentence_text = " ".join(line_texts)
                sentence_dict = dict(sentence_num=sentence_num, sentence_text=sentence_text)
                sentence_list.append(sentence_dict)
                for c in chars_list_reconstructed[sentence_start_idx:]:
                    c["in_sentence_number"] = sentence_num
                    c["in_sentence"] = sentence_text
                sentence_start_idx = len(chars_list_reconstructed)
                sentence_num += 1
            else:
                sentence_list[-1]["sentence_text"] += chars_list_reconstructed[sentence_start_idx]["char"]
                chars_list_reconstructed[idx]["in_sentence_number"] = sentence_list[-1]["sentence_num"]
                chars_list_reconstructed[idx]["in_sentence"] = sentence_list[-1]["sentence_text"]
    for cidx, char_dict in enumerate(chars_list_reconstructed):
        if (
            char_dict["char"] == " "
            and (cidx + 1) < len(chars_list_reconstructed)
            and char_dict["assigned_line"] == chars_list_reconstructed[cidx + 1]["assigned_line"]
        ):
            char_dict["in_word_number"] = chars_list_reconstructed[cidx + 1]["in_word_number"]
            char_dict["in_word"] = chars_list_reconstructed[cidx + 1]["in_word"]

    last_letter_in_word = words_list[-1]["word"][-1]
    last_letter_in_chars_list_reconstructed = char_dict["char"]
    if last_letter_in_word != last_letter_in_chars_list_reconstructed:
        if last_letter_in_chars_list_reconstructed in [".", "!", "?"]:
            words_list[-1] = dict(
                word_number=len(words_list),
                word=words_list[-1]["word"] + char_dict["char"],
                word_length=len(words_list[-1]["word"] + char_dict["char"]),
                word_xmin=words_list[-1]["word_xmin"],
                word_xmax=char_dict["char_xmax"],
                word_ymin=words_list[-1]["word_ymin"],
                word_ymax=words_list[-1]["word_ymax"],
                assigned_line=assigned_line,
            )

            word_x_center = round(
                (words_list[-1]["word_xmax"] - words_list[-1]["word_xmin"]) / 2 + words_list[-1]["word_xmin"], ndigits=2
            )
            word_y_center = round(
                (words_list[-1]["word_ymax"] - word_dict["word_ymin"]) / 2 + words_list[-1]["word_ymin"], ndigits=2
            )
            words_list[-1]["word_x_center"] = word_x_center
            words_list[-1]["word_y_center"] = word_y_center
        else:
            word_dict = dict(
                word_number=len(words_list),
                word=char_dict["char"],
                word_length=1,
                word_xmin=char_dict["char_xmin"],
                word_xmax=char_dict["char_xmax"],
                word_ymin=char_dict["char_ymin"],
                word_ymax=char_dict["char_ymax"],
                word_x_center=char_dict["char_x_center"],
                word_y_center=char_dict["char_y_center"],
                assigned_line=assigned_line,
            )
            words_list.append(word_dict)
        chars_list_reconstructed[-1]["in_word_number"] = len(words_list) - 1
        chars_list_reconstructed[-1]["in_word"] = word_dict["word"]
        chars_list_reconstructed[-1]["num_letters_from_start_of_word"] = 0
        if len(sentence_list) > 0:
            chars_list_reconstructed[-1]["in_sentence_number"] = sentence_num - 1
            chars_list_reconstructed[-1]["in_sentence"] = sentence_list[-1]["sentence_text"]
        else:
            ic(f"Warning Sentence list empty: {sentence_list}")

    return words_list, chars_list_reconstructed


def read_ias_file(ias_file, prefix):

    if isinstance(ias_file, UploadedFile):
        lines = StringIO(ias_file.getvalue().decode("utf-8")).readlines()
        ias_dicts = []
        for l in lines:
            lsplit = l.strip().split("\t")
            ldict = {
                f"{prefix}_number": float(lsplit[1]),
                f"{prefix}_xmin": float(lsplit[2]),
                f"{prefix}_xmax": float(lsplit[4]),
                f"{prefix}_ymin": float(lsplit[3]),
                f"{prefix}_ymax": float(lsplit[5]),
                prefix: lsplit[6],
            }
            ias_dicts.append(ldict)
        ias_df = pd.DataFrame(ias_dicts)
    else:
        ias_df = pd.read_csv(ias_file, delimiter="\t", header=None)
        ias_df = ias_df.rename(
            {
                1: f"{prefix}_number",
                2: f"{prefix}_xmin",
                4: f"{prefix}_xmax",
                3: f"{prefix}_ymin",
                5: f"{prefix}_ymax",
                6: prefix,
            },
            axis=1,
        )
    first_line_df = ias_df[ias_df[f"{prefix}_ymin"] == ias_df.loc[0, f"{prefix}_ymin"]]
    words_include_spaces = (
        first_line_df[f"{prefix}_xmax"].values == first_line_df[f"{prefix}_xmin"].shift(-1).values
    ).any()
    ias_df[f"{prefix}_width"] = ias_df[f"{prefix}_xmax"] - ias_df[f"{prefix}_xmin"]
    if words_include_spaces:
        ias_df[f"{prefix}_length"] = ias_df[prefix].map(lambda x: len(x) + 1)
        ias_df[f"{prefix}_width_per_length"] = ias_df[f"{prefix}_width"] / ias_df[f"{prefix}_length"]
        ias_df[f"{prefix}_xmax"] = (ias_df[f"{prefix}_xmax"] - ias_df[f"{prefix}_width_per_length"]).round(2)

    ias_df[f"{prefix}_x_center"] = (
        (ias_df[f"{prefix}_xmax"] - ias_df[f"{prefix}_xmin"]) / 2 + ias_df[f"{prefix}_xmin"]
    ).round(2)
    ias_df[f"{prefix}_y_center"] = (
        (ias_df[f"{prefix}_ymax"] - ias_df[f"{prefix}_ymin"]) / 2 + ias_df[f"{prefix}_ymin"]
    ).round(2)
    unique_midlines = list(np.unique(ias_df[f"{prefix}_y_center"]))
    assigned_lines = [unique_midlines.index(x) for x in ias_df[f"{prefix}_y_center"]]
    ias_df["assigned_line"] = assigned_lines
    ias_df[f"{prefix}_number"] = np.arange(ias_df.shape[0])
    return ias_df


def get_chars_list_from_words_list(ias_df, prefix="word"):
    ias_df.reset_index(inplace=True, drop=True)
    unique_midlines = list(np.unique(ias_df[f"{prefix}_y_center"]))
    chars_list = []
    for (idx, row), (next_idx, next_row) in zip(ias_df.iterrows(), ias_df.shift(-1).iterrows()):
        word = str(row[prefix])
        letter_width = (row[f"{prefix}_xmax"] - row[f"{prefix}_xmin"]) / len(word)
        for i_w, letter in enumerate(word):
            char_dict = dict(
                in_word_number=idx,
                in_word=word,
                char_xmin=round(row[f"{prefix}_xmin"] + i_w * letter_width, 2),
                char_xmax=round(row[f"{prefix}_xmin"] + (i_w + 1) * letter_width, 2),
                char_ymin=row[f"{prefix}_ymin"],
                char_ymax=row[f"{prefix}_ymax"],
                char=letter,
            )

            char_dict["char_x_center"] = round(
                (char_dict["char_xmax"] - char_dict["char_xmin"]) / 2 + char_dict["char_xmin"], ndigits=2
            )
            char_dict["char_y_center"] = round(
                (row[f"{prefix}_ymax"] - row[f"{prefix}_ymin"]) / 2 + row[f"{prefix}_ymin"], ndigits=2
            )

            if i_w >= len(word) + 1:
                break
            char_dict["assigned_line"] = unique_midlines.index(char_dict["char_y_center"])
            chars_list.append(char_dict)
        if chars_list[-1]["char"] != " " and row.assigned_line == next_row.assigned_line:
            char_dict = dict(
                char_xmin=chars_list[-1]["char_xmax"],
                char_xmax=round(chars_list[-1]["char_xmax"] + letter_width, 2),
                char_ymin=row[f"{prefix}_ymin"],
                char_ymax=row[f"{prefix}_ymax"],
                char=" ",
            )

            char_dict["char_x_center"] = round(
                (char_dict["char_xmax"] - char_dict["char_xmin"]) / 2 + char_dict["char_xmin"], ndigits=2
            )
            char_dict["char_y_center"] = round(
                (row[f"{prefix}_ymax"] - row[f"{prefix}_ymin"]) / 2 + row[f"{prefix}_ymin"], ndigits=2
            )

            char_dict["assigned_line"] = unique_midlines.index(char_dict["char_y_center"])
            chars_list.append(char_dict)
    chars_df = pd.DataFrame(chars_list)
    chars_df.loc[:, ["in_word_number", "in_word"]] = chars_df.loc[:, ["in_word_number", "in_word"]].copy().ffill(axis=0)
    return chars_df.to_dict("records")


def check_values(v1, v2):
    """Function that compares two lists for equality.

    Returns True if both lists are the same; False if they are not; and None if either is None."""

    # Check if any of the lists is None
    if v1 is None or v2 is None or pd.isna(v1) or pd.isna(v2):
        return None

    # Compare elements in v1 with corresponding elements in v2
    if v1 != v2:
        return False
    if v1 != v2:
        return False
    return True


def asc_lines_to_trials_by_trail_id(
    lines: list,
    paragraph_trials_only=True,
    filename: str = "",
    close_gap_between_words=True,
    ias_files=[],
    start_trial_at_keyword="START",
    end_trial_at_keyword="END",
) -> dict:

    if len(ias_files) > 0:
        ias_files_dict = {pl.Path(f.name).stem: f for f in ias_files}
    else:
        ias_files_dict = {}
    if hasattr(filename, "name"):
        filename = filename.name
    subject = pl.Path(filename).stem
    y_px = []
    x_px = []
    calibration_offset = []
    calibration_max_error = []
    calibration_time = []
    calibration_avg_error = []
    trial_var_block_lines = None
    question_answer = None
    question_correct = None
    condition = "UNKNOWN"
    item = "UNKNOWN"
    depend = "UNKNOWN"
    trial_index = None
    fps = None
    display_coords = None
    trial_var_block_idx = -1
    trials_dict = dict(paragraph_trials=[], paragraph_trial_IDs=[])
    trial_idx = -1
    trial_var_block_start_idx = -1
    removed_trial_ids = []
    ias_file = ""
    trial_var_block_lines_list = []
    if "\n".join(map(str.strip, lines)).find("TRIAL_VAR") != -1:
        for idx, l in enumerate(tqdm(lines, desc=f"Checking for TRIAL_VAR lines for {filename}")):
            if trial_var_block_start_idx == -1 and "MSG" not in l:
                continue
            if "TRIAL_VAR" in l:
                if trial_var_block_start_idx == -1:
                    trial_var_block_start_idx = idx
                continue
            else:
                if trial_var_block_start_idx != -1:
                    trial_var_block_stop_idx = idx
                    trial_var_block_lines = [
                        x.strip() for x in lines[trial_var_block_start_idx:trial_var_block_stop_idx]
                    ]
                    trial_var_block_lines_list.append(trial_var_block_lines)
                trial_var_block_start_idx = -1
        has_trial_var_lines = len(trial_var_block_lines_list) > 0
    else:
        has_trial_var_lines = False

    for idx, l in enumerate(lines):
        if "MSG" not in l:
            continue
        parts = l.strip().split(" ")
        if "TRIALID" in l:
            trial_id = re.split(r"[ :\t]+", l.strip())[-1]
            trial_id_timestamp = parts[1]
            trial_idx += 1
            if trial_id[0] in ["F", "P", "E"]:

                parse_dict = emf.parse_itemID(trial_id)
                condition = parse_dict["condition"]
                item = parse_dict["item"]
                depend = parse_dict["depend"]
            else:
                parse_dict = {}
            if trial_id[0] == "F":
                trial_is = "question"
            elif trial_id[0] == "P":
                trial_is = "practice"
            else:
                if has_trial_var_lines:
                    trial_var_block_idx += 1
                    trial_var_block_lines = trial_var_block_lines_list[trial_var_block_idx]
                    image_lines = [s for s in trial_var_block_lines if "img" in s]
                    if len(image_lines) > 0:
                        item = image_lines[0].split(" ")[-1]
                    cond_lines = [s for s in trial_var_block_lines if "cond" in s]
                    if len(cond_lines) > 0:
                        condition = cond_lines[0].split(" ")[-1]
                    item_lines = [s for s in trial_var_block_lines if "item" in s]
                    if len(item_lines) > 0:
                        item = item_lines[0].split(" ")[-1]
                    trial_index_lines = [s for s in trial_var_block_lines if "Trial_Index" in s]
                    if len(trial_index_lines) > 0:
                        trial_index = trial_index_lines[0].split(" ")[-1]
                    question_key_lines = [s for s in trial_var_block_lines if "QUESTION_KEY_PRESSED" in s]
                    if len(question_key_lines) > 0:
                        question_answer = question_key_lines[0].split(" ")[-1]
                    question_response_lines = [s for s in trial_var_block_lines if " RESPONSE" in s]
                    if len(question_response_lines) > 0:
                        question_answer = question_response_lines[0].split(" ")[-1]
                    question_correct_lines = [
                        s for s in trial_var_block_lines if ("QUESTION_ACCURACY" in s) | (" ACCURACY" in s)
                    ]
                    if len(question_correct_lines) > 0:
                        question_correct = question_correct_lines[0].split(" ")[-1]
                    trial_is_lines = [s for s in trial_var_block_lines if "trial" in s]
                    if len(trial_is_lines) > 0:
                        trial_is_line = trial_is_lines[0].split(" ")[-1]
                        if "pract" in trial_is_line or "end" in trial_is_line:
                            trial_is = "practice"
                            trial_id = f"{trial_is}_{trial_id}"
                        else:
                            trial_is = "paragraph"
                            trial_id = f"{condition}_{trial_is}_{trial_id}"
                            trials_dict["paragraph_trials"].append(trial_idx)
                            trials_dict["paragraph_trial_IDs"].append(trial_id)
                    else:
                        trial_is = "paragraph"
                        trial_id = f"{condition}_{trial_is}_{trial_id}_{trial_idx}"
                        trials_dict["paragraph_trials"].append(trial_idx)
                        trials_dict["paragraph_trial_IDs"].append(trial_id)
                else:
                    if len(trial_id) > 1:
                        condition = trial_id[1]
                    trial_is = "paragraph"
                    trials_dict["paragraph_trials"].append(trial_idx)
                    trials_dict["paragraph_trial_IDs"].append(trial_id)
            trials_dict[trial_idx] = dict(
                subject=subject,
                filename=filename,
                trial_idx=trial_idx,
                trial_id=trial_id,
                trial_id_idx=idx,
                trial_id_timestamp=trial_id_timestamp,
                trial_is=trial_is,
                trial_var_block_lines=trial_var_block_lines,
                seq=trial_idx,
                item=item,
                depend=depend,
                condition=condition,
                parse_dict=parse_dict,
            )
            if question_answer is not None:
                trials_dict[trial_idx]["question_answer"] = question_answer
            if question_correct is not None:
                trials_dict[trial_idx]["question_correct"] = question_correct
            if trial_index is not None:
                trials_dict[trial_idx]["trial_index"] = trial_index
            last_trial_skipped = False

        elif "TRIAL_RESULT" in l or "stop_trial" in l:
            trials_dict[trial_idx]["trial_result_idx"] = idx
            trials_dict[trial_idx]["trial_result_timestamp"] = int(parts[0].split("\t")[1])
            if len(parts) > 2:
                trials_dict[trial_idx]["trial_result_number"] = int(parts[2])
        elif "QUESTION_ANSWER" in l and not has_trial_var_lines:
            trials_dict[trial_idx]["question_answer_idx"] = idx
            trials_dict[trial_idx]["question_answer_timestamp"] = int(parts[0].split("\t")[1])
            if len(parts) > 2:
                trials_dict[trial_idx]["question_answer_question_trial"] = int(
                    pd.to_numeric(l.strip().split(" ")[-1].strip(), errors="coerce")
                )
        elif "KEYBOARD" in l:
            trials_dict[trial_idx]["keyboard_press_idx"] = idx
            trials_dict[trial_idx]["keyboard_press_timestamp"] = int(parts[0].split("\t")[1])
        elif "DISPLAY COORDS" in l and display_coords is None:
            display_coords = (float(parts[-4]), float(parts[-3]), float(parts[-2]), float(parts[-1]))
        elif "GAZE_COORDS" in l and display_coords is None:
            display_coords = (float(parts[-4]), float(parts[-3]), float(parts[-2]), float(parts[-1]))
        elif "FRAMERATE" in l:
            l_idx = parts.index(metadata_strs[2])
            fps = float(parts[l_idx + 1])
        elif "TRIAL ABORTED" in l or "TRIAL REPEATED" in l:
            if not last_trial_skipped:
                if trial_is == "paragraph":
                    trials_dict["paragraph_trials"].remove(trial_idx)
                trial_idx -= 1
                removed_trial_ids.append(trial_id)
                last_trial_skipped = True
        elif "IAREA FILE" in l:
            ias_file = parts[-1]
            ias_file_stem = ias_file.split("/")[-1].split("\\")[-1].split(".")[0]
            trials_dict[trial_idx]["ias_file_from_asc"] = ias_file
            trials_dict[trial_idx]["ias_file"] = ias_file_stem
            if item == "UNKNOWN":
                trials_dict[trial_idx]["item"] = ias_file_stem
            if ias_file_stem in ias_files_dict:
                try:
                    ias_file = ias_files_dict[ias_file_stem]
                    ias_df = read_ias_file(ias_file, prefix="word")  # TODO make option if word or chars in ias
                    trials_dict[trial_idx]["words_list"] = ias_df.to_dict("records")
                    trials_dict[trial_idx]["chars_list"] = get_chars_list_from_words_list(ias_df, prefix="word")
                except Exception as e:
                    ic(f"Reading ias file failed")
                    ic(e)
            else:
                ic(f"IAS file {ias_file_stem} not found")
        elif "CALIBRATION" in l and "MSG" in l:
            calibration_method = parts[3].strip()
            if trial_idx > -1:
                trials_dict[trial_idx]["calibration_method"] = calibration_method
        elif "VALIDATION" in l and "MSG" in l and "ABORTED" not in l:
            try:
                calibration_time_line_parts = re.split(r"[ :\t]+", l.strip())
                calibration_time.append(float(calibration_time_line_parts[1]))
                calibration_avg_error.append(float(calibration_time_line_parts[9]))
                calibration_max_error.append(float(calibration_time_line_parts[11]))
                calibration_offset.append(float(calibration_time_line_parts[14]))
                x_px.append(float(calibration_time_line_parts[-2].split(",")[0]))
                y_px.append(float(calibration_time_line_parts[-2].split(",")[1]))
            except Exception as e:
                ic(f"parsing VALIDATION failed for line {l}")
    trials_df = pd.DataFrame([trials_dict[i] for i in range(trial_idx) if i in trials_dict])

    if (
        question_correct is None
        and "trial_result_number" in trials_df.columns
        and "question_answer_question_trial" in trials_df.columns
    ):
        trials_df["question_answer_selection"] = trials_df["trial_result_number"].shift(-1).values
        trials_df["correct_trial_answer_would_be"] = trials_df["question_answer_question_trial"].shift(-1).values
        trials_df["question_correct"] = [
            check_values(a, b)
            for a, b in zip(trials_df["question_answer_selection"], trials_df["correct_trial_answer_would_be"])
        ]
        for pidx, prow in trials_df.loc[trials_df.trial_is == "paragraph", :].iterrows():
            trials_dict[pidx]["question_correct"] = prow["question_correct"]
            if prow["question_correct"] is not None:
                trials_dict[pidx]["question_answer_selection"] = prow["question_answer_selection"]
                trials_dict[pidx]["correct_trial_answer_would_be"] = prow["correct_trial_answer_would_be"]
            else:
                trials_dict[pidx]["question_answer_selection"] = None
                trials_dict[pidx]["correct_trial_answer_would_be"] = None
    if "question_correct" in trials_df.columns:
        paragraph_trials_df = trials_df.loc[trials_df.trial_is == "paragraph", :]
        overall_question_answer_value_counts = (
            paragraph_trials_df["question_correct"].dropna().astype(int).value_counts().to_dict()
        )
        overall_question_answer_value_counts_normed = (
            paragraph_trials_df["question_correct"].dropna().astype(int).value_counts(normalize=True).to_dict()
        )
    else:
        overall_question_answer_value_counts = None
        overall_question_answer_value_counts_normed = None
    if paragraph_trials_only:
        trials_dict_temp = trials_dict.copy()
        for k in trials_dict_temp.keys():
            if k not in ["paragraph_trials"] + trials_dict_temp["paragraph_trials"]:
                trials_dict.pop(k)
        if len(trials_dict_temp["paragraph_trials"]):
            trial_idx = trials_dict_temp["paragraph_trials"][-1]
        else:
            return trials_dict
    trials_dict["display_coords"] = display_coords
    trials_dict["fps"] = fps
    trials_dict["max_trial_idx"] = trial_idx
    trials_dict["overall_question_answer_value_counts"] = overall_question_answer_value_counts
    trials_dict["overall_question_answer_value_counts_normed"] = overall_question_answer_value_counts_normed
    enum = (
        trials_dict["paragraph_trials"]
        if ("paragraph_trials" in trials_dict.keys() and paragraph_trials_only)
        else range(len(trials_dict))
    )
    for trial_idx in enum:
        if trial_idx not in trials_dict.keys():
            continue
        if "chars_list" in trials_dict[trial_idx]:
            chars_list = trials_dict[trial_idx]["chars_list"]
        else:
            chars_list = []
        if "display_coords" not in trials_dict[trial_idx].keys():
            trials_dict[trial_idx]["display_coords"] = trials_dict["display_coords"]
        trials_dict[trial_idx]["overall_question_answer_value_counts"] = trials_dict[
            "overall_question_answer_value_counts"
        ]
        trials_dict[trial_idx]["overall_question_answer_value_counts_normed"] = trials_dict[
            "overall_question_answer_value_counts_normed"
        ]
        trial_start_idx = trials_dict[trial_idx]["trial_id_idx"]
        trial_end_idx = trials_dict[trial_idx]["trial_result_idx"]
        trial_lines = lines[trial_start_idx:trial_end_idx]
        if len(y_px) > 0:
            trials_dict[trial_idx]["y_px"] = y_px
            trials_dict[trial_idx]["x_px"] = x_px
            if "calibration_method" not in trials_dict[trial_idx]:
                trials_dict[trial_idx]["calibration_method"] = calibration_method
            trials_dict[trial_idx]["calibration_offset"] = calibration_offset
            trials_dict[trial_idx]["calibration_max_error"] = calibration_max_error
            trials_dict[trial_idx]["calibration_time"] = calibration_time
            trials_dict[trial_idx]["calibration_avg_error"] = calibration_avg_error
        for idx, l in enumerate(trial_lines):
            parts = l.strip().split(" ")
            if "START" in l and " MSG" not in l:
                trials_dict[trial_idx]["text_end_idx"] = trial_start_idx + idx
                trials_dict[trial_idx]["start_idx"] = trial_start_idx + idx + 7
                trials_dict[trial_idx]["start_time"] = int(parts[0].split("\t")[1])
            elif "END" in l and "ENDBUTTON" not in l and " MSG" not in l:
                trials_dict[trial_idx]["end_idx"] = trial_start_idx + idx - 2
                trials_dict[trial_idx]["end_time"] = int(parts[0].split("\t")[1])
            elif "MSG" not in l:
                continue
            elif "ENDBUTTON" in l:
                trials_dict[trial_idx]["endbutton_idx"] = trial_start_idx + idx
                trials_dict[trial_idx]["endbutton_time"] = int(parts[0].split("\t")[1])
            elif "SYNCTIME" in l:
                trials_dict[trial_idx]["synctime"] = trial_start_idx + idx
                trials_dict[trial_idx]["synctime_time"] = int(parts[0].split("\t")[1])
            elif start_trial_at_keyword in l:
                trials_dict[trial_idx][f"{start_trial_at_keyword}_line_idx"] = trial_start_idx + idx
                trials_dict[trial_idx][f"{start_trial_at_keyword}_time"] = int(parts[0].split("\t")[1])
            elif "GAZE TARGET OFF" in l:
                trials_dict[trial_idx]["gaze_targ_off_time"] = int(parts[0].split("\t")[1])
            elif "GAZE TARGET ON" in l:
                trials_dict[trial_idx]["gaze_targ_on_time"] = int(parts[0].split("\t")[1])
                trials_dict[trial_idx]["gaze_targ_on_time_idx"] = trial_start_idx + idx
            elif "DISPLAY_SENTENCE" in l:  # some .asc files seem to use this
                trials_dict[trial_idx]["gaze_targ_on_time"] = int(parts[0].split("\t")[1])
                trials_dict[trial_idx]["gaze_targ_on_time_idx"] = trial_start_idx + idx
            elif "DISPLAY TEXT" in l:
                trials_dict[trial_idx]["text_start_idx"] = trial_start_idx + idx
            elif "REGION CHAR" in l:
                rg_idx = parts.index("CHAR")
                if len(parts[rg_idx:]) > 8:
                    char = " "
                    idx_correction = 1
                elif len(parts[rg_idx:]) == 3:
                    char = " "
                    if "REGION CHAR" not in trial_lines[idx + 1]:
                        parts = trial_lines[idx + 1].strip().split(" ")
                        idx_correction = -rg_idx - 4
                else:
                    char = parts[rg_idx + 3]
                    idx_correction = 0
                try:
                    char_dict = {
                        "char": char,
                        "char_xmin": float(parts[rg_idx + 4 + idx_correction]),
                        "char_ymin": float(parts[rg_idx + 5 + idx_correction]),
                        "char_xmax": float(parts[rg_idx + 6 + idx_correction]),
                        "char_ymax": float(parts[rg_idx + 7 + idx_correction]),
                    }
                    char_dict["char_y_center"] = round(
                        (char_dict["char_ymax"] - char_dict["char_ymin"]) / 2 + char_dict["char_ymin"], ndigits=2
                    )
                    char_dict["char_x_center"] = round(
                        (char_dict["char_xmax"] - char_dict["char_xmin"]) / 2 + char_dict["char_xmin"], ndigits=2
                    )
                    chars_list.append(char_dict)
                except Exception as e:
                    ic(f"char_dict creation failed for parts {parts}")
                    ic(e)

        if start_trial_at_keyword == "SYNCTIME" and "synctime_time" in trials_dict[trial_idx]:
            trials_dict[trial_idx]["trial_start_time"] = trials_dict[trial_idx]["synctime_time"]
            trials_dict[trial_idx]["trial_start_idx"] = trials_dict[trial_idx]["synctime"]
        elif start_trial_at_keyword == "GAZE TARGET ON" and "gaze_targ_on_time" in trials_dict[trial_idx]:
            trials_dict[trial_idx]["trial_start_time"] = trials_dict[trial_idx]["gaze_targ_on_time"]
            trials_dict[trial_idx]["trial_start_idx"] = trials_dict[trial_idx]["gaze_targ_on_time_idx"]
        elif start_trial_at_keyword == "START":
            trials_dict[trial_idx]["trial_start_time"] = trials_dict[trial_idx]["start_time"]
            trials_dict[trial_idx]["trial_start_idx"] = trials_dict[trial_idx]["start_idx"]
        elif f"{start_trial_at_keyword}_time" in trials_dict[trial_idx]:
            trials_dict[trial_idx]["trial_start_time"] = trials_dict[trial_idx][f"{start_trial_at_keyword}_time"]
            trials_dict[trial_idx]["trial_start_idx"] = trials_dict[trial_idx][f"{start_trial_at_keyword}_line_idx"]
        else:
            trials_dict[trial_idx]["trial_start_time"] = trials_dict[trial_idx]["start_time"]
            trials_dict[trial_idx]["trial_start_idx"] = trials_dict[trial_idx]["start_idx"]
        if end_trial_at_keyword == "ENDBUTTON" and "endbutton_time" in trials_dict[trial_idx]:
            trials_dict[trial_idx]["trial_end_time"] = trials_dict[trial_idx]["endbutton_time"]
            trials_dict[trial_idx]["trial_end_idx"] = trials_dict[trial_idx]["endbutton_idx"]
        elif end_trial_at_keyword == "END" and "end_idx" in trials_dict[trial_idx]:
            trials_dict[trial_idx]["trial_end_time"] = trials_dict[trial_idx]["end_time"]
            trials_dict[trial_idx]["trial_end_idx"] = trials_dict[trial_idx]["end_idx"]
        elif end_trial_at_keyword == "KEYBOARD" and "keyboard_press_idx" in trials_dict[trial_idx]:
            trials_dict[trial_idx]["trial_end_idx"] = trials_dict[trial_idx]["keyboard_press_idx"]
        else:
            trials_dict[trial_idx]["trial_end_idx"] = trials_dict[trial_idx]["trial_result_idx"]
        if trials_dict[trial_idx]["trial_end_idx"] < trials_dict[trial_idx]["trial_start_idx"]:
            raise ValueError(f"trial_start_idx is larger than trial_end_idx for trial_idx {trial_idx}")
        if len(chars_list) > 0:
            line_ycoords = []
            for idx in range(len(chars_list)):
                chars_list[idx]["char_y_center"] = round(
                    (chars_list[idx]["char_ymax"] - chars_list[idx]["char_ymin"]) / 2 + chars_list[idx]["char_ymin"],
                    ndigits=2,
                )
                if chars_list[idx]["char_y_center"] not in line_ycoords:
                    line_ycoords.append(chars_list[idx]["char_y_center"])
            for idx in range(len(chars_list)):
                chars_list[idx]["assigned_line"] = line_ycoords.index(chars_list[idx]["char_y_center"])

            letter_width_avg = np.mean(
                [x["char_xmax"] - x["char_xmin"] for x in chars_list if x["char_xmax"] > x["char_xmin"]]
            )
            line_heights = [round(abs(x["char_ymax"] - x["char_ymin"]), 3) for x in chars_list]
            line_xcoords_all = [x["char_x_center"] for x in chars_list]
            line_xcoords_no_pad = np.unique(line_xcoords_all)

            line_ycoords_all = [x["char_y_center"] for x in chars_list]
            line_ycoords_no_pad = np.unique(line_ycoords_all)

            trials_dict[trial_idx]["x_char_unique"] = list(line_xcoords_no_pad)
            trials_dict[trial_idx]["y_char_unique"] = list(line_ycoords_no_pad)
            x_diff, y_diff = calc_xdiff_ydiff(
                line_xcoords_no_pad, line_ycoords_no_pad, line_heights, allow_multiple_values=False
            )
            trials_dict[trial_idx]["x_diff"] = float(x_diff)
            trials_dict[trial_idx]["y_diff"] = float(y_diff)
            trials_dict[trial_idx]["num_char_lines"] = len(line_ycoords_no_pad)
            trials_dict[trial_idx]["letter_width_avg"] = letter_width_avg
            trials_dict[trial_idx]["line_heights"] = line_heights
            words_list_from_func, chars_list_reconstructed = add_words(chars_list)
            words_list = words_list_from_func

            if close_gap_between_words:  # TODO this may need to change the "in_word" col for the chars_df
                for widx in range(1, len(words_list)):
                    if words_list[widx]["assigned_line"] == words_list[widx - 1]["assigned_line"]:
                        word_sep_half_width = (words_list[widx]["word_xmin"] - words_list[widx - 1]["word_xmax"]) / 2
                        words_list[widx - 1]["word_xmax"] = words_list[widx - 1]["word_xmax"] + word_sep_half_width
                        words_list[widx]["word_xmin"] = words_list[widx]["word_xmin"] - word_sep_half_width
            else:
                chars_df = pd.DataFrame(chars_list_reconstructed)
                chars_df.loc[
                    chars_df["char"] == " ", ["in_word", "in_word_number", "num_letters_from_start_of_word"]
                ] = pd.NA
                chars_list_reconstructed = chars_df.to_dict("records")
            trials_dict[trial_idx]["words_list"] = words_list
            trials_dict[trial_idx]["chars_list"] = chars_list_reconstructed
    return trials_dict


def get_lines_from_file(uploaded_file, asc_encoding="ISO-8859-15"):
    if isinstance(uploaded_file, str) or isinstance(uploaded_file, pl.Path):
        with open(uploaded_file, "r", encoding=asc_encoding) as f:
            lines = f.readlines()
    else:
        stringio = StringIO(uploaded_file.getvalue().decode(asc_encoding))
        loaded_str = stringio.read()
        lines = loaded_str.split("\n")
    return lines


def file_to_trials_and_lines(
    uploaded_file,
    asc_encoding: str = "ISO-8859-15",
    close_gap_between_words=True,
    paragraph_trials_only=True,
    uploaded_ias_files=[],
    trial_start_keyword="START",
    end_trial_at_keyword="END",
):
    lines = get_lines_from_file(uploaded_file, asc_encoding=asc_encoding)
    trials_dict = asc_lines_to_trials_by_trail_id(
        lines,
        paragraph_trials_only,
        uploaded_file,
        close_gap_between_words=close_gap_between_words,
        ias_files=uploaded_ias_files,
        start_trial_at_keyword=trial_start_keyword,
        end_trial_at_keyword=end_trial_at_keyword,
    )

    if "paragraph_trials" not in trials_dict.keys() and "trial_is" in trials_dict[0].keys():
        paragraph_trials = []
        for k in range(trials_dict["max_trial_idx"]):
            if trials_dict[k]["trial_is"] == "paragraph":
                paragraph_trials.append(k)
        trials_dict["paragraph_trials"] = paragraph_trials

    enum = (
        trials_dict["paragraph_trials"]
        if paragraph_trials_only and "paragraph_trials" in trials_dict.keys()
        else range(trials_dict["max_trial_idx"])
    )
    for k in enum:
        if "chars_list" in trials_dict[k].keys():
            max_line = trials_dict[k]["chars_list"][-1]["assigned_line"]
            words_on_lines = {x: [] for x in range(max_line + 1)}
            [words_on_lines[x["assigned_line"]].append(x["char"]) for x in trials_dict[k]["chars_list"]]
            line_list = ["".join([s for s in v]) for idx, v in words_on_lines.items()]
            sentences_temp = "".join([x["char"] for x in trials_dict[k]["chars_list"]])
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?)", sentences_temp)
            text = "\n".join([x for x in line_list])
            trials_dict[k]["sentence_list"] = [s for s in sentences if len(s) > 0]
            trials_dict[k]["line_list"] = line_list
            trials_dict[k]["text"] = text
            trials_dict[k]["max_line"] = max_line

    return trials_dict, lines


def discard_empty_str_from_list(l):
    return [x for x in l if len(x) > 0]


def make_folders(gradio_temp_folder, gradio_temp_unzipped_folder, PLOTS_FOLDER):
    gradio_temp_folder.mkdir(exist_ok=True)
    gradio_temp_unzipped_folder.mkdir(exist_ok=True)
    PLOTS_FOLDER.mkdir(exist_ok=True)
    return 0


def plotly_plot_with_image(
    dffix,
    trial,
    algo_choice,
    saccade_df=None,
    to_plot_list=["Uncorrected Fixations", "Corrected Fixations", "Word boxes"],
    lines_in_plot="Uncorrected",
    scale_factor=0.5,
    font="DejaVu Sans Mono",
    box_annotations: list = None,
):
    mpl_fig, img_width, img_height = matplotlib_plot_df(
        dffix,
        trial,
        algo_choice,
        None,
        desired_dpi=300,
        fix_to_plot=[],
        stim_info_to_plot=to_plot_list,
        font=font,
        box_annotations=box_annotations,
    )
    mpl_fig.savefig(TEMP_FIGURE_STIMULUS_PATH)
    plt.close(mpl_fig)
    if lines_in_plot == "Uncorrected":
        uncorrected_plot_mode = "markers+lines+text"
    else:
        uncorrected_plot_mode = "markers+text"

    if lines_in_plot == "Corrected":
        corrected_plot_mode = "markers+lines+text"
    else:
        corrected_plot_mode = "markers+text"

    if lines_in_plot == "Both":
        uncorrected_plot_mode = "markers+lines+text"
        corrected_plot_mode = "markers+lines+text"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[img_height * scale_factor, 0],
            mode="markers",
            marker_opacity=0,
            name="scale_helper",
        )
    )

    fig.update_xaxes(visible=False, range=[0, img_width * scale_factor])

    fig.update_yaxes(
        visible=False,
        range=[img_height * scale_factor, 0],
        scaleanchor="x",
    )
    if (
        "Words" in to_plot_list
        or "Word boxes" in to_plot_list
        or "Character boxes" in to_plot_list
        or "Characters" in to_plot_list
    ):
        imsource = Image.open(str(TEMP_FIGURE_STIMULUS_PATH))
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=0,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=imsource,
            )
        )

    duration_scaled = dffix.duration - dffix.duration.min()
    duration_scaled = ((duration_scaled / duration_scaled.max()) - 0.5) * 3
    duration = sigmoid(duration_scaled) * 50 * scale_factor
    if "Uncorrected Fixations" in to_plot_list:
        fig.add_trace(
            go.Scatter(
                x=dffix.x * scale_factor,
                y=dffix.y * scale_factor,
                mode=uncorrected_plot_mode,
                name="Raw fixations",
                marker=dict(
                    color=COLORS[-1],
                    symbol="arrow",
                    size=duration.values,
                    angleref="previous",
                ),
                line=dict(color=COLORS[-1], width=2 * scale_factor),
                text=np.arange(dffix.shape[0]),
                textposition="top right",
                textfont=dict(
                    family="sans serif",
                    size=23 * scale_factor,
                    color=COLORS[-1],
                ),
                hovertext=[f"x:{x}, y:{y}, n:{num}" for x, y, num in zip(dffix.x, dffix[f"y"], range(dffix.shape[0]))],
                opacity=0.9,
            )
        )

    if "Corrected Fixations" in to_plot_list:
        if isinstance(algo_choice, list):
            algo_choices = algo_choice
            repeats = range(len(algo_choice))
        else:
            algo_choices = [algo_choice]
            repeats = range(1)
        for algoIdx in repeats:
            algo_choice = algo_choices[algoIdx]
            if f"y_{algo_choice}" in dffix.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dffix.x * scale_factor,
                        y=dffix.loc[:, f"y_{algo_choice}"] * scale_factor,
                        mode=corrected_plot_mode,
                        name=algo_choice,
                        marker=dict(
                            color=COLORS[algoIdx],
                            symbol="arrow",
                            size=duration.values,
                            angleref="previous",
                        ),
                        line=dict(color=COLORS[algoIdx], width=1.5 * scale_factor),
                        text=np.arange(dffix.shape[0]),
                        textposition="top center",
                        textfont=dict(
                            family="sans serif",
                            size=22 * scale_factor,
                            color=COLORS[algoIdx],
                        ),
                        hovertext=[
                            f"x:{x}, y:{y}, n:{num}"
                            for x, y, num in zip(dffix.x, dffix[f"y_{algo_choice}"], range(dffix.shape[0]))
                        ],
                        opacity=0.9,
                    )
                )
    if "Saccades" in to_plot_list:

        duration_scaled = saccade_df.duration - saccade_df.duration.min()
        duration_scaled = ((duration_scaled / duration_scaled.max()) - 0.5) * 3
        duration = sigmoid(duration_scaled) * 65 * scale_factor
        starting_coordinates = [tuple(row * scale_factor) for row in saccade_df.loc[:, ["xs", "ys"]].values]
        ending_coordinates = [tuple(row * scale_factor) for row in saccade_df.loc[:, ["xe", "ye"]].values]
        for sidx, (start, end) in enumerate(zip(starting_coordinates, ending_coordinates)):
            if sidx == 0:
                show_legend = True
            else:
                show_legend = False

            fig.add_trace(
                go.Scatter(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    mode="markers+lines+text",
                    line=dict(color=COLORS[-1], width=1.5 * scale_factor, dash="dash"),
                    showlegend=show_legend,
                    legendgroup="1",
                    name="Saccades",
                    text=sidx,
                    textposition="top center",
                    textfont=dict(family="sans serif", size=22 * scale_factor, color=COLORS[-1]),
                    marker=dict(
                        color=COLORS[-1],
                        symbol="arrow",
                        size=duration.values,
                        angleref="previous",
                    ),
                )
            )
    if "Saccades snapped to line" in to_plot_list:

        duration_scaled = saccade_df.duration - saccade_df.duration.min()
        duration_scaled = ((duration_scaled / duration_scaled.max()) - 0.5) * 3
        duration = sigmoid(duration_scaled) * 65 * scale_factor

        if isinstance(algo_choice, list):
            algo_choices = algo_choice
            repeats = range(len(algo_choice))
        else:
            algo_choices = [algo_choice]
            repeats = range(1)
        for algoIdx in repeats:
            algo_choice = algo_choices[algoIdx]
            if f"ys_{algo_choice}" in saccade_df.columns:
                starting_coordinates = [
                    tuple(row * scale_factor) for row in saccade_df.loc[:, ["xs", f"ys_{algo_choice}"]].values
                ]
                ending_coordinates = [
                    tuple(row * scale_factor) for row in saccade_df.loc[:, ["xe", f"ye_{algo_choice}"]].values
                ]
                for sidx, (start, end) in enumerate(zip(starting_coordinates, ending_coordinates)):
                    if sidx == 0:
                        show_legend = True
                    else:
                        show_legend = False
                    fig.add_trace(
                        go.Scatter(
                            x=[start[0], end[0]],
                            y=[start[1], end[1]],
                            mode="markers+lines",
                            line=dict(color=COLORS[algoIdx], width=1.5 * scale_factor, dash="dash"),
                            showlegend=show_legend,
                            legendgroup="2",
                            text=sidx,
                            textposition="top center",
                            textfont=dict(family="sans serif", size=22 * scale_factor, color=COLORS[algoIdx]),
                            name="Saccades snapped to line",
                            marker=dict(
                                color=COLORS[algoIdx],
                                symbol="arrow",
                                size=duration.values,
                                angleref="previous",
                            ),
                        )
                    )
    fig.update_layout(
        plot_bgcolor=None,
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="right", x=0.8),
    )

    for trace in fig["data"]:
        if trace["name"] == "scale_helper":
            trace["showlegend"] = False
    return fig


def plot_fix_measure(
    dffix,
    plot_choices,
    x_axis_selection,
    margin=dict(t=40, l=10, r=10, b=1),
    label_start="Fixation",
):
    y_label = f"{label_start} Feature"
    if x_axis_selection == "Index":
        num_datapoints = dffix.shape[0]
        x_label = f"{label_start} Number"
        x_nums = np.arange(num_datapoints)
    elif x_axis_selection == "Start Time":
        x_label = f"{label_start} Start Time"
        x_nums = dffix["start_time"]

    layout = dict(
        plot_bgcolor="white",
        autosize=True,
        margin=margin,
        xaxis=dict(
            title=x_label,
            linecolor="black",
            range=[x_nums.min() - 1, x_nums.max() + 1],
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        yaxis=dict(
            title=y_label,
            side="left",
            linecolor="black",
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        legend=dict(orientation="v", yanchor="middle", y=0.95, xanchor="left", x=1.05),
    )

    fig = go.Figure(layout=layout)
    for pidx, plot_choice in enumerate(plot_choices):
        fig.add_trace(
            go.Scatter(
                x=x_nums,
                y=dffix.loc[:, plot_choice],
                mode="markers",
                name=plot_choice,
                marker_color=COLORS[pidx],
                marker_size=3,
                showlegend=True,
            )
        )
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")

    return fig


def plot_y_corr(dffix, algo_choice, margin=dict(t=40, l=10, r=10, b=1)):
    num_datapoints = len(dffix.x)

    layout = dict(
        plot_bgcolor="white",
        autosize=True,
        margin=margin,
        xaxis=dict(
            title="Fixation Index",
            linecolor="black",
            range=[-1, num_datapoints + 1],
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        yaxis=dict(
            title="y correction",
            side="left",
            linecolor="black",
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        legend=dict(orientation="v", yanchor="middle", y=0.95, xanchor="left", x=1.05),
    )
    if isinstance(dffix, dict):
        dffix = dffix["value"]
    algo_string = algo_choice[0] if isinstance(algo_choice, list) else algo_choice
    if f"y_{algo_string}_correction" not in dffix.columns:
        ic("No line-assignment column found in dataframe")
        return go.Figure(layout=layout)
    if isinstance(dffix, dict):
        dffix = dffix["value"]

    fig = go.Figure(layout=layout)

    if isinstance(algo_choice, list):
        algo_choices = algo_choice
        repeats = range(len(algo_choice))
    else:
        algo_choices = [algo_choice]
        repeats = range(1)
    for algoIdx in repeats:
        algo_choice = algo_choices[algoIdx]
        fig.add_trace(
            go.Scatter(
                x=np.arange(num_datapoints),
                y=dffix.loc[:, f"y_{algo_choice}_correction"],
                mode="markers",
                name=f"{algo_choice} y correction",
                marker_color=COLORS[algoIdx],
                marker_size=3,
                showlegend=True,
            )
        )
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")

    return fig


def download_example_ascs(EXAMPLES_FOLDER, EXAMPLES_ASC_ZIP_FILENAME, OSF_DOWNLAOD_LINK, EXAMPLES_FOLDER_PATH):
    if not os.path.isdir(EXAMPLES_FOLDER):
        os.mkdir(EXAMPLES_FOLDER)

    if not os.path.exists(EXAMPLES_ASC_ZIP_FILENAME):
        download_url(OSF_DOWNLAOD_LINK, EXAMPLES_ASC_ZIP_FILENAME)

    if os.path.exists(EXAMPLES_ASC_ZIP_FILENAME):
        if EXAMPLES_FOLDER_PATH.exists():
            EXAMPLE_ASC_FILES = [x for x in EXAMPLES_FOLDER_PATH.glob("*.asc")]
        if len(EXAMPLE_ASC_FILES) != 4:
            try:
                with zipfile.ZipFile(EXAMPLES_ASC_ZIP_FILENAME, "r") as zip_ref:
                    zip_ref.extractall(EXAMPLES_FOLDER)
            except Exception as e:
                ic(e)
                ic(f"Extracting {EXAMPLES_ASC_ZIP_FILENAME} failed")

        EXAMPLE_ASC_FILES = [x for x in EXAMPLES_FOLDER_PATH.glob("*.asc")]
    else:
        EXAMPLE_ASC_FILES = []
    return EXAMPLE_ASC_FILES
