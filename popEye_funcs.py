"""
Mostly adapted from: https://github.com/sascha2schroeder/popEye
"""

import numpy as np
import pandas as pd
from icecream import ic
from scipy import stats
import pathlib as pl

RESULTS_FOLDER = pl.Path("results")


def compute_velocity(xy):
    samp = 1000

    N = xy.shape[0]
    v = pd.DataFrame(data=np.zeros((N, 3)), columns=["time", "vx", "vy"])
    v["time"] = xy["time"]

    v.iloc[2 : (N - 2), 1:3] = (
        samp
        / 6
        * (
            xy.iloc[4:N, 1:3].values
            + xy.iloc[3 : (N - 1), 1:3].values
            - xy.iloc[1 : (N - 3), 1:3].values
            - xy.iloc[0 : (N - 4), 1:3].values
        )
    )
    v.iloc[1, 1:3] = samp / 2 * (xy.iloc[2, 1:3].values - xy.iloc[0, 1:3].values)
    v.iloc[(N - 2), 1:3] = samp / 2 * (xy.iloc[N - 1, 1:3].values - xy.iloc[N - 4, 1:3].values)

    xy = pd.concat([xy.set_index("time"), v.set_index("time")], axis=1).reset_index()
    return xy


def event_long(events_df):
    events_df["duration"] = events_df["stop"] - events_df["start"]
    events_df = events_df[events_df["duration"] > 0]
    events_df = events_df.drop(columns=["duration"])
    events_df.reset_index(drop=True, inplace=True)
    tmplong_cols = list(events_df.columns)
    tmplong_cols.remove("msg")
    events_df["del"] = 0
    for i in events_df.index:
        if events_df.loc[i, "msg"] == "BLINK":
            if i == 0:
                continue
            for col in tmplong_cols:
                events_df.loc[i, col] = events_df.loc[i - 1, col]
            events_df.loc[i - 1, "del"] = 1

    events_df = events_df[events_df["del"] == 0]
    events_df = events_df.drop(columns=["del"])
    events_df.reset_index(drop=True, inplace=True)
    events_df["num"] = range(len(events_df))
    # compute blinks
    # ---------------

    events_df["blink_before"] = 0
    events_df["blink_after"] = 0

    for i in events_df.index:
        if events_df.loc[i, "msg"] == "BLINK":
            events_df.loc[i - 1, "blink_after"] = 1
            if i < len(events_df) - 1:
                events_df.loc[i + 1, "blink_before"] = 1

    # combine
    events_df["blink"] = (events_df["blink_before"] == 1) | (events_df["blink_after"] == 1)
    return events_df.copy()


def compute_non_line_dependent_saccade_measures(saccade_df, trial_dict):

    saccade_df["trial_id"] = trial_dict["trial_id"]
    gaze_df = trial_dict["gaze_df"]
    for s in range(len(saccade_df)):
        is_directional_deviation = False
        a = saccade_df["start_time"][s]
        b = saccade_df["end_time"][s]

        if not gaze_df["x"][[True if (a <= x <= b) else False for x in gaze_df["time"]]].any():
            gaze_df.loc[a:b, "x"] = np.nan

        bool_vec = (gaze_df["time"] >= a) & (gaze_df["time"] <= b)
        if (not gaze_df["x"][bool_vec].isna().any()) and bool_vec.any():
            # saccade amplitude (dX, dY)
            minx = min(gaze_df.loc[bool_vec, "x"])
            maxx = max(gaze_df.loc[bool_vec, "x"])
            if "calibration_method" not in trial_dict or trial_dict["calibration_method"] != "H3":
                miny = min(gaze_df.loc[bool_vec, "y"])
                maxy = max(gaze_df.loc[bool_vec, "y"])
            ix1 = gaze_df.loc[bool_vec, "x"].index[np.argmin(gaze_df.loc[bool_vec, "x"])]
            ix2 = gaze_df.loc[bool_vec, "x"].index[np.argmax(gaze_df.loc[bool_vec, "x"])]
            if "calibration_method" not in trial_dict or trial_dict["calibration_method"] != "H3":
                iy1 = gaze_df.loc[bool_vec, "y"].index[np.argmin(gaze_df.loc[bool_vec, "y"])]
                iy2 = gaze_df.loc[bool_vec, "y"].index[np.argmax(gaze_df.loc[bool_vec, "y"])]
            saccade_df.loc[s, "dX"] = round(np.sign(ix2 - ix1) * (maxx - minx))
            if "calibration_method" not in trial_dict or trial_dict["calibration_method"] != "H3":
                saccade_df.loc[s, "dY"] = round(np.sign(iy2 - iy1) * (maxy - miny))

            # saccade amplitude/angle
            if "calibration_method" not in trial_dict or trial_dict["calibration_method"] != "H3":
                saccade_df.loc[s, "amp_px"] = round(
                    np.sqrt(saccade_df.loc[s, "dX"] ** 2 + saccade_df.loc[s, "dY"] ** 2)
                )
                saccade_df.loc[s, "amp_angle"] = round(np.arctan2(saccade_df.loc[s, "dY"], saccade_df.loc[s, "dX"]), 2)
                saccade_df.loc[s, "amp_angle_deg"] = round(
                    np.arctan2(saccade_df.loc[s, "dY"], saccade_df.loc[s, "dX"]) * (180 / np.pi), 2
                )

            else:
                saccade_df.loc[s, "amp_px"] = np.nan
                saccade_df.loc[s, "amp_angle"] = np.nan
                saccade_df.loc[s, "amp_angle_deg"] = np.nan

        if 35 <= abs(saccade_df.loc[s, "angle"]) <= 145:
            if saccade_df.loc[s, "xe"] - saccade_df.loc[s, "xs"] > 0 and not (
                "blink_before" in saccade_df.columns
                and (saccade_df.loc[s, "blink_before"] or saccade_df.loc[s, "blink_after"])
            ):
                is_directional_deviation = True

        saccade_df.loc[s, "is_directional_deviation"] = is_directional_deviation

    return saccade_df


def compute_saccade_measures(saccade_df, trial_dict, algo_choice):

    if algo_choice is not None:
        algo_str = f"_{algo_choice}"
    else:
        algo_str = ""
    gaze_df = trial_dict["gaze_df"]
    saccade_df.reset_index(drop=True, inplace=True)
    saccade_df.loc[:, f"has_line_change{algo_str}"] = (
        saccade_df.loc[:, f"lines{algo_str}"] != saccade_df.loc[:, f"linee{algo_str}"]
    )
    saccade_df.loc[:, f"goes_to_next_line{algo_str}"] = saccade_df.loc[:, f"linee{algo_str}"] == (
        saccade_df.loc[:, f"lines{algo_str}"] + 1
    )
    saccade_df.loc[:, f"is_directional_deviation{algo_str}"] = False
    saccade_df.loc[:, f"is_return_sweep{algo_str}"] = False

    for sidx, subdf in saccade_df.groupby(f"lines{algo_str}"):
        if subdf.iloc[-1][f"goes_to_next_line{algo_str}"]:
            saccade_df.loc[subdf.index[-1], f"is_return_sweep{algo_str}"] = True

    for s in range(len(saccade_df)):
        is_directional_deviation = False
        a = saccade_df["start_time"][s]
        b = saccade_df["end_time"][s]

        if not gaze_df["x"][[True if (a <= x <= b) else False for x in gaze_df["time"]]].any():
            gaze_df.loc[a:b, "x"] = np.nan

        # saccade distance in letters
        if saccade_df.loc[s, f"lete{algo_str}"] is None or saccade_df.loc[s, f"lets{algo_str}"] is None:
            ic(
                f"None found for compute_saccade_measures at index {s} for subj {trial_dict['subject']} and trial {trial_dict['trial_id']}"
            )
        else:
            saccade_df.loc[s, f"dist_let{algo_str}"] = (
                saccade_df.loc[s, f"lete{algo_str}"] - saccade_df.loc[s, f"lets{algo_str}"]
            )

        bool_vec = (gaze_df["time"] >= a) & (gaze_df["time"] <= b)
        if (not gaze_df["x"][bool_vec].isna().any()) and bool_vec.any():
            # saccade peak velocity (vpeak)
            if "calibration_method" not in trial_dict or trial_dict["calibration_method"] != "H3":
                vx = gaze_df.vx[bool_vec]
                vy = gaze_df.vy[bool_vec]
                if not vx.empty and not vy.empty:
                    saccade_df.loc[s, f"peak_vel{algo_str}"] = round(np.nanmax(np.sqrt(vx**2 + vy**2)))
            else:
                saccade_df.loc[s, f"peak_vel{algo_str}"] = round(np.nanmax(np.sqrt(gaze_df.vx[bool_vec] ** 2)))

        if 35 <= abs(saccade_df.loc[s, f"angle{algo_str}"]) <= 145:
            if saccade_df.loc[s, "xe"] - saccade_df.loc[s, "xs"] > 0 and not (
                "blink_before" in saccade_df.columns
                and (saccade_df.loc[s, "blink_before"] or saccade_df.loc[s, "blink_after"])
            ):
                is_directional_deviation = True

        saccade_df.loc[s, f"is_directional_deviation{algo_str}"] = is_directional_deviation
    return saccade_df.copy()


def get_angle_and_eucl_dist(saccade_df, algo_choice=None):
    if algo_choice is not None:
        algo_str = f"_{algo_choice}"
    else:
        algo_str = ""
    saccade_df["xe_minus_xs"] = saccade_df["xe"] - saccade_df["xs"]
    saccade_df[f"ye_minus_ys{algo_str}"] = saccade_df[f"ye{algo_str}"] - saccade_df[f"ys{algo_str}"]
    saccade_df["eucledian_distance"] = (
        saccade_df["xe_minus_xs"].map(np.square) + saccade_df[f"ye_minus_ys{algo_str}"].map(np.square)
    ).map(np.sqrt)
    saccade_df[f"angle{algo_str}"] = np.arctan2(
        saccade_df.loc[:, f"ye_minus_ys{algo_str}"], saccade_df.loc[:, "xe_minus_xs"]
    ) * (180 / np.pi)
    return saccade_df


def compute_saccade_length(dffix, stimulus_df, algo_choice):

    for j in dffix.index:
        if (
            j == 0
            or pd.isna(dffix.at[j, f"line_num_{algo_choice}"])
            or pd.isna(dffix.at[j - 1, f"line_num_{algo_choice}"])
            or dffix.at[j, f"letternum_{algo_choice}"] is None
            or dffix.at[j - 1, f"letternum_{algo_choice}"] is None
        ):
            continue

        # Same line, calculate saccade length as difference in letter numbers
        if dffix.at[j - 1, f"line_num_{algo_choice}"] == dffix.at[j, f"line_num_{algo_choice}"]:
            dffix.at[j, f"sac_in_{algo_choice}"] = (
                dffix.at[j, f"letternum_{algo_choice}"] - dffix.at[j - 1, f"letternum_{algo_choice}"]
            )

        # Go to line ahead, calculate saccade length as difference in minimum letter numbers in target and previous lines, respectively
        elif dffix.at[j - 1, f"line_num_{algo_choice}"] < dffix.at[j, f"line_num_{algo_choice}"]:
            min_stim_j = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j, f"line_num_{algo_choice}"]]["letternum"]
            )
            min_stim_j_1 = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j - 1, f"line_num_{algo_choice}"]]["letternum"]
            )
            dffix.at[j, f"sac_in_{algo_choice}"] = (dffix.at[j, f"letternum_{algo_choice}"] - min_stim_j) - (
                dffix.at[j - 1, f"letternum_{algo_choice}"] - min_stim_j_1
            )

        # Return to line visited before, calculate saccade length as difference in minimum letter numbers in target and next lines, respectively
        elif dffix.at[j - 1, f"line_num_{algo_choice}"] > dffix.at[j, f"line_num_{algo_choice}"]:
            min_stim_j_1 = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j - 1, f"line_num_{algo_choice}"]]["letternum"]
            )
            min_stim_j = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j, f"line_num_{algo_choice}"]]["letternum"]
            )
            dffix.at[j, f"sac_in_{algo_choice}"] = (dffix.at[j - 1, f"letternum_{algo_choice}"] - min_stim_j_1) - (
                dffix.at[j, f"letternum_{algo_choice}"] - min_stim_j
            )

    for j in range(len(dffix) - 1):
        if (
            pd.isna(dffix.at[j, f"line_num_{algo_choice}"])
            or pd.isna(dffix.at[j + 1, f"line_num_{algo_choice}"])
            or dffix.at[j + 1, f"letternum_{algo_choice}"] is None
            or dffix.at[j, f"letternum_{algo_choice}"] is None
        ):
            continue

        # Same line, calculate saccade length as difference in letter numbers
        if dffix.at[j + 1, f"line_num_{algo_choice}"] == dffix.at[j, f"line_num_{algo_choice}"]:
            dffix.at[j, f"sac_out_{algo_choice}"] = (
                dffix.at[j + 1, f"letternum_{algo_choice}"] - dffix.at[j, f"letternum_{algo_choice}"]
            )

        elif dffix.at[j + 1, f"line_num_{algo_choice}"] > dffix.at[j, f"line_num_{algo_choice}"]:
            min_stim_j_1 = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j + 1, f"line_num_{algo_choice}"]]["letternum"]
            )
            min_stim_j = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j, f"line_num_{algo_choice}"]]["letternum"]
            )
            dffix.at[j, f"sac_out_{algo_choice}"] = (dffix.at[j + 1, f"letternum_{algo_choice}"] - min_stim_j_1) - (
                dffix.at[j, f"letternum_{algo_choice}"] - min_stim_j
            )

        elif dffix.at[j + 1, f"line_num_{algo_choice}"] < dffix.at[j, f"line_num_{algo_choice}"]:
            min_stim_j_1 = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j, f"line_num_{algo_choice}"]]["letternum"]
            )
            min_stim_j = np.min(
                stimulus_df[stimulus_df["assigned_line"] == dffix.at[j + 1, f"line_num_{algo_choice}"]]["letternum"]
            )
            dffix.at[j, f"sac_out_{algo_choice}"] = (dffix.at[j, f"letternum_{algo_choice}"] - min_stim_j) - (
                dffix.at[j + 1, f"letternum_{algo_choice}"] - min_stim_j_1
            )

    return dffix


def compute_launch_distance(dffix, algo_choice):

    for i in range(1, dffix.shape[0]):
        if pd.isna(dffix.loc[i, f"sac_in_{algo_choice}"]):
            continue

        if dffix.loc[i, f"sac_in_{algo_choice}"] >= 0:
            dffix.loc[i, f"word_launch_{algo_choice}"] = (
                dffix.loc[i, f"sac_in_{algo_choice}"] - dffix.loc[i, f"word_land_{algo_choice}"]
            )

        else:
            dffix.loc[i, f"word_launch_{algo_choice}"] = (
                dffix.loc[i, f"sac_in_{algo_choice}"] + dffix.loc[i - 1, f"word_land_{algo_choice}"]
            )

    return dffix


def compute_refixation(dffix, algo_choice):
    dffix.loc[:, f"word_refix_{algo_choice}"] = False
    dffix.loc[:, f"sentence_refix_{algo_choice}"] = False
    for j in dffix.index:
        if (
            j == 0
            or pd.isna(dffix.loc[j, f"on_word_number_{algo_choice}"])
            or pd.isna(dffix.loc[j - 1, f"on_word_number_{algo_choice}"])
        ):
            continue
        dffix.loc[j, f"word_refix_{algo_choice}"] = (
            dffix.loc[j, f"on_word_number_{algo_choice}"] == dffix.loc[j - 1, f"on_word_number_{algo_choice}"]
        )
        dffix.loc[j, f"sentence_refix_{algo_choice}"] = (
            dffix.loc[j, f"on_sentence_num_{algo_choice}"] == dffix.loc[j - 1, f"on_sentence_num_{algo_choice}"]
        )
    return dffix


def compute_regression(dffix, algo_choice):
    tmp = dffix.copy()
    tmp.reset_index(drop=True, inplace=True)
    tmp.loc[:, f"word_reg_out_{algo_choice}"] = False
    tmp.loc[:, f"word_reg_in_{algo_choice}"] = False
    tmp.loc[:, f"word_reg_out_to_{algo_choice}"] = float("nan")
    tmp.loc[:, f"word_reg_in_from_{algo_choice}"] = float("nan")
    tmp.loc[:, f"sentence_reg_out_{algo_choice}"] = False
    tmp.loc[:, f"sentence_reg_in_{algo_choice}"] = False
    tmp.loc[:, f"sentence_reg_out_to_{algo_choice}"] = float("nan")
    tmp.loc[:, f"sentence_reg_in_from_{algo_choice}"] = float("nan")

    if len(tmp) > 1:
        for j in range(1, len(tmp)):
            # Skip outliers
            if pd.isnull(tmp.iloc[j][f"on_word_number_{algo_choice}"]) or pd.isnull(
                tmp.iloc[j - 1][f"on_word_number_{algo_choice}"]
            ):
                continue

            # Word
            if tmp.iloc[j][f"on_word_number_{algo_choice}"] < tmp.iloc[j - 1][f"on_word_number_{algo_choice}"]:
                tmp.loc[j, f"word_reg_in_{algo_choice}"] = True
                tmp.loc[j - 1, f"word_reg_out_{algo_choice}"] = True
                tmp.loc[j, f"word_reg_in_from_{algo_choice}"] = tmp.iloc[j - 1][f"on_word_number_{algo_choice}"]
                tmp.loc[j - 1, f"word_reg_out_to_{algo_choice}"] = tmp.iloc[j][f"on_word_number_{algo_choice}"]

            # Sentence
            if tmp.iloc[j][f"on_sentence_num_{algo_choice}"] < tmp.iloc[j - 1][f"on_sentence_num_{algo_choice}"]:
                tmp.loc[j, f"sentence_reg_in_{algo_choice}"] = True
                tmp.loc[j - 1, f"sentence_reg_out_{algo_choice}"] = True
                tmp.loc[j, f"sentence_reg_in_from_{algo_choice}"] = tmp.iloc[j - 1][f"on_sentence_num_{algo_choice}"]
                tmp.loc[j - 1, f"sentence_reg_out_to_{algo_choice}"] = tmp.iloc[j][f"on_sentence_num_{algo_choice}"]

    extra_cols = list(set(tmp.columns) - set(dffix.columns))
    # select these columns from tmp and add the 'fixation_number'
    cols_to_add = ["fixation_number"] + extra_cols

    # merge selected columns to dffix with 'outer' how and 'fixation_number' as common key
    dffix = pd.merge(dffix, tmp[cols_to_add], on="fixation_number", how="outer")
    return dffix


def compute_firstskip(dffix, algo_choice):
    dffix[f"word_firstskip_{algo_choice}"] = 0
    word_mem = []

    dffix[f"sentence_firstskip_{algo_choice}"] = 0
    sentence_mem = []
    dffix.reset_index(inplace=True)
    for j in range(dffix.shape[0]):

        # word
        if (
            dffix.loc[j, f"on_word_number_{algo_choice}"] < np.max(word_mem, initial=0)
            and dffix.loc[j, f"on_word_number_{algo_choice}"] not in word_mem
        ):
            dffix.loc[j, f"word_firstskip_{algo_choice}"] = 1

        # sent
        if (
            dffix.loc[j, f"on_sentence_num_{algo_choice}"] < np.max(sentence_mem, initial=0)
            and dffix.loc[j, f"on_sentence_num_{algo_choice}"] not in sentence_mem
        ):
            dffix.loc[j, f"sentence_firstskip_{algo_choice}"] = 1

        word_mem.append(dffix.loc[j, f"on_word_number_{algo_choice}"])
        sentence_mem.append(dffix.loc[j, f"on_sentence_num_{algo_choice}"])

    # set NA values for missing line numbers
    dffix.loc[dffix[f"line_num_{algo_choice}"].isna(), f"word_firstskip_{algo_choice}"] = np.nan
    dffix.loc[dffix[f"line_num_{algo_choice}"].isna(), f"sentence_firstskip_{algo_choice}"] = np.nan
    dffix.set_index("index", inplace=True)
    return dffix


def compute_run(dffix, algo_choice):
    if "fixation_number" not in dffix.columns and "num" in dffix.columns:
        dffix["fixation_number"] = dffix["num"]
    tmp = dffix.copy()
    tmp.reset_index(inplace=True, drop=True)
    # initialize
    tmp.loc[~tmp[f"on_word_{algo_choice}"].isna(), f"word_runid_{algo_choice}"] = 0
    tmp[f"sentence_runid_{algo_choice}"] = 0

    # fixation loop
    if len(tmp) > 1:
        for j in range(1, len(tmp)):

            # word
            if tmp[f"word_reg_in_{algo_choice}"][j] == 1 and tmp[f"word_reg_in_{algo_choice}"][j - 1] != 1:
                tmp.loc[j, f"word_runid_{algo_choice}"] = tmp[f"word_runid_{algo_choice}"][j - 1] + 1
            else:
                tmp.loc[j, f"word_runid_{algo_choice}"] = tmp.loc[j - 1, f"word_runid_{algo_choice}"]

            # sentence
            if tmp[f"sentence_reg_in_{algo_choice}"][j] == 1 and tmp[f"sentence_reg_in_{algo_choice}"][j - 1] != 1:
                tmp.loc[j, f"sentence_runid_{algo_choice}"] = tmp[f"sentence_runid_{algo_choice}"][j - 1] + 1
            else:
                tmp.loc[j, f"sentence_runid_{algo_choice}"] = tmp[f"sentence_runid_{algo_choice}"][j - 1]
    tmp[f"word_runid_{algo_choice}"] = tmp[f"word_runid_{algo_choice}"] - 1
    tmp[f"sentence_runid_{algo_choice}"] = tmp[f"sentence_runid_{algo_choice}"] - 1
    # fixid in word
    tmp[f"word_fix_{algo_choice}"] = tmp.groupby(f"on_word_number_{algo_choice}")["fixation_number"].transform(
        lambda x: stats.rankdata(x, method="min")
    )
    # fixid in sent
    tmp[f"sentence_fix_{algo_choice}"] = tmp.groupby(f"on_sentence_num_{algo_choice}")["fixation_number"].transform(
        lambda x: stats.rankdata(x, method="min")
    )

    # runid in word
    tmp["id"] = tmp[f"on_word_number_{algo_choice}"].astype(str) + ":" + tmp[f"word_runid_{algo_choice}"].astype(str)
    fix_tmp = tmp.copy().drop_duplicates(subset="id")
    fix_tmp[f"word_run_{algo_choice}"] = fix_tmp.groupby(f"on_word_number_{algo_choice}")[
        f"word_runid_{algo_choice}"
    ].transform(lambda x: stats.rankdata(x, method="min"))

    if f"word_run_{algo_choice}" in tmp.columns:
        tmp = tmp.drop(columns=[f"word_run_{algo_choice}"])
    tmp = pd.merge(tmp, fix_tmp[["id", f"word_run_{algo_choice}"]], on="id")
    del tmp["id"]
    tmp = tmp.sort_values("fixation_number")

    # runid in sentence
    tmp["id"] = (
        tmp[f"on_sentence_num_{algo_choice}"].astype(str) + ":" + tmp[f"sentence_runid_{algo_choice}"].astype(str)
    )
    fix_tmp = tmp.copy().drop_duplicates(subset="id")
    fix_tmp[f"sentence_run_{algo_choice}"] = fix_tmp.groupby(f"on_sentence_num_{algo_choice}")["id"].transform(
        lambda x: stats.rankdata(x, method="min")
    )
    if f"sentence_run_{algo_choice}" in tmp.columns:
        tmp = tmp.drop(columns=[f"sentence_run_{algo_choice}"])
    tmp = pd.merge(tmp, fix_tmp[["id", f"sentence_run_{algo_choice}"]], on="id")
    del tmp["id"]
    tmp = tmp.sort_values("fixation_number")

    # fixnum in word_run
    tmp["id"] = tmp[f"on_word_number_{algo_choice}"].astype(str) + ":" + tmp[f"word_run_{algo_choice}"].astype(str)
    tmp[f"word_run_fix_{algo_choice}"] = tmp.groupby(["id"])["fixation_number"].rank("first").values
    del tmp["id"]
    tmp = tmp.sort_values("fixation_number")

    # fixnum in sentence_run
    tmp["id"] = tmp[f"on_sentence_num_{algo_choice}"].astype(str) + ":" + tmp[f"sentence_run_{algo_choice}"].astype(str)
    tmp[f"sentence_run_fix_{algo_choice}"] = tmp.groupby(["id"])["fixation_number"].rank("first").values
    del tmp["id"]
    tmp = tmp.sort_values("fixation_number")
    names = [
        "fixation_number",
        f"word_runid_{algo_choice}",
        f"sentence_runid_{algo_choice}",
        f"word_fix_{algo_choice}",
        f"sentence_fix_{algo_choice}",
        f"word_run_{algo_choice}",
        f"sentence_run_{algo_choice}",
        f"word_run_fix_{algo_choice}",
        f"sentence_run_fix_{algo_choice}",
    ]
    dffix = pd.merge(dffix, tmp[names], on="fixation_number", how="left")
    return dffix.copy()


def compute_landing_position(dffix, algo_choice):
    dffix[f"word_cland_{algo_choice}"] = (
        dffix[f"word_land_{algo_choice}"] - (dffix[f"on_word_{algo_choice}"].str.len() + 1) / 2
    )
    return dffix


def aggregate_words_firstrun(
    fix,
    algo_choice,
    measures_to_calculate=[
        "firstrun_blink",
        "firstrun_skip",
        "firstrun_refix",
        "firstrun_reg_in",
        "firstrun_reg_out",
        "firstrun_dur",
        "firstrun_gopast",
        "firstrun_gopast_sel",
    ],
):
    firstruntmp = fix.loc[fix[f"word_run_{algo_choice}"] == 1].copy()

    firstrun = firstruntmp.drop_duplicates(subset=f"on_word_number_{algo_choice}", keep="first").copy()

    names = [
        "subject",
        "trial_id",
        "item",
        "condition",
        f"on_word_number_{algo_choice}",
        f"on_word_{algo_choice}",
        "fixation_number",
    ]
    firstrun = firstrun[names].sort_values(f"on_word_number_{algo_choice}")

    # compute measures
    firstrun[f"firstrun_nfix_{algo_choice}"] = firstruntmp.groupby(f"on_word_number_{algo_choice}")[
        "fixation_number"
    ].transform(
        "count"
    )  # Required for many other measures
    firstrun[f"firstrun_nfix_{algo_choice}"] = firstrun[f"firstrun_nfix_{algo_choice}"].fillna(0)
    if "firstrun_blink" in measures_to_calculate:
        if "blink" in firstruntmp:
            firstrun[f"firstrun_blink_{algo_choice}"] = firstruntmp.groupby(f"on_word_number_{algo_choice}")[
                "blink"
            ].transform("max")
        else:
            firstrun[f"firstrun_blink_{algo_choice}"] = 0

    if "firstrun_skip" in measures_to_calculate:
        firstrun[f"firstrun_skip_{algo_choice}"] = firstruntmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_firstskip_{algo_choice}"
        ].transform("max")
    if "firstrun_refix" in measures_to_calculate:
        firstrun[f"firstrun_refix_{algo_choice}"] = firstruntmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_refix_{algo_choice}"
        ].transform("max")
    if "firstrun_reg_in" in measures_to_calculate:
        firstrun[f"firstrun_reg_in_{algo_choice}"] = firstruntmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_reg_out_{algo_choice}"
        ].transform("max")
    if "firstrun_reg_out" in measures_to_calculate:
        firstrun[f"firstrun_reg_out_{algo_choice}"] = firstruntmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_reg_in_{algo_choice}"
        ].transform("max")
    if "firstrun_dur" in measures_to_calculate:
        firstrun[f"firstrun_dur_{algo_choice}"] = firstruntmp.groupby(f"on_word_number_{algo_choice}")[
            "duration"
        ].transform("sum")
    firstrun = firstrun.sort_values(["trial_id", f"on_word_number_{algo_choice}"]).copy()

    return firstrun


def compute_gopast_word(fixations_dataframe, algo_choice):

    ias = np.unique(fixations_dataframe.loc[:, f"on_word_number_{algo_choice}"])

    for j in range(len(ias) - 1):
        fixations_dataframe.loc[
            (fixations_dataframe[f"on_word_number_{algo_choice}"] == ias[j]), f"gopast_{algo_choice}"
        ] = np.nansum(
            fixations_dataframe.loc[
                (
                    fixations_dataframe["fixation_number"]
                    >= np.min(
                        fixations_dataframe.loc[
                            (fixations_dataframe[f"on_word_number_{algo_choice}"] == ias[j]), "fixation_number"
                        ]
                    )
                )
                & (
                    fixations_dataframe["fixation_number"]
                    < np.min(
                        fixations_dataframe.loc[
                            (fixations_dataframe[f"on_word_number_{algo_choice}"] > ias[j]), "fixation_number"
                        ]
                    )
                )
                & (~fixations_dataframe[f"on_word_number_{algo_choice}"].isna())
            ]["duration"]
        )

        fixations_dataframe.loc[
            (fixations_dataframe[f"on_word_number_{algo_choice}"] == ias[j]), f"selgopast_{algo_choice}"
        ] = np.nansum(
            fixations_dataframe.loc[
                (
                    fixations_dataframe["fixation_number"]
                    >= np.min(
                        fixations_dataframe.loc[
                            (fixations_dataframe[f"on_word_number_{algo_choice}"] == ias[j]), "fixation_number"
                        ]
                    )
                )
                & (
                    fixations_dataframe["fixation_number"]
                    < np.min(
                        fixations_dataframe.loc[
                            (fixations_dataframe[f"on_word_number_{algo_choice}"] > ias[j]), "fixation_number"
                        ]
                    )
                )
                & (fixations_dataframe[f"on_word_number_{algo_choice}"] == ias[j])
                & (~fixations_dataframe[f"on_word_number_{algo_choice}"].isna())
            ]["duration"]
        )
    return fixations_dataframe


def aggregate_words(
    fix,
    word_item,
    algo_choice,
    measures_to_calculate=[
        "blink",
    ],
):
    wordtmp = fix.copy()

    word = wordtmp.drop_duplicates(subset=f"on_word_number_{algo_choice}", keep="first").copy()
    names = [
        f"on_sentence_num_{algo_choice}",
        f"on_word_number_{algo_choice}",
        f"on_word_{algo_choice}",
    ]
    word = word.loc[:, names].sort_values(by=f"on_word_number_{algo_choice}")

    wordtmp = compute_gopast_word(wordtmp, algo_choice)

    if "blink" in measures_to_calculate:
        if "blink" in wordtmp:
            word[f"blink_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")["blink"].transform("max")
        else:
            word[f"blink_{algo_choice}"] = 0
    if "nrun" in measures_to_calculate or "reread" in measures_to_calculate:
        word[f"nrun_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_run_{algo_choice}"
        ].transform("max")
    if "reread" in measures_to_calculate:
        word[f"reread_{algo_choice}"] = word[f"nrun_{algo_choice}"] > 1
    word[f"number_of_fixations_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
        "fixation_number"
    ].transform("count")
    if "refix" in measures_to_calculate:
        word[f"refix_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_refix_{algo_choice}"
        ].transform("max")
    if "reg_in" in measures_to_calculate:
        word[f"reg_in_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_reg_in_{algo_choice}"
        ].transform("max")
    if "reg_out" in measures_to_calculate:
        word[f"reg_out_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
            f"word_reg_out_{algo_choice}"
        ].transform("max")
    if "total_fixation_duration" in measures_to_calculate:
        word[f"total_fixation_duration_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
            "duration"
        ].transform("sum")
    if "gopast" in measures_to_calculate and f"gopast_{algo_choice}" in wordtmp.columns:
        word[f"gopast_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
            f"gopast_{algo_choice}"
        ].transform("max")
        word[f"gopast_{algo_choice}"] = word[f"gopast_{algo_choice}"].fillna(0)

    if "gopast_sel" in measures_to_calculate and f"selgopast_{algo_choice}" in wordtmp.columns:
        word[f"gopast_sel_{algo_choice}"] = wordtmp.groupby(f"on_word_number_{algo_choice}")[
            f"selgopast_{algo_choice}"
        ].transform("max")
        word[f"gopast_sel_{algo_choice}"] = word[f"gopast_sel_{algo_choice}"].fillna(0)

    word.rename({f"on_word_number_{algo_choice}": "word_number"}, axis=1, inplace=True)
    word = pd.merge(
        word.reset_index(drop=True), word_item.reset_index(drop=True), on="word_number", how="right", validate="1:1"
    )
    word[f"number_of_fixations_{algo_choice}"] = word[f"number_of_fixations_{algo_choice}"].fillna(0)
    if "total_fixation_duration" in measures_to_calculate:
        word[f"total_fixation_duration_{algo_choice}"] = word[f"total_fixation_duration_{algo_choice}"].fillna(0)

    word[f"skip_{algo_choice}"] = 0
    if "blink" in measures_to_calculate:
        word.loc[word[f"blink_{algo_choice}"].isna(), f"skip_{algo_choice}"] = 1
    word.loc[word[f"number_of_fixations_{algo_choice}"] == 0, f"skip_{algo_choice}"] = 1
    word[f"skip_{algo_choice}"] = word[f"skip_{algo_choice}"].astype("boolean")

    if "number_of_fixations" not in measures_to_calculate:
        word = word.drop(columns=f"number_of_fixations_{algo_choice}")
    if "blink" in measures_to_calculate:
        word[f"blink_{algo_choice}"] = word[f"blink_{algo_choice}"].astype("boolean")

    word = word.sort_values(by=["word_number"])

    if "condition" in wordtmp.columns and "condition" not in word.columns:
        word.insert(loc=0, column="condition", value=wordtmp["condition"].iloc[0])
    if "item" in wordtmp.columns and "item" not in word.columns:
        word.insert(loc=0, column="item", value=wordtmp["item"].iloc[0])
    if "trial_id" in wordtmp.columns and "trial_id" not in word.columns:
        word.insert(loc=0, column="trial_id", value=wordtmp["trial_id"].iloc[0])
    if "subject" in wordtmp.columns and "subject" not in word.columns:
        word.insert(loc=0, column="subject", value=wordtmp["subject"].iloc[0])

    return word


def combine_words(fix, wordfirst, wordtmp, algo_choice, measures_to_calculate):

    subject = wordtmp["subject"].values[0]
    trial_id = wordtmp["trial_id"].values[0]
    item = wordtmp["item"].values[0]
    condition = wordtmp["condition"].values[0]
    wordtmp = wordtmp.loc[
        :,
        [
            c
            for c in [
                "word_number",
                "word",
                f"blink_{algo_choice}",
                f"skip_{algo_choice}",
                f"nrun_{algo_choice}",
                f"reread_{algo_choice}",
                f"number_of_fixations_{algo_choice}",
                f"refix_{algo_choice}",
                f"reg_in_{algo_choice}",
                f"reg_out_{algo_choice}",
                f"total_fixation_duration_{algo_choice}",
                f"gopast_{algo_choice}",
                f"gopast_sel_{algo_choice}",
            ]
            if c in wordtmp.columns
        ],
    ]

    wordfirsttmp = wordfirst.loc[
        :,
        [
            c
            for c in [
                f"on_word_number_{algo_choice}",
                f"firstrun_skip_{algo_choice}",
                f"firstrun_nfix_{algo_choice}",
                f"firstrun_refix_{algo_choice}",
                f"firstrun_reg_in_{algo_choice}",
                f"firstrun_reg_out_{algo_choice}",
                f"firstrun_dur_{algo_choice}",
                f"firstrun_gopast_{algo_choice}",
                f"firstrun_gopast_sel_{algo_choice}",
            ]
            if c in wordfirst.columns
        ],
    ]

    fixtmp = fix[(fix[f"word_run_{algo_choice}"] == 1) & (fix[f"word_run_fix_{algo_choice}"] == 1)].copy()
    names = [
        c
        for c in [
            f"on_word_number_{algo_choice}",
            f"sac_in_{algo_choice}",
            f"sac_out_{algo_choice}",
            f"word_launch_{algo_choice}",
            f"word_land_{algo_choice}",
            f"word_cland_{algo_choice}",
            f"duration",
        ]
        if c in fixtmp.columns
    ]
    fixtmp = fixtmp[names].copy()
    fixtmp.rename(
        {
            f"sac_in_{algo_choice}": f"firstfix_sac_in_{algo_choice}",
            f"sac_out_{algo_choice}": f"firstfix_sac_out_{algo_choice}",
            f"word_launch_{algo_choice}": f"firstfix_launch_{algo_choice}",
            f"word_land_{algo_choice}": f"firstfix_land_{algo_choice}",
            f"word_cland_{algo_choice}": f"firstfix_cland_{algo_choice}",
            f"duration": f"firstfix_dur_{algo_choice}",
        },
        axis=1,
        inplace=True,
    )
    comb = pd.merge(
        pd.merge(
            wordtmp,
            wordfirsttmp.rename({f"on_word_number_{algo_choice}": "word_number"}, axis=1),
            on="word_number",
            how="left",
        ),
        fixtmp.rename({f"on_word_number_{algo_choice}": "word_number"}, axis=1),
        on="word_number",
        how="left",
    )

    dropcols = [
        c
        for c in [
            f"firstrun_skip_{algo_choice}",
            f"firstrun_refix_{algo_choice}",
            f"firstrun_reg_in_{algo_choice}",
            f"firstrun_reg_out_{algo_choice}",
            f"firstrun_dur_{algo_choice}",
            f"firstrun_gopast_{algo_choice}",
            f"firstrun_gopast_sel_{algo_choice}",
            f"firstfix_sac_in_{algo_choice}",
            f"firstfix_sac_out_{algo_choice}",
            f"firstfix_launch_{algo_choice}",
            f"firstfix_land_{algo_choice}",
            f"firstfix_cland_{algo_choice}",
            f"firstfix_dur_{algo_choice}",
        ]
        if ((c.replace(f"_{algo_choice}", "") not in measures_to_calculate) & (c in comb.columns))
    ]
    comb = comb.drop(columns=dropcols).copy()
    comb.sort_values(by="word_number", inplace=True)

    # recompute firstrun skip (skips are also firstkips)
    if f"skip_{algo_choice}" in comb.columns and f"firstrun_skip_{algo_choice}" in comb.columns:
        comb.loc[comb[f"skip_{algo_choice}"] == 1, f"firstrun_skip_{algo_choice}"] = 1

    # gopast time in firstrun
    if f"gopast_{algo_choice}" in comb.columns and "firstrun_gopast" in measures_to_calculate:
        comb[f"firstrun_gopast_{algo_choice}"] = comb[f"gopast_{algo_choice}"]
    if f"gopast_sel_{algo_choice}" in comb.columns and "firstrun_gopast_sel" in measures_to_calculate:
        comb[f"firstrun_gopast_sel_{algo_choice}"] = comb[f"gopast_sel_{algo_choice}"]
    if f"gopast_{algo_choice}" in comb.columns:
        comb.drop(columns=[f"gopast_{algo_choice}"], inplace=True)

    if f"gopast_sel_{algo_choice}" in comb.columns:
        comb.drop(columns=[f"gopast_sel_{algo_choice}"], inplace=True)

    if f"firstrun_nfix_{algo_choice}" in comb.columns and "singlefix" in measures_to_calculate:
        comb[f"singlefix_{algo_choice}"] = 0
        comb.loc[(comb[f"firstrun_nfix_{algo_choice}"] == 1), f"singlefix_{algo_choice}"] = 1

    if f"firstfix_sac_in_{algo_choice}" in comb.columns and "singlefix_sac_in" in measures_to_calculate:
        comb.loc[(comb[f"firstrun_nfix_{algo_choice}"] == 1), f"singlefix_sac_in_{algo_choice}"] = comb[
            f"firstfix_sac_in_{algo_choice}"
        ][(comb[f"firstrun_nfix_{algo_choice}"] == 1)]

    if f"firstfix_sac_out_{algo_choice}" in comb.columns and "singlefix_sac_out" in measures_to_calculate:
        comb.loc[(comb[f"firstrun_nfix_{algo_choice}"] == 1), f"singlefix_sac_out_{algo_choice}"] = comb[
            f"firstfix_sac_out_{algo_choice}"
        ][(comb[f"firstrun_nfix_{algo_choice}"] == 1)]

    if f"firstfix_launch_{algo_choice}" in comb.columns and "singlefix_launch" in measures_to_calculate:
        comb.loc[(comb[f"firstrun_nfix_{algo_choice}"] == 1), f"singlefix_launch_{algo_choice}"] = comb[
            f"firstfix_launch_{algo_choice}"
        ][(comb[f"firstrun_nfix_{algo_choice}"] == 1)]

    if f"firstfix_land_{algo_choice}" in comb.columns and "singlefix_land" in measures_to_calculate:
        comb.loc[(comb[f"firstrun_nfix_{algo_choice}"] == 1), f"singlefix_land_{algo_choice}"] = comb[
            f"firstfix_land_{algo_choice}"
        ][(comb[f"firstrun_nfix_{algo_choice}"] == 1)]

    if f"firstfix_cland_{algo_choice}" in comb.columns and "singlefix_cland" in measures_to_calculate:
        comb.loc[(comb[f"firstrun_nfix_{algo_choice}"] == 1), f"singlefix_cland_{algo_choice}"] = comb[
            f"firstfix_cland_{algo_choice}"
        ][(comb[f"firstrun_nfix_{algo_choice}"] == 1)]

    if f"firstfix_dur_{algo_choice}" in comb.columns and "singlefix_dur" in measures_to_calculate:
        comb.loc[(comb[f"firstrun_nfix_{algo_choice}"] == 1), f"singlefix_dur_{algo_choice}"] = comb[
            f"firstfix_dur_{algo_choice}"
        ][(comb[f"firstrun_nfix_{algo_choice}"] == 1)]

    if "condition" not in comb.columns:
        comb.insert(loc=0, column="condition", value=condition)
    if "item" not in comb.columns:
        comb.insert(loc=0, column="item", value=item)
    if "trial_id" not in comb.columns:
        comb.insert(loc=0, column="trial_id", value=trial_id)
    if "subject" not in comb.columns:
        comb.insert(loc=0, column="subject", value=subject)
    return comb.copy()


def compute_sentence_measures(fix, stimmat, algo_choice, measures_to_calc, save_to_csv=False):
    sentitem = stimmat.drop_duplicates(
        subset="in_sentence_number", keep="first"
    )  # TODO check why there are rows with sent number None
    fixin = fix.copy().reset_index(drop=True)

    fixin["on_sentence_num2"] = fixin[f"on_sentence_num_{algo_choice}"].copy()

    # Recompute sentence number (two fixation exception rule)
    for j in range(1, len(fixin) - 1):
        if fixin.loc[j, "on_sentence_num2"] != fixin.loc[j - 1, "on_sentence_num2"]:
            if j + 1 in fixin.index and fixin.loc[j + 1, "on_sentence_num2"] == fixin.loc[j - 1, "on_sentence_num2"]:
                fixin.loc[j, "on_sentence_num2"] = fixin.loc[j - 1, "on_sentence_num2"]
            elif j + 2 in fixin.index and fixin.loc[j + 2, "on_sentence_num2"] == fixin.loc[j - 1, "on_sentence_num2"]:
                fixin.loc[j, "on_sentence_num2"] = fixin.loc[j - 1, "on_sentence_num2"]

    fixin["id"] = fixin.apply(lambda row: f"{row['on_sentence_num2']}", axis=1)

    fixin[f"sent_reg_in2_{algo_choice}"] = 0
    fixin[f"sent_reg_out2_{algo_choice}"] = 0

    fixin[f"sent_runid2_{algo_choice}"] = 1

    fixin.loc[0, "last"] = fixin.loc[0, "id"]
    fixin.loc[0, f"firstpass_{algo_choice}"] = 1
    mem = [fixin.loc[0, "on_sentence_num2"]]
    wordmem = [fixin.loc[0, f"on_word_number_{algo_choice}"]]
    fixin.loc[0, f"forward_{algo_choice}"] = 1

    for j in range(1, len(fixin)):
        fixin.loc[j, "last"] = fixin.loc[j - 1, "id"]

        if fixin.loc[j, "on_sentence_num2"] != fixin.loc[j - 1, "on_sentence_num2"]:
            fixin.loc[j, f"sent_reg_in2_{algo_choice}"] = 1
            fixin.loc[j - 1, f"sent_reg_out2_{algo_choice}"] = 1
            fixin.loc[j, f"sent_reg_in_from2_{algo_choice}"] = fixin.loc[j - 1, "on_sentence_num2"]
            fixin.loc[j - 1, f"sent_reg_out_to2_{algo_choice}"] = fixin.loc[j, "on_sentence_num2"]

        if fixin.loc[j, f"sent_reg_in2_{algo_choice}"] == 1 and fixin.loc[j - 1, f"sent_reg_in2_{algo_choice}"] != 1:
            fixin.loc[j, f"sent_runid2_{algo_choice}"] = fixin.loc[j - 1, f"sent_runid2_{algo_choice}"] + 1
        else:
            fixin.loc[j, f"sent_runid2_{algo_choice}"] = fixin.loc[j - 1, f"sent_runid2_{algo_choice}"]

        if fixin.loc[j, "on_sentence_num2"] >= fixin.loc[j - 1, "on_sentence_num2"]:
            if fixin.loc[j, "on_sentence_num2"] in mem:
                if fixin.loc[j, "on_sentence_num2"] == max(mem):
                    fixin.loc[j, f"firstpass_{algo_choice}"] = 1
                else:
                    fixin.loc[j, f"firstpass_{algo_choice}"] = 0
            else:
                mem.append(fixin.loc[j, "on_sentence_num2"])
                fixin.loc[j, f"firstpass_{algo_choice}"] = 1
        else:
            fixin.loc[j, f"firstpass_{algo_choice}"] = 0

        if fixin.loc[j, f"on_word_number_{algo_choice}"] > max(wordmem):
            wordmem.append(fixin.loc[j, f"on_word_number_{algo_choice}"])
            fixin.loc[j, f"forward_{algo_choice}"] = 1
        elif fixin.loc[j, f"on_word_number_{algo_choice}"] < max(wordmem):
            fixin.loc[j, f"forward_{algo_choice}"] = 0

    for i in range(len(fixin) - 3):
        if fixin.loc[i, f"line_change_{algo_choice}"] > 0:
            fixin.loc[i, "on_word_number"] = 0
            fixin.loc[i + 1, f"forward_{algo_choice}"] = 1
            fixin.loc[i + 2, f"forward_{algo_choice}"] = 1
            fixin.loc[i + 3, f"forward_{algo_choice}"] = 1

    for i in range(1, len(fixin) - 3):
        if fixin.loc[i, "on_sentence_num2"] > fixin.loc[i - 1, "on_sentence_num2"]:
            fixin.loc[i + 1, f"forward_{algo_choice}"] = 1
            fixin.loc[i + 2, f"forward_{algo_choice}"] = 1

    fixin["id2"] = fixin["id"] + ":" + fixin[f"sent_runid2_{algo_choice}"].astype(str)

    fixin = fixin.sort_values(["trial_id", "fixation_number"])

    sent = fixin.copy().drop_duplicates(subset="id", keep="first")
    names = [
        "id",
        "subject",
        "trial_id",
        "item",
        "condition",
        "on_sentence_num2",
        f"on_sentence_num_{algo_choice}",
        f"on_sentence_{algo_choice}",
        "num_words_in_sentence",
    ]
    sent = sent[names].reset_index(drop=True)

    sent[f"firstrun_skip_{algo_choice}"] = 0

    mem = []
    for j in range(len(sent)):
        if not pd.isna(sent.loc[j, f"on_sentence_num_{algo_choice}"]):
            if len(mem) > 0 and sent.loc[j, f"on_sentence_num_{algo_choice}"] < max(mem) and not pd.isna(max(mem)):
                sent.loc[j, f"firstrun_skip_{algo_choice}"] = 1
        if (
            not pd.isna(sent.loc[j, f"on_sentence_num_{algo_choice}"])
            and sent.loc[j, f"on_sentence_num_{algo_choice}"] not in mem
        ):
            mem.append(sent.loc[j, f"on_sentence_num_{algo_choice}"])

    if "total_n_fixations" in measures_to_calc:
        tmp = fixin.groupby("id")["duration"].count().reset_index()
        tmp.columns = ["id", f"total_n_fixations_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"total_n_fixations_{algo_choice}": 0}, inplace=True)

    tmp = fixin.groupby("id")["duration"].sum().reset_index()
    tmp.columns = ["id", f"total_dur_{algo_choice}"]
    sent = pd.merge(sent, tmp, on="id", how="left")
    sent.fillna({f"total_dur_{algo_choice}": 0}, inplace=True)

    if "firstpass_n_fixations" in measures_to_calc:
        tmp = fixin[fixin[f"firstpass_{algo_choice}"] == 1].groupby("id")["duration"].count().reset_index()
        tmp.columns = ["id", f"firstpass_n_fixations_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstpass_n_fixations_{algo_choice}": 0}, inplace=True)

    if "firstpass_dur" in measures_to_calc:
        tmp = fixin[fixin[f"firstpass_{algo_choice}"] == 1].groupby("id")["duration"].sum().reset_index()
        tmp.columns = ["id", f"firstpass_dur_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstpass_dur_{algo_choice}": 0}, inplace=True)

    if "firstpass_forward_n_fixations" in measures_to_calc:
        tmp = (
            fixin[(fixin[f"firstpass_{algo_choice}"] == 1) & (fixin[f"forward_{algo_choice}"] == 1)]
            .groupby("id")["duration"]
            .count()
            .reset_index()
        )
        tmp.columns = ["id", f"firstpass_forward_n_fixations_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstpass_forward_n_fixations_{algo_choice}": 0}, inplace=True)

    if "firstpass_forward_dur" in measures_to_calc:
        tmp = (
            fixin[(fixin[f"firstpass_{algo_choice}"] == 1) & (fixin[f"forward_{algo_choice}"] == 1)]
            .groupby("id")["duration"]
            .sum()
            .reset_index()
        )
        tmp.columns = ["id", f"firstpass_forward_dur_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstpass_forward_dur_{algo_choice}": 0}, inplace=True)

    if "firstpass_reread_n_fixations" in measures_to_calc:
        tmp = (
            fixin[(fixin[f"firstpass_{algo_choice}"] == 1) & (fixin[f"forward_{algo_choice}"] == 0)]
            .groupby("id")["duration"]
            .count()
            .reset_index()
        )
        tmp.columns = ["id", f"firstpass_reread_n_fixations_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstpass_reread_n_fixations_{algo_choice}": 0}, inplace=True)

    if "firstpass_reread_dur" in measures_to_calc:
        tmp = (
            fixin[(fixin[f"firstpass_{algo_choice}"] == 1) & (fixin[f"forward_{algo_choice}"] == 0)]
            .groupby("id")["duration"]
            .sum()
            .reset_index()
        )
        tmp.columns = ["id", f"firstpass_reread_dur_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstpass_reread_dur_{algo_choice}": 0}, inplace=True)

    if sum(fixin[f"firstpass_{algo_choice}"] == 0) != 0:
        if "lookback_n_fixations" in measures_to_calc:
            tmp = fixin[fixin[f"firstpass_{algo_choice}"] == 0].groupby("id")["duration"].count().reset_index()
            tmp.columns = ["id", f"lookback_n_fixations_{algo_choice}"]
            sent = pd.merge(sent, tmp, on="id", how="left")
            sent.fillna({f"lookback_n_fixations_{algo_choice}": 0}, inplace=True)

        if "lookback_dur" in measures_to_calc:
            tmp = fixin[fixin[f"firstpass_{algo_choice}"] == 0].groupby("id")["duration"].sum().reset_index()
            tmp.columns = ["id", f"lookback_dur_{algo_choice}"]
            sent = pd.merge(sent, tmp, on="id", how="left")
            sent.fillna({f"lookback_dur_{algo_choice}": 0}, inplace=True)

        fixin["id2"] = fixin.apply(lambda row: f"{row['id']}:{row[f'sent_runid2_{algo_choice}']}", axis=1)
        sent2 = fixin.drop_duplicates(subset="id2", keep="first")
        sent3 = sent2[(sent2[f"firstpass_{algo_choice}"] == 0) & (~pd.isna(sent2[f"sent_reg_in_from2_{algo_choice}"]))]

        tmp = fixin[fixin["id2"].isin(sent3["id2"])].groupby("id")["duration"].count().reset_index()
        tmp.columns = ["id", f"lookfrom_n_fixations_{algo_choice}"]
        tmp2 = pd.merge(tmp, sent3)
        tmp3 = tmp2.groupby("last")[f"lookfrom_n_fixations_{algo_choice}"].sum().reset_index()
        tmp3.columns = ["last", f"lookfrom_n_fixations_{algo_choice}"]
        sent = pd.merge(sent, tmp3, left_on="id", right_on="last", how="left")
        sent.fillna({f"lookfrom_n_fixations_{algo_choice}": 0}, inplace=True)

        if "lookfrom_dur" in measures_to_calc:
            tmp = fixin[fixin["id2"].isin(sent3["id2"])].groupby("id")["duration"].sum().reset_index()
            tmp.columns = ["id", f"lookfrom_dur_{algo_choice}"]
            tmp2 = pd.merge(tmp, sent3)
            tmp3 = tmp2.groupby("last")[f"lookfrom_dur_{algo_choice}"].sum().reset_index()
            tmp3.columns = ["last", f"lookfrom_dur_{algo_choice}"]
            sent = pd.merge(sent, tmp3, left_on="id", right_on="last", how="left")
            sent.fillna({f"lookfrom_dur_{algo_choice}": 0}, inplace=True)

    # Firstrun
    firstruntmp = fixin[fixin[f"sentence_run_{algo_choice}"] == 1]

    if "firstrun_reg_in" in measures_to_calc:
        tmp = firstruntmp.groupby("id")[f"sent_reg_in2_{algo_choice}"].max().reset_index()
        tmp.columns = ["id", f"firstrun_reg_in_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstrun_reg_in_{algo_choice}": 0}, inplace=True)

    if "firstrun_reg_out" in measures_to_calc:
        tmp = firstruntmp.groupby("id")[f"sent_reg_out2_{algo_choice}"].max().reset_index()
        tmp.columns = ["id", f"firstrun_reg_out_{algo_choice}"]
        sent = pd.merge(sent, tmp, on="id", how="left")
        sent.fillna({f"firstrun_reg_out_{algo_choice}": 0}, inplace=True)

    # Complete sentence
    gopasttmp = fixin.copy()
    gopasttmp[f"on_sentence_num_{algo_choice}"] = gopasttmp["on_sentence_num2"]
    tmp = compute_gopast_sentence(gopasttmp, algo_choice)
    names = ["id", f"gopast_{algo_choice}", f"selgopast_{algo_choice}"]
    tmp = tmp[names]
    tmp = tmp.drop_duplicates(subset="id", keep="first")
    tmp.columns = ["id", f"gopast_{algo_choice}", f"gopast_sel_{algo_choice}"]
    sent = pd.merge(sent, tmp, on="id", how="left")

    # Nrun
    tmp = fixin.groupby("id")[f"sentence_run_{algo_choice}"].max().reset_index()
    tmp.columns = ["id", f"nrun_{algo_choice}"]
    sent = pd.merge(sent, tmp, on="id", how="left")

    # Reread
    sent[f"reread_{algo_choice}"] = sent.apply(lambda row: 1 if row[f"nrun_{algo_choice}"] > 1 else 0, axis=1)

    # Reg_in
    tmp = fixin.groupby("id")[f"sent_reg_in2_{algo_choice}"].max().reset_index()
    tmp.columns = ["id", f"reg_in_{algo_choice}"]
    sent = pd.merge(sent, tmp, on="id", how="left")

    # Reg_out
    tmp = fixin.groupby("id")[f"sent_reg_out2_{algo_choice}"].max().reset_index()
    tmp.columns = ["id", f"reg_out_{algo_choice}"]
    sent = pd.merge(sent, tmp, on="id", how="left")

    sent = sent.sort_values(by=f"on_sentence_num_{algo_choice}").reset_index(drop=True)

    # Rate
    sent[f"rate_{algo_choice}"] = round(60000 / (sent[f"total_dur_{algo_choice}"] / sent["num_words_in_sentence"]))

    # Write out
    item = sentitem.copy()

    sent = pd.merge(
        sent,
        item.rename({"in_sentence_number": f"on_sentence_num_{algo_choice}"}, axis=1),
        on=f"on_sentence_num_{algo_choice}",
        how="left",
    )
    sent[f"skip_{algo_choice}"] = 0
    sent.loc[pd.isna(sent[f"nrun_{algo_choice}"]), f"skip_{algo_choice}"] = 1

    names = [
        "subject",
        "trial_id",
        "item",
        "condition",
    ] + [
        c
        for c in [
            f"on_sentence_num_{algo_choice}",
            f"on_sentence_{algo_choice}",
            "num_words_in_sentence",
            f"skip_{algo_choice}",
            f"nrun_{algo_choice}",
            f"reread_{algo_choice}",
            f"reg_in_{algo_choice}",
            f"reg_out_{algo_choice}",
            f"total_n_fixations_{algo_choice}",
            f"total_dur_{algo_choice}",
            f"rate_{algo_choice}",
            f"gopast_{algo_choice}",
            f"gopast_sel_{algo_choice}",
            f"firstrun_skip_{algo_choice}",
            f"firstrun_reg_in_{algo_choice}",
            f"firstrun_reg_out_{algo_choice}",
            f"firstpass_n_fixations_{algo_choice}",
            f"firstpass_dur_{algo_choice}",
            f"firstpass_forward_n_fixations_{algo_choice}",
            f"firstpass_forward_dur_{algo_choice}",
            f"firstpass_reread_n_fixations_{algo_choice}",
            f"firstpass_reread_dur_{algo_choice}",
            f"lookback_n_fixations_{algo_choice}",
            f"lookback_dur_{algo_choice}",
            f"lookfrom_n_fixations_{algo_choice}",
            f"lookfrom_dur_{algo_choice}",
        ]
        if (c in sent.columns and c.replace(f"_{algo_choice}", "") in measures_to_calc)
    ]
    sent = sent[names].copy()
    sent.rename(
        {
            f"on_sentence_num_{algo_choice}": "sentence_number",
            f"on_sentence_{algo_choice}": "sentence",
            "num_words_in_sentence": "number_of_words",
        },
        axis=1,
        inplace=True,
    )

    if save_to_csv:
        subj = fix["subject"].iloc[0]
        trial_id = fix["trial_id"].iloc[0]
        sent.to_csv(RESULTS_FOLDER / f"{subj}_{trial_id}_{algo_choice}_sentence_measures.csv")
    return sent.copy()


def compute_gopast_sentence(fixin, algo_choice):
    # create response vectors
    fixin[f"gopast_{algo_choice}"] = np.nan
    fixin[f"selgopast_{algo_choice}"] = np.nan

    # compute trialid within person
    ias = fixin[f"on_sentence_num_{algo_choice}"].unique()

    # compute measures
    for j in ias:
        min_fixation_number_j = fixin.loc[fixin[f"on_sentence_num_{algo_choice}"] == j, "fixation_number"].min(
            skipna=True
        )
        next_min_fixation_number = (
            fixin.loc[fixin[f"on_sentence_num_{algo_choice}"] > j, "fixation_number"].min(skipna=True)
            if j != ias[-1]
            else float("inf")
        )

        mask = (
            (fixin["fixation_number"] >= min_fixation_number_j)
            & (fixin["fixation_number"] < next_min_fixation_number)
            & (~fixin[f"on_sentence_num_{algo_choice}"].isna())
        )
        fixin.loc[fixin[f"on_sentence_num_{algo_choice}"] == j, f"gopast_{algo_choice}"] = fixin.loc[
            mask, "duration"
        ].sum(skipna=True)

        mask_j = (
            (fixin["fixation_number"] >= min_fixation_number_j)
            & (fixin["fixation_number"] < next_min_fixation_number)
            & (~fixin[f"on_sentence_num_{algo_choice}"].isna())
            & (fixin[f"on_sentence_num_{algo_choice}"] == j)
        )
        fixin.loc[fixin[f"on_sentence_num_{algo_choice}"] == j, f"selgopast_{algo_choice}"] = fixin.loc[
            mask_j, "duration"
        ].sum(skipna=True)

    return fixin


def aggregate_trials(dffix_combined, wordcomb, all_trials_by_subj, algo_choices):
    tmp = dffix_combined.copy()

    trial = tmp.drop_duplicates(subset="subject_trialID", keep="first")
    names = ["subject_trialID", "subject", "trial_id", "item", "condition"]
    trial = trial[names].copy()

    for index, row in trial.iterrows():
        selected_trial = all_trials_by_subj[row["subject"]][row["trial_id"]]
        info_keys = [
            k for k in selected_trial.keys() if k in ["trial_start_time", "trial_end_time", "question_correct"]
        ]
        if row["subject"] in all_trials_by_subj and row["trial_id"] in all_trials_by_subj[row["subject"]]:
            if selected_trial["Fixation Cleaning Stats"]["Discard fixation before or after blinks"]:
                trial.at[index, "blink"] = selected_trial["Fixation Cleaning Stats"][
                    "Number of discarded fixations due to blinks"
                ]
            for key, value in selected_trial.items():
                if key in info_keys:
                    trial.at[index, key] = value

    subdf = wordcomb.copy().loc[:, ["subject_trialID"]].drop_duplicates(subset=["subject_trialID"], keep="first")
    trial = pd.merge(trial, subdf, on="subject_trialID", how="left")
    for sub, subdf in wordcomb.groupby("subject"):
        for trialid, trialdf in subdf.groupby("trial_id"):
            trial.loc[((trial["subject"] == sub) & (trial["trial_id"] == trialid)), "number_of_words_in_trial"] = (
                trialdf["word"].count()
            )
    trial.sort_values(by="subject_trialID", inplace=True)

    if "blink" in tmp.columns:
        blink = tmp.groupby("subject_trialID")["blink"].sum() / 2
        blink = blink.round().reset_index()
        trial = pd.merge(trial, blink, on="subject_trialID", how="left")

    trial["nfix"] = tmp.groupby("subject_trialID")["fixation_number"].agg("count").values
    new_col_dfs = []
    new_col_dfs.append(tmp.groupby("subject_trialID")["duration"].agg("mean").reset_index(name="mean_fix_duration"))

    new_col_dfs.append(tmp.groupby("subject_trialID")["duration"].agg("sum").reset_index(name="total_fix_duration"))
    for algo_choice in algo_choices:
        new_col_dfs.append(
            tmp.groupby("subject_trialID")[f"word_runid_{algo_choice}"]
            .agg("max")
            .reset_index(name=f"nrun_{algo_choice}")
        )
        tmp[f"saccade_length_{algo_choice}"] = tmp[f"word_land_{algo_choice}"] + tmp[f"word_launch_{algo_choice}"]
        new_col_dfs.append(
            tmp[(tmp[f"saccade_length_{algo_choice}"] >= 0) & tmp[f"saccade_length_{algo_choice}"].notna()]
            .groupby("subject_trialID")[f"saccade_length_{algo_choice}"]
            .agg("mean")
            .reset_index(name=f"saccade_length_{algo_choice}")
        )

        word = wordcomb.copy()
        if f"firstrun_skip_{algo_choice}" in wordcomb.columns:
            new_col_dfs.append(
                word.groupby("subject_trialID")[f"firstrun_skip_{algo_choice}"]
                .agg("mean")
                .reset_index(name=f"skip_{algo_choice}")
            )
        if f"refix_{algo_choice}" in wordcomb.columns:
            new_col_dfs.append(
                word.groupby("subject_trialID")[f"refix_{algo_choice}"]
                .agg("mean")
                .reset_index(name=f"refix_{algo_choice}")
            )
        if f"reg_in_{algo_choice}" in wordcomb.columns:
            new_col_dfs.append(
                word.groupby("subject_trialID")[f"reg_in_{algo_choice}"]
                .agg("mean")
                .reset_index(name=f"reg_{algo_choice}")
            )

        if f"firstrun_dur_{algo_choice}" in wordcomb.columns:
            new_col_dfs.append(
                word.groupby("subject_trialID")[f"firstrun_dur_{algo_choice}"]
                .agg("sum")
                .reset_index(name=f"firstpass_{algo_choice}")
            )

        if f"total_fixation_duration_{algo_choice}" in wordcomb.columns:
            new_col_dfs.append(
                (word[f"total_fixation_duration_{algo_choice}"] - word[f"firstrun_dur_{algo_choice}"])
                .groupby(word["subject_trialID"])
                .agg("sum")
                .reset_index(name=f"rereading_{algo_choice}")
            )
    trial = pd.concat(
        [trial.set_index("subject_trialID")] + [df.set_index("subject_trialID") for df in new_col_dfs], axis=1
    ).reset_index()
    trial[f"reading_rate_{algo_choice}"] = (
        60000 / (trial["total_fix_duration"] / trial["number_of_words_in_trial"])
    ).round()

    return trial.copy()


def aggregate_subjects(trials, algo_choices):
    trial_aggregates = trials.groupby("subject")[["nfix", "blink"]].mean().round(3).reset_index()
    trial_aggregates = trial_aggregates.merge(
        trials.groupby("subject")["question_correct"].sum().reset_index(name="n_question_correct"), on="subject"
    )
    trial_aggregates = trial_aggregates.merge(
        trials.groupby("subject")["trial_id"].count().reset_index(name="ntrial"), on="subject"
    )
    for algo_choice in algo_choices:
        cols_to_do = [
            c
            for c in [
                f"saccade_length_{algo_choice}",
                f"reg_{algo_choice}",
                f"mean_fix_duration_{algo_choice}",
                f"total_fix_duration_{algo_choice}",
                f"reading_rate_{algo_choice}",
                f"refix_{algo_choice}",
                f"nrun_{algo_choice}",
                f"skip_{algo_choice}",
            ]
            if c in trials.columns
        ]
        trial_aggregates_temp = trials.groupby("subject")[cols_to_do].mean().round(3).reset_index()
        trial_aggregates = pd.merge(trial_aggregates, trial_aggregates_temp, how="left", on="subject")

    return trial_aggregates
