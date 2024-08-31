"""
Partially taken and adapted from: https://github.com/jwcarr/eyekit/blob/1db1913411327b108b87e097a00278b6e50d0751/eyekit/measure.py
Functions for calculating common reading measures, such as gaze duration or initial landing position.
"""

import pandas as pd
from icecream import ic

ic.configureOutput(includeContext=True)


def fix_in_ia(fix_x, fix_y, ia_x_min, ia_x_max, ia_y_min, ia_y_max):
    in_x = ia_x_min <= fix_x <= ia_x_max
    in_y = ia_y_min <= fix_y <= ia_y_max
    if in_x and in_y:
        return True
    else:
        return False


def fix_in_ia_default(fixation, ia_row, prefix):
    return fix_in_ia(
        fixation.x,
        fixation.y,
        ia_row[f"{prefix}_xmin"],
        ia_row[f"{prefix}_xmax"],
        ia_row[f"{prefix}_ymin"],
        ia_row[f"{prefix}_ymax"],
    )


def number_of_fixations_own(trial, dffix, prefix, correction_algo):
    """
    Return the number of fixations on that interest area.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    counts = []
    for cidx, ia_row in ia_df.iterrows():
        count = 0
        for idx, fixation in dffix.iterrows():
            if fix_in_ia(
                fixation.x,
                fixation.y,
                ia_row[f"{prefix}_xmin"],
                ia_row[f"{prefix}_xmax"],
                ia_row[f"{prefix}_ymin"],
                ia_row[f"{prefix}_ymax"],
            ):
                count += 1
        counts.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"number_of_fixations_{correction_algo}": count,
            }
        )
    return pd.DataFrame(counts)


def initial_fixation_duration_own(trial, dffix, prefix, correction_algo):
    """
    The duration of the initial fixation on that interest area for each word.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    durations = []

    for cidx, ia_row in ia_df.iterrows():
        initial_duration = 0
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                initial_duration = fixation.duration
                break  # Exit the loop after finding the initial fixation for the word
        durations.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"initial_fixation_duration_{correction_algo}": initial_duration,
            }
        )

    return pd.DataFrame(durations)


def first_of_many_duration_own(trial, dffix, prefix, correction_algo):
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    durations = []
    for cidx, ia_row in ia_df.iterrows():
        fixation_durations = []
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                fixation_durations.append(fixation.duration)
        if len(fixation_durations) > 1:
            durations.append(
                {
                    f"{prefix}_number": cidx,
                    prefix: ia_row[f"{prefix}"],
                    f"first_of_many_duration_{correction_algo}": fixation_durations[0],
                }
            )
        else:
            durations.append(
                {
                    f"{prefix}_number": cidx,
                    prefix: ia_row[f"{prefix}"],
                    f"first_of_many_duration_{correction_algo}": None,
                }
            )
    if len(durations) > 0:
        return pd.DataFrame(durations)
    else:
        return pd.DataFrame()


def total_fixation_duration_own(trial, dffix, prefix, correction_algo):
    """
    sum duration of all fixations on that interest area.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    durations = []
    for cidx, ia_row in ia_df.iterrows():
        total_duration = 0
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                total_duration += fixation.duration
        durations.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"total_fixation_duration_{correction_algo}": total_duration,
            }
        )
    return pd.DataFrame(durations)


def gaze_duration_own(trial, dffix, prefix, correction_algo):
    """
    Gaze duration is the sum duration of all fixations
    inside an interest area until the area is exited for the first time.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    durations = []
    for cidx, ia_row in ia_df.iterrows():
        duration = 0
        in_ia = False
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                duration += fixation.duration
                in_ia = True
            elif in_ia:
                break
        durations.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"gaze_duration_{correction_algo}": duration,
            }
        )
    return pd.DataFrame(durations)


def go_past_duration_own(trial, dffix, prefix, correction_algo):
    """
    Given an interest area and fixation sequence, return the go-past time on
    that interest area. Go-past time is the sum duration of all fixations from
    when the interest area is first entered until when it is first exited to
    the right, including any regressions to the left that occur during that
    time period (and vice versa in the case of right-to-left text).
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    results = []

    for cidx, ia_row in ia_df.iterrows():
        entered = False
        go_past_time = 0

        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                if not entered:
                    entered = True
                go_past_time += fixation.duration
            elif entered:
                if ia_row[f"{prefix}_xmax"] < fixation.x:  # Interest area has been exited to the right
                    break
                go_past_time += fixation.duration

        results.append(
            {f"{prefix}_number": cidx, prefix: ia_row[f"{prefix}"], f"go_past_duration_{correction_algo}": go_past_time}
        )

    return pd.DataFrame(results)


def second_pass_duration_own(trial, dffix, prefix, correction_algo):
    """
    Given an interest area and fixation sequence, return the second pass
    duration on that interest area for each word.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    durations = []

    for cidx, ia_row in ia_df.iterrows():
        current_pass = None
        next_pass = 1
        pass_duration = 0
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                if current_pass is None:  # first fixation in a new pass
                    current_pass = next_pass
                if current_pass == 2:
                    pass_duration += fixation.duration
            elif current_pass == 1:  # first fixation to exit the first pass
                current_pass = None
                next_pass += 1
            elif current_pass == 2:  # first fixation to exit the second pass
                break
        durations.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"second_pass_duration_{correction_algo}": pass_duration,
            }
        )

    return pd.DataFrame(durations)


def initial_landing_position_own(trial, dffix, prefix, correction_algo):
    """
    initial landing position (expressed in character positions) on that interest area.
    Counting is from 1. Returns `None` if no fixation
    landed on the interest area.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    if prefix == "word":
        chars_df = pd.DataFrame(trial[f"chars_list"])
    else:
        chars_df = None
    results = []
    for cidx, ia_row in ia_df.iterrows():
        landing_position = None
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                if prefix == "char":
                    landing_position = 1
                else:
                    prefix_temp = "char"
                    matched_chars_df = chars_df.loc[
                        (chars_df.char_xmin >= ia_row[f"{prefix}_xmin"])
                        & (chars_df.char_xmax <= ia_row[f"{prefix}_xmax"])
                        & (chars_df.char_ymin >= ia_row[f"{prefix}_ymin"])
                        & (chars_df.char_ymax <= ia_row[f"{prefix}_ymax"]),
                        :,
                    ]  # TODO need to find way to count correct letter number
                    for char_idx, (rowidx, char_row) in enumerate(matched_chars_df.iterrows()):
                        if fix_in_ia_default(fixation, char_row, prefix_temp):
                            landing_position = char_idx + 1  # starts at 1
                            break
                break
        results.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"initial_landing_position_{correction_algo}": landing_position,
            }
        )
    return pd.DataFrame(results)


def initial_landing_distance_own(trial, dffix, prefix, correction_algo):
    """
    Given an interest area and fixation sequence, return the initial landing
    distance on that interest area. The initial landing distance is the pixel
    distance between the first fixation to land in an interest area and the
    left edge of that interest area (or, in the case of right-to-left text,
    the right edge). Technically, the distance is measured from the text onset
    without including any padding. Returns `None` if no fixation landed on the
    interest area.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    distances = []
    for cidx, ia_row in ia_df.iterrows():
        initial_distance = None
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                distance = abs(ia_row[f"{prefix}_xmin"] - fixation.x)
                if initial_distance is None:
                    initial_distance = distance
                    break
        distances.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"initial_landing_distance_{correction_algo}": initial_distance,
            }
        )
    return pd.DataFrame(distances)


def landing_distances_own(trial, dffix, prefix, correction_algo):
    """
    Given an interest area and fixation sequence, return a dataframe with
    landing distances for each word in the interest area.
    """
    ia_df = pd.DataFrame(trial[f"{prefix}s_list"])
    distances = []
    for cidx, ia_row in ia_df.iterrows():
        landing_distances = []
        for idx, fixation in dffix.iterrows():
            if fix_in_ia_default(fixation, ia_row, prefix):
                landing_distance = abs(ia_row[f"{prefix}_xmin"] - fixation.x)
                landing_distances.append(round(landing_distance, ndigits=2))
        distances.append(
            {
                f"{prefix}_number": cidx,
                prefix: ia_row[f"{prefix}"],
                f"landing_distances_{correction_algo}": landing_distances,
            }
        )
    return pd.DataFrame(distances)


def number_of_regressions_in_own(trial, dffix, prefix, correction_algo):
    word_reg_in_count = (
        dffix.groupby([f"on_{prefix}_number_{correction_algo}", f"on_{prefix}_{correction_algo}"])[
            f"{prefix}_reg_in_{correction_algo}"
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                f"on_{prefix}_number_{correction_algo}": f"{prefix}_number",
                f"{prefix}_reg_in_{correction_algo}": f"number_of_regressions_in_{correction_algo}",
                f"on_{prefix}_{correction_algo}": prefix,
            }
        )
    )

    return word_reg_in_count
