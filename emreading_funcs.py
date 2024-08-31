"""Mostly adapted from https://github.com/martin-vasilev/EMreading
Moslty deprecated in favour of alternative methods."""

from icecream import ic
from io import StringIO
import re
import numpy as np
import pandas as pd


def assign_chars_to_words(df):
    df.reset_index(inplace=True, names="index_temp")
    df["wordID"] = ""
    df["char_word"] = -1
    word_list = []
    cols = []
    sent_list = df["sent"].unique()

    for i in range(len(sent_list)):  # for each sentence
        word_list = df[df["sent"] == i]["word"].unique()
        for j in range(len(word_list)):
            cols = df[(df["sent"] == i) & (df["word"] == word_list[j])].index
            df.loc[cols, "wordID"] = "".join(df["char"].loc[cols])
            df.loc[(df["sent"] == i) & (df["word"] == word_list[j]), "char_word"] = [k for k in range(len(cols))]
    df.set_index("index_temp", inplace=True)
    return df


def round_and_int(value):
    if not pd.isna(value):
        return int(round(value))
    else:
        return None


def get_coord_map(coords, x=1920, y=1080):
    """
    Original R version:
    ```R
    # Use stimuli information to create a coordinate map_arr for each pixel on the screen
    # This makes it possible to find exactly what participants were fixating
    coord_map_arr<- function(coords, x=resolution_x, y= resolution_y){

    coords$id<- 1:nrow(coords)
    map_arr<- data.frame(matrix(NA, nrow = y, ncol = x))

    for(i in 1:nrow(coords)){
        map_arr[coords$y1[i]:coords$y2[i],coords$x1[i]:coords$x2[i]]<- coords$id[i]

    }

    return(map_arr)
    }```
    """
    coords.reset_index(drop=True, inplace=True)
    y1 = coords["char_ymin"].map(round_and_int)
    y2 = coords["char_ymax"].map(round_and_int)
    x1 = coords["char_xmin"].map(round_and_int)
    x2 = coords["char_xmax"].map(round_and_int)
    coords["id"] = np.arange(len(coords))
    map_arr = np.full((y, x), np.nan)

    for i in range(len(coords)):
        map_arr[y1[i] : y2[i] + 1, x1[i] : x2[i] + 1] = coords["id"].iloc[i]

    np.sum(pd.isna(map_arr), axis=None)
    return map_arr


def get_char_num_for_each_line(df):
    df.reset_index(inplace=True, names="index_temp")
    df["line_char"] = np.nan
    unq_line = df["assigned_line"].unique()
    for i in unq_line:
        assigned_line = df[df["assigned_line"] == i].index
        df.loc[assigned_line, "line_char"] = range(len(assigned_line))
    df.set_index("index_temp", inplace=True)
    return df


def parse_fix(
    file,
    trial_db,
):

    indexrange = list(range(trial_db["trial_start_idx"], trial_db["trial_end_idx"] + 1))

    sfix_stamps = [i for i in indexrange if re.search(r"(?i)(SFIX)", file[i])]

    efix_stamps = [i for i in indexrange if re.search(r"(?i)EFIX", file[i])]

    if len(sfix_stamps) > (len(efix_stamps) + 1):
        ic(f"length mismatch parse_fix of {len(sfix_stamps) - (len(efix_stamps))}")

    if not sfix_stamps or not efix_stamps:
        raw_fix = None
        return raw_fix
    for safe_num in range(25):
        if efix_stamps[0] < sfix_stamps[0]:
            efix_stamps = efix_stamps[1:]
        elif efix_stamps[-1] <= sfix_stamps[-1]:
            sfix_stamps = sfix_stamps[:-1]
        elif efix_stamps[0] >= sfix_stamps[0]:
            sfix_stamps = sfix_stamps[1:]
        if not (len(efix_stamps) != len(sfix_stamps) and len(efix_stamps) > 1 and len(sfix_stamps) > 1):
            break

    def parse_sacc(string):
        a = string.split("	")
        return float(a[2])

    esacc_flag = [file[f - 1] if "ESACC" in file[f - 1] else None for f in sfix_stamps]
    saccDur = []
    for k in esacc_flag:
        if k is None:
            saccDur.append(None)
        else:
            saccDur.append(parse_sacc(k))

    s_time = [int(file[s].strip().split(" ")[-1]) for s in sfix_stamps]

    e_time = [int(file[s - 1].strip().split(" ")[0]) for s in efix_stamps]
    if len(s_time) != len(e_time):
        if s_time[-1] > e_time[-1]:
            s_time = s_time[:-1]

    fixDur = [e_time[index] - s_time[index] for index in range(len(s_time))]
    fixDur = [e - s for e, s in zip(e_time, s_time)]
    assert ~(np.asarray(fixDur) < 0).any()
    x = [float(file[fidx].split("\t")[3]) for fidx in efix_stamps]
    y = [float(file[fidx].split("\t")[4]) for fidx in efix_stamps]
    blink_stamp = [index for index in indexrange if "EBLINK" in file[index]]
    blink_time = [float(file[index].strip().replace("\t", " ").split(" ")[2]) - 1 for index in blink_stamp]
    index = np.searchsorted(s_time, blink_time, side="right") - 1
    blink = np.zeros((len(s_time)))
    blink[index] = -1
    raw_fix = pd.DataFrame(
        {"s_time": s_time, "e_time": e_time, "fixDur": fixDur, "saccDur": saccDur, "x": x, "y": y, "blink": blink}
    )
    return raw_fix


def process_fix_EM(fix, coords_map, coords, SL):
    resolution_y, resolution_x = coords_map.shape
    loc = None
    raw_fix = pd.DataFrame()
    num_fixations = len(fix)
    SFIX = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    EFIX = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    x = np.full(num_fixations, np.nan)
    y = np.full(num_fixations, np.nan)
    fix_num = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    fix_dur = np.full(num_fixations, None)
    sent = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    line = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    word = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    char_trial = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    char_line = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    word_line = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    max_sent = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    max_word = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    regress = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    blink = pd.array([None] * num_fixations, dtype=pd.BooleanDtype())
    outOfBnds = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    outsideText = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    wordID = np.full(num_fixations, None)
    land_pos = pd.array([None] * num_fixations, dtype=pd.Int64Dtype())
    sacc_len = np.full(num_fixations, np.nan)

    max_sentence = coords["in_sentence_number"].max()

    curr_sent = np.zeros((max_sentence + 1, 2))
    curr_sent[: max_sentence + 1, 0] = np.arange(0, max_sentence + 1)

    if isinstance(coords["index"], str):
        coords["index"] = pd.to_numeric(coords["index"], errors="coerce")

    for j in range(len(fix)):
        if (fix["y"][j] > 0) and (fix["x"][j] > 0) and (fix["y"][j] <= resolution_y) and (fix["x"][j] <= resolution_x):
            loc = coords_map[round(fix["y"][j]), round(fix["x"][j])]
            if pd.isnull(loc):
                loc = None
        else:
            loc = None

        fix_num[j] = j
        fix_dur[j] = fix["duration"][j]
        SFIX[j] = fix["start_uncorrected"][j]
        EFIX[j] = fix["stop_uncorrected"][j]
        x[j] = fix["x"][j]
        y[j] = fix["y"][j]
        blink[j] = fix["blink"][j]

        if x[j] < 1 or x[j] > resolution_x or y[j] < 1 or y[j] > resolution_y:
            outOfBnds[j] = 1
        else:
            outOfBnds[j] = 0
            outsideText[j] = 1 if loc is None else 0

        if fix["x"][j] < 0:
            loc = None
            outOfBnds[j] = 1
            outsideText[j] = 1

        if loc is not None:
            sent[j] = coords["in_sentence_number"][loc]
            line[j] = coords["assigned_line"][loc]
            word[j] = coords["in_word_number"][loc]
            word_line[j] = coords["wordline"][loc]
            char_trial[j] = coords["index"][loc] + 1
            char_line[j] = coords["letline"][loc]
            wordID[j] = coords["in_word"][loc]
            land_pos[j] = coords["letword"][loc]

            if j > 0 and not pd.isna(char_trial[j]) and not pd.isna(char_trial[j - 1]):
                sacc_len[j] = abs(char_trial[j] - char_trial[j - 1])
            else:
                sacc_len[j] = np.nan
        else:
            sent[j] = np.nan
            line[j] = np.nan
            word[j] = np.nan
            word_line[j] = np.nan
            char_trial[j] = np.nan
            char_line[j] = np.nan
            wordID[j] = np.nan
            land_pos[j] = np.nan
            sacc_len[j] = np.nan

        if SL:
            if loc is not None:
                if j == 0:
                    max_sent[j] = sent[j]
                else:
                    max_sent[j] = max_sent[j - 1] if pd.isna(sent[j]) or pd.isna(max_sent[j - 1]) else max_sent[j - 1]
                    if not (pd.isna(max_sent[j]) or pd.isna(sent[j])) and sent[j] > max_sent[j]:
                        max_sent[j] = sent[j]

                if j == 0:
                    max_word[j] = abs(word[j])
                    curr_sent[sent[j] - 1, 1] = abs(word[j])
                else:
                    max_word[j] = (
                        curr_sent[sent[j] - 1, 1]
                        if pd.isna(word[j]) or pd.isna(curr_sent[sent[j] - 1, 1])
                        else curr_sent[sent[j] - 1, 1]
                    )
                    if not (pd.isna(word[j]) or pd.isna(max_word[j])) and abs(word[j]) > curr_sent[sent[j] - 1, 1]:
                        max_word[j] = abs(word[j])
                        curr_sent[sent[j] - 1, 1] = abs(word[j])

                if not (pd.isna(word[j]) or pd.isna(max_word[j])) and abs(word[j]) < max_word[j]:
                    regress[j] = 1
                else:
                    regress[j] = 0

                if j > 0 and not pd.isna(word[j]):
                    if pd.isna(regress[j - 1]):
                        regress[j] = np.nan
                    else:
                        if abs(word[j]) == max_word[j] and regress[j - 1] == 1 and word[j] in np.unique(word[:j]):
                            regress[j] = 1

    raw_fix = pd.DataFrame(
        {
            "start_uncorrected": SFIX,
            "stop_uncorrected": EFIX,
            "x": x,
            "y": y,
            "fixation_number": fix_num,
            "on_sentence_number_EM": sent,
            "line_EM": line,
            "word_EM": word,
            "word_line_EM": word_line,
            "char_trial_EM": char_trial,
            "char_line_EM": char_line,
            "regress_EM": regress,
            "wordID_EM": wordID,
            "land_pos_EM": land_pos,
            "sacc_len_EM": sacc_len,
            "blink_EM": blink,
            "outOfBnds_EM": outOfBnds,
            "outsideText_EM": outsideText,
        }
    )

    fix2 = fix.merge(
        raw_fix,
        on=[
            "start_uncorrected",
            "stop_uncorrected",
            "x",
            "y",
            "fixation_number",
        ],
        how="left",
    )
    return fix2


def RS(i, rawfix, coords, reqYthresh, reqXthresh, Ythresh, Xthresh, threshSimilar):

    if i == 0:
        return 0

    lw = coords["char_xmax"][0] - coords["char_xmin"][0]
    lh = coords["char_ymax"][0] - coords["char_ymin"][0]
    meetXthresh = False
    meetYthresh = False

    leftSacc = rawfix["x"][i] < rawfix["x"][i - 1]
    downSacc = rawfix["y"][i] > rawfix["y"][i - 1]

    if downSacc & reqYthresh:
        Ydiff = lh * Ythresh
        trueYdiff = rawfix["y"][i] - rawfix["y"][i - 1]
        meetYthresh = trueYdiff >= Ydiff

    if leftSacc & reqXthresh:
        Xdiff = lw * Xthresh
        trueXdiff = rawfix["x"][i - 1] - rawfix["x"][i]
        meetXthresh = trueXdiff >= Xdiff

    maxPoints = 1 + 2
    if reqYthresh:
        maxPoints += 1
    if reqXthresh:
        maxPoints += 1

    currPoints = 0
    if leftSacc:
        currPoints = currPoints + (1 / maxPoints)
        if meetXthresh:
            currPoints = currPoints + (1 / maxPoints)

    if downSacc:
        currPoints = currPoints + 2 * (1 / maxPoints)
        if meetYthresh:
            currPoints = currPoints + (1 / maxPoints)

    return round(currPoints, 2)


def reMap(rawfix, i, coords_map, coords, newY=None):
    rawfix.set_index("fixation_number", inplace=True)
    assert i in rawfix.index, "Not in index"
    rawfix.loc[i, "reAligned"] = True
    rawfix.loc[i, "previous_line"] = rawfix.loc[i, "line_EM"]
    rawfix.loc[i, "previous_y"] = rawfix.loc[i, "y"]
    if newY != None:
        rawfix.loc[i, "y"] = newY
    loc = coords_map[round(rawfix.loc[i, "y"]), round(rawfix.loc[i, "x"])]
    if pd.isnull(loc):
        return rawfix
    rawfix.loc[i, "on_sentence_number_EM"] = coords["in_sentence_number"][loc]
    rawfix.loc[i, "word_EM"] = coords["in_word_number"][loc]
    rawfix.loc[i, "line_EM"] = coords["assigned_line"][loc]

    return rawfix.reset_index(drop=False, names=["fixation_number"])


def reAlign(rawfix, coords, coords_map, RSpar):

    ystart = coords["char_ymin"].min()
    yend = coords["char_ymax"].max()
    nlines = coords["assigned_line"].max()
    letterHeight = coords["char_ymax"][0] - coords["char_ymin"][0]
    xstart = pd.DataFrame(columns=["1", "2"])
    xstart["1"] = np.arange(nlines + 1)
    ystart = pd.DataFrame(columns=["1", "2"])
    ystart["1"] = np.arange(nlines + 1)
    xend = pd.DataFrame(columns=["1", "2"])
    xend["1"] = np.arange(nlines + 1)
    yend = pd.DataFrame(columns=["1", "2"])
    yend["1"] = np.arange(nlines + 1)
    rawfix["previous_x"] = np.nan

    for i in coords["assigned_line"].unique():
        a = coords[coords["assigned_line"] == i]
        xstart.loc[i, "2"] = a["char_xmin"].min()
        xend.loc[i, "2"] = a["char_xmax"].max()
        ystart.loc[i, "2"] = a["char_ymin"].min()
        yend.loc[i, "2"] = a["char_ymax"].min()

    lineCenter = ystart["2"] + letterHeight / 2

    rawfix["prob_return_sweep"] = np.nan
    rawfix["prob_interline_saccade"] = np.nan
    rawfix["reAligned"] = False
    rawfix["previous_y"] = np.nan
    rawfix["previous_line"] = np.nan

    for i in range(rawfix.shape[0]):
        rawfix.loc[i, "prob_return_sweep"] = RS(
            i,
            rawfix,
            coords,
            reqYthresh=True,
            reqXthresh=True,
            Ythresh=RSpar[0],
            Xthresh=RSpar[1],
            threshSimilar=RSpar[2],
        )

        if i > 0:
            if (rawfix["prob_return_sweep"][i] < 1) & (rawfix["y"][i] > rawfix["y"][i - 1] + letterHeight / 2):
                rawfix.loc[i, "prob_return_sweep"] = 1

        rawfix.loc[i, "previous_x"] = rawfix["x"][i]
        rawfix.loc[i, "previous_y"] = rawfix["y"][i]

        if i > 0:
            if rawfix["y"][i] < rawfix["y"][i - 1] - letterHeight / 2:
                rawfix.loc[i, "prob_interline_saccade"] = 1
            else:
                rawfix.loc[i, "prob_interline_saccade"] = 0

    RsweepFix = np.sort(
        np.concatenate(
            (np.where(rawfix["prob_return_sweep"] == 1)[0], np.where(rawfix["prob_interline_saccade"] == 1)[0])
        )
    )

    for i in range(len(RsweepFix)):
        if i == 0:
            linePass = rawfix.loc[: RsweepFix[0] - 1]

        elif i >= len(RsweepFix):
            linePass = rawfix.loc[RsweepFix[-1] :]

        else:
            linePass = rawfix.loc[RsweepFix[i - 1] : RsweepFix[i] - 1]

        if linePass.shape[0] == 1:
            continue

        avgYpos = linePass["y"].mean(skipna=True)
        whichLine = min(range(len(lineCenter)), key=lambda index: abs(lineCenter[index] - avgYpos))
        linePass.reset_index(inplace=True, drop=True)
        for j in range(linePass.shape[0]):
            onLine = (linePass["y"][j] >= ystart["2"][whichLine]) & (linePass["y"][j] <= yend["2"][whichLine])

            if not onLine:
                if linePass["y"][j] < ystart["2"][whichLine]:
                    rawfix = reMap(
                        rawfix, linePass.loc[j, "fixation_number"], coords_map, coords, newY=ystart["2"][whichLine] + 5
                    )
                else:
                    rawfix = reMap(
                        rawfix, linePass.loc[j, "fixation_number"], coords_map, coords, newY=yend["2"][whichLine] - 5
                    )
                rawfix.loc[linePass.loc[j, "fixation_number"], "reAligned"] = True
            else:
                rawfix.loc[linePass.loc[j, "fixation_number"], "reAligned"] = False

    return rawfix


def cleanData(
    raw_fix,
    algo_choice,
    removeBlinks=True,
    combineNearbySmallFix=True,
    combineMethod="char",
    combineDist=1,
    removeSmallFix=True,
    smallFixCutoff=80,
    remove_duration_outliers=True,
    outlierMethod="ms",
    outlierCutoff=800,
    keepRS=False,
):

    if combineNearbySmallFix:
        nbefore = raw_fix.shape[0]
        which_comb = []

        for i, _ in enumerate(raw_fix):
            prev_line_same = False
            next_line_same = False

            if (i > 0) and (i < nbefore - 1):
                if combineMethod == "char":
                    if (
                        pd.isna(raw_fix[f"letternum_{algo_choice}"][i])
                        or pd.isna(raw_fix[f"letternum_{algo_choice}"][i - 1])
                        or pd.isna(raw_fix[f"letternum_{algo_choice}"][i + 1])
                    ):
                        continue

                if raw_fix["duration"][i] < smallFixCutoff:
                    if (
                        not pd.isna(raw_fix[f"line_num_{algo_choice}"][i])
                        and not pd.isna(raw_fix[f"line_num_{algo_choice}"][i - 1])
                        and not pd.isna(raw_fix[f"line_num_{algo_choice}"][i + 1])
                    ):

                        if raw_fix[f"line_num_{algo_choice}"][i] == raw_fix[f"line_num_{algo_choice}"][i - 1]:
                            prev_line_same = True
                    if raw_fix[f"line_num_{algo_choice}"][i] == raw_fix[f"line_num_{algo_choice}"][i + 1]:
                        next_line_same = True

                    if combineMethod == "char":
                        prev = abs(raw_fix[f"letternum_{algo_choice}"][i] - raw_fix[f"letternum_{algo_choice}"][i - 1])
                        after = abs(raw_fix[f"letternum_{algo_choice}"][i] - raw_fix[f"letternum_{algo_choice}"][i + 1])

                    else:
                        prev = abs(round(raw_fix["x"][i]) - round(raw_fix["x"][i - 1]))
                        after = abs(round(raw_fix["x"][i]) - round(raw_fix["x"][i + 1]))

                    if prev <= combineDist:
                        which_comb.append(i)

                        if prev_line_same:

                            raw_fix["duration"][i - 1] += raw_fix["duration"][i]

                        if keepRS and (raw_fix["Rtn_sweep"][i] == 1):

                            raw_fix["Rtn_sweep"][i - 1] = 1

                    if after <= combineDist:
                        which_comb.append(i)

                        if next_line_same:

                            raw_fix["duration"][i + 1] += raw_fix["duration"][i]

                        if keepRS and (raw_fix["Rtn_sweep"][i] == 1):

                            raw_fix["Rtn_sweep"][i + 1] = 1

        which_comb = list(set(which_comb))

        if len(which_comb) > 0:
            raw_fix = raw_fix.drop(labels=which_comb, axis=0)
    nstart = raw_fix.shape[0]

    if removeBlinks:
        raw_fix = raw_fix[~raw_fix["blink"]].copy()
    nblink = nstart - raw_fix.shape[0]

    if remove_duration_outliers:
        if outlierMethod == "ms":
            outIndices = np.where(raw_fix["duration"] > outlierCutoff)[0]
            if len(outIndices) > 0:
                raw_fix = raw_fix.drop(outIndices).copy()
        elif outlierMethod == "std":
            nSubCutoff, nOutliers = [], 0
            subM = np.mean(raw_fix["duration"])
            subSTD = np.std(raw_fix["duration"])
            cutoff = subM + outlierCutoff * subSTD
            nSubCutoff.append((len(np.where(raw_fix[raw_fix["duration"] > cutoff])[0])))
            nOutliers = sum(nSubCutoff)

    return raw_fix.reset_index(drop=True)


def get_space(s):
    if len(s) == 0 or s == " ":
        return 1
    else:
        return None


def get_num(string):
    strr = "".join([i for i in string if i.isdigit()])
    if len(strr) > 0:
        return int(strr)
    else:
        ic(string)
        return strr


def parse_itemID(trialid):
    I = re.search(r"I", trialid).start()
    condition = get_num(trialid[:I])

    D = re.search(r"D", trialid).start()
    item = get_num(trialid[I + 1 : D])
    depend = get_num(trialid[D:])

    E = trialid[0]

    return {"trialid": trialid, "condition": condition, "item": item, "depend": depend, "trial_is": E}


def get_coord(str_input):
    string = "\n".join(
        [l.split("\t")[1].strip() for l in str_input if (("DELAY" not in l) & ("BUTTON" not in l) & ("REGION" in l))]
    )

    df = pd.read_table(
        StringIO(string),
        sep=" ",
        names=["X" + str(i) for i in range(1, 12)],
    )
    df.loc[:, ["char_xmin", "char_ymin", "char_xmax", "char_ymax", "X11"]] = df[
        ["char_xmin", "char_ymin", "char_xmax", "char_ymax", "X11"]
    ].apply(pd.to_numeric, errors="coerce")
    df.char = df.char.fillna("")

    a = df[df["char"] == ""].index
    for i in a:
        if "space" not in df.columns:
            df.loc[:, "space"] = None
        df.at[i, "space"] = 1

        if "char_xmin" in df.columns and "char_ymin" in df.columns:
            df.at[i, "char_xmin"], df.at[i, "char_ymin"] = df.at[i, "char_ymin"], df.at[i, "char_xmax"]

        if "char_ymin" in df.columns and "char_xmax" in df.columns:
            df.at[i, "char_ymin"], df.at[i, "char_xmax"] = df.at[i, "char_xmax"], df.at[i, "char_ymax"]

        if "char_xmax" in df.columns and "char_ymax" in df.columns:
            df.at[i, "char_xmax"], df.at[i, "char_ymax"] = df.at[i, "char_ymax"], df.at[i, "X11"]
    df = df.drop(columns=["X1", "X2", "X3", "X5"])
    return df


def map_sent(df):

    sent_bnd = df[(df.char == ".") | (df.char == "?") | (df.char == "!")].index.tolist()

    if len(sent_bnd) > 0:
        sent = pd.Series([-1] * len(df))

        for i, eidx in enumerate(sent_bnd):
            sidx = sent_bnd[i - 1] if i > 0 else 0
            if i == len(sent_bnd) - 1:
                sent.loc[sidx:] = len(sent_bnd) - 1
            else:
                sent.loc[sidx:eidx] = i
        df["sent"] = sent
    else:
        df["sent"] = 1
    return df


def map_line(df):
    df = df[~pd.isnull(df["char_ymin"])].reset_index(names="index_temp")

    lines = sorted(set(df["char_ymin"].values))

    assigned_line = np.array([], dtype=int)

    for i in range(len(lines)):
        loc_lines = np.where(df["char_ymin"].values == lines[i])[0]
        assigned_line = np.concatenate((assigned_line, np.full(len(loc_lines), fill_value=i)))
        df.loc[len(assigned_line) - 1, "space"] = 2

    df["assigned_line"] = assigned_line
    df.set_index("index_temp", inplace=True)

    return df


def map_words(df):
    curr_sent, curr_line, curr_word = 0, 0, 0
    df["space"] == 2

    for i in df.index:
        newSent = curr_sent != df.loc[i, "sent"]
        newLine = curr_line != df.loc[i, "assigned_line"]

        df.loc[i, "word"] = curr_word
        if df.loc[i, "char"] == "" and not newSent:
            curr_word += 1
            df.loc[i, "word"] = curr_word

        elif newLine:
            if df.loc[i, "char"] != ".":
                curr_word += 1
            df.loc[i, "word"] = curr_word
            curr_line += 1

        elif newSent:
            curr_sent += 1
            curr_word = 0
            df.loc[i, "word"] = curr_word

    return df


def get_return_sweeps(raw_fix_new, coords, algo_choice):  # TODO Check if covered by popEye
    currentSent = 0
    currentLine = 0
    maxLine = 0
    inReg = False

    curr_sent = np.zeros((max(coords["in_sentence_number"]) + 1, 4))
    curr_sent[:, 0] = np.arange(0, max(coords["in_sentence_number"]) + 1)

    diff_sent = coords["in_sentence_number"].diff().fillna(0)
    last_words = coords.loc[np.where(diff_sent == 1), "in_word_number"]
    curr_sent[:, 2] = np.append(last_words.values, coords["in_word_number"].iloc[-1])
    for m in range(1, len(raw_fix_new)):
        if not (pd.isna(raw_fix_new["char_line_EM"][m - 1]) or pd.isna(raw_fix_new["char_line_EM"][m])):
            raw_fix_new.at[m, "sacc_len_EM"] = abs(raw_fix_new["char_line_EM"][m] - raw_fix_new["char_line_EM"][m - 1])

        if not pd.isna(raw_fix_new["line_EM"][m]):
            currentLine = raw_fix_new["line_EM"][m]

        if currentLine > maxLine:
            maxLine = currentLine
            raw_fix_new.at[m, "Rtn_sweep"] = 1

            if m < len(raw_fix_new) - 1:
                sameLine = (
                    not (pd.isna(raw_fix_new["line_EM"][m + 1]) or pd.isna(raw_fix_new["line_EM"][m]))
                    and raw_fix_new["line_EM"][m + 1] == raw_fix_new["line_EM"][m]
                )

                if raw_fix_new["x"][m + 1] < raw_fix_new["x"][m]:
                    raw_fix_new.at[m, "Rtn_sweep_type"] = "undersweep" if sameLine else None
                else:
                    raw_fix_new.at[m, "Rtn_sweep_type"] = "accurate" if sameLine else None
            else:
                raw_fix_new.at[m, "Rtn_sweep_type"] = np.nan
        else:
            raw_fix_new.at[m, "Rtn_sweep"] = 0

        if not pd.isna(raw_fix_new["on_sentence_number_EM"][m]):
            if m == 1:
                curr_sent[int(raw_fix_new["on_sentence_number_EM"][m]), 2] = raw_fix_new["word_EM"][m]
                raw_fix_new.at[m, "regress_EM"] = 0
            else:
                if raw_fix_new["word_EM"][m] > curr_sent[int(raw_fix_new["on_sentence_number_EM"][m]), 2]:
                    curr_sent[int(raw_fix_new["on_sentence_number_EM"][m]), 2] = raw_fix_new["word_EM"][m]
                    inReg = False

                if currentSent < raw_fix_new["on_sentence_number_EM"][m]:
                    curr_sent[currentSent, 3] = 1
                    currentSent = raw_fix_new["on_sentence_number_EM"][m]

                if (
                    not pd.isna(raw_fix_new["on_sentence_number_EM"][m - 1])
                    and raw_fix_new["on_sentence_number_EM"][m] > raw_fix_new["on_sentence_number_EM"][m - 1]
                ):
                    curr_sent[int(raw_fix_new["on_sentence_number_EM"][m - 1]), 3] = 1

                if (
                    raw_fix_new["word_EM"][m] < curr_sent[int(raw_fix_new["on_sentence_number_EM"][m]), 2]
                    and curr_sent[int(raw_fix_new["on_sentence_number_EM"][m]), 3] == 0
                ):
                    raw_fix_new.at[m, "regress_EM"] = 1
                    inReg = True
                else:
                    if curr_sent[int(raw_fix_new["on_sentence_number_EM"][m]), 3] == 0:
                        raw_fix_new.at[m, "regress_EM"] = 0

                        if (
                            raw_fix_new["word_EM"][m] == curr_sent[int(raw_fix_new["on_sentence_number_EM"][m]), 2]
                            and inReg
                        ):
                            raw_fix_new.at[m, "regress_EM"] = 1
                    else:
                        raw_fix_new.at[m, "regress_EM"] = 1
                        raw_fix_new.at[m, "regress2nd_EM"] = 1
                        inReg = True
    return raw_fix_new


def word_m_EM(n2):
    sub_list = []
    item_list = []
    cond_list = []
    seq_list = []
    word_list = []
    wordID_list = []
    sent_list = []
    FFD_list = []
    SFD_list = []
    GD_list = []
    TVT_list = []
    nfix1_list = []
    nfix2_list = []
    nfixAll_list = []
    regress_list = []
    o = n2["sent"].unique()
    for k in range(len(o)):
        q = n2[n2["sent"] == o[k]]
        r = sorted(q["word"].unique())

        for l in range(len(r)):
            word_list.append(r[l])
            sub_list.append(n2["sub"].iloc[0])
            item_list.append(n2["item"].iloc[0])
            seq_list.append(n2["seq"].iloc[0])
            cond_list.append(n2["cond"].iloc[0])
            sent_list.append(o[k])

            p = q[q["word"] == r[l]]

            if p.shape[0] == 0:
                FFD_list.append(None)
                SFD_list.append(None)
                GD_list.append(None)
                TVT_list.append(None)
                nfix1_list.append(0)
                nfix2_list.append(0)
                nfixAll_list.append(0)
            else:
                p1 = p[p["regress"] == 0]
                p2 = p[p["regress"] == 1]

                if p1.shape[0] == 0:
                    FFD_list.append(None)
                    SFD_list.append(None)
                    GD_list.append(None)
                elif p1.shape[0] == 1:
                    FFD_list.append(p1["fix_dur"].iloc[0])
                    SFD_list.append(p1["fix_dur"].iloc[0])
                    GD_list.append(p1["fix_dur"].iloc[0])
                else:
                    FFD_list.append(p1["fix_dur"].iloc[0])
                    SFD_list.append(None)
                    GD_list.append(p1["fix_dur"].sum())

                TVT_list.append(p["fix_dur"].sum())
                nfix1_list.append(p1.shape[0])
                nfix2_list.append(p2.shape[0])
                nfixAll_list.append(p1.shape[0] + p2.shape[0])

                wordID_list.append(p["wordID"].iloc[0])

                if nfix2_list[-1] == 0:
                    regress_list.append(0)
                else:
                    regress_list.append(1)

        dataT = pd.DataFrame(
            {
                "sub": sub_list,
                "item": item_list,
                "cond": cond_list,
                "seq": seq_list,
                "word": word_list,
                "wordID": wordID_list,
                "sent": sent_list,
                "FFD": FFD_list,
                "SFD": SFD_list,
                "GD": GD_list,
                "TVT": TVT_list,
                "nfix1": nfix1_list,
                "nfix2": nfix2_list,
                "nfixAll": nfixAll_list,
                "regress": regress_list,
            }
        )

        sub_list = []
        item_list = []
        cond_list = []
        seq_list = []
        word_list = []
        wordID_list = []
        sent_list = []
        FFD_list = []
        SFD_list = []
        GD_list = []
        TVT_list = []
        nfix1_list = []
        nfix2_list = []
        nfixAll_list = []
        regress_list = []

        if "dataN" in locals():
            dataN = pd.concat([dataN, dataT], ignore_index=True)
        else:
            dataN = dataT


def word_measures_EM(data, algo_choice, include_time_stamps=False):
    add_blanks = False

    if "blink" in data.columns:
        required_columns = ["blink", "prev_blink", "after_blink"]
        if all(col in data.columns for col in required_columns):
            if (data["blink"] + data["prev_blink"] + data["after_blink"]).sum() == 0:
                ic("Blinks appear to be already excluded! \n\n")
            else:
                add_blanks = True
                ic("There appears to be valid blink data! We will map blinks to individual words. \n\n")

                regress_blinks = data[data["blink"] == 1 & ~data["regress_EM"].isna()].index

                if len(regress_blinks) < 1:
                    BlinkFixTypeNotMapped = True
                    ic(
                        "Fixation type is not mapped for observations with blinks. Therefore, blinks can't be mapped in terms of 1st and 2nd pass reading."
                    )
                    ic(
                        "Please note that, by default, blink fixation durations will also not be added to fixation duration measures for that word since it's assumed you will delete this word from analysis.\n"
                    )
                    ic("If you need to change this, see settings in the pre-processing function.\n\n")

    data_n = pd.DataFrame()

    o_k = sorted(np.unique(data[f"on_sentence_num_{algo_choice}"]))

    for k, sent_k in enumerate(o_k):
        q_k = data[data[f"on_sentence_num_{algo_choice}"] == sent_k]

        p1_k = q_k[q_k["regress_EM"] == 0].copy()
        p2_k = q_k[q_k["regress_EM"] == 1].copy()

        RS_word = np.nan
        check_next = False

        if max(data[f"line_num_{algo_choice}"]) > 1:
            for z, q_row in q_k.iterrows():
                if not pd.isna(q_row["Rtn_sweep"]):
                    if q_row["Rtn_sweep"] == 1:
                        check_next = True
                        RS_Word = (
                            q_row[f"line_word_{algo_choice}"]
                            if not pd.isna(q_row[f"line_word_{algo_choice}"])
                            else np.nan
                        )
                    elif check_next and (pd.notna(q_row[f"line_word_{algo_choice}"])) and (q_row["regress_EM"]):
                        break

        word_l = []
        sub_l = [data.loc[0, "subject"]] * len(q_k)
        item_l = [data.loc[0, "item"]] * len(q_k)
        cond_l = [1] * len(q_k)

        for l, q_row in q_k.iterrows():
            word_l.append(q_row[f"on_word_number_{algo_choice}"])

            if add_blanks:
                sum_1st_pass = (
                    q_row["blink"]
                    + p1_k[p1_k.index[q_row.name]]["prev_blink"]
                    + p2_k[p2_k.index[q_row.name]]["after_blink"]
                ).sum()

                blinks_l = [0] * len(word_l)
                if sum_1st_pass > 0:
                    blinks_l[l] = 1

        for l, q_row in q_k.iterrows():
            word_line_l = [q_row[f"line_word_{algo_choice}"]]

            line_l = [q_row[f"line_num_{algo_choice}"]]

            if include_time_stamps:
                EFIX_SFD_l = [np.nan]

            for l, q_row in q_k.iterrows():
                word_line_l.append(q_row[f"line_word_{algo_choice}"])
                line_l.append(q_row[f"line_num_{algo_choice}"])

        if include_time_stamps:

            if len(p1_k) > 0:

                if len(p1_k) == 1:
                    EFIX_SFD_l.append(p1_k["stop_uncorrected"][0])

        data_t = pd.DataFrame(
            list(
                zip(
                    sub_l,
                    item_l,
                    cond_l,
                    word_l,
                    line_l,
                )
            ),
            columns=[
                "subject",
                "item",
                "condition",
                f"on_word_number_{algo_choice}",
                f"line_num_{algo_choice}",
                "FFD",
                "SFD",
                "GD",
                "TVT",
                "nfix1",
                "nfix2",
                "nfixAll",
                "regress",
            ],
        )

        if add_blanks:
            data_t["blinks_1stPass"] = blinks_l

        data_n = pd.concat([data_n, data_t], ignore_index=True)

    return data_n
