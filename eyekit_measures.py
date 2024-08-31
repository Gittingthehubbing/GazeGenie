import copy
import eyekit as ek
import numpy as np
import pandas as pd
from PIL import Image
from icecream import ic
import time

ic.configureOutput(includeContext=True)
MEASURES_DICT = {
    "number_of_fixations": [],
    "initial_fixation_duration": [],
    "first_of_many_duration": [],
    "total_fixation_duration": [],
    "gaze_duration": [],
    "go_past_duration": [],
    "second_pass_duration": [],
    "initial_landing_position": [],
    "initial_landing_distance": [],
    "landing_distances": [],
    "number_of_regressions_in": [],
}


def get_fix_seq_and_text_block(
    dffix,
    trial,
    x_txt_start=None,
    y_txt_start=None,
    font_face="Courier New",
    font_size=None,
    line_height=None,
    use_corrected_fixations=True,
    correction_algo="warp",
):
    if use_corrected_fixations and correction_algo is not None:
        fixations_tuples = [
            (
                (x[1]["x"], x[1][f"y_{correction_algo}"], x[1]["corrected_start_time"], x[1]["corrected_end_time"])
                if x[1]["corrected_start_time"] < x[1]["corrected_end_time"]
                else (x[1]["x"], x[1]["y"], x[1]["corrected_start_time"], x[1]["corrected_end_time"] + 1)
            )
            for x in dffix.iterrows()
        ]
    else:
        fixations_tuples = [
            (
                (x[1]["x"], x[1]["y"], x[1]["corrected_start_time"], x[1]["corrected_end_time"])
                if x[1]["corrected_start_time"] < x[1]["corrected_end_time"]
                else (x[1]["x"], x[1]["y"], x[1]["corrected_start_time"], x[1]["corrected_end_time"] + 1)
            )
            for x in dffix.iterrows()
        ]

    if "display_coords" in trial:
        display_coords = trial["display_coords"]
    else:
        display_coords = (0, 0, 1920, 1080)
    screen_size = ((display_coords[2] - display_coords[0]), (display_coords[3] - display_coords[1]))

    try:
        fixation_sequence = ek.FixationSequence(fixations_tuples)
    except Exception as e:
        ic(e)
        ic(f"Creating fixation failed for {trial['trial_id']} {trial['filename']}")
        return None, None, screen_size

    y_diffs = np.unique(trial["line_heights"])
    if len(y_diffs) == 1:
        y_diff = y_diffs[0]
    else:
        y_diff = np.min(y_diffs)
    chars_list = trial["chars_list"]
    max_line = int(chars_list[-1]["assigned_line"])
    words_on_lines = {x: [] for x in range(int(max_line) + 1)}
    [words_on_lines[x["assigned_line"]].append(x["char"]) for x in chars_list]
    sentence_list = ["".join([s for s in v]) for idx, v in words_on_lines.items()]

    if x_txt_start is None:
        x_txt_start = float(chars_list[0]["char_xmin"])
    if y_txt_start is None:
        y_txt_start = float(chars_list[0]["char_ymax"])

    if font_face is None and "font" in trial:
        font_face = trial["font"]
    elif font_face is None:
        font_face = "DejaVu Sans Mono"

    if font_size is None and "font_size" in trial:
        font_size = trial["font_size"]
    elif font_size is None:
        font_size = float(y_diff * 0.333)  # pixel to point conversion
    if line_height is None:
        line_height = float(y_diff)
    textblock_input_dict = dict(
        text=sentence_list,
        position=(float(x_txt_start), float(y_txt_start)),
        font_face=font_face,
        line_height=line_height,
        font_size=font_size,
        anchor="left",
        align="left",
    )
    textblock = ek.TextBlock(**textblock_input_dict)

    ek.io.save(fixation_sequence, f'results/fixation_sequence_eyekit_{trial["trial_id"]}.json', compress=False)
    ek.io.save(textblock, f'results/textblock_eyekit_{trial["trial_id"]}.json', compress=False)

    return fixations_tuples, textblock_input_dict, screen_size


def eyekit_plot(fixations_tuples, textblock_input_dict, screen_size):
    textblock = ek.TextBlock(**textblock_input_dict)
    img = ek.vis.Image(*screen_size)
    img.draw_text_block(textblock)
    for word in textblock.words():
        img.draw_rectangle(word, color="hotpink")
    fixation_sequence = ek.FixationSequence(fixations_tuples)
    img.draw_fixation_sequence(fixation_sequence)
    img.save("temp_eyekit_img.png", crop_margin=200)
    img_png = Image.open("temp_eyekit_img.png")
    return img_png


def plot_with_measure(fixations_tuples, textblock_input_dict, screen_size, measure, use_characters=False):
    textblock = ek.TextBlock(**textblock_input_dict)
    fixation_sequence = ek.FixationSequence(fixations_tuples)

    eyekitplot_img = eyekit_plot(fixations_tuples, textblock_input_dict, screen_size)
    eyekitplot_img = ek.vis.Image(*screen_size)
    eyekitplot_img.draw_text_block(textblock)
    if use_characters:
        measure_results = getattr(ek.measure, measure)(textblock.characters(), fixation_sequence)
        enum = textblock.characters()
    else:
        measure_results = getattr(ek.measure, measure)(textblock.words(), fixation_sequence)
        enum = textblock.words()
    for word in enum:
        eyekitplot_img.draw_rectangle(word, color="lightseagreen")
        x = word.onset
        y = word.y_br - 3
        label = f"{measure_results[word.id]}"
        eyekitplot_img.draw_annotation((x, y), label, color="lightseagreen", font_face="Arial bold", font_size=15)
    eyekitplot_img.draw_fixation_sequence(fixation_sequence, color="gray")
    eyekitplot_img.save("multiline_passage_piccol.png", crop_margin=100)
    img_png = Image.open("multiline_passage_piccol.png")
    return img_png


def get_eyekit_measures(fixations_tuples, textblock_input_dict, trial, get_char_measures=False):
    textblock = ek.TextBlock(**textblock_input_dict)
    fixation_sequence = ek.FixationSequence(fixations_tuples)
    measures = copy.deepcopy(MEASURES_DICT)
    words = []
    for w in textblock.words():
        words.append(w.text)
        for m in measures.keys():
            measures[m].append(getattr(ek.measure, m)(w, fixation_sequence))
    word_measures_df = pd.DataFrame(measures)
    word_measures_df["word_number"] = np.arange(0, len(words))
    word_measures_df["word"] = words

    first_column = word_measures_df.pop("word")
    word_measures_df.insert(0, "word", first_column)
    first_column = word_measures_df.pop("word_number")
    word_measures_df.insert(0, "word_number", first_column)

    if "item" in trial and "item" not in word_measures_df.columns:
        word_measures_df.insert(loc=0, column="item", value=trial["item"])
    if "condition" in trial and "condition" not in word_measures_df.columns:
        word_measures_df.insert(loc=0, column="condition", value=trial["condition"])
    if "trial_id" in trial and "trial_id" not in word_measures_df.columns:
        word_measures_df.insert(loc=0, column="trial_id", value=trial["trial_id"])
    if "subject" in trial and "subject" not in word_measures_df.columns:
        word_measures_df.insert(loc=0, column="subject", value=trial["subject"])
    if get_char_measures:
        measures = copy.deepcopy(MEASURES_DICT)

        characters = []
        for c in textblock.characters():
            characters.append(c.text)
            for m in measures.keys():
                measures[m].append(getattr(ek.measure, m)(c, fixation_sequence))
        character_measures_df = pd.DataFrame(measures)
        character_measures_df["char_number"] = np.arange(0, len(characters))
        character_measures_df["character"] = characters

        first_column = character_measures_df.pop("character")
        character_measures_df.insert(0, "character", first_column)
        first_column = character_measures_df.pop("char_number")
        character_measures_df.insert(0, "char_number", first_column)
    else:
        character_measures_df = None
    return word_measures_df, character_measures_df
