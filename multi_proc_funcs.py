from icecream import ic
from matplotlib import pyplot as plt
import pathlib as pl
import json
from PIL import Image
from torch.utils.data.dataloader import DataLoader as dl
import matplotlib.patches as patches
from torch.utils.data import Dataset as torch_dset
import torchvision.transforms.functional as tvfunc
import einops as eo
from collections.abc import Iterable
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm
import torch as t
import plotly.express as px
import copy

import yaml
import classic_correction_algos as calgo
import analysis_funcs as anf
import models
import popEye_funcs as pf
from loss_functions import corn_label_from_logits

ic.configureOutput(includeContext=True)

PLOTS_FOLDER = pl.Path("plots")
event_strs = [
    "EFIX",
    "EFIX R",
    "EFIX L",
    "SSACC",
    "ESACC",
    "SFIX",
    "MSG",
    "SBLINK",
    "EBLINK",
    "BUTTON",
    "INPUT",
    "END",
    "START",
    "DISPLAY ON",
]
AVAILABLE_FONTS = [x.name for x in font_manager.fontManager.ttflist]
COLORS = px.colors.qualitative.Alphabet
RESULTS_FOLDER = pl.Path("results")
PLOTS_FOLDER = pl.Path("plots")

DIST_MODELS_FOLDER = pl.Path("models")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_FIX_MEASURES = [
    "letternum",
    "letter",
    "on_word_number",
    "on_word",
    "on_sentence",
    "num_words_in_sentence",
    "on_sentence_num",
    "word_land",
    "line_let",
    "line_word",
    "sac_in",
    "sac_out",
    "word_launch",
    "word_refix",
    "word_reg_in",
    "word_reg_out",
    "sentence_reg_in",
    "word_firstskip",
    "word_run",
    "sentence_run",
    "word_run_fix",
    "word_cland",
]
ALL_FIX_MEASURES = DEFAULT_FIX_MEASURES + [
    "angle_incoming",
    "angle_outgoing",
    "line_let_from_last_letter",
    "sentence_word",
    "line_let_previous",
    "line_let_next",
    "sentence_refix",
    "word_reg_out_to",
    "word_reg_in_from",
    "sentence_reg_out",
    "sentence_reg_in_from",
    "sentence_reg_out_to",
    "sentence_firstskip",
    "word_runid",
    "sentence_runid",
    "word_fix",
    "sentence_fix",
    "sentence_run_fix",
]


class DSet(torch_dset):
    def __init__(
        self,
        in_sequence: t.Tensor,
        chars_center_coords_padded: t.Tensor,
        out_categories: t.Tensor,
        trialslist: list,
        padding_list: list = None,
        padding_at_end: bool = False,
        return_images_for_conv: bool = False,
        im_partial_string: str = "fixations_chars_channel_sep",
        input_im_shape=[224, 224],
    ) -> None:
        super().__init__()

        self.in_sequence = in_sequence
        self.chars_center_coords_padded = chars_center_coords_padded
        self.out_categories = out_categories
        self.padding_list = padding_list
        self.padding_at_end = padding_at_end
        self.trialslist = trialslist
        self.return_images_for_conv = return_images_for_conv
        self.input_im_shape = input_im_shape
        if return_images_for_conv:
            self.im_partial_string = im_partial_string
            self.plot_files = [
                str(x["plot_file"]).replace("fixations_words", im_partial_string) for x in self.trialslist
            ]

    def __getitem__(self, index):

        if self.return_images_for_conv:
            im = Image.open(self.plot_files[index])
            if [im.size[1], im.size[0]] != self.input_im_shape:
                im = tvfunc.resize(im, self.input_im_shape)
            im = tvfunc.normalize(tvfunc.to_tensor(im), IMAGENET_MEAN, IMAGENET_STD)
        if self.chars_center_coords_padded is not None:
            if self.padding_list is not None:
                attention_mask = t.ones(self.in_sequence[index].shape[:-1], dtype=t.long)
                if self.padding_at_end:
                    if self.padding_list[index] > 0:
                        attention_mask[-self.padding_list[index] :] = 0
                else:
                    attention_mask[: self.padding_list[index]] = 0
                if self.return_images_for_conv:
                    return (
                        self.in_sequence[index],
                        self.chars_center_coords_padded[index],
                        im,
                        attention_mask,
                        self.out_categories[index],
                    )
                return (
                    self.in_sequence[index],
                    self.chars_center_coords_padded[index],
                    attention_mask,
                    self.out_categories[index],
                )
            else:
                if self.return_images_for_conv:
                    return (
                        self.in_sequence[index],
                        self.chars_center_coords_padded[index],
                        im,
                        self.out_categories[index],
                    )
                else:
                    return (self.in_sequence[index], self.chars_center_coords_padded[index], self.out_categories[index])

        if self.padding_list is not None:
            attention_mask = t.ones(self.in_sequence[index].shape[:-1], dtype=t.long)
            if self.padding_at_end:
                if self.padding_list[index] > 0:
                    attention_mask[-self.padding_list[index] :] = 0
            else:
                attention_mask[: self.padding_list[index]] = 0
            if self.return_images_for_conv:
                return (self.in_sequence[index], im, attention_mask, self.out_categories[index])
            else:
                return (self.in_sequence[index], attention_mask, self.out_categories[index])
        if self.return_images_for_conv:
            return (self.in_sequence[index], im, self.out_categories[index])
        else:
            return (self.in_sequence[index], self.out_categories[index])

    def __len__(self):
        if isinstance(self.in_sequence, t.Tensor):
            return self.in_sequence.shape[0]
        else:
            return len(self.in_sequence)


def remove_compile_from_model(model):
    if hasattr(model.project, "_orig_mod"):
        model.project = model.project._orig_mod
        model.chars_conv = model.chars_conv._orig_mod
        model.chars_classifier = model.chars_classifier._orig_mod
        model.layer_norm_in = model.layer_norm_in._orig_mod
        model.bert_model = model.bert_model._orig_mod
        model.linear = model.linear._orig_mod
    return model


def remove_compile_from_dict(state_dict):
    for key in list(state_dict.keys()):
        newkey = key.replace("._orig_mod.", ".")
        state_dict[newkey] = state_dict.pop(key)
    return state_dict


def load_model(model_file, cfg):
    try:
        model_loaded = t.load(model_file, map_location="cpu")
        if "hyper_parameters" in model_loaded.keys():
            model_cfg_temp = model_loaded["hyper_parameters"]["cfg"]
        else:
            model_cfg_temp = cfg
        model_state_dict = model_loaded["state_dict"]
    except Exception as e:
        ic(e)
        ic(f"Failed to load {model_file}")
        return None
    model = models.LitModel(
        [1, 500, 3],
        model_cfg_temp["hidden_dim_bert"],
        model_cfg_temp["num_attention_heads"],
        model_cfg_temp["n_layers_BERT"],
        model_cfg_temp["loss_function"],
        1e-4,
        model_cfg_temp["weight_decay"],
        model_cfg_temp,
        model_cfg_temp["use_lr_warmup"],
        model_cfg_temp["use_reduce_on_plateau"],
        track_gradient_histogram=model_cfg_temp["track_gradient_histogram"],
        register_forw_hook=model_cfg_temp["track_activations_via_hook"],
        char_dims=model_cfg_temp["char_dims"],
    )
    model = remove_compile_from_model(model)
    model_state_dict = remove_compile_from_dict(model_state_dict)
    with t.no_grad():
        model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    model.freeze()
    return model


def find_and_load_model(model_date: str):
    model_cfg_file = list(DIST_MODELS_FOLDER.glob(f"*{model_date}*.yaml"))
    if len(model_cfg_file) == 0:
        ic(f"No model cfg yaml found for {model_date}")
        return None, None
    model_cfg_file = model_cfg_file[0]
    with open(model_cfg_file) as f:
        model_cfg = yaml.safe_load(f)

    model_file = list(pl.Path("models").glob(f"*{model_date}*.ckpt"))[0]
    model = load_model(model_file, model_cfg)

    return model, model_cfg


def set_up_models(dist_models_folder):
    out_dict = {}
    dist_models_with_norm = list(dist_models_folder.glob("*normalize_by_line_height_and_width_True*.ckpt"))
    dist_models_without_norm = list(dist_models_folder.glob("*normalize_by_line_height_and_width_False*.ckpt"))
    DIST_MODEL_DATE_WITH_NORM = dist_models_with_norm[0].stem.split("_")[1]

    models_without_norm_df = [find_and_load_model(m_file.stem.split("_")[1]) for m_file in dist_models_without_norm]
    models_with_norm_df = [find_and_load_model(m_file.stem.split("_")[1]) for m_file in dist_models_with_norm]

    model_cfg_without_norm_df = [x[1] for x in models_without_norm_df if x[1] is not None][0]
    model_cfg_with_norm_df = [x[1] for x in models_with_norm_df if x[1] is not None][0]

    models_without_norm_df = [x[0] for x in models_without_norm_df if x[0] is not None]
    models_with_norm_df = [x[0] for x in models_with_norm_df if x[0] is not None]

    ensemble_model_avg = models.EnsembleModel(
        models_without_norm_df, models_with_norm_df, learning_rate=0.0058, use_simple_average=True
    )
    out_dict["ensemble_model_avg"] = ensemble_model_avg

    out_dict["model_cfg_without_norm_df"] = model_cfg_without_norm_df
    out_dict["model_cfg_with_norm_df"] = model_cfg_with_norm_df

    single_DIST_model, single_DIST_model_cfg = find_and_load_model(model_date=DIST_MODEL_DATE_WITH_NORM)
    out_dict["single_DIST_model"] = single_DIST_model
    out_dict["single_DIST_model_cfg"] = single_DIST_model_cfg
    return out_dict


def reorder_columns(
    df,
    cols=[
        "subject",
        "trial_id",
        "item",
        "condition",
        "fixation_number",
        "num",
        "word_number",
        "sentence_number",
        "duration",
        "start_uncorrected",
        "stop_uncorrected",
        "start_time",
        "end_time",
        "corrected_start_time",
        "corrected_end_time",
        "dX",
        "dY",
    ],
):
    existing_cols = [col for col in cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in cols]
    return df[existing_cols + other_cols]


def nan_or_int_minus_one(x):
    if not pd.isna(x):
        return int(x - 1.0)
    else:
        return pd.NA


def add_popEye_cols_to_chars_df(chars_df):

    if "letternum" not in chars_df.columns or "letline" not in chars_df.columns:
        chars_df.reset_index(drop=False, inplace=True)
        chars_df.rename({"index": "letternum"}, axis=1, inplace=True)
        chars_df.loc[:, "letline"] = -1
        chars_df["wordline"] = (
            chars_df.groupby("assigned_line")["in_word_number"].rank(method="dense").map(nan_or_int_minus_one)
        )
        chars_df["wordsent"] = (
            chars_df.groupby("in_sentence_number")["in_word_number"].rank(method="dense").map(nan_or_int_minus_one)
        )
        chars_df["letword"] = (
            chars_df.groupby("in_word_number")["letternum"].rank(method="dense").map(nan_or_int_minus_one)
        )
        for line_idx in chars_df.assigned_line.unique():
            chars_df.loc[chars_df.assigned_line == line_idx, "letline"] = (
                chars_df.loc[chars_df.assigned_line == line_idx, "char"].reset_index().index
            )
    return chars_df


def add_boxes_to_ax(
    chars_list,
    ax,
    font_to_use="DejaVu Sans Mono",
    fontsize=21,
    prefix="char",
    box_annotations: list = None,
    edgecolor="grey",
    linewidth=0.8,
):
    if box_annotations is None:
        enum = chars_list
    else:
        enum = zip(chars_list, box_annotations)
    for v in enum:
        if box_annotations is not None:
            v, annot_text = v
        x0, y0 = v[f"{prefix}_xmin"], v[f"{prefix}_ymin"]
        xdiff, ydiff = v[f"{prefix}_xmax"] - v[f"{prefix}_xmin"], v[f"{prefix}_ymax"] - v[f"{prefix}_ymin"]
        ax.add_patch(Rectangle((x0, y0), xdiff, ydiff, edgecolor=edgecolor, facecolor="none", lw=linewidth, alpha=0.4))
        if box_annotations is not None:
            ax.annotate(
                str(annot_text),
                (x0 + xdiff / 2, y0),
                horizontalalignment="center",
                verticalalignment="center",
                fontproperties=FontProperties(family=font_to_use, style="normal", size=fontsize / 1.5),
            )


def add_text_to_ax(
    chars_list,
    ax,
    font_to_use="DejaVu Sans Mono",
    fontsize=21,
    prefix="char",
):
    font_props = FontProperties(family=font_to_use, style="normal", size=fontsize)
    enum = chars_list
    for v in enum:
        ax.text(
            v[f"{prefix}_x_center"],
            v[f"{prefix}_y_center"],
            v[prefix],
            horizontalalignment="center",
            verticalalignment="center",
            fontproperties=font_props,
        )


def set_font_from_chars_list(trial):

    if "chars_list" in trial:
        chars_df = pd.DataFrame(trial["chars_list"])
        line_diffs = np.diff(chars_df.char_y_center.unique())
        y_diffs = np.unique(line_diffs)
        if len(y_diffs) == 1:
            y_diff = y_diffs[0]
        else:
            y_diff = np.min(y_diffs)
        y_diff = round(y_diff * 2) / 2

    else:
        y_diff = 1 / 0.333 * 18
    font_size = y_diff * 0.333  # pixel to point conversion
    return round((font_size) * 4, ndigits=0) / 4


def get_plot_props(trial, available_fonts):
    if "font" in trial.keys():
        font = trial["font"]
        font_size = trial["font_size"]
        if font not in available_fonts:
            font = "DejaVu Sans Mono"
    else:
        font = "DejaVu Sans Mono"
        font_size = 21
    dpi = 96
    if "display_coords" in trial.keys() and trial["display_coords"] is not None:
        screen_res = (trial["display_coords"][2], trial["display_coords"][3])
    else:
        screen_res = (1920, 1080)
    return font, font_size, dpi, screen_res


def get_font_and_font_size_from_trial(trial):
    font_face, font_size, dpi, screen_res = get_plot_props(trial, AVAILABLE_FONTS)

    if font_size is None and "font_size" in trial:
        font_size = trial["font_size"]
    elif font_size is None:
        font_size = set_font_from_chars_list(trial)
    return font_face, font_size


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def matplotlib_plot_df(
    dffix,
    trial,
    algo_choice,
    dffix_no_clean=None,
    desired_dpi=300,
    fix_to_plot=[],
    stim_info_to_plot=["Characters", "Word boxes"],
    box_annotations: list = None,
    font=None,
    use_duration_arrow_sizes=True,
):
    chars_df = pd.DataFrame(trial["chars_list"]) if "chars_list" in trial else None

    if chars_df is not None:
        font_face, font_size = get_font_and_font_size_from_trial(trial)
        font_size = font_size * 0.65
    else:
        ic("No character or word information available to plot")

    if "display_coords" in trial:
        desired_width_in_pixels = trial["display_coords"][2] + 1
        desired_height_in_pixels = trial["display_coords"][3] + 1
    else:
        desired_width_in_pixels = 1920
        desired_height_in_pixels = 1080

    figure_width = desired_width_in_pixels / desired_dpi
    figure_height = desired_height_in_pixels / desired_dpi

    fig = plt.figure(figsize=(figure_width, figure_height), dpi=desired_dpi)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    if font is None:
        if "font" in trial and trial["font"] in AVAILABLE_FONTS:
            font_to_use = trial["font"]
        else:
            font_to_use = "DejaVu Sans Mono"
    else:
        font_to_use = font
    if "font_size" in trial:
        font_size = trial["font_size"]
    else:
        font_size = 20

    if "Words" in stim_info_to_plot and "words_list" in trial:
        add_text_to_ax(
            trial["words_list"],
            ax,
            font_to_use,
            prefix="word",
            fontsize=font_size / 3.89,
        )
    if "Word boxes" in stim_info_to_plot and "words_list" in trial:
        add_boxes_to_ax(
            trial["words_list"],
            ax,
            font_to_use,
            prefix="word",
            fontsize=font_size / 3.89,
            box_annotations=box_annotations,
            edgecolor="black",
            linewidth=0.9,
        )

    if "Characters" in stim_info_to_plot and "chars_list" in trial:
        add_text_to_ax(
            trial["chars_list"],
            ax,
            font_to_use,
            prefix="char",
            fontsize=font_size / 3.89,
        )
    if "Character boxes" in stim_info_to_plot and "chars_list" in trial:
        add_boxes_to_ax(
            trial["chars_list"],
            ax,
            font_to_use,
            prefix="char",
            fontsize=font_size / 3.89,
            box_annotations=box_annotations,
        )

    if "Uncorrected Fixations" in fix_to_plot and dffix_no_clean is None:
        if use_duration_arrow_sizes and "duration" in dffix.columns:
            duration_scaled = dffix.duration - dffix.duration.min()
            duration_scaled = (((duration_scaled / duration_scaled.max()) - 0.5) * 3).values
            durations = sigmoid(duration_scaled) * 50 * 0.5
        if use_duration_arrow_sizes:
            ax.plot(
                dffix.x,
                dffix.y,
                label="Raw fixations",
                color="blue",
                alpha=0.5,
            )
            add_arrow_annotations(dffix, "y", ax, "blue", durations[:-1])
        else:
            ax.plot(
                dffix.x,
                dffix.y,
                label="Remaining fixations",
                color="blue",
                alpha=0.5,
            )
            add_arrow_annotations(dffix, "y", ax, "blue", 4)

    if dffix_no_clean is not None and "Uncorrected Fixations" in fix_to_plot:

        ax.plot(
            dffix_no_clean.x,
            dffix_no_clean.y,
            # marker='.',
            label="All fixations",
            color="k",
            alpha=0.5,
            lw=1,
        )
        add_arrow_annotations(dffix_no_clean, "y", ax, "k", 4)
        if "was_discarded_due_blinks" in dffix_no_clean.columns and dffix_no_clean["was_discarded_due_blinks"].any():
            discarded_blink_fix = dffix_no_clean.loc[dffix_no_clean["was_discarded_due_blinks"], :].copy()
            ax.scatter(
                discarded_blink_fix.x,
                discarded_blink_fix.y,
                s=12,
                label="Discarded due to blinks",
                lw=1.5,
                edgecolors="orange",
                facecolors="none",
            )
        if (
            "was_discarded_due_to_long_duration" in dffix_no_clean.columns
            and dffix_no_clean["was_discarded_due_to_long_duration"].any()
        ):
            discarded_long_fix = dffix_no_clean.loc[dffix_no_clean["was_discarded_due_to_long_duration"], :].copy()
            ax.scatter(
                discarded_long_fix.x,
                discarded_long_fix.y,
                s=18,
                label="Overly long fixations",
                lw=0.8,
                edgecolors="purple",
                facecolors="none",
            )
        if "was_merged" in dffix_no_clean.columns:
            merged_fix = dffix_no_clean.loc[dffix_no_clean["was_merged"], :].copy()
            if not merged_fix.empty:
                ax.scatter(
                    merged_fix.x,
                    merged_fix.y,
                    s=7,
                    label="Merged short fixations",
                    lw=1,
                    edgecolors="red",
                    facecolors="none",
                )
        if "was_discarded_outside_text" in dffix_no_clean.columns:
            was_discarded_outside_text_fix = dffix_no_clean.loc[dffix_no_clean["was_discarded_outside_text"], :].copy()
            if not was_discarded_outside_text_fix.empty:
                ax.scatter(
                    was_discarded_outside_text_fix.x,
                    was_discarded_outside_text_fix.y,
                    s=8,
                    label="Outside text fixations",
                    lw=1.2,
                    edgecolors="blue",
                    facecolors="none",
                )
        if "was_discarded_short_fix" in dffix_no_clean.columns:
            was_discarded_short_fix_fix = dffix_no_clean.loc[dffix_no_clean["was_discarded_short_fix"], :].copy()
            if not was_discarded_short_fix_fix.empty:
                ax.scatter(
                    was_discarded_short_fix_fix.x,
                    was_discarded_short_fix_fix.y,
                    label="Discarded short fixations",
                    s=9,
                    lw=1.5,
                    edgecolors="green",
                    facecolors="none",
                )
    if "Corrected Fixations" in fix_to_plot:
        if isinstance(algo_choice, list):
            algo_choices = algo_choice
            repeats = range(len(algo_choice))
        else:
            algo_choices = [algo_choice]
            repeats = range(1)
        for algoIdx in repeats:
            algo_choice = algo_choices[algoIdx]
            if f"y_{algo_choice}" in dffix.columns:
                ax.plot(
                    dffix.x,
                    dffix.loc[:, f"y_{algo_choice}"],
                    label=algo_choice,
                    color=COLORS[algoIdx],
                    alpha=0.6,
                    linewidth=0.6,
                )

                add_arrow_annotations(dffix, f"y_{algo_choice}", ax, COLORS[algoIdx], 6)

    ax.set_xlim((0, desired_width_in_pixels))
    ax.set_ylim((0, desired_height_in_pixels))
    ax.invert_yaxis()
    if "Corrected Fixations" in fix_to_plot or "Uncorrected Fixations" in fix_to_plot:
        ax.legend(prop={"size": 5})

    return fig, desired_width_in_pixels, desired_height_in_pixels


def add_arrow_annotations(dffix, y_col, ax, color, size):
    x = dffix.x.values

    y = dffix.loc[:, y_col].values

    x = x[:-1]
    y = y[:-1]
    dX = -(x[1:] - x[:-1])
    dY = -(y[1:] - y[:-1])

    xpos = x[1:]
    ypos = y[1:]
    if isinstance(size, Iterable):
        use_size_idx = True
    else:
        use_size_idx = False
        s = size
    for fidx, (X, Y, dX, dY) in enumerate(zip(xpos, ypos, dX, dY)):
        if use_size_idx:
            s = size[fidx]
        ax.annotate(
            "",
            xytext=(X + 0.001 * dX, Y + 0.001 * dY),
            xy=(X, Y),
            arrowprops=dict(arrowstyle="fancy", color=color),
            size=s,
            alpha=0.3,
        )


def plot_saccade_df(fix_df, sac_df, trial, show_numbers=False, add_lines_to_fix_df=False):
    stim_only_fig, _, _ = matplotlib_plot_df(
        fix_df,
        trial,
        None,
        dffix_no_clean=None,
        desired_dpi=300,
        fix_to_plot=[],
        stim_info_to_plot=["Characters", "Word boxes"],
        box_annotations=None,
        font=None,
    )
    if stim_only_fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
        invert_ax_needed = True
    else:
        fig = stim_only_fig
        ax = fig.axes[0]
        invert_ax_needed = False

    def plot_arrow(x1, y1, x2, y2, scale_factor):
        """Plot an arrow from (x1,y1) to (x2,y2) with adjustable size"""
        ax.arrow(
            x1,
            y1,
            (x2 - x1),
            (y2 - y1),
            color="k",
            alpha=0.7,
            length_includes_head=True,
            width=3 * scale_factor,
            head_width=15 * scale_factor,
            head_length=15 * scale_factor,
        )

    xs = sac_df["xs"].values
    ys = sac_df["ys"].values
    xe = sac_df["xe"].values
    ye = sac_df["ye"].values
    extent = np.sqrt((xs.min() - xe.max()) ** 2 + (ys.min() - ye.max()) ** 2)
    scale_factor = 0.0005 * extent
    for i in range(len(xs)):
        plot_arrow(xs[i], ys[i], xe[i], ye[i], scale_factor=scale_factor)
    if add_lines_to_fix_df:
        plotfunc = ax.plot
    else:
        plotfunc = ax.scatter
    if "x" in fix_df.columns:
        plotfunc(fix_df["x"], fix_df["y"], marker=".")
    else:
        plotfunc(fix_df["xs"], fix_df["ys"], marker=".")

    if invert_ax_needed:
        ax.invert_yaxis()
    if show_numbers:
        size = 8 * scale_factor

        xytext = (
            1,
            -1,
        )
        for index, row in fix_df.iterrows():
            ax.annotate(
                index,
                xy=(row["x"], row["y"]),
                textcoords="offset points",
                ha="center",
                xytext=xytext,
                va="bottom",
                color="k",
                size=size,
            )

        for index, row in sac_df.iterrows():
            ax.annotate(
                index,
                xy=(row["xs"], row["ys"]),
                textcoords="offset points",
                ha="center",
                xytext=xytext,
                va="top",
                color="r",
                size=size,
            )
    return fig


def get_events_df_from_lines_and_trial_selection(trial, trial_lines, discard_fixations_without_sfix):

    line_dicts = []
    fixations_dicts = []
    events_dicts = []
    blink_started = False

    fixation_started = False
    esac_count = 0
    efix_count = 0
    sfix_count = 0
    sblink_count = 0
    eblink_times = []

    eye_to_use = "R"
    for l in trial_lines:
        if "EFIX R" in l:
            eye_to_use = "R"
            break
        elif "EFIX L" in l:
            eye_to_use = "L"
            break
    for l in trial_lines:
        parts = [x.strip() for x in l.split("\t")]
        if f"EFIX {eye_to_use}" in l:
            efix_count += 1
            if fixation_started:
                had_SFIX_before_it = True
                if parts[1] == "." and parts[2] == ".":
                    continue
                fixation_started = False
            else:
                had_SFIX_before_it = False
            fix_dict = {
                "fixation_number": efix_count,
                "start_time": float(pd.to_numeric(parts[0].split()[-1].strip(), errors="coerce")),
                "end_time": float(pd.to_numeric(parts[1].strip(), errors="coerce")),
                "duration": float(pd.to_numeric(parts[2].strip(), errors="coerce")),
                "x": float(pd.to_numeric(parts[3].strip(), errors="coerce")),
                "y": float(pd.to_numeric(parts[4].strip(), errors="coerce")),
                "pupil_size": float(pd.to_numeric(parts[5].strip(), errors="coerce")),
                "had_SFIX_before_it": had_SFIX_before_it,
                "msg": "FIX",
            }
            if not discard_fixations_without_sfix or had_SFIX_before_it:
                fixations_dicts.append(fix_dict)
                events_dicts.append(
                    {
                        "num": efix_count - 1,
                        "start": float(pd.to_numeric(parts[0].split()[-1].strip(), errors="coerce")),
                        "stop": float(pd.to_numeric(parts[1].strip(), errors="coerce")),
                        "duration": float(pd.to_numeric(parts[2].strip(), errors="coerce")),
                        "xs": float(pd.to_numeric(parts[3].strip(), errors="coerce")),
                        "xe": None,
                        "ys": float(pd.to_numeric(parts[4].strip(), errors="coerce")),
                        "ye": None,
                        "ampl": None,
                        "pv": None,
                        "pupil_size": float(pd.to_numeric(parts[5].strip(), errors="coerce")),
                        "msg": "FIX",
                    }
                )
            if len(fixations_dicts) >= 2:
                assert fixations_dicts[-1]["start_time"] > fixations_dicts[-2]["start_time"], "start times not in order"
        elif f"SFIX {eye_to_use}" in l:
            sfix_count += 1
            fixation_started = True
        elif f"SBLINK {eye_to_use}" in l:
            sblink_count += 1
            blink_started = True
        elif f"EBLINK {eye_to_use}" in l:
            blink_started = False
            blink_dict = {
                "num": len(eblink_times),
                "start": float(pd.to_numeric(parts[0].split()[-1].strip(), errors="coerce")),
                "stop": float(pd.to_numeric(parts[1].strip(), errors="coerce")),
                "duration": float(pd.to_numeric(parts[2].strip(), errors="coerce")),
                "xs": None,
                "xe": None,
                "ys": None,
                "ye": None,
                "ampl": None,
                "pv": None,
                "pupil_size": None,
                "msg": "BLINK",
            }
            events_dicts.append(blink_dict)
            eblink_times.append(float(pd.to_numeric(parts[-1], errors="coerce")))
        elif "ESACC" in l:
            sac_dict = {
                "num": esac_count,
                "start": float(pd.to_numeric(parts[0].split()[-1].strip(), errors="coerce")),
                "stop": float(pd.to_numeric(parts[1].strip(), errors="coerce")),
                "duration": float(pd.to_numeric(parts[2].strip(), errors="coerce")),
                "xs": float(pd.to_numeric(parts[3].strip(), errors="coerce")),
                "ys": float(pd.to_numeric(parts[4].strip(), errors="coerce")),
                "xe": float(pd.to_numeric(parts[5].strip(), errors="coerce")),
                "ye": float(pd.to_numeric(parts[6].strip(), errors="coerce")),
                "ampl": float(pd.to_numeric(parts[7].strip(), errors="coerce")),
                "pv": float(pd.to_numeric(parts[8].strip(), errors="coerce")),
                "pupil_size": None,
                "msg": "SAC",
            }
            events_dicts.append(sac_dict)
            esac_count += 1
        if not blink_started and not any([True for x in event_strs if x in l]):
            if len(parts) < 3 or (parts[1] == "." and parts[2] == "."):
                continue
            line_dicts.append(
                {
                    "idx": float(pd.to_numeric(parts[0].strip(), errors="coerce")),
                    "x": float(pd.to_numeric(parts[1].strip(), errors="coerce")),
                    "y": float(pd.to_numeric(parts[2].strip(), errors="coerce")),
                    "p": float(pd.to_numeric(parts[3].strip(), errors="coerce")),
                    "part_of_fixation": fixation_started,
                    "fixation_number": sfix_count,
                    "part_of_blink": blink_started,
                    "blink_number": sblink_count,
                }
            )

    trial["eblink_times"] = eblink_times
    df = pd.DataFrame(line_dicts)
    df["x_smoothed"] = np.convolve(df.x, np.ones((5,)) / 5, mode="same")  # popEye smoothes this way
    df["y_smoothed"] = np.convolve(df.y, np.ones((5,)) / 5, mode="same")
    df["time"] = df["idx"] - df["idx"].iloc[0]
    df = pf.compute_velocity(df)
    events_df = pd.DataFrame(events_dicts)
    events_df["start_uncorrected"] = events_df.start
    events_df["stop_uncorrected"] = events_df.stop
    events_df["start"] = events_df.start - trial["trial_start_time"]
    events_df["stop"] = events_df.stop - trial["trial_start_time"]
    events_df["start"] = events_df["start"].clip(0, events_df["start"].max())
    events_df.sort_values(by="start", inplace=True)  # Needed because blinks can happen during other events, I think
    events_df.reset_index(drop=True, inplace=True)
    events_df = pf.event_long(events_df)
    events_df["duration"] = events_df["stop"] - events_df["start"]

    trial["efix_count"] = efix_count
    trial["eye_to_use"] = eye_to_use
    trial["sfix_count"] = sfix_count
    trial["sblink_count"] = sblink_count
    return trial, df, events_df


def add_default_font_and_character_props_to_state(trial):
    chars_list = trial["chars_list"]
    chars_df = pd.DataFrame(trial["chars_list"])
    line_diffs = np.diff(chars_df.char_y_center.unique())
    y_diffs = np.unique(line_diffs)
    if len(y_diffs) > 1:
        y_diff = np.min(y_diffs)
    else:
        y_diff = y_diffs[0]

    y_diff = round(y_diff * 2) / 2
    x_txt_start = chars_list[0]["char_xmin"]
    y_txt_start = chars_list[0]["char_y_center"]

    font_face, font_size = get_font_and_font_size_from_trial(trial)

    line_height = y_diff
    return y_diff, x_txt_start, y_txt_start, font_face, font_size, line_height


def get_raw_events_df_and_trial(trial, discard_fixations_without_sfix):
    fname = pl.Path(trial["filename"]).stem
    trial_id = trial["trial_id"]
    trial_lines = trial.pop("trial_lines")

    trial["plot_file"] = str(PLOTS_FOLDER.joinpath(f"{fname}_{trial_id}_2ndInput_chars_channel_sep.png"))

    trial, df, events_df = get_events_df_from_lines_and_trial_selection(
        trial, trial_lines, discard_fixations_without_sfix
    )
    trial["gaze_df"] = df
    font, font_size, dpi, screen_res = get_plot_props(trial, AVAILABLE_FONTS)
    trial["font"] = font
    trial["font_size"] = font_size
    trial["dpi"] = dpi
    trial["screen_res"] = screen_res
    if "chars_list" in trial:
        chars_df = pd.DataFrame(trial["chars_list"])

        chars_df = add_popEye_cols_to_chars_df(chars_df)

        if "index" not in chars_df.columns:
            chars_df.reset_index(inplace=True)
        trial["chars_df"] = chars_df.to_dict()
        trial["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())
    return reorder_columns(events_df), trial


def get_outlier_indeces(
    dffix, chars_df, x_thres_in_chars, y_thresh_in_heights, xcol, ycol, letter_width_avg, line_heights_avg
):
    indeces_out = []
    for linenum, line_chars_subdf in chars_df.groupby("assigned_line"):
        left = line_chars_subdf["char_xmin"].min()
        right = line_chars_subdf["char_xmax"].max()
        top = line_chars_subdf["char_ymin"].min()
        bottom = line_chars_subdf["char_ymax"].max()
        left_min = left - (x_thres_in_chars * letter_width_avg)
        right_max = right + (x_thres_in_chars * letter_width_avg)
        top_max = top - (line_heights_avg * y_thresh_in_heights)
        bottom_min = bottom + (line_heights_avg * y_thresh_in_heights)
        indeces_out_line = []
        indeces_out_line.extend(list(dffix.loc[dffix[xcol] < left_min, :].index))
        indeces_out_line.extend(list(dffix.loc[dffix[xcol] > right_max, :].index))
        indeces_out_line.extend(list(dffix.loc[dffix[ycol] < top_max, :].index))
        indeces_out_line.extend(list(dffix.loc[dffix[ycol] > bottom_min, :].index))
        indeces_out_line_set = set(indeces_out_line)
        indeces_out.append(indeces_out_line_set)
    return list(set.intersection(*indeces_out))


def get_distance_between_fixations_in_characters_and_recalc_duration(
    fix, letter_width_avg, start_colname="start", stop_colname="stop", xcol="xs"
):
    fix.reset_index(drop=True, inplace=True)
    fix.loc[:, "duration"] = fix[stop_colname] - fix[start_colname]
    fix.loc[:, "distance_in_char_widths"] = 0.0
    for i in range(1, len(fix)):
        fix.loc[i, "distance_in_char_widths"] = np.round(
            np.abs(fix.loc[i, xcol] - fix.loc[i - 1, xcol]) / letter_width_avg, decimals=3
        )
    return fix


def clean_fixations_popeye_no_sacc(fix, trial, duration_threshold, distance_threshold):
    if "letter_width_avg" in trial:
        letter_width_avg = trial["letter_width_avg"]
    else:
        letter_width_avg = 12

    stop_time_col, start_time_col = get_time_cols(fix)
    if "xs" in fix.columns:
        x_colname = "xs"
        y_colname = "ys"
    else:
        x_colname = "x"
        y_colname = "y"
    if "blink" not in fix.columns:
        fix["blink"] = 0
    fix.dropna(subset=[x_colname, y_colname], how="any", axis=0, inplace=True)
    fix.reset_index(drop=True, inplace=True)
    fix = get_distance_between_fixations_in_characters_and_recalc_duration(
        fix, letter_width_avg, start_time_col, stop_time_col, x_colname
    )

    fix["num"] = np.arange(len(fix), dtype=int)
    i = 0
    while i <= len(fix) - 1:

        merge_before = False
        merge_after = False

        if fix["duration"].iloc[i] <= duration_threshold:

            # check fixation n - 1
            if i > 1:
                if (
                    fix["duration"].iloc[i - 1] > duration_threshold
                    and fix["blink"].iloc[i - 1] == 0
                    and fix["distance_in_char_widths"].iloc[i] <= distance_threshold
                ):
                    merge_before = True
            # check fixation n + 1
            if i < len(fix) - 1:
                if (
                    fix["duration"].iloc[i + 1] > duration_threshold
                    and fix["blink"].iloc[i + 1] == 0
                    and fix["distance_in_char_widths"].iloc[i + 1] <= distance_threshold
                ):
                    merge_after = True

            # check merge.status
            if merge_before and not merge_after:
                merge = -1
            elif not merge_before and merge_after:
                merge = 1
            elif not merge_before and not merge_after:
                merge = 0
            elif merge_before and merge_after:
                if fix["duration"].iloc[i - 1] >= fix["duration"].iloc[i + 1]:
                    merge = -1
                else:
                    merge = 1

        # close if above duration threshold
        else:
            merge = 0

        if merge == 0:
            i += 1

        elif merge == -1:

            fix.loc[i - 1, stop_time_col] = fix.loc[i, stop_time_col]
            fix.loc[i - 1, x_colname] = round((fix.loc[i - 1, x_colname] + fix.loc[i, x_colname]) / 2)
            fix.loc[i - 1, y_colname] = round((fix.loc[i - 1, y_colname] + fix.loc[i, y_colname]) / 2)

            fix = fix.drop(i, axis=0)
            fix.reset_index(drop=True, inplace=True)

            start = fix[start_time_col].iloc[i - 1]
            stop = fix[stop_time_col].iloc[i - 1]

            fix = get_distance_between_fixations_in_characters_and_recalc_duration(
                fix, letter_width_avg, start_time_col, stop_time_col, x_colname
            )

        elif merge == 1:
            fix.loc[i + 1, start_time_col] = fix.loc[i, start_time_col]
            fix.loc[i + 1, x_colname] = round((fix.loc[i, x_colname] + fix.loc[i + 1, x_colname]) / 2)
            fix.loc[i + 1, y_colname] = round((fix.loc[i, y_colname] + fix.loc[i + 1, y_colname]) / 2)

            fix.drop(index=i, inplace=True)
            fix.reset_index(drop=True, inplace=True)

            start = fix.loc[i, start_time_col]
            stop = fix.loc[i, stop_time_col]

            fix = get_distance_between_fixations_in_characters_and_recalc_duration(
                fix, letter_width_avg, start_time_col, stop_time_col, x_colname
            )

    fix.loc[:, "num"] = np.arange(len(fix), dtype=int)

    # delete last fixation
    if fix.iloc[-1]["duration"] < duration_threshold:
        fix = fix.iloc[:-1]
        trial["last_fixation_was_discarded_because_too_short"] = True
    else:
        trial["last_fixation_was_discarded_because_too_short"] = False
    fix.reset_index(drop=True, inplace=True)
    return fix.copy()


def clean_dffix_own(
    trial: dict,
    choice_handle_short_and_close_fix: str,
    discard_far_out_of_text_fix,
    x_thres_in_chars,
    y_thresh_in_heights,
    short_fix_threshold,
    merge_distance_threshold: float,
    discard_long_fix: bool,
    discard_long_fix_threshold: int,
    discard_blinks: bool,
    dffix: pd.DataFrame,
):
    dffix = dffix.dropna(how="all", axis=1).copy()
    if dffix.empty:
        return dffix, trial
    dffix = dffix.rename(
        {
            k: v
            for k, v in {
                "xs": "x",
                "ys": "y",
                "num": "fixation_number",
            }.items()
            if v not in dffix.columns
        },
        axis=1,
    )
    stop_time_col, start_time_col = get_time_cols(dffix)
    add_time_cols(dffix, stop_time_col, start_time_col)
    if "dffix_no_clean" not in trial:
        trial["dffix_no_clean"] = (
            dffix.copy()
        )  # TODO check if cleaning can be dialed in or if dffix get overwritten every time
    add_time_cols(trial["dffix_no_clean"], stop_time_col, start_time_col)

    trial["dffix_no_clean"]["was_merged"] = False
    trial["dffix_no_clean"]["was_discarded_short_fix"] = False
    trial["dffix_no_clean"]["was_discarded_outside_text"] = False

    num_fix_before_clean = trial["dffix_no_clean"].shape[0]
    trial["Fixation Cleaning Stats"] = {}
    trial["Fixation Cleaning Stats"]["Number of fixations before cleaning"] = num_fix_before_clean

    trial["Fixation Cleaning Stats"]["Discard fixation before or after blinks"] = discard_blinks

    if discard_blinks and "blink" in dffix.columns:
        trial["dffix_no_clean"]["was_discarded_due_blinks"] = False
        dffix = dffix[dffix["blink"] == False].copy()
        trial["dffix_no_clean"].loc[
            ~trial["dffix_no_clean"]["start_time"].isin(dffix["start_time"]), "was_discarded_due_blinks"
        ] = True
        trial["Fixation Cleaning Stats"]["Number of discarded fixations due to blinks"] = (
            num_fix_before_clean - dffix.shape[0]
        )
        trial["Fixation Cleaning Stats"]["Number of discarded fixations due to blinks (%)"] = round(
            100
            * (trial["Fixation Cleaning Stats"]["Number of discarded fixations due to blinks"] / num_fix_before_clean),
            2,
        )

    trial["Fixation Cleaning Stats"]["Discard long fixations"] = discard_long_fix

    if discard_long_fix and not dffix.empty:
        dffix_before_long_fix_removal = dffix.copy()
        trial["dffix_no_clean"]["was_discarded_due_to_long_duration"] = False
        dffix = dffix[dffix["duration"] < discard_long_fix_threshold].copy()
        dffix_after_long_fix_removal = dffix.copy()
        trial["dffix_no_clean"].loc[
            (
                ~trial["dffix_no_clean"]["start_time"].isin(dffix_after_long_fix_removal["start_time"])
                & (trial["dffix_no_clean"]["start_time"].isin(dffix_before_long_fix_removal["start_time"]))
            ),
            "was_discarded_due_to_long_duration",
        ] = True
        trial["Fixation Cleaning Stats"]["Number of discarded long fixations"] = num_fix_before_clean - dffix.shape[0]
        trial["Fixation Cleaning Stats"]["Number of discarded long fixations (%)"] = round(
            100 * (trial["Fixation Cleaning Stats"]["Number of discarded long fixations"] / num_fix_before_clean), 2
        )
    num_fix_before_merge = dffix.shape[0]
    trial["Fixation Cleaning Stats"]["How short and close fixations were handled"] = choice_handle_short_and_close_fix
    if (
        choice_handle_short_and_close_fix == "Merge" or choice_handle_short_and_close_fix == "Merge then discard"
    ) and not dffix.empty:
        dffix_before_merge = dffix.copy()
        dffix = clean_fixations_popeye_no_sacc(dffix, trial, short_fix_threshold, merge_distance_threshold)
        dffix_after_merge = dffix.copy()
        trial["dffix_no_clean"].loc[
            (~trial["dffix_no_clean"]["start_time"].isin(dffix_after_merge["start_time"]))
            & (trial["dffix_no_clean"]["start_time"].isin(dffix_before_merge["start_time"])),
            "was_merged",
        ] = True
        if trial["last_fixation_was_discarded_because_too_short"]:
            trial["dffix_no_clean"].iloc[-1, trial["dffix_no_clean"].columns.get_loc("was_merged")] = False
            trial["dffix_no_clean"].iloc[-1, trial["dffix_no_clean"].columns.get_loc("was_discarded_short_fix")] = True
        trial["Fixation Cleaning Stats"]["Number of merged fixations"] = (
            num_fix_before_merge - dffix_after_merge.shape[0]
        )
        trial["Fixation Cleaning Stats"]["Number of merged fixations (%)"] = round(
            100 * (trial["Fixation Cleaning Stats"]["Number of merged fixations"] / num_fix_before_merge), 2
        )

    if not dffix.empty:
        dffix.reset_index(drop=True, inplace=True)
        dffix.loc[:, "fixation_number"] = np.arange(dffix.shape[0])
    trial["x_thres_in_chars"], trial["y_thresh_in_heights"] = x_thres_in_chars, y_thresh_in_heights
    if "chars_list" in trial and not dffix.empty:
        indeces_out = get_outlier_indeces(
            dffix,
            pd.DataFrame(trial["chars_list"]),
            x_thres_in_chars,
            y_thresh_in_heights,
            "x",
            "y",
            trial["letter_width_avg"],
            np.mean(trial["line_heights"]),
        )
    else:
        indeces_out = []
    dffix["is_far_out_of_text_uncorrected"] = "in"
    if len(indeces_out) > 0:
        times_out = dffix.loc[indeces_out, "start_time"].copy()
        dffix.loc[indeces_out, "is_far_out_of_text_uncorrected"] = "out"
    trial["Fixation Cleaning Stats"]["Far out of text fixations were discarded"] = discard_far_out_of_text_fix
    if discard_far_out_of_text_fix and len(indeces_out) > 0:
        num_fix_before_clean_via_discard_far_out_of_text_fix = dffix.shape[0]
        trial["dffix_no_clean"].loc[
            trial["dffix_no_clean"]["start_time"].isin(times_out), "was_discarded_outside_text"
        ] = True
        dffix = dffix.loc[dffix["is_far_out_of_text_uncorrected"] == "in", :].reset_index(drop=True).copy()
        trial["Fixation Cleaning Stats"]["Number of discarded far-out-of-text fixations"] = (
            num_fix_before_clean_via_discard_far_out_of_text_fix - dffix.shape[0]
        )
        trial["Fixation Cleaning Stats"]["Number of discarded far-out-of-text fixations (%)"] = round(
            100
            * (
                trial["Fixation Cleaning Stats"]["Number of discarded far-out-of-text fixations"]
                / num_fix_before_clean_via_discard_far_out_of_text_fix
            ),
            2,
        )
    dffix = dffix.drop(columns="is_far_out_of_text_uncorrected")
    if (
        choice_handle_short_and_close_fix == "Discard"
        or choice_handle_short_and_close_fix == "Merge then discard"
        and not dffix.empty
    ):
        num_fix_before_clean_via_discard_short = dffix.shape[0]
        times_out = dffix.loc[(dffix["duration"] < short_fix_threshold), "start_time"].copy()
        if len(times_out) > 0:
            trial["dffix_no_clean"].loc[
                trial["dffix_no_clean"]["start_time"].isin(times_out), "was_discarded_short_fix"
            ] = True
            dffix = dffix[(dffix["duration"] >= short_fix_threshold)].reset_index(drop=True).copy()
            trial["Fixation Cleaning Stats"]["Number of discarded short fixations"] = (
                num_fix_before_clean_via_discard_short - dffix.shape[0]
            )
            trial["Fixation Cleaning Stats"]["Number of discarded short fixations (%)"] = round(
                100
                * (trial["Fixation Cleaning Stats"]["Number of discarded short fixations"])
                / num_fix_before_clean_via_discard_short,
                2,
            )

    trial["Fixation Cleaning Stats"]["Total number of discarded and merged fixations"] = (
        num_fix_before_clean - dffix.shape[0]
    )
    trial["Fixation Cleaning Stats"]["Total number of discarded and merged fixations (%)"] = round(
        100 * trial["Fixation Cleaning Stats"]["Total number of discarded and merged fixations"] / num_fix_before_clean,
        2,
    )

    if not dffix.empty:
        droplist = ["num", "msg"]
        if discard_blinks:
            droplist += ["blink", "blink_before", "blink_after"]
        for col in droplist:
            if col in dffix.columns:
                dffix = dffix.drop(col, axis=1)

        if "start" in dffix.columns:
            dffix = dffix.drop(axis=1, labels=["start", "stop"])
        if "corrected_start_time" not in dffix.columns:
            min_start_time = min(dffix["start_uncorrected"])
            dffix["corrected_start_time"] = dffix["start_uncorrected"] - min_start_time
            dffix["corrected_end_time"] = dffix["stop_uncorrected"] - min_start_time
        assert all(np.diff(dffix["corrected_start_time"]) > 0), "start times not in order"

        dffix_no_clean_fig, _, _ = matplotlib_plot_df(
            dffix,
            trial,
            None,
            trial["dffix_no_clean"],
            box_annotations=None,
            fix_to_plot=["Uncorrected Fixations"],
            stim_info_to_plot=["Characters", "Word boxes"],
        )
        savename = f"{trial['subject']}_{trial['trial_id']}_clean_compare.png"
        dffix_no_clean_fig.savefig(RESULTS_FOLDER.joinpath(savename), dpi=300, bbox_inches="tight")
        plt.close(dffix_no_clean_fig)

        dffix_clean_fig, _, _ = matplotlib_plot_df(
            dffix,
            trial,
            None,
            None,
            box_annotations=None,
            fix_to_plot=["Uncorrected Fixations"],
            stim_info_to_plot=["Characters", "Word boxes"],
            use_duration_arrow_sizes=False,
        )
        savename = f"{trial['subject']}_{trial['trial_id']}_after_clean.png"
        dffix_clean_fig.savefig(RESULTS_FOLDER.joinpath(savename), dpi=300, bbox_inches="tight")
        plt.close(dffix_clean_fig)
        if "item" not in dffix.columns and "item" in trial:
            dffix.insert(loc=0, column="item", value=trial["item"])
        if "condition" not in dffix.columns and "condition" in trial:
            dffix.insert(loc=0, column="condition", value=trial["condition"])
        if "subject" not in dffix.columns and "subject" in trial:
            dffix.insert(loc=0, column="subject", value=trial["subject"])
        if "trial_id" not in dffix.columns and "trial_id" in trial:
            dffix.insert(loc=0, column="trial_id", value=trial["trial_id"])
        dffix = reorder_columns(dffix)
    return dffix, trial


def add_time_cols(dffix, stop_time_col, start_time_col):
    if "start_time" not in dffix.columns:
        dffix["start_time"] = dffix[start_time_col]
    if "end_time" not in dffix.columns:
        dffix["end_time"] = dffix[stop_time_col]
    if "duration" not in dffix.columns:
        dffix["duration"] = dffix["end_time"] - dffix["start_time"]


def get_time_cols(dffix):
    if "stop" in dffix.columns:
        stop_time_col = "stop"
    elif "end_time" in dffix.columns:
        stop_time_col = "end_time"
    elif "corrected_end_time" in dffix.columns:
        stop_time_col = "corrected_end_time"
    if "start" in dffix.columns:
        start_time_col = "start"
    elif "start_time" in dffix.columns:
        start_time_col = "start_time"
    elif "corrected_start_time" in dffix.columns:
        start_time_col = "corrected_start_time"
    return stop_time_col, start_time_col


def trial_to_dfs(
    trial: dict,
    discard_fixations_without_sfix,
    choice_handle_short_and_close_fix,
    discard_far_out_of_text_fix,
    x_thres_in_chars,
    y_thresh_in_heights,
    short_fix_threshold,
    merge_distance_threshold,
    discard_long_fix,
    discard_long_fix_threshold,
    discard_blinks,
):
    events_df, trial = get_raw_events_df_and_trial(trial, discard_fixations_without_sfix)
    dffix, trial = clean_dffix_own(
        trial,
        choice_handle_short_and_close_fix,
        discard_far_out_of_text_fix,
        x_thres_in_chars,
        y_thresh_in_heights,
        short_fix_threshold,
        merge_distance_threshold,
        discard_long_fix,
        discard_long_fix_threshold,
        discard_blinks,
        events_df[events_df["msg"] == "FIX"].copy(),
    )

    dffix = dffix.dropna(how="all", axis=1).copy()
    trial["dffix"] = dffix
    trial["events_df"] = events_df
    return dffix, trial


def get_all_measures(
    trial,
    dffix,
    prefix,
    use_corrected_fixations=True,
    correction_algo="Wisdom_of_Crowds",
    measures_to_calculate=["initial_landing_position"],
    include_coords=False,
    save_to_csv=False,
):
    stim_df = pd.DataFrame(trial[f"{prefix}s_list"])
    if f"{prefix}_number" not in stim_df.columns:
        stim_df[f"{prefix}_number"] = np.arange(stim_df.shape[0])
    if use_corrected_fixations:
        dffix_copy = copy.deepcopy(dffix)
        dffix_copy["y"] = dffix_copy[f"y_{correction_algo}"]
    else:
        dffix_copy = dffix
        correction_algo = "uncorrected"
    res_dfs = []
    for measure in measures_to_calculate:
        if hasattr(anf, f"{measure}_own"):
            function = getattr(anf, f"{measure}_own")
            result = function(trial, dffix_copy, prefix, correction_algo)
            res_dfs.append(result)
    dfs_list = [df for df in [stim_df] + res_dfs if not df.empty]
    own_measure_df = stim_df
    if len(dfs_list) > 1:
        for df in dfs_list[1:]:
            droplist = [col for col in df.columns if (col != f"{prefix}_number" and col in stim_df.columns)]
            own_measure_df = own_measure_df.merge(df.drop(columns=droplist), how="left", on=[f"{prefix}_number"])
    first_column = own_measure_df.pop(prefix)
    own_measure_df.insert(0, prefix, first_column)
    wordfirst = pf.aggregate_words_firstrun(dffix_copy, correction_algo, measures_to_calculate)
    wordtmp = pf.aggregate_words(dffix_copy, pd.DataFrame(trial["words_list"]), correction_algo, measures_to_calculate)
    out = pf.combine_words(
        dffix_copy,
        wordfirst=wordfirst,
        wordtmp=wordtmp,
        algo_choice=correction_algo,
        measures_to_calculate=measures_to_calculate,
    )

    extra_cols = list(set(out.columns) - set(own_measure_df.columns))
    cols_to_add = ["word_number"] + extra_cols
    own_measure_df = pd.merge(own_measure_df, out.loc[:, cols_to_add], on="word_number", how="left")

    first_cols = [
        "subject",
        "trial_id",
        "item",
        "condition",
        "question_correct",
        "word_number",
        "word",
    ]
    for col in first_cols:
        if col in trial and col not in own_measure_df.columns:
            own_measure_df.insert(loc=0, column=col, value=trial[col])

    own_measure_df = own_measure_df.dropna(how="all", axis=1).copy()
    if not include_coords:
        word_cols = ["word_xmin", "word_xmax", "word_ymax", "word_xmin", "word_ymin", "word_x_center", "word_y_center"]
        own_measure_df = own_measure_df.drop(columns=word_cols)

    own_measure_df = reorder_columns(own_measure_df)
    if "question_correct" in own_measure_df.columns:
        own_measure_df = own_measure_df.drop(columns=["question_correct"])
    if save_to_csv:
        own_measure_df.to_csv(
            RESULTS_FOLDER / f"{trial['subject']}_{trial['trial_id']}_{correction_algo}_word_measures.csv"
        )
    return own_measure_df


def add_line_overlaps_to_sample(trial, sample):
    char_df = pd.DataFrame(trial["chars_list"])
    line_overlaps = []
    for arr in sample:
        y_val = arr[1]
        line_overlap = t.tensor(-1, dtype=t.float32)
        for idx, (x1, x2) in enumerate(zip(char_df.char_ymin.unique(), char_df.char_ymax.unique())):
            if x1 <= y_val <= x2:
                line_overlap = t.tensor(idx, dtype=t.float32)
                break
        line_overlaps.append(line_overlap)
    line_olaps_tensor = t.stack(line_overlaps, dim=0)
    sample = t.cat([sample, line_olaps_tensor.unsqueeze(1)], dim=1)
    return sample


def norm_coords_by_letter_min_x_y(
    sample_idx: int,
    trialslist: list,
    samplelist: list,
    chars_center_coords_list: list = None,
):
    chars_df = pd.DataFrame(trialslist[sample_idx]["chars_list"])
    trialslist[sample_idx]["x_char_unique"] = list(chars_df.char_xmin.unique())

    min_x_chars = chars_df.char_xmin.min()
    min_y_chars = chars_df.char_ymin.min()

    norm_vector_substract = t.zeros(
        (1, samplelist[sample_idx].shape[1]), dtype=samplelist[sample_idx].dtype, device=samplelist[sample_idx].device
    )
    norm_vector_substract[0, 0] = norm_vector_substract[0, 0] + 1 * min_x_chars
    norm_vector_substract[0, 1] = norm_vector_substract[0, 1] + 1 * min_y_chars

    samplelist[sample_idx] = samplelist[sample_idx] - norm_vector_substract

    if chars_center_coords_list is not None:
        norm_vector_substract = norm_vector_substract.squeeze(0)[:2]
        if chars_center_coords_list[sample_idx].shape[-1] == norm_vector_substract.shape[-1] * 2:
            chars_center_coords_list[sample_idx][:, :2] -= norm_vector_substract
            chars_center_coords_list[sample_idx][:, 2:] -= norm_vector_substract
        else:
            chars_center_coords_list[sample_idx] -= norm_vector_substract
    return trialslist, samplelist, chars_center_coords_list


def norm_coords_by_letter_positions(
    sample_idx: int,
    trialslist: list,
    samplelist: list,
    meanlist: list = None,
    stdlist: list = None,
    return_mean_std_lists=False,
    norm_by_char_averages=False,
    chars_center_coords_list: list = None,
    add_normalised_values_as_features=False,
):
    chars_df = pd.DataFrame(trialslist[sample_idx]["chars_list"])
    trialslist[sample_idx]["x_char_unique"] = list(chars_df.char_xmin.unique())

    min_x_chars = chars_df.char_xmin.min()
    max_x_chars = chars_df.char_xmax.max()

    norm_vector_multi = t.ones(
        (1, samplelist[sample_idx].shape[1]), dtype=samplelist[sample_idx].dtype, device=samplelist[sample_idx].device
    )
    if norm_by_char_averages:
        chars_list = trialslist[sample_idx]["chars_list"]
        char_widths = np.asarray([x["char_xmax"] - x["char_xmin"] for x in chars_list])
        char_heights = np.asarray([x["char_ymax"] - x["char_ymin"] for x in chars_list])
        char_widths_average = np.mean(char_widths[char_widths > 0])
        char_heights_average = np.mean(char_heights[char_heights > 0])

        norm_vector_multi[0, 0] = norm_vector_multi[0, 0] * char_widths_average
        norm_vector_multi[0, 1] = norm_vector_multi[0, 1] * char_heights_average

    else:
        line_height = min(np.unique(trialslist[sample_idx]["line_heights"]))
        line_width = max_x_chars - min_x_chars
        norm_vector_multi[0, 0] = norm_vector_multi[0, 0] * line_width
        norm_vector_multi[0, 1] = norm_vector_multi[0, 1] * line_height
    assert ~t.any(t.isnan(norm_vector_multi)), "Nan found in char norming vector"

    norm_vector_multi = norm_vector_multi.squeeze(0)
    if add_normalised_values_as_features:
        norm_vector_multi = norm_vector_multi[norm_vector_multi != 1]
        normed_features = samplelist[sample_idx][:, : norm_vector_multi.shape[0]] / norm_vector_multi
        samplelist[sample_idx] = t.cat([samplelist[sample_idx], normed_features], dim=1)
    else:
        samplelist[sample_idx] = samplelist[sample_idx] / norm_vector_multi  #  in case time or pupil size is included
    if chars_center_coords_list is not None:
        norm_vector_multi = norm_vector_multi[:2]
        if chars_center_coords_list[sample_idx].shape[-1] == norm_vector_multi.shape[-1] * 2:
            chars_center_coords_list[sample_idx][:, :2] /= norm_vector_multi
            chars_center_coords_list[sample_idx][:, 2:] /= norm_vector_multi
        else:
            chars_center_coords_list[sample_idx] /= norm_vector_multi
    if return_mean_std_lists:
        mean_val = samplelist[sample_idx].mean(axis=0).cpu().numpy()
        meanlist.append(mean_val)
        std_val = samplelist[sample_idx].std(axis=0).cpu().numpy()
        stdlist.append(std_val)
        assert ~any(pd.isna(mean_val)), "Nan found in mean_val"
        assert ~any(pd.isna(mean_val)), "Nan found in std_val"

        return trialslist, samplelist, meanlist, stdlist, chars_center_coords_list
    return trialslist, samplelist, chars_center_coords_list


def get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, dffix=None, prefix="word"):
    fig = plt.figure(figsize=(screen_res[0] / dpi, screen_res[1] / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    if dffix is not None:
        ax.set_ylim((dffix.y.min(), dffix.y.max()))
        ax.set_xlim((dffix.x.min(), dffix.x.max()))
    else:
        ax.set_ylim((words_df[f"{prefix}_y_center"].min() - y_margin, words_df[f"{prefix}_y_center"].max() + y_margin))
        ax.set_xlim((words_df[f"{prefix}_x_center"].min() - x_margin, words_df[f"{prefix}_x_center"].max() + x_margin))
    ax.invert_yaxis()
    fig.add_axes(ax)
    return fig, ax


def get_save_path(fpath, fname_ending):
    save_path = PLOTS_FOLDER.joinpath(f"{fpath.stem}_{fname_ending}.png")
    return save_path


def save_im_load_convert(fpath, fig, fname_ending, mode):
    save_path = get_save_path(fpath, fname_ending)
    fig.savefig(save_path)
    im = Image.open(save_path).convert(mode)
    im.save(save_path)
    return im


def plot_text_boxes_fixations(
    fpath,
    dpi,
    screen_res,
    set_font_size: bool,
    font_size: int,
    dffix=None,
    trial=None,
):
    if isinstance(fpath, str):
        fpath = pl.Path(fpath)
    prefix = "char"

    if dffix is None:
        dffix = pd.read_csv(fpath)
    if trial is None:
        json_fpath = str(fpath).replace("_fixations.csv", "_trial.json")
        with open(json_fpath, "r") as f:
            trial = json.load(f)
    words_df = pd.DataFrame(trial[f"{prefix}s_list"])
    x_right = words_df[f"{prefix}_xmin"]
    x_left = words_df[f"{prefix}_xmax"]
    y_top = words_df[f"{prefix}_ymax"]
    y_bottom = words_df[f"{prefix}_ymin"]

    if f"{prefix}_x_center" not in words_df.columns:
        words_df[f"{prefix}_x_center"] = (words_df[f"{prefix}_xmax"] - words_df[f"{prefix}_xmin"]) / 2 + words_df[
            f"{prefix}_xmin"
        ]
        words_df[f"{prefix}_y_center"] = (words_df[f"{prefix}_ymax"] - words_df[f"{prefix}_ymin"]) / 2 + words_df[
            f"{prefix}_ymin"
        ]

    x_margin = words_df[f"{prefix}_x_center"].mean() / 8
    y_margin = words_df[f"{prefix}_y_center"].mean() / 4
    times = dffix.corrected_start_time - dffix.corrected_start_time.min()
    times = times / times.max()
    times = np.linspace(0.25, 1, len(times))

    if set_font_size:
        font = "monospace"
    else:
        font_size = trial["font_size"] * 27 // dpi

    font_props = FontProperties(family=font, style="normal", size=font_size)

    fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)

    ax.scatter(words_df[f"{prefix}_x_center"], words_df[f"{prefix}_y_center"], s=1, facecolor="k", alpha=0.01)
    for idx in range(len(x_left)):
        ax.text(
            words_df[f"{prefix}_x_center"][idx],
            words_df[f"{prefix}_y_center"][idx],
            words_df[prefix][idx],
            horizontalalignment="center",
            verticalalignment="center",
            fontproperties=font_props,
        )
    fname_ending = f"{prefix}s_grey"
    words_grey_im = save_im_load_convert(fpath, fig, fname_ending, "L")

    plt.close("all")
    fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)

    ax.scatter(words_df[f"{prefix}_x_center"], words_df[f"{prefix}_y_center"], s=1, facecolor="k", alpha=0.1)
    for idx in range(len(x_left)):
        xdiff = x_right[idx] - x_left[idx]
        ydiff = y_top[idx] - y_bottom[idx]
        rect = patches.Rectangle(
            (x_left[idx] - 1, y_bottom[idx] - 1), xdiff, ydiff, alpha=0.9, linewidth=1, edgecolor="k", facecolor="grey"
        )  # seems to need one pixel offset
        ax.add_patch(rect)
    fname_ending = f"{prefix}_boxes_grey"
    word_boxes_grey_im = save_im_load_convert(fpath, fig, fname_ending, "L")

    plt.close("all")

    fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)

    ax.scatter(dffix.x, dffix.y, facecolor="k", alpha=times)
    fname_ending = "fix_scatter_grey"
    fix_scatter_grey_im = save_im_load_convert(fpath, fig, fname_ending, "L")

    plt.close("all")

    arr_combo = np.stack(
        [
            np.asarray(words_grey_im),
            np.asarray(word_boxes_grey_im),
            np.asarray(fix_scatter_grey_im),
        ],
        axis=2,
    )

    im_combo = Image.fromarray(arr_combo)
    fname_ending = f"{prefix}s_channel_sep"

    im_combo.save(fpath)

    return im_combo


def prep_data_for_dist(model_cfg, dffix, trial):
    if isinstance(dffix, dict):
        dffix = dffix["value"]
    sample_tensor = t.tensor(dffix.loc[:, model_cfg["sample_cols"]].to_numpy(), dtype=t.float32)

    if model_cfg["add_line_overlap_feature"]:
        sample_tensor = add_line_overlaps_to_sample(trial, sample_tensor)

    has_nans = t.any(t.isnan(sample_tensor))
    assert not has_nans, "NaNs found in sample tensor"
    samplelist_eval = [sample_tensor]
    trialslist_eval = [trial]
    chars_center_coords_list_eval = None
    if model_cfg["norm_coords_by_letter_min_x_y"]:
        for sample_idx, _ in enumerate(samplelist_eval):
            trialslist_eval, samplelist_eval, chars_center_coords_list_eval = norm_coords_by_letter_min_x_y(
                sample_idx,
                trialslist_eval,
                samplelist_eval,
                chars_center_coords_list=chars_center_coords_list_eval,
            )

    if model_cfg["normalize_by_line_height_and_width"]:
        meanlist_eval, stdlist_eval = [], []
        for sample_idx, _ in enumerate(samplelist_eval):
            (
                trialslist_eval,
                samplelist_eval,
                meanlist_eval,
                stdlist_eval,
                chars_center_coords_list_eval,
            ) = norm_coords_by_letter_positions(
                sample_idx,
                trialslist_eval,
                samplelist_eval,
                meanlist_eval,
                stdlist_eval,
                return_mean_std_lists=True,
                norm_by_char_averages=model_cfg["norm_by_char_averages"],
                chars_center_coords_list=chars_center_coords_list_eval,
                add_normalised_values_as_features=model_cfg["add_normalised_values_as_features"],
            )
    sample_tensor = samplelist_eval[0]
    sample_means = t.tensor(model_cfg["sample_means"], dtype=t.float32)
    sample_std = t.tensor(model_cfg["sample_std"], dtype=t.float32)
    sample_tensor = (sample_tensor - sample_means) / sample_std
    sample_tensor = sample_tensor.unsqueeze(0)
    if not pl.Path(trial["plot_file"]).exists():
        plot_text_boxes_fixations(
            fpath=trial["plot_file"],
            dpi=250,
            screen_res=(1024, 768),
            set_font_size=True,
            font_size=4,
            dffix=dffix,
            trial=trial,
        )

    val_set = DSet(
        sample_tensor,
        None,
        t.zeros((1, sample_tensor.shape[1])),
        trialslist_eval,
        padding_list=[0],
        padding_at_end=model_cfg["padding_at_end"],
        return_images_for_conv=True,
        im_partial_string=model_cfg["im_partial_string"],
        input_im_shape=model_cfg["char_plot_shape"],
    )
    val_loader = dl(val_set, batch_size=1, shuffle=False, num_workers=0)
    return val_loader, val_set


def fold_in_seq_dim(out, y=None):
    batch_size, seq_len, num_classes = out.shape

    out = eo.rearrange(out, "b s c -> (b s) c", s=seq_len)
    if y is None:
        return out, None
    if len(y.shape) > 2:
        y = eo.rearrange(y, "b s c -> (b s) c", s=seq_len)
    else:
        y = eo.rearrange(y, "b s -> (b s)", s=seq_len)
    return out, y


def logits_to_pred(out, y=None):
    seq_len = out.shape[1]
    out, y = fold_in_seq_dim(out, y)
    preds = corn_label_from_logits(out)
    preds = eo.rearrange(preds, "(b s) -> b s", s=seq_len)
    if y is not None:
        y = eo.rearrange(y.squeeze(), "(b s) -> b s", s=seq_len)
        y = y
    return preds, y


def get_DIST_preds(dffix, trial, models_dict):
    algo_choice = "DIST"

    model = models_dict["single_DIST_model"]
    loader, dset = prep_data_for_dist(models_dict["single_DIST_model_cfg"], dffix, trial)
    batch = next(iter(loader))

    if "cpu" not in str(model.device):
        batch = [x.cuda() for x in batch]
    try:
        out = model(batch)
        preds, y = logits_to_pred(out, y=None)
        if len(trial["y_char_unique"]) < 1:
            y_char_unique = pd.DataFrame(trial["chars_list"]).char_y_center.sort_values().unique()
        else:
            y_char_unique = trial["y_char_unique"]
        num_lines = trial["num_char_lines"] - 1
        preds = t.clamp(preds, 0, num_lines).squeeze().cpu().numpy()
        y_pred_DIST = [y_char_unique[idx] for idx in preds]

        dffix[f"line_num_{algo_choice}"] = preds
        dffix[f"y_{algo_choice}"] = np.round(y_pred_DIST, decimals=2)
        dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(2)
    except Exception as e:
        ic(f"Exception on model(batch) for DIST \n{e}")
    return dffix


def get_DIST_ensemble_preds(
    dffix,
    trial,
    model_cfg_without_norm_df,
    model_cfg_with_norm_df,
    ensemble_model_avg,
):
    algo_choice = "DIST-Ensemble"
    loader_without_norm, dset_without_norm = prep_data_for_dist(model_cfg_without_norm_df, dffix, trial)
    loader_with_norm, dset_with_norm = prep_data_for_dist(model_cfg_with_norm_df, dffix, trial)
    batch_without_norm = next(iter(loader_without_norm))
    batch_with_norm = next(iter(loader_with_norm))
    out = ensemble_model_avg((batch_without_norm, batch_with_norm))
    preds, y = logits_to_pred(out[0]["out_avg"], y=None)
    if len(trial["y_char_unique"]) < 1:
        y_char_unique = pd.DataFrame(trial["chars_list"]).char_y_center.sort_values().unique()
    else:
        y_char_unique = trial["y_char_unique"]
    num_lines = trial["num_char_lines"] - 1
    preds = t.clamp(preds, 0, num_lines).squeeze().cpu().numpy()
    y_pred_DIST = [y_char_unique[idx] for idx in preds]

    dffix[f"line_num_{algo_choice}"] = preds
    dffix[f"y_{algo_choice}"] = np.round(y_pred_DIST, decimals=1)
    dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(1)
    return dffix


def get_EDIST_preds_with_model_check(dffix, trial, models_dict):

    dffix = get_DIST_ensemble_preds(
        dffix,
        trial,
        models_dict["model_cfg_without_norm_df"],
        models_dict["model_cfg_with_norm_df"],
        models_dict["ensemble_model_avg"],
    )
    return dffix


def get_all_classic_preds(dffix, trial, classic_algos_cfg):
    corrections = []
    for algo, classic_params in copy.deepcopy(classic_algos_cfg).items():
        dffix = calgo.apply_classic_algo(dffix, trial, algo, classic_params)
        corrections.append(np.asarray(dffix.loc[:, f"y_{algo}"]))
    return dffix, corrections


def apply_woc(dffix, trial, corrections, algo_choice):

    corrected_Y = calgo.wisdom_of_the_crowd(corrections)
    dffix.loc[:, f"y_{algo_choice}"] = corrected_Y
    dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(1)
    corrected_line_nums = [trial["y_char_unique"].index(y) for y in corrected_Y]
    dffix.loc[:, f"line_num_y_{algo_choice}"] = corrected_line_nums
    dffix.loc[:, f"line_num_{algo_choice}"] = corrected_line_nums
    return dffix


def apply_correction_algo(dffix, algo_choice, trial, models_dict, classic_algos_cfg):

    if algo_choice == "DIST":
        dffix = get_DIST_preds(dffix, trial, models_dict=models_dict)

    elif algo_choice == "DIST-Ensemble":
        dffix = get_EDIST_preds_with_model_check(dffix, trial, models_dict=models_dict)
    elif algo_choice == "Wisdom_of_Crowds_with_DIST":
        dffix, corrections = get_all_classic_preds(dffix, trial, classic_algos_cfg)
        dffix = get_DIST_preds(dffix, trial, models_dict=models_dict)
        for _ in range(3):
            corrections.append(np.asarray(dffix.loc[:, "y_DIST"]))
        dffix = apply_woc(dffix, trial, corrections, algo_choice)
    elif algo_choice == "Wisdom_of_Crowds_with_DIST_Ensemble":
        dffix, corrections = get_all_classic_preds(dffix, trial, classic_algos_cfg)
        dffix = get_EDIST_preds_with_model_check(dffix, trial, models_dict=models_dict)
        for _ in range(3):
            corrections.append(np.asarray(dffix.loc[:, "y_DIST-Ensemble"]))
        dffix = apply_woc(dffix, trial, corrections, algo_choice)
    elif algo_choice == "Wisdom_of_Crowds":
        dffix, corrections = get_all_classic_preds(dffix, trial, classic_algos_cfg)
        dffix = apply_woc(dffix, trial, corrections, algo_choice)

    else:
        algo_cfg = classic_algos_cfg[algo_choice]
        dffix = calgo.apply_classic_algo(dffix, trial, algo_choice, algo_cfg)
        dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(1)
    dffix = dffix.copy()  # apparently helps with fragmentation
    return dffix


def add_popEye_cols_to_dffix(dffix, algo_choice, chars_df, trial, xcol, cols_to_add: list):
    """
    Required for word or sentence measures:
    - letternum
    - letter
    - on_word_number
    - on_word
    - on_sentence
    - num_words_in_sentence
    - on_sentence_num
    - word_land
    - line_let
    - line_word
    - sac_in
    - sac_out
    - word_launch
    - word_refix
    - word_reg_in
    - word_reg_out
    - sentence_reg_in
    - word_firstskip
    - word_run
    - sentence_run
    - word_run_fix
    - word_cland
    Optional:
    - line_let_from_last_letter
    - sentence_word
    - line_let_previous
    - line_let_next
    - sentence_refix
    - word_reg_out_to
    - word_reg_in_from
    - sentence_reg_out
    - sentence_reg_in_from
    - sentence_reg_out_to
    - sentence_firstskip
    - word_runid
    - sentence_runid
    - word_fix
    - sentence_fix
    """
    if "angle_incoming" in cols_to_add:
        x_diff_incoming = dffix[xcol].values - dffix[xcol].shift(1).values
        y_diff_incoming = dffix["y"].values - dffix["y"].shift(1).values
        angle_incoming = np.arctan2(y_diff_incoming, x_diff_incoming) * (180 / np.pi)
        dffix["angle_incoming"] = angle_incoming
    if "angle_outgoing" in cols_to_add:
        x_diff_outgoing = dffix[xcol].shift(-1).values - dffix[xcol].values
        y_diff_outgoing = dffix["y"].shift(-1).values - dffix["y"].values
        angle_outgoing = np.arctan2(y_diff_outgoing, x_diff_outgoing) * (180 / np.pi)
        dffix["angle_outgoing"] = angle_outgoing
    dffix[f"line_change_{algo_choice}"] = np.concatenate(
        ([0], np.diff(dffix[f"line_num_{algo_choice}"])), axis=0
    ).astype(int)

    for i in list(dffix.index):
        if dffix.loc[i, f"line_num_{algo_choice}"] > -1 and not pd.isna(dffix.loc[i, f"line_num_{algo_choice}"]):
            selected_stimmat = chars_df[
                chars_df["assigned_line"] == dffix.loc[i, f"line_num_{algo_choice}"]
            ].reset_index()
            selected_stimmat.loc[:, "letword"] = selected_stimmat.groupby("in_word_number")["letternum"].rank()
            letters_on_line = selected_stimmat.shape[0]
            out = dffix.loc[i, xcol] - selected_stimmat["char_x_center"]
            min_idx = out.abs().idxmin()
            dffix.loc[i, f"letternum_{algo_choice}"] = selected_stimmat.loc[min_idx, "letternum"]
            dffix.loc[i, f"letter_{algo_choice}"] = selected_stimmat.loc[min_idx, "char"]
            dffix.loc[i, f"line_let_{algo_choice}"] = selected_stimmat.loc[min_idx, "letline"]
            if "line_let_from_last_letter" in cols_to_add:
                dffix.loc[i, f"line_let_from_last_letter_{algo_choice}"] = (
                    letters_on_line - dffix.loc[i, f"line_let_{algo_choice}"]
                )
            word_min_idx = min_idx
            if (
                selected_stimmat.loc[min_idx, "char"] == " "
                and (min_idx - 1) in selected_stimmat.index
                and (min_idx + 1) in selected_stimmat.index
            ):
                dist_to_previous_letter = np.abs(
                    dffix.loc[i, xcol] - selected_stimmat.loc[min_idx - 1, "char_x_center"]
                )
                dist_to_following_letter = np.abs(
                    dffix.loc[i, xcol] - selected_stimmat.loc[min_idx + 1, "char_x_center"]
                )
                if dist_to_previous_letter < dist_to_following_letter:
                    word_min_idx = min_idx - 1
            if not pd.isna(selected_stimmat.loc[min_idx, "in_word_number"]):
                dffix.loc[i, f"on_word_number_{algo_choice}"] = selected_stimmat.loc[word_min_idx, "in_word_number"]
                dffix.loc[i, f"on_word_{algo_choice}"] = selected_stimmat.loc[word_min_idx, "in_word"]
                dffix.loc[i, f"word_land_{algo_choice}"] = selected_stimmat.loc[
                    word_min_idx, "num_letters_from_start_of_word"
                ]
                dffix.loc[i, f"line_word_{algo_choice}"] = selected_stimmat.loc[word_min_idx, "wordline"]
                if "sentence_word" in cols_to_add:
                    dffix.loc[i, f"sentence_word_{algo_choice}"] = selected_stimmat.loc[word_min_idx, "wordsent"]
            dffix.loc[i, "num_words_in_sentence"] = len(selected_stimmat.loc[word_min_idx, "in_sentence"].split(" "))
            dffix.loc[i, f"on_sentence_num_{algo_choice}"] = selected_stimmat.loc[word_min_idx, "in_sentence_number"]
            dffix.loc[i, f"on_sentence_{algo_choice}"] = selected_stimmat.loc[word_min_idx, "in_sentence"]
    if "line_let_previous" in cols_to_add:
        dffix[f"line_let_previous_{algo_choice}"] = dffix[f"line_let_{algo_choice}"].shift(-1)
    if "line_let_next" in cols_to_add:
        dffix[f"line_let_next_{algo_choice}"] = dffix[f"line_let_{algo_choice}"].shift(1)
    dffix = pf.compute_saccade_length(dffix, chars_df, algo_choice)
    dffix = pf.compute_launch_distance(dffix, algo_choice)
    dffix = pf.compute_refixation(dffix, algo_choice)
    dffix = pf.compute_regression(dffix, algo_choice)
    dffix = pf.compute_firstskip(dffix, algo_choice)
    dffix = pf.compute_run(dffix, algo_choice)
    dffix = pf.compute_landing_position(dffix, algo_choice)
    dffix = dffix.loc[:, ~dffix.columns.duplicated()]
    return dffix


def export_dataframe(df: pd.DataFrame, csv_name: str):
    if isinstance(df, dict):
        df = df["value"]
    df.to_csv(csv_name)
    return csv_name


def _convert_to_json(obj):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: _convert_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_convert_to_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_to_json(val) for k, val in obj.items()}
    elif hasattr(obj, "to_dict"):
        return _convert_to_json(obj.to_dict())
    elif hasattr(obj, "tolist"):
        return _convert_to_json(obj.tolist())
    elif obj is None:
        return None
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_trial_to_json(trial, savename):
    filtered_trial = {}
    for key, value in trial.items():
        try:
            filtered_trial[key] = _convert_to_json(value)
        except TypeError as e:
            ic(f"Warning: Skipping non-serializable value for key '{key}' due to error: {e}")

    with open(savename, "w", encoding="utf-8") as f:
        json.dump(filtered_trial, f, ensure_ascii=False, indent=4)


def export_trial(trial: dict):

    trial_id = trial["trial_id"]
    savename = RESULTS_FOLDER.joinpath(pl.Path(trial["filename"]).stem)
    trial_name = f"{savename}_{trial_id}_trial_info.json"

    filtered_trial = copy.deepcopy(trial)
    _ = [filtered_trial.pop(k) for k in list(filtered_trial.keys()) if isinstance(filtered_trial[k], pd.DataFrame)]
    _ = [
        filtered_trial.pop(k)
        for k in list(filtered_trial.keys())
        if k
        in [
            "words_list",
            "chars_list",
            "chars_df_alt",
            "EMReading_fix",
            "chars_df",
            "dffix_sacdf_popEye",
            "fixdf_popEye",
            "sacdf_popEye",
            "saccade_df",
            "combined_df",
            "own_sentence_measures_dfs_for_algo",
            "own_word_measures_dfs_for_algo",
        ]
    ]

    filtered_trial["line_heights"] = list(np.unique(filtered_trial["line_heights"]))
    save_trial_to_json(filtered_trial, trial_name)
    return trial_name


def add_cols_from_trial(trial, df, cols=["item", "condition", "trial_id", "subject"]):
    for col in cols:
        if col not in df.columns:
            df.insert(loc=0, column=col, value=trial[col])


def correct_df(
    dffix,
    algo_choice,
    trial,
    for_multi,
    is_outside_of_streamlit,
    classic_algos_cfg,
    models_dict,
    measures_to_calculate_multi_asc=[],
    include_coords_multi_asc=False,
    sent_measures_to_calc_multi=[],
    fix_cols_to_add=[],
):
    if is_outside_of_streamlit:
        stqdm = tqdm
    else:
        from stqdm import stqdm

    if isinstance(dffix, dict):
        dffix = dffix["value"]
    if "x" not in dffix.keys() or "x" not in dffix.keys():
        ic(f"x or y not in dffix")
        ic(dffix.columns)
        return dffix

    if isinstance(algo_choice, list):
        algo_choices = algo_choice
        repeats = range(len(algo_choice))
    else:
        algo_choices = [algo_choice]
        repeats = range(1)

    chars_df = pd.DataFrame(trial["chars_df"]) if "chars_df" in trial else pd.DataFrame(trial["chars_list"])
    if for_multi:
        own_word_measures_dfs_for_algo = []
        own_sentence_measures_dfs_for_algo = []
    trial["average_y_corrections"] = []
    for algoIdx in stqdm(repeats, desc="Applying line-assignment algorithms"):
        algo_choice = algo_choices[algoIdx]
        dffix = apply_correction_algo(dffix, algo_choice, trial, models_dict, classic_algos_cfg)
        average_y_correction = (dffix[f"y_{algo_choice}"] - dffix["y"]).mean().round(1)
        trial["average_y_corrections"].append({"Algorithm": algo_choice, "average_y_correction": average_y_correction})
        fig, desired_width_in_pixels, desired_height_in_pixels = matplotlib_plot_df(
            dffix,
            trial,
            algo_choice,
            None,
            box_annotations=None,
            fix_to_plot=["Uncorrected Fixations", "Corrected Fixations"],
            stim_info_to_plot=["Characters", "Word boxes"],
        )
        savename = f"{trial['subject']}_{trial['trial_id']}_corr_{algo_choice}_fix.png"
        fig.savefig(RESULTS_FOLDER.joinpath(savename), dpi=300)
        plt.close(fig)
        dffix = add_popEye_cols_to_dffix(dffix, algo_choice, chars_df, trial, "x", cols_to_add=fix_cols_to_add)

        if for_multi and len(measures_to_calculate_multi_asc) > 0 and dffix.shape[0] > 1:
            own_word_measures = get_all_measures(
                trial,
                dffix,
                prefix="word",
                use_corrected_fixations=True,
                correction_algo=algo_choice,
                measures_to_calculate=measures_to_calculate_multi_asc,
                include_coords=include_coords_multi_asc,
            )
            own_word_measures_dfs_for_algo.append(own_word_measures)
            sent_measures_multi = pf.compute_sentence_measures(
                dffix, pd.DataFrame(trial["chars_df"]), algo_choice, sent_measures_to_calc_multi
            )
            own_sentence_measures_dfs_for_algo.append(sent_measures_multi)

    if for_multi and len(own_word_measures_dfs_for_algo) > 0:
        words_df = (
            pd.DataFrame(trial["chars_df"])
            .drop_duplicates(subset="in_word_number", keep="first")
            .loc[:, ["in_word_number", "in_word"]]
            .rename({"in_word_number": "word_number", "in_word": "word"}, axis=1)
            .reset_index(drop=True)
        )
        add_cols_from_trial(trial, words_df, cols=["item", "condition", "trial_id", "subject"])
        words_df["subject_trialID"] = [f"{id}_{num}" for id, num in zip(words_df["subject"], words_df["trial_id"])]
        words_df = words_df.merge(
            own_word_measures_dfs_for_algo[0],
            how="left",
            on=["subject", "trial_id", "item", "condition", "word_number", "word"],
        )
        for word_measure_df in own_word_measures_dfs_for_algo[1:]:
            words_df = words_df.merge(
                word_measure_df, how="left", on=["subject", "trial_id", "item", "condition", "word_number", "word"]
            )
        words_df = reorder_columns(words_df, ["subject", "trial_id", "item", "condition", "word_number", "word"])

        sentence_df = (
            pd.DataFrame(trial["chars_df"])
            .drop_duplicates(subset="in_sentence_number", keep="first")
            .loc[
                :,
                [
                    "in_sentence_number",
                    "in_sentence",
                ],
            ]
            .rename({"in_sentence_number": "sentence_number", "in_sentence": "sentence"}, axis=1)
            .reset_index(drop=True)
        )
        add_cols_from_trial(trial, sentence_df, cols=["item", "condition", "trial_id", "subject"])
        sentence_df["subject_trialID"] = [
            f"{id}_{num}" for id, num in zip(sentence_df["subject"], sentence_df["trial_id"])
        ]
        sentence_df = sentence_df.merge(
            own_sentence_measures_dfs_for_algo[0],
            how="left",
            on=["item", "condition", "trial_id", "subject", "sentence_number", "sentence"],
        )
        for sent_measure_df in own_sentence_measures_dfs_for_algo[1:]:
            sentence_df = sentence_df.merge(
                sent_measure_df,
                how="left",
                on=["subject", "trial_id", "item", "condition", "sentence_number", "sentence", "number_of_words"],
            )
        sentence_df = reorder_columns(
            sentence_df, ["subject", "trial_id", "item", "condition", "sentence_number", "sentence", "number_of_words"]
        )

        trial["own_word_measures_dfs_for_algo"] = words_df

        trial["own_sentence_measures_dfs_for_algo"] = sentence_df
    dffix = reorder_columns(dffix)
    if for_multi:
        return dffix
    else:
        fix_cols_to_keep = [
            c
            for c in dffix.columns
            if (
                (any([lname in c for lname in ALL_FIX_MEASURES]) and any([lname in c for lname in fix_cols_to_add]))
                or (not any([lname in c for lname in ALL_FIX_MEASURES]))
            )
        ]

        savename = RESULTS_FOLDER.joinpath(pl.Path(trial["filename"]).stem)
        csv_name = f"{savename}_{trial['trial_id']}_corrected_fixations.csv"
        csv_name = export_dataframe(dffix.loc[:, fix_cols_to_keep].copy(), csv_name)

        export_trial(trial)
        return dffix


def process_trial_choice(
    trial: dict,
    algo_choice: str,
    choice_handle_short_and_close_fix,
    for_multi,
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
    classic_algos_cfg,
    models_dict,
    fix_cols_to_add,
):

    dffix, trial = trial_to_dfs(
        trial=trial,
        choice_handle_short_and_close_fix=choice_handle_short_and_close_fix,
        discard_fixations_without_sfix=discard_fixations_without_sfix,
        discard_far_out_of_text_fix=discard_far_out_of_text_fix,
        x_thres_in_chars=x_thres_in_chars,
        y_thresh_in_heights=y_thresh_in_heights,
        short_fix_threshold=short_fix_threshold,
        discard_long_fix=discard_long_fix,
        discard_long_fix_threshold=discard_long_fix_threshold,
        merge_distance_threshold=merge_distance_threshold,
        discard_blinks=discard_blinks,
    )
    if "chars_list" in trial:
        chars_df = pd.DataFrame(trial["chars_df"])

        trial["chars_df"] = chars_df.to_dict()
        trial["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())
    if algo_choice is not None and ("chars_list" in trial or "words_list" in trial):
        if dffix.shape[0] > 1:
            dffix = correct_df(
                dffix,
                algo_choice,
                trial,
                for_multi=for_multi,
                is_outside_of_streamlit=False,
                classic_algos_cfg=classic_algos_cfg,
                models_dict=models_dict,
                measures_to_calculate_multi_asc=measures_to_calculate_multi_asc,
                include_coords_multi_asc=include_coords_multi_asc,
                sent_measures_to_calc_multi=sent_measures_to_calculate_multi_asc,
                fix_cols_to_add=fix_cols_to_add,
            )

            saccade_df = get_saccade_df(dffix, trial, algo_choice, trial.pop("events_df"))
            trial["saccade_df"] = saccade_df.to_dict()

            fig = plot_saccade_df(dffix, saccade_df, trial, True, False)
            fig.savefig(RESULTS_FOLDER / f"{trial['subject']}_{trial['trial_id']}_saccades.png")
            plt.close(fig)
        else:
            ic(
                f"🚨 Only {dffix.shape[0]} fixation left after processing. saccade_df not created for trial {trial['trial_id']} 🚨"
            )

    else:
        ic("🚨 Stimulus information needed for fixation line-assignment 🚨")
    for c in ["gaze_df", "dffix"]:
        if c in trial:
            trial.pop(c)
    return dffix, trial


def get_saccade_df(dffix, trial, algo_choices, events_df):
    if not isinstance(algo_choices, list):
        algo_choices = [algo_choices]
    sac_df_as_detected = events_df[events_df["msg"] == "SAC"].copy()
    last_sacc_stop_time = sac_df_as_detected["stop_uncorrected"].iloc[-1]
    dffix_after_last_sacc = dffix.loc[dffix["start_uncorrected"] > last_sacc_stop_time, :].copy()
    if not dffix_after_last_sacc.empty:
        dffix_before_last_sacc = dffix.loc[dffix["start_uncorrected"] < last_sacc_stop_time, :].copy()
        dffix = pd.concat([dffix_before_last_sacc, dffix_after_last_sacc.iloc[[0], :]], axis=0)
    sac_df_as_detected = sac_df_as_detected[sac_df_as_detected["start"] >= dffix["end_time"].iloc[0]]
    sac_df_as_detected = sac_df_as_detected[sac_df_as_detected["stop"] <= dffix["start_time"].iloc[-1]]

    sac_index_keep = [
        i for i, row in sac_df_as_detected.iterrows() if np.abs(row["start"] - dffix["start_time"].values).min() < 100
    ]
    sac_df_as_detected = sac_df_as_detected.loc[sac_index_keep, :]

    starts = pd.Series(dffix["start_time"].values, dffix["start_time"])
    ends = pd.Series(dffix["end_time"].values, dffix["end_time"])
    starts_reind = starts.reindex(sac_df_as_detected["stop"], method="bfill").dropna()
    ends_reind = ends.reindex(sac_df_as_detected["start"], method="ffill").dropna()

    sac_df_as_detected_start_indexed = sac_df_as_detected.copy().set_index("start")
    saccade_df = (
        sac_df_as_detected_start_indexed.loc[ends_reind.index, :]
        .reset_index(drop=False)
        .rename({"start": "start_time", "stop": "end_time"}, axis=1)
    )

    saccade_df = pf.get_angle_and_eucl_dist(saccade_df)
    # TODO maybe add incoming outgoing angle from sacc_df to dffix

    dffix_start_indexed = dffix.copy().set_index("start_time")
    dffix_end_indexed = dffix.copy().set_index("end_time")
    for algo_choice in algo_choices:

        saccade_df[f"ys_{algo_choice}"] = dffix_end_indexed.loc[ends_reind.values, f"y_{algo_choice}"].values
        saccade_df[f"ye_{algo_choice}"] = dffix_start_indexed.loc[starts_reind.values, f"y_{algo_choice}"].values
        saccade_df = pf.get_angle_and_eucl_dist(saccade_df, algo_choice)

        saccade_df[f"lines_{algo_choice}"] = dffix_end_indexed.loc[ends_reind.values, f"line_num_{algo_choice}"].values
        saccade_df[f"linee_{algo_choice}"] = dffix_start_indexed.loc[
            starts_reind.values, f"line_num_{algo_choice}"
        ].values

        saccade_df[f"line_word_s_{algo_choice}"] = dffix_end_indexed.loc[
            ends_reind.values, f"line_word_{algo_choice}"
        ].values
        saccade_df[f"line_word_e_{algo_choice}"] = dffix_start_indexed.loc[
            starts_reind.values, f"line_word_{algo_choice}"
        ].values

        saccade_df[f"lets_{algo_choice}"] = dffix_end_indexed.loc[ends_reind.values, f"letternum_{algo_choice}"].values
        saccade_df[f"lete_{algo_choice}"] = dffix_start_indexed.loc[
            starts_reind.values, f"letternum_{algo_choice}"
        ].values

    blink_df = events_df[events_df["msg"] == "BLINK"]
    for i in range(len(saccade_df)):
        if saccade_df.loc[i, "start_time"] in blink_df["start"]:
            saccade_df.loc[i, "blink"] = True

    saccade_df = pf.compute_non_line_dependent_saccade_measures(saccade_df, trial)
    for algo_choice in algo_choices:
        saccade_df = pf.compute_saccade_measures(saccade_df, trial, algo_choice)

    if "msg" in saccade_df.columns:
        saccade_df = saccade_df.drop(axis=1, labels=["msg"])
    saccade_df = reorder_columns(saccade_df)
    return saccade_df.dropna(how="all", axis=1).copy()
