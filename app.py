import os
import traceback
from datetime import datetime
import hashlib
if os.name == 'posix' and os.uname().sysname == "Darwin" and os.path.exists("/opt/homebrew/lib"):
    # Fix for macOS https://github.com/Kozea/CairoSVG/issues/354#issuecomment-1072905204
    from ctypes.macholib import dyld
    dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")
import subprocess
import copy
from io import StringIO, BytesIO
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd
import numpy as np
import time
from icecream import ic
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pathlib as pl
import json
import logging
import zipfile
from stqdm import stqdm
import jellyfish as jf
import shutil
import eyekit_measures as ekm
import zipfile
from matplotlib import font_manager
import os
import multiprocessing
import torch
import lovely_tensors as lt
lt.monkey_patch()

try:
    from create_interest_areas_from_image import recognize_text
except Exception as e:
    print(e)

from multi_proc_funcs import (
    ALL_FIX_MEASURES,
    COLORS,
    DEFAULT_FIX_MEASURES,
    add_default_font_and_character_props_to_state,
    clean_dffix_own,
    export_dataframe,
    export_trial,
    get_plot_props,
    get_raw_events_df_and_trial,
    get_saccade_df,
    plot_saccade_df,
    process_trial_choice,
    reorder_columns,
    set_font_from_chars_list,
    correct_df,
    get_font_and_font_size_from_trial,
    matplotlib_plot_df,
    get_all_measures,
    add_popEye_cols_to_chars_df,
    AVAILABLE_FONTS,
    PLOTS_FOLDER,
    RESULTS_FOLDER,
    set_up_models,
    add_cols_from_trial,
)
import utils as ut
import popEye_funcs as pf
from process_asc_files_in_multi_p import process_asc_files_in_multi_proc, get_cpu_count

ic.configureOutput(includeContext=True)
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

st.set_page_config("Correction", page_icon=":eye:", layout="wide")

try:
    AVAILABLE_FONTS = st.session_state["AVAILABLE_FONTS"] = AVAILABLE_FONTS
except:
    AVAILABLE_FONTS = [x.name for x in font_manager.fontManager.ttflist]


if "Consolas" in AVAILABLE_FONTS:
    FONT_INDEX = AVAILABLE_FONTS.index("Consolas")
elif "Courier New" in AVAILABLE_FONTS:
    FONT_INDEX = AVAILABLE_FONTS.index("Courier New")
elif "DejaVu Sans Mono" in AVAILABLE_FONTS:
    FONT_INDEX = AVAILABLE_FONTS.index("DejaVu Sans Mono")
else:
    FONT_INDEX = 0
DEFAULT_PLOT_FONT = "DejaVu Sans Mono"
EXAMPLES_FOLDER = "./testfiles/"
EXAMPLES_ASC_ZIP_FILENAME = "asc_files.zip"
DEFAULT_OSF_DOWNLOAD_LINK = "https://osf.io/download/us97f/"
DEFAULT_CONTACT_EMAIL = "tmercier@bournemouth.ac.uk"
try:
    OSF_DOWNLOAD_LINK = st.secrets.get("osf_download_link", DEFAULT_OSF_DOWNLOAD_LINK)
    CONTACT_EMAIL = st.secrets.get("contact_email", DEFAULT_CONTACT_EMAIL)
except StreamlitSecretNotFoundError:
    OSF_DOWNLOAD_LINK = DEFAULT_OSF_DOWNLOAD_LINK
    CONTACT_EMAIL = DEFAULT_CONTACT_EMAIL
EXAMPLES_FOLDER_PATH = pl.Path(EXAMPLES_FOLDER)

EXAMPLE_CUSTOM_CSV_FILE = EXAMPLES_FOLDER_PATH / "ABREV13_trial_id_E1I21D0_fixations.csv"
EXAMPLE_CUSTOM_JSON_FILE = EXAMPLES_FOLDER_PATH / "ABREV13_trial_id_E1I21D0_trial.json"

UNZIPPED_FOLDER = pl.Path("unzipped")

TEMP_FIGURE_STIMULUS_PATH = PLOTS_FOLDER.joinpath("temp_matplotlib_plot_stimulus.png")
ut.make_folders(RESULTS_FOLDER, UNZIPPED_FOLDER, PLOTS_FOLDER)
OCR_CACHE_FOLDER = RESULTS_FOLDER.joinpath("ocr_cache")
OCR_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


def _read_image_bytes(image_obj) -> bytes:
    if hasattr(image_obj, "getvalue"):
        image_bytes = image_obj.getvalue()
    elif isinstance(image_obj, (str, pl.Path)):
        image_bytes = pl.Path(image_obj).read_bytes()
    elif hasattr(image_obj, "read"):
        previous_pos = image_obj.tell() if hasattr(image_obj, "tell") else None
        if hasattr(image_obj, "seek"):
            image_obj.seek(0)
        image_bytes = image_obj.read()
        if previous_pos is not None and hasattr(image_obj, "seek"):
            image_obj.seek(previous_pos)
    else:
        raise TypeError(f"Unsupported image object type for OCR: {type(image_obj)}")

    if hasattr(image_obj, "seek"):
        image_obj.seek(0)

    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError("Could not read image as bytes for OCR caching")
    return bytes(image_bytes)


def get_cached_or_run_ocr(image_obj, *, image_name: str | None = None, trial_label: str | None = None) -> pd.DataFrame:
    image_bytes = _read_image_bytes(image_obj)
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    cache_file = OCR_CACHE_FOLDER.joinpath(f"{image_hash}.csv")

    if cache_file.exists():
        stim_df_raw = pd.read_csv(cache_file)
        st.session_state["logger"].info(
            f"OCR cache hit for {trial_label or image_name or image_hash[:12]} ({cache_file.name})"
        )
        return stim_df_raw

    start_time_ocr = time.time()
    image_stream = BytesIO(image_bytes)
    image_stream.name = image_name or getattr(image_obj, "name", f"{image_hash}.png")
    stim_df_raw = recognize_text(image_stream)
    stim_df_raw.to_csv(cache_file, index=False)
    st.session_state["logger"].info(
        f"OCR cache miss for {trial_label or image_stream.name}; computed in {time.time() - start_time_ocr:.2f}s and cached to {cache_file.name}"
    )
    return stim_df_raw


def handle_single_csv_analysis_selection(algo_choice: str | None) -> bool:
    """Persist the selected algorithm for single-CSV analysis or warn when missing."""

    if algo_choice is None:
        st.warning("Please select an algorithm before running the analysis.")
        st.session_state.pop("algo_choice_analysis_single_csv", None)
        return False

    st.session_state["algo_choice_analysis_single_csv"] = algo_choice
    return True


@st.cache_data
def get_classic_cfg(filename):
    with open(filename, "r") as f:
        jsonsstring = f.read()
    classic_algos_cfg = json.loads(jsonsstring)
    classic_algos_cfg["slice"] = classic_algos_cfg["slice"]
    classic_algos_cfg = classic_algos_cfg
    return classic_algos_cfg


CLASSIC_ALGOS_CFGS = get_classic_cfg("algo_cfgs_all.json")

DIST_MODELS_FOLDER = st.session_state["DIST_MODELS_FOLDER"] = pl.Path("models")
STIM_FIX_PLOT_OPTIONS = [
    "Uncorrected Fixations",
    "Corrected Fixations",
    "Word boxes",
    "Characters",
    "Character boxes",
]
ALGO_CHOICES = [
    "warp",
    "regress",
    "compare",
    "attach",
    "segment",
    "split",
    "stretch",
    "chain",
    "slice",
    "cluster",
    "merge",
    "Wisdom_of_Crowds",
    "DIST",
    "DIST-Ensemble",
    "Wisdom_of_Crowds_with_DIST",
    "Wisdom_of_Crowds_with_DIST_Ensemble",
]

DEFAULT_ALGO_CHOICE = ["slice", "DIST"]
START_KEYWORD_OPTIONS = ["SYNCTIME", "START", "GAZE TARGET ON", "custom"]
END_KEYWORD_OPTIONS = ["ENDBUTTON", "END", "KEYBOARD", "custom"]
ALL_MEASURES_OWN = [
    "blink",
    "first_of_many_duration",
    "firstfix_cland",
    "firstfix_dur",
    "firstfix_land",
    "firstfix_launch",
    "firstfix_sac_in",
    "firstfix_sac_out",
    "firstrun_blink",
    "firstrun_dur",
    "firstrun_gopast",
    "firstrun_gopast_sel",
    "firstrun_nfix",
    "firstrun_refix",
    "firstrun_reg_in",
    "firstrun_reg_out",
    "firstrun_skip",
    "gopast",
    "gopast_sel",
    "initial_landing_distance",
    "initial_landing_position",
    "landing_distances",
    "nrun",
    "number_of_fixations",
    "number_of_regressions_in",
    "refix",
    "skip",
    "reg_in",
    "reg_out",
    "reread",
    "second_pass_duration",
    "singlefix",
    "singlefix_cland",
    "singlefix_dur",
    "singlefix_land",
    "singlefix_launch",
    "singlefix_sac_in",
    "singlefix_sac_out",
    "total_fixation_duration",
]
DEFAULT_WORD_MEASURES = [
    "firstrun_dur",
    "firstrun_nfix",
    "firstfix_dur",
    "singlefix_dur",
    "total_fixation_duration",
    "firstrun_gopast",
    "skip",
    "reg_in",
    "reg_out",
    "number_of_fixations",
    "number_of_regressions_in",  # TODO Check why it does not always agree with reg_in
]

ALL_SENT_MEASURES = [
    "on_sentence_num",
    "on_sentence",
    "num_words_in_sentence",
    "skip",
    "nrun",
    "reread",
    "reg_in",
    "reg_out",
    "total_n_fixations",
    "total_dur",
    "rate",
    "gopast",
    "gopast_sel",
    "firstrun_skip",
    "firstrun_reg_in",
    "firstrun_reg_out",
    "firstpass_n_fixations",
    "firstpass_dur",
    "firstpass_forward_n_fixations",
    "firstpass_forward_dur",
    "firstpass_reread_n_fixations",
    "firstpass_reread_dur",
    "lookback_n_fixations",
    "lookback_dur",
    "lookfrom_n_fixations",
    "lookfrom_dur",
]
DEFAULT_SENT_MEASURES = ["on_sentence_num", "on_sentence", "num_words_in_sentence", "total_n_fixations", "total_dur"]

COLNAMES_CUSTOM_CSV_FIX = {
    "x_col_name_fix": "x",
    "y_col_name_fix": "y",
    "x_col_name_fix_stim": "char_x_center",
    "x_start_col_name_fix_stim": "char_xmin",
    "x_end_col_name_fix_stim": "char_xmax",
    "y_col_name_fix_stim": "char_y_center",
    "y_start_col_name_fix_stim": "char_ymin",
    "y_end_col_name_fix_stim": "char_ymax",
    "char_col_name_fix_stim": "char",
    "trial_id_col_name_fix": "trial_id",
    "trial_id_col_name_stim": "trial_id",
    "subject_col_name_fix": "subject",
    "line_num_col_name_stim": "assigned_line",
    "time_start_col_name_fix": "start",
    "time_stop_col_name_fix": "stop",
}

COLNAME_CANDIDATES_CUSTOM_CSV_FIX = {
    "x_col_name_fix": ["x", "xs"],
    "y_col_name_fix": ["y", "ys"],
    "trial_id_col_name_fix": ["trial_id", "trialid", "trial", "trial_num", "id"],
    "subject_col_name_fix": ["subject", "sub", "subid", "sub_id"],
    "time_start_col_name_fix": ["start", "start_time", "ts", "t_start", "starttime"],
    "time_stop_col_name_fix": ["end","stop", "stop_time", "te", "t_end", "t_stop", "stoptime"],
}
COLNAME_CANDIDATES_CUSTOM_CSV_FIX_DEFAULT = {k: v[0] for k, v in COLNAME_CANDIDATES_CUSTOM_CSV_FIX.items()}

COLNAMES_CUSTOM_CSV_STIM = {
    "x_col_name_fix_stim": ["char_x_center", "xm"],
    "x_start_col_name_fix_stim": ["char_xmin", "xs", "xstart", "xmin"],
    "x_end_col_name_fix_stim": ["char_xmax", "xe", "xend", "xstop", "xmax"],
    "y_col_name_fix_stim": ["char_y_center", "ym"],
    "y_start_col_name_fix_stim": ["char_ymin", "ys", "ystart", "ymin"],
    "y_end_col_name_fix_stim": ["char_ymax", "ye", "yend", "ystop", "ymax"],
    "char_col_name_fix_stim": ["char", "letter", "let", "character"],
    "trial_id_col_name_stim": ["trial_id", "trialid", "trial", "trial_num", "id"],
    "line_num_col_name_stim": ["assigned_line", "line"],
}
COLNAMES_CUSTOM_CSV_STIM_DEFAULT = {k: v[0] for k, v in COLNAMES_CUSTOM_CSV_STIM.items()}
FIX_COL_NAMES_FOR_SEARCH = [
    "x",
    "y",
    "start_time",
    "end_time",
    "stop_time",
    "line",
    "subject",
    "trialid",
    "fixid",
    "fixnum",
    "fixation_number",
    "num",
]
STIM_COL_NAMES_FOR_SEARCH = [
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "xcenter",
    "ycenter",
    "char",
    "line",
    "subject",
    "trialid",
    "num",
]

SHORT_FIX_CLEAN_OPTIONS = ["Merge", "Merge then discard", "Discard", "Leave unchanged"]
DEFAULT_LONG_FIX_THRESHOLD = 800
DEFAULT_MERGE_DISTANCE_THRESHOLD = 1

if "results" not in st.session_state:
    st.session_state["results"] = {}


@st.cache_resource
def create_logger(name, level="DEBUG", file=None):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    if sum([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]) == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                "%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s >>> %(message)s",
                "%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(ch)
    if file is not None:
        if sum([isinstance(handler, logging.FileHandler) for handler in logger.handlers]) == 0:
            ch = logging.FileHandler(file, "a")
            ch.setFormatter(
                logging.Formatter(
                    "%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s >>> %(message)s",
                    "%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(ch)
    logger.debug("Logger added")
    return logger


if "logger" not in st.session_state:
    st.session_state["logger"] = create_logger(name="app", level="DEBUG", file="log_for_app.log")


def add_fonts(font_dirs=["fonts"]):
    try:
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        if len(font_files) > 0:
            for font_file in font_files:
                font_manager.fontManager.addfont(font_file)
            st.session_state["logger"].info(f"done importing font_files {font_files}")
            st.session_state["fonts imported"] = font_files
    except Exception as e:
        st.session_state["logger"].warning(f"Adding fonts failed for {font_dirs}, please add font files to ./fonts")
        st.session_state["logger"].warning(e)
        st.session_state["fonts imported"] = None


pl.Path("fonts").mkdir(exist_ok=True)
if "fonts imported" not in st.session_state or st.session_state["fonts imported"] is None:
    add_fonts(font_dirs=["fonts"])


@st.cache_data
def download_example_ascs(examples_folder, asc_zip_filename, osf_download_link, examples_folder_path):
    return ut.download_example_ascs(examples_folder, asc_zip_filename, osf_download_link, examples_folder_path)


EXAMPLE_ASC_FILES = download_example_ascs(
    EXAMPLES_FOLDER, EXAMPLES_ASC_ZIP_FILENAME, OSF_DOWNLOAD_LINK, EXAMPLES_FOLDER_PATH
)


@st.cache_data
def unzip_testfiles(folderpath):
    for f in folderpath.glob("*.zip"):
        with zipfile.ZipFile(f, "r") as zip_ref:
            zip_ref.extractall(EXAMPLES_FOLDER)
    return list(folderpath.glob("*.asc"))


EXAMPLE_ASC_FILES = unzip_testfiles(EXAMPLES_FOLDER_PATH)

matplotlib_plot_df = st.cache_data(matplotlib_plot_df)


def in_st_nn(name):
    if name in st.session_state and st.session_state[name] is not None:
        return True
    else:
        return False


def get_trial_skip_reason(trial: dict, default_reason: str) -> str:
    status = trial.get("correction_status", {}) if isinstance(trial, dict) else {}
    trial_reasons = status.get("trial_level_reasons", []) if isinstance(status, dict) else []
    algo_status = status.get("algorithms", {}) if isinstance(status, dict) else {}
    algo_reasons = []
    if isinstance(algo_status, dict):
        for algo_name, info in algo_status.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") in {"failed", "skipped"}:
                reason = info.get("reason")
                if reason:
                    algo_reasons.append(f"{algo_name}: {reason}")

    reasons = [r for r in [*trial_reasons, *algo_reasons] if isinstance(r, str) and len(r) > 0]
    if len(reasons) == 0:
        return default_reason
    return f"{default_reason} Reason: {' | '.join(reasons)}"


@st.cache_resource
def get_cached_models(dist_models_folder: pl.Path):
    return set_up_models(dist_models_folder)


plotly_plot_with_image = st.cache_data(ut.plotly_plot_with_image)
plot_y_corr = st.cache_data(ut.plot_y_corr)
plot_fix_measure = st.cache_data(ut.plot_fix_measure)


def save_to_zips(folder, pattern, savename, delete_after_zip=False, required_string: str = None):
    if os.path.exists(RESULTS_FOLDER.joinpath(savename)):
        mode = "a"
    else:
        mode = "w"
    with zipfile.ZipFile(RESULTS_FOLDER.joinpath(savename), mode=mode) as archive:
        for idx, f in enumerate(folder.glob(pattern)):
            if (required_string is None or required_string in str(f)) and f.stem not in [
                pl.Path(x).stem for x in archive.namelist()
            ]:
                archive.write(f)
                if delete_after_zip:
                    try:
                        os.remove(f)
                    except Exception as e:
                        st.session_state["logger"].warning(e)
                        st.session_state["logger"].warning(f"Failed to delete {f}")
            if idx == 1:
                mode = "a"
    st.session_state["logger"].info(f"Done zipping for pattern {pattern}")


def create_master_results_zipfile(
    folder, 
    individual_zips, 
    master_zipname, 
    file_type_prefix: str = "asc"
):
    """
    Create a master zipfile containing both individual file zips and combined results in organized subfolders.
    
    Args:
        folder: The results folder path
        individual_zips: List of individual zipfile paths to include
        master_zipname: Name for the master zipfile (e.g., 'all_asc_results_YYYYMMDD.zip')
        file_type_prefix: Either 'asc' or 'csv' for logging purposes
    
    Returns:
        Path to the created master zipfile
    """
    master_zip_path = folder.joinpath(master_zipname)
    
    st.session_state["logger"].info(f"Creating master {file_type_prefix} results zipfile: {master_zipname}")
    
    with zipfile.ZipFile(master_zip_path, mode="w") as master_archive:
        # Add individual file zips to /individual/ subfolder
        for individual_zip in individual_zips:
            if os.path.exists(individual_zip):
                individual_zip_path = pl.Path(individual_zip)
                arcname = f"individual/{individual_zip_path.name}"
                master_archive.write(individual_zip_path, arcname=arcname)
                st.session_state["logger"].info(f"Added {individual_zip_path.name} to master zip at {arcname}")
        
        # Add combined result files to /combined/ subfolder
        combined_patterns = [
            "*_comb_fixations.csv",
            "*_comb_saccades.csv",
            "*_comb_chars.csv",
            "*_comb_words.csv",
            "*_comb_sentences.csv",
            "*_comb_metadata.json",
            "*_trials_summary.csv",
            "*_subjects_summary.csv",
        ]
        
        added_combined_files = set()
        for pattern in combined_patterns:
            for file_path in folder.glob(pattern):
                if file_path.name not in added_combined_files:
                    arcname = f"combined/{file_path.name}"
                    master_archive.write(file_path, arcname=arcname)
                    added_combined_files.add(file_path.name)
                    st.session_state["logger"].info(f"Added {file_path.name} to master zip at {arcname}")
    
    st.session_state["logger"].info(f"Master {file_type_prefix} results zipfile created: {master_zip_path}")
    return str(master_zip_path)


def make_json_serializable(obj, _stats=None):
    """
    Recursively convert non-JSON-serializable objects to serializable types.
    
    Handles:
    - numpy integers (int8, int16, int32, int64, etc.) -> int
    - numpy floats (float16, float32, float64, etc.) -> float
    - numpy arrays -> list
    - numpy bool -> bool
    - pandas Series -> list
    - pandas DataFrame -> dict
    - sets -> list
    - bytes -> string (decoded)
    - nested dicts and lists
    
    Args:
        obj: The object to make JSON serializable
        _stats: Internal dict to track conversion statistics (optional)
        
    Returns:
        A tuple of (serializable_object, conversion_stats) if called at top level,
        or just the serializable object for recursive calls
    """
    # Initialize stats tracking on first call
    is_top_level = _stats is None
    if is_top_level:
        _stats = {
            'numpy_int': 0,
            'numpy_float': 0,
            'numpy_array': 0,
            'numpy_bool': 0,
            'pandas_series': 0,
            'pandas_dataframe': 0,
            'set': 0,
            'bytes': 0,
            'fallback_str': 0,
            'fallback_none': 0,
        }
    
    if isinstance(obj, dict):
        result = {key: make_json_serializable(value, _stats) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [make_json_serializable(item, _stats) for item in obj]
    elif isinstance(obj, set):
        _stats['set'] += 1
        result = [make_json_serializable(item, _stats) for item in obj]
    elif isinstance(obj, np.integer):
        _stats['numpy_int'] += 1
        result = int(obj)
    elif isinstance(obj, np.floating):
        _stats['numpy_float'] += 1
        result = float(obj)
    elif isinstance(obj, np.ndarray):
        _stats['numpy_array'] += 1
        result = make_json_serializable(obj.tolist(), _stats)
    elif isinstance(obj, np.bool_):
        _stats['numpy_bool'] += 1
        result = bool(obj)
    elif isinstance(obj, pd.Series):
        _stats['pandas_series'] += 1
        result = make_json_serializable(obj.tolist(), _stats)
    elif isinstance(obj, pl.Path):
        result = str(obj)
    elif isinstance(obj, pd.DataFrame):
        _stats['pandas_dataframe'] += 1
        result = make_json_serializable(obj.to_dict(orient='records'), _stats)
    elif isinstance(obj, bytes):
        _stats['bytes'] += 1
        result = obj.decode('utf-8', errors='replace')
    elif isinstance(obj, (str, int, float, bool, type(None))):
        result = obj
    else:
        # For any other type, try to convert to string as a fallback
        try:
            result = str(obj)
            _stats['fallback_str'] += 1
            if "logger" in st.session_state:
                st.session_state["logger"].debug(f"Fallback conversion to str: {type(obj).__name__}")
        except:
            result = None
            _stats['fallback_none'] += 1
            if "logger" in st.session_state:
                st.session_state["logger"].warning(f"Fallback conversion to None: {type(obj).__name__}")
    
    # Return stats on top-level call
    if is_top_level:
        return result, _stats
    return result


def call_subprocess(script_path, data):
    try:
        # Convert data to JSON-serializable format
        serializable_data, conversion_stats = make_json_serializable(data)
        st.session_state["logger"].info(f"Converted data to JSON-serializable format with stats: {conversion_stats}")
        
        # Log conversion statistics
        if "logger" in st.session_state:
            total_conversions = sum(conversion_stats.values())
            if total_conversions > 0:
                st.session_state["logger"].info(f"JSON serialization conversions: {conversion_stats}")
                if conversion_stats['fallback_str'] > 0 or conversion_stats['fallback_none'] > 0:
                    st.session_state["logger"].warning(
                        f"Used fallback conversions: {conversion_stats['fallback_str']} to str, "
                        f"{conversion_stats['fallback_none']} to None"
                    )
        
        json_data_in = json.dumps(serializable_data, cls=ut.NpEncoder)

        result = subprocess.run(["python", script_path], input=json_data_in, capture_output=True, text=True)
        ic(result.stderr)
        st.session_state["logger"].info(f"Got result from call_subprocess with return code {result.returncode}")
        if result.stdout and "error" not in result.stdout[:9]:
            result_data = json.loads(result.stdout)
        else:
            if result.stdout:
                st.session_state["logger"].warning("Subprocess returned error")
                st.session_state["logger"].warning(result.stdout)
            result_data = None
        if isinstance(result_data, dict) and "error" in result_data:
            st.session_state["logger"].warning(f"Subprocess returned error:\n---\n{result_data['error']}")
            result_data = None

        return result_data
    except Exception as e:
        st.session_state["logger"].error(f"ERROR: {e}\n traceback:\n{traceback.format_exc()}")
        return None


def ensure_dataframe(obj, *, label="data", context: str | None = None):
    if isinstance(obj, pd.DataFrame):
        return obj
    if obj is None:
        parts = [f"{label} is missing"]
        if context:
            parts.append(f"({context})")
        message = " ".join(parts) + "."
        logger = st.session_state.get("logger")
        if logger is not None and hasattr(logger, "error"):
            logger.error(message)
        st.warning(message)
        raise ValueError(message)
    return pd.DataFrame(obj)


def key_val_to_dataframe(obj):
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        try:
            df = pd.DataFrame(obj)
        except Exception as e:
            return obj
        return df
    else:
        return obj


def trial_vals_to_dfs(trial):
    trial2 = {}
    for k, v in trial.items():
        if "list" in k:
            trial2[k] = v
        elif "_df" in k:
            trial2[k] = pd.DataFrame(v)
        else:
            trial2[k] = key_val_to_dataframe(v)
    return trial2


def process_all_asc_files(
    asc_files,
    algo_choice_multi_asc,
    ias_files,
    close_gap_between_words,
    close_gap_between_lines,
    trial_start_keyword,
    end_trial_at_keyword,
    paragraph_trials_only,
    choice_handle_short_and_close_fix,
    discard_fixations_without_sfix,
    discard_far_out_of_text_fix,
    x_thres_in_chars,
    y_thresh_in_heights,
    short_fix_threshold,
    merge_distance_threshold: float,
    discard_long_fix: bool,
    discard_long_fix_threshold: int,
    discard_blinks: bool,
    measures_to_calculate_multi_asc: list,
    include_coords_multi_asc: bool,
    sent_measures_to_calculate_multi_asc: list,
    use_multiprocessing: bool,
    fix_cols_to_add_multi_asc: list,
    save_files_for_each_trial_individually: bool,
):
    asc_files_to_do = get_asc_filelist(asc_files)
    if len(asc_files_to_do) > 0:

        zipfiles_with_results = []
        asc_files_for_log = [a.name if hasattr(a, "name") else a for a in asc_files]
        st.session_state["logger"].info(f"found asc_files {asc_files_for_log}")

        all_fix_dfs_list = []
        all_sacc_dfs_list = []
        all_chars_dfs_list = []
        all_words_dfs_list = []
        all_sentence_dfs_list = []
        asc_files_so_far = []
        all_trials_by_subj = {}
        list_of_trial_lists = []
        list_of_lines = []
        total_num_trials = 0
        for asc_file in stqdm(asc_files_to_do, desc="Processing .asc files"):
            st.session_state["asc_file"] = asc_file
            if hasattr(asc_file, "name"):
                asc_file_stem = pl.Path(asc_file.name).stem
            else:
                asc_file_stem = pl.Path(asc_file).stem
            asc_files_so_far.append(asc_file_stem)
            st.session_state["logger"].info(f"processing asc_file {asc_file_stem}")
            trial_choices_single_asc, trials_by_ids, lines, asc_file, trials_dict = ut.get_trials_list(
                asc_file,
                close_gap_between_words=close_gap_between_words,
                close_gap_between_lines=close_gap_between_lines,
                ias_files=ias_files,
                trial_start_keyword=trial_start_keyword,
                end_trial_at_keyword=end_trial_at_keyword,
                paragraph_trials_only=paragraph_trials_only,
            )

            st.session_state["logger"].info(f"Found {len(trials_by_ids)} trials in {asc_file_stem}.")
            st.info(f"Found {len(trials_by_ids)} trials in {asc_file_stem}.")
            if len(trials_by_ids) > 0:
                total_num_trials += len(trials_by_ids)
                list_of_trial_lists.append(trials_by_ids)
                list_of_lines.append(lines)
                savestring = "-".join([f for f in asc_files_so_far])[:100]
                # Note: per-subject initialization happens when processing each trial
            else:
                st.info(f"No trials found in {asc_file_stem}. Skipping file.")
                continue

            for trial_id, trial in trials_by_ids.items():
                trial_start_idx, trial_end_idx = trial["trial_start_idx"] + 1, trial["trial_end_idx"]
                trial_lines = lines[trial_start_idx:trial_end_idx + 1]
                trial["trial_lines"] = trial_lines
            models_dict = {}
            if use_multiprocessing:
                st.session_state["logger"].info("Using multiprocessing")
                args = (
                    algo_choice_multi_asc,
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
                    CLASSIC_ALGOS_CFGS,
                    models_dict,
                    fix_cols_to_add_multi_asc,
                )
                out2 = call_subprocess("process_asc_files_in_multi_p.py", args)
                if out2 is None:
                    st.session_state["logger"].warning("Multiprocessing failed, falling back on single process")
                    out = out2
                else:
                    st.session_state["logger"].info(
                        f"Multiprocessing produced output of type {type(out2)} with length {len(out2)}"
                    )
                    out = []
                    for dffix, trial in out2:
                        dffix = pd.DataFrame(dffix)
                        trial = trial_vals_to_dfs(trial)
                        out.append((dffix, trial))

            if not use_multiprocessing or out is None:
                if (
                    "DIST" in algo_choice_multi_asc
                    or "Wisdom_of_Crowds_with_DIST" in algo_choice_multi_asc
                    or "DIST-Ensemble" in algo_choice_multi_asc
                    or "Wisdom_of_Crowds_with_DIST_Ensemble" in algo_choice_multi_asc
                ):
                    models_dict = get_cached_models(DIST_MODELS_FOLDER)
                dffixs = []
                trials = []
                for trial_id, trial in stqdm(trials_by_ids.items(), desc=f"Processing trials in {asc_file_stem}"):
                    dffix, trial = process_trial_choice(
                        trial,
                        algo_choice_multi_asc,
                        choice_handle_short_and_close_fix,
                        True,
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
                        CLASSIC_ALGOS_CFGS,
                        models_dict,
                        fix_cols_to_add_multi_asc,
                    )
                    dffixs.append(dffix)
                    trials.append(trial)
                out = zip(dffixs, trials)
            for dffix, trial in stqdm(out, desc=f"Aggregating results for file {asc_file_stem}"):
                if dffix.shape[0] < 2:
                    trial_id_for_msg = trial.get("trial_id", "unknown") if isinstance(trial, dict) else "unknown"
                    base_reason = (
                        f"trial {trial_id_for_msg} for file {asc_file_stem} failed because fixation dataframe only had {dffix.shape[0]} fixation after processing."
                    )
                    full_reason = get_trial_skip_reason(trial, base_reason)
                    st.warning(
                        full_reason
                    )
                    st.session_state["logger"].warning(
                        full_reason
                    )
                    continue
                fix_cols_to_keep = [
                    c
                    for c in dffix.columns
                    if (
                        (
                            any([lname in c for lname in ALL_FIX_MEASURES])
                            and any([lname in c for lname in fix_cols_to_add_multi_asc])
                        )
                        or (not any([lname in c for lname in ALL_FIX_MEASURES]))
                    )
                ]
                dffix = dffix.loc[:, fix_cols_to_keep].copy()
                trial_id = trial["trial_id"]
                saccade_df = ensure_dataframe(
                    trial.get("saccade_df"),
                    label="Saccade dataframe",
                    context=f"{asc_file_stem} / trial {trial_id}",
                )
                chars_source = trial.get("chars_df")
                if chars_source is None and "chars_list" in trial:
                    chars_source = trial["chars_list"]
                chars_df = ensure_dataframe(
                    chars_source,
                    label="Character stimulus data",
                    context=f"{asc_file_stem} / trial {trial_id}",
                )
                
                # Get subject from trial, fallback to asc_file_stem if not present
                subject = trial.get("subject", asc_file_stem)
                
                # Initialize subject entry in all_trials_by_subj if needed
                if subject not in all_trials_by_subj:
                    all_trials_by_subj[subject] = {}
                
                trial_for_comb = pop_large_trial_entries(all_trials_by_subj, subject, trial_id, trial)
                if "words_list" in trial:
                    if "own_word_measures_dfs_for_algo" in trial:
                        words_df = trial.pop("own_word_measures_dfs_for_algo")
                    else:
                        words_df = ensure_dataframe(
                            trial.get("words_list"),
                            label="Word stimulus data",
                            context=f"{asc_file_stem} / trial {trial_id}",
                        )
                else:
                    words_df = None
                if "own_sentence_measures_dfs_for_algo" in trial:
                    sent_measures_multi = trial["own_sentence_measures_dfs_for_algo"]
                else:
                    sent_measures_multi = None

                if "subject" in trial:
                    add_cols_from_trial_info(
                        asc_file_stem, trial_id, trial, dffix, saccade_df, chars_df, words_df, sent_measures_multi
                    )

                st.session_state["results"][f"{asc_file_stem}_{trial_id}"] = {
                    "trial": trial,
                    "dffix": dffix.copy(),
                }
                all_fix_dfs_list.append(dffix)
                all_sacc_dfs_list.append(saccade_df)
                st.session_state["results"][f"{asc_file_stem}_{trial_id}"]["chars_df"] = chars_df
                all_chars_dfs_list.append(chars_df)
                if words_df is not None:
                    st.session_state["results"][f"{asc_file_stem}_{trial_id}"]["words_df"] = words_df
                    all_words_dfs_list.append(words_df)
                if sent_measures_multi is not None:
                    st.session_state["results"][f"{asc_file_stem}_{trial_id}"][
                        "sent_measures_multi"
                    ] = sent_measures_multi
                    all_sentence_dfs_list.append(sent_measures_multi)

                if save_files_for_each_trial_individually:
                    savename = RESULTS_FOLDER.joinpath(asc_file_stem)  # TODO save word_measures here?
                    csv_name = f"{savename}_{trial_id}_fixations_df.csv"
                    csv_name = export_dataframe(dffix, csv_name)
                    csv_name = f"{savename}_{trial_id}_saccade_df.csv"
                    csv_name = export_dataframe(pd.DataFrame(trial["saccade_df"]), csv_name)
                    export_trial(trial)
                    csv_name = f"{savename}_{trial_id}_stimulus_df.csv"
                    export_dataframe(pd.DataFrame(trial["chars_list"]), csv_name)
                    ut.save_trial_to_json(trial_for_comb, RESULTS_FOLDER.joinpath(f"{asc_file_stem}_{trial_id}.json"))

            if os.path.exists(RESULTS_FOLDER.joinpath(f"{asc_file_stem}.zip")):
                os.remove(RESULTS_FOLDER.joinpath(f"{asc_file_stem}.zip"))
            save_to_zips(RESULTS_FOLDER, f"*{asc_file_stem}*.csv", f"{asc_file_stem}.zip", delete_after_zip=True)
            save_to_zips(RESULTS_FOLDER, f"*{asc_file_stem}*.json", f"{asc_file_stem}.zip", delete_after_zip=True)
            save_to_zips(RESULTS_FOLDER, f"*{asc_file_stem}*.png", f"{asc_file_stem}.zip", delete_after_zip=True)
            zipfiles_with_results += [str(x) for x in RESULTS_FOLDER.glob(f"{asc_file_stem}*.zip")]
        if len(all_fix_dfs_list) == 0:
            st.warning("All .asc files failed")
            st.session_state["logger"].info("All .asc files failed")
            return None, None, None, None, None, None, None, None, None, None
        results_keys = list(st.session_state["results"].keys())
        st.session_state["logger"].info(f"results_keys are {results_keys}")
        all_fix_dfs_concat = pd.concat(all_fix_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)
        droplist = ["num", "msg"]
        if discard_blinks:
            droplist += ["blink", "blink_before", "blink_after"]
        for col in droplist:
            if col in all_fix_dfs_concat.columns:
                all_fix_dfs_concat = all_fix_dfs_concat.drop(col, axis=1)
        all_sacc_dfs_concat = pd.concat(all_sacc_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)
        all_chars_dfs_concat = pd.concat(all_chars_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)
        if len(all_words_dfs_list) > 0:
            all_words_dfs_concat = pd.concat(all_words_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)

            word_cols = [
                c
                for c in [
                    "word_xmin",
                    "word_xmax",
                    "word_ymax",
                    "word_xmin",
                    "word_ymin",
                    "word_x_center",
                    "word_y_center",
                ]
                if c in all_words_dfs_concat.columns
            ]
            all_words_dfs_concat = all_words_dfs_concat.drop(columns=word_cols)
        else:
            all_words_dfs_concat = pd.DataFrame()
        if len(all_sentence_dfs_list) > 0:
            all_sentence_dfs_concat = pd.concat(all_sentence_dfs_list, axis=0).reset_index(
                drop=True, allow_duplicates=True
            )
            # all_sentence_dfs_concat = all_sentence_dfs_concat.dropna(axis=0,how='any',subset=['sentence_number']) #TODO this should now be needed
        else:
            all_sentence_dfs_concat = pd.DataFrame()
        if not all_fix_dfs_concat.empty:
            savestring = "-".join(
                [pl.Path(f.name).stem if hasattr(f, "name") else pl.Path(str(f)).stem for f in asc_files_to_do]
            )[:100]
            correction_summary_df_all_multi, cleaning_summary_df_all_multi, trials_quick_meta_df = (
                get_summaries_from_trials(all_trials_by_subj)
            )
            correction_summary_df_all_multi = correction_summary_df_all_multi.merge(
                cleaning_summary_df_all_multi, on=["subject", "trial_id"]
            )
            if "question_correct" in all_words_dfs_concat.columns:
                all_words_dfs_concat["question_correct"] = all_words_dfs_concat["question_correct"].astype("boolean")
            trials_summary = pf.aggregate_trials(
                all_fix_dfs_concat, all_words_dfs_concat, all_trials_by_subj, algo_choice_multi_asc
            )
            trials_summary = trials_summary.drop(columns="subject_trialID")
            trials_summary = correction_summary_df_all_multi.merge(trials_summary, on=["subject", "trial_id"])
            trials_summary = reorder_columns(trials_summary, ["subject", "trial_id", "item", "condition"])
            trials_summary.to_csv(RESULTS_FOLDER / f"{savestring}_trials_summary.csv")
            subjects_summary = pf.aggregate_subjects(trials_summary, algo_choice_multi_asc)
            subjects_summary.to_csv(RESULTS_FOLDER / f"{savestring}_subjects_summary.csv")
            ut.save_trial_to_json(
                {
                    k_outer: {
                        k: {
                            prop: val
                            for prop, val in v.items()
                            if isinstance(val, (int, float, str, list, tuple, bool, dict))
                        }
                        for k, v in v_outer.items()
                    }
                    for k_outer, v_outer in all_trials_by_subj.items()
                },
                RESULTS_FOLDER / f"{savestring}_comb_metadata.json",
            )
            if "msg" in all_fix_dfs_concat.columns:
                all_fix_dfs_concat = all_fix_dfs_concat.drop(columns="msg")
            all_fix_dfs_concat = all_fix_dfs_concat.drop(columns="subject_trialID")
            all_fix_dfs_concat = reorder_columns(
                all_fix_dfs_concat,
                [
                    "subject",
                    "trial_id",
                    "item",
                    "condition",
                    "fixation_number",
                    "duration",
                    "start_uncorrected",
                    "stop_uncorrected",
                    "start_time",
                    "stop_time",
                    "corrected_start_time",
                    "corrected_end_time",
                ],
            )
            all_fix_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_fixations.csv")
            if "msg" in all_sacc_dfs_concat.columns:
                all_sacc_dfs_concat = all_sacc_dfs_concat.drop(columns="msg")
            all_sacc_dfs_concat = reorder_columns(
                all_sacc_dfs_concat, ["subject", "trial_id", "item", "condition", "num"]
            )
            all_sacc_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_saccades.csv")
            all_chars_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_chars.csv")
            if not all_words_dfs_concat.empty:
                all_words_dfs_concat = all_words_dfs_concat.drop(columns="subject_trialID")
                all_words_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_words.csv")
            if not all_sentence_dfs_concat.empty:
                all_sentence_dfs_concat = all_sentence_dfs_concat.drop(columns="subject_trialID")
                all_sentence_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_sentences.csv")

            # Create master zipfile with organized subfolders
            date_str = datetime.now().strftime("%Y%m%d")
            master_zip_name = f"all_asc_results_{date_str}.zip"
            master_zip_path = create_master_results_zipfile(
                RESULTS_FOLDER,
                zipfiles_with_results,
                master_zip_name,
                file_type_prefix="asc"
            )
            # Add master zip to the list for user download
            zipfiles_with_results.append(master_zip_path)
        else:
            trials_summary = None
            subjects_summary = None
    return (
        list_of_trial_lists,
        list_of_lines,
        results_keys,
        zipfiles_with_results,
        all_fix_dfs_concat,
        all_sacc_dfs_concat,
        all_chars_dfs_concat,
        all_words_dfs_concat,
        all_sentence_dfs_concat,
        all_trials_by_subj,
        trials_summary,
        subjects_summary,
        trials_quick_meta_df,
    )


def process_all_csv_files(
    csv_files,
    image_files,
    algo_choice_multi_csv,
    close_gap_between_words,
    close_gap_between_lines,
    choice_handle_short_and_close_fix,
    discard_fixations_without_sfix,
    discard_far_out_of_text_fix,
    x_thres_in_chars,
    y_thresh_in_heights,
    short_fix_threshold,
    merge_distance_threshold: float,
    discard_long_fix: bool,
    discard_long_fix_threshold: int,
    discard_blinks: bool,
    measures_to_calculate_multi_csv: list,
    include_coords_multi_csv: bool,
    sent_measures_to_calculate_multi_csv: list,
    fix_cols_to_add_multi_csv: list,
    save_files_for_each_trial_individually: bool,
):
    image_files_dict = {pl.Path(img.name).stem: img for img in image_files} if image_files else {}

    zipfiles_with_results = []
    csv_files_for_log = [c.name if hasattr(c, "name") else c for c in csv_files]
    st.session_state["logger"].info(f"found csv_files {csv_files_for_log}")

    all_fix_dfs_list = []
    all_sacc_dfs_list = []
    all_chars_dfs_list = []
    all_words_dfs_list = []
    all_sentence_dfs_list = []
    csv_files_so_far = []
    all_trials_by_subj = {}
    list_of_trial_lists = []
    total_num_trials = 0

    requires_models = any(
        algo
        for algo in algo_choice_multi_csv
        if algo
        and (
            "DIST" in algo
            or "Wisdom_of_Crowds_with_DIST" in algo
            or "DIST-Ensemble" in algo
            or "Wisdom_of_Crowds_with_DIST_Ensemble" in algo
        )
    )
    models_dict = get_cached_models(DIST_MODELS_FOLDER) if requires_models else {}

    for csv_file in stqdm(csv_files, desc="Processing .csv files"):
        st.session_state["csv_file"] = csv_file
        if hasattr(csv_file, "name"):
            csv_file_stem = pl.Path(csv_file.name).stem
        else:
            csv_file_stem = pl.Path(str(csv_file)).stem
        csv_files_so_far.append(csv_file_stem)
        st.session_state["logger"].info(f"processing csv_file {csv_file_stem}")

        try:
            dffix_raw = load_csv_delim_agnostic(csv_file)
        except Exception as exc:
            st.session_state["logger"].warning(f"Failed to load {csv_file_stem}: {exc}")
            st.warning(f"Loading {csv_file_stem} failed. See log for details.")
            continue

        try:
            dffix_prepped = prepare_fixation_dataframe_for_multi_csv(dffix_raw)
        except ValueError as err:
            st.session_state["logger"].warning(f"{csv_file_stem} skipped: {err}")
            st.warning(f"{csv_file_stem}: {err}")
            continue

        # Store whether subject column existed in the original CSV
        has_subject_column = "subject" in dffix_prepped.columns
        
        # Process subject column: clean it if it exists, otherwise use csv_file_stem as default
        if not has_subject_column:
            # No subject column - use CSV filename as default subject for all rows
            dffix_prepped["subject"] = csv_file_stem
        else:
            # Clean existing subject column
            dffix_prepped["subject"] = stringify_column_keep_decimal_if_needed(dffix_prepped["subject"])
            dffix_prepped["subject"] = dffix_prepped["subject"].replace(
                {"nan": np.nan, "NaN": np.nan, "": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan}
            )
            # Fill missing subjects with csv_file_stem
            dffix_prepped["subject"] = dffix_prepped["subject"].fillna(csv_file_stem)
            dffix_prepped["subject"] = dffix_prepped["subject"].astype(str).str.strip()
            # Replace empty/invalid string values with csv_file_stem
            dffix_prepped.loc[
                dffix_prepped["subject"].isin(["", "nan", "NaN", "None", "NA", "N/A"]), "subject"
            ] = csv_file_stem
        
        dffix_prepped = dffix_prepped.sort_values("corrected_start_time").reset_index(drop=True)
        
        # Now group by BOTH subject and trial_id to handle all subject-trial combinations
        grouped = dffix_prepped.groupby(["subject", "trial_id"], dropna=False)
        trials_by_ids: dict[str, dict] = {}
        num_trials_in_file = 0

        # Iterate over each (subject, trial_id) group
        for (subject_value, trial_id_value), subdf in grouped:
            if subdf.empty:
                continue
            
            # The trial_id should match an image filename
            trial_id_str = str(trial_id_value)
            subject_str = str(subject_value) if pd.notna(subject_value) else csv_file_stem
            
            # Ensure subject is valid (this should already be handled above, but double-check)
            if subject_str.strip().lower() in {"", "nan", "none", "n/a"}:
                subject_str = csv_file_stem

            # Look up the image using the trial_id
            image_obj = image_files_dict.get(trial_id_str)
            if image_obj is None:
                st.warning(f"No image found for trial {trial_id_str} (subject: {subject_str}) in {csv_file_stem}. Skipping trial.")
                st.session_state["logger"].warning(
                    f"Missing stimulus image for trial {trial_id_str} (subject: {subject_str}) in {csv_file_stem}"
                )
                continue
            
            if hasattr(image_obj, "seek"):
                image_obj.seek(0)

            try:
                stim_df_raw = get_cached_or_run_ocr(
                    image_obj,
                    image_name=getattr(image_obj, "name", f"{trial_id_str}.png"),
                    trial_label=f"{csv_file_stem}:{subject_str}:{trial_id_str}",
                )
                stim_df_raw.to_csv(RESULTS_FOLDER / f"{trial_id_str}_stimdf_from_OCR.csv")
                stim_df = prepare_stimulus_dataframe_for_multi_csv(stim_df_raw, trial_id_str)
            except ValueError as err:
                st.warning(f"Stimulus extraction failed for trial {trial_id_str}: {err}")
                st.session_state["logger"].warning(
                    f"Stimulus extraction failed for trial {trial_id_str} in {csv_file_stem}: {err}"
                )
                continue

            trial = make_trial_from_stimulus_df(
                stim_df,
                csv_file_stem,
                trial_id_str,
                close_gaps_between_words=close_gap_between_words,
                close_gaps_between_lines=close_gap_between_lines,
            )
            trial["filename"] = csv_file_stem
            trial["subject"] = subject_str
            trial["plot_file"] = str(
                PLOTS_FOLDER.joinpath(f"{subject_str}_{trial_id_str}_2ndInput_chars_channel_sep.png")
            )

            words_df_for_trial = pd.DataFrame(trial["words_list"])
            if "word_number" not in words_df_for_trial.columns:
                words_df_for_trial["word_number"] = np.arange(words_df_for_trial.shape[0])
            trial["words_list"] = words_df_for_trial.to_dict("records")

            chars_df_full = pd.DataFrame(trial["chars_list"])
            chars_df_full = add_popEye_cols_to_chars_df(chars_df_full)
            trial["chars_df"] = chars_df_full
            trial["chars_list"] = chars_df_full.to_dict("records")
            trial["y_char_unique"] = list(chars_df_full["char_y_center"].sort_values().unique())

            subdf = subdf.reset_index(drop=True).copy()
            subdf["subject"] = subject_str
            subdf["trial_id"] = trial_id_str
            if "blink" not in subdf.columns:
                subdf["blink"] = False
            if "start_uncorrected" not in subdf.columns:
                subdf["start_uncorrected"] = subdf["corrected_start_time"]
            if "stop_uncorrected" not in subdf.columns:
                subdf["stop_uncorrected"] = subdf["corrected_end_time"]

            for meta_col in ["item", "condition"]:
                if meta_col in subdf.columns and meta_col not in trial:
                    trial[meta_col] = subdf[meta_col].iloc[0]

            if "item" not in trial:
                trial["item"] = None
            if "condition" not in trial:
                trial["condition"] = None

            trial["dffix"] = subdf.copy()
            start_time_clean = time.time()
            dffix_cleaned, trial = clean_dffix_own(
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
                subdf.copy(),
            )
            st.session_state["logger"].info(f"Cleaning for {trial_id_str} took {time.time() - start_time_clean:.2f}s")

            if dffix_cleaned.shape[0] < 2:
                trial.setdefault("correction_status", {}).setdefault("trial_level_reasons", []).append(
                    f"Only {dffix_cleaned.shape[0]} fixation(s) remained after cleaning; minimum of 2 required for correction."
                )
                base_reason = (
                    f"trial {trial_id_str} for file {csv_file_stem} and subject {trial.get('subject', 'unknown')} discarded because fewer than 2 fixations remained after cleaning ({dffix_cleaned.shape[0]})."
                )
                full_reason = get_trial_skip_reason(trial, base_reason)
                st.warning(
                    full_reason
                )
                st.session_state["logger"].warning(
                    full_reason
                )
                continue

            fix_cols_to_keep = [
                c
                for c in dffix_cleaned.columns
                if (
                    (
                        any([lname in c for lname in ALL_FIX_MEASURES])
                        and any([lname in c for lname in fix_cols_to_add_multi_csv])
                    )
                    or (not any([lname in c for lname in ALL_FIX_MEASURES]))
                )
            ]
            dffix_cleaned = dffix_cleaned.loc[:, fix_cols_to_keep].copy()

            try:
                start_time_corr = time.time()
                dffix_corrected = correct_df(
                    dffix_cleaned,
                    algo_choice_multi_csv,
                    trial,
                    for_multi=True,
                    is_outside_of_streamlit=False,
                    classic_algos_cfg=CLASSIC_ALGOS_CFGS,
                    models_dict=models_dict,
                    measures_to_calculate_multi_asc=measures_to_calculate_multi_csv,
                    include_coords_multi_asc=include_coords_multi_csv,
                    sent_measures_to_calc_multi=sent_measures_to_calculate_multi_csv,
                    fix_cols_to_add=fix_cols_to_add_multi_csv,
                )
                st.session_state["logger"].info(f"Correction for subject {trial.get('subject', 'unknown')} and trial {trial_id_str} took {time.time() - start_time_corr:.2f}s")
            except Exception as err:
                st.session_state["logger"].warning(
                    f"Correction failed for subject {trial.get('subject', 'unknown')} and trial {trial_id_str} in {csv_file_stem}: {err}", exc_info=True
                )
                st.warning(f"Correction failed for subject {trial.get('subject', 'unknown')} and trial {trial_id_str}. Skipping. See log for details.")
                continue

            requested_algos = algo_choice_multi_csv if isinstance(algo_choice_multi_csv, list) else [algo_choice_multi_csv]
            requested_algos = [algo for algo in requested_algos if algo]
            valid_algos = [
                algo
                for algo in requested_algos
                if (f"y_{algo}" in dffix_corrected.columns and not dffix_corrected[f"y_{algo}"].isna().all())
            ]
            if len(requested_algos) > 0 and len(valid_algos) == 0:
                base_reason = (
                    f"Skipping subject {subject_str} trial {trial_id_str} because no selected algorithm produced valid corrected y values."
                )
                full_reason = get_trial_skip_reason(trial, base_reason)
                st.warning(full_reason)
                st.session_state["logger"].warning(full_reason)
                continue
            if len(valid_algos) < len(requested_algos):
                skipped_algos = [algo for algo in requested_algos if algo not in valid_algos]
                st.warning(
                    f"Subject {subject_str} trial {trial_id_str}: some algorithms were skipped ({', '.join(skipped_algos)})."
                )

            trial["dffix"] = dffix_corrected.copy()
            saccade_df_dict = trial.get("saccade_df", {})
            saccade_df = pd.DataFrame(saccade_df_dict) if len(saccade_df_dict) > 0 else pd.DataFrame()
            chars_df = pd.DataFrame(trial["chars_list"])
            
            # Initialize subject entry in all_trials_by_subj if needed
            if subject_str not in all_trials_by_subj:
                all_trials_by_subj[subject_str] = {}
            
            trial_for_comb = pop_large_trial_entries(all_trials_by_subj, subject_str, trial_id_str, trial)

            if "own_word_measures_dfs_for_algo" in trial:
                words_df = trial.pop("own_word_measures_dfs_for_algo")
            elif "words_list" in trial:
                words_df = pd.DataFrame(trial["words_list"])
            else:
                words_df = None
            sent_measures_multi = trial.pop("own_sentence_measures_dfs_for_algo", None)

            add_cols_from_trial_info(
                csv_file_stem,
                trial_id_str,
                trial,
                dffix_corrected,
                saccade_df.copy() if not saccade_df.empty else pd.DataFrame(),
                chars_df.copy(),
                words_df.copy() if words_df is not None else None,
                sent_measures_multi.copy() if isinstance(sent_measures_multi, pd.DataFrame) else sent_measures_multi,
            )

            result_key = f"{subject_str}_{trial_id_str}"
            st.session_state["results"][result_key] = {"trial": trial, "dffix": dffix_corrected.copy()}
            st.session_state["results"][result_key]["chars_df"] = chars_df.copy()
            if words_df is not None:
                st.session_state["results"][result_key]["words_df"] = words_df.copy()
            if sent_measures_multi is not None:
                st.session_state["results"][result_key]["sent_measures_multi"] = (
                    sent_measures_multi.copy() if isinstance(sent_measures_multi, pd.DataFrame) else sent_measures_multi
                )

            all_fix_dfs_list.append(dffix_corrected.copy())
            if not saccade_df.empty:
                all_sacc_dfs_list.append(saccade_df.copy())
            all_chars_dfs_list.append(chars_df.copy())
            if words_df is not None and not words_df.empty:
                all_words_dfs_list.append(words_df.copy())
            if isinstance(sent_measures_multi, pd.DataFrame) and not sent_measures_multi.empty:
                all_sentence_dfs_list.append(sent_measures_multi.copy())

            if save_files_for_each_trial_individually:
                start_time_save = time.time()
                savename = RESULTS_FOLDER.joinpath(csv_file_stem)
                csv_name = f"{savename}_{trial_id_str}_fixations_df.csv"
                csv_name = export_dataframe(dffix_corrected.copy(), csv_name)
                if not saccade_df.empty:
                    csv_name = f"{savename}_{trial_id_str}_saccade_df.csv"
                    csv_name = export_dataframe(saccade_df.copy(), csv_name)
                export_trial(trial)
                csv_name = f"{savename}_{trial_id_str}_stimulus_df.csv"
                export_dataframe(chars_df.copy(), csv_name)
                ut.save_trial_to_json(trial_for_comb, RESULTS_FOLDER.joinpath(f"{csv_file_stem}_{trial_id_str}.json"))
                st.session_state["logger"].info(f"Saving files for {trial_id_str} took {time.time() - start_time_save:.2f}s")

            trials_by_ids[trial_id_str] = trial
            num_trials_in_file += 1

        if num_trials_in_file == 0:
            st.info(f"No trials found in {csv_file_stem}. Skipping file.")
            continue

        total_num_trials += num_trials_in_file
        list_of_trial_lists.append(trials_by_ids)

        if os.path.exists(RESULTS_FOLDER.joinpath(f"{csv_file_stem}.zip")):
            os.remove(RESULTS_FOLDER.joinpath(f"{csv_file_stem}.zip"))
        
        start_time_zip = time.time()
        save_to_zips(RESULTS_FOLDER, f"*{csv_file_stem}*.csv", f"{csv_file_stem}.zip", delete_after_zip=True)
        save_to_zips(RESULTS_FOLDER, f"*{csv_file_stem}*.json", f"{csv_file_stem}.zip", delete_after_zip=True)
        save_to_zips(RESULTS_FOLDER, f"*{csv_file_stem}*.png", f"{csv_file_stem}.zip", delete_after_zip=True)
        st.session_state["logger"].info(f"Zipping {csv_file_stem} took {time.time() - start_time_zip:.2f}s")
        zipfiles_with_results += [str(x) for x in RESULTS_FOLDER.glob(f"{csv_file_stem}*.zip")]

    if len(all_fix_dfs_list) == 0:
        st.warning("All .csv files failed")
        st.session_state["logger"].info("All .csv files failed")
        return None, None, None, None, None, None, None, None, None, None, None, None, None

    results_keys = list(st.session_state["results"].keys())
    st.session_state["logger"].info(f"results_keys are {results_keys}")
    all_fix_dfs_concat = pd.concat(all_fix_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)
    droplist = ["num", "msg"]
    if discard_blinks:
        droplist += ["blink", "blink_before", "blink_after"]
    for col in droplist:
        if col in all_fix_dfs_concat.columns:
            all_fix_dfs_concat = all_fix_dfs_concat.drop(col, axis=1)

    if "subject_trialID" in all_fix_dfs_concat.columns:
        duplicate_mask = all_fix_dfs_concat["subject_trialID"].duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_info = (
                all_fix_dfs_concat.loc[duplicate_mask, ["subject", "trial_id", "subject_trialID"]]
                .drop_duplicates()
                .sort_values("subject_trialID")
            )
            logger = st.session_state.get("logger")
            if logger is not None:
                logger.warning(
                    "Duplicate subject_trialID entries detected in combined fixations: %s",
                    duplicate_info.to_dict("records"),
                )

    all_sacc_dfs_concat = (
        pd.concat(all_sacc_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)
        if len(all_sacc_dfs_list) > 0
        else pd.DataFrame()
    )
    all_chars_dfs_concat = pd.concat(all_chars_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)
    if len(all_words_dfs_list) > 0:
        all_words_dfs_concat = pd.concat(all_words_dfs_list, axis=0).reset_index(drop=True, allow_duplicates=True)

        word_cols = [
            c
            for c in [
                "word_xmin",
                "word_xmax",
                "word_ymax",
                "word_xmin",
                "word_ymin",
                "word_x_center",
                "word_y_center",
            ]
            if c in all_words_dfs_concat.columns
        ]
        all_words_dfs_concat = all_words_dfs_concat.drop(columns=word_cols)
    else:
        all_words_dfs_concat = pd.DataFrame()
    if len(all_sentence_dfs_list) > 0:
        all_sentence_dfs_concat = pd.concat(all_sentence_dfs_list, axis=0).reset_index(
            drop=True, allow_duplicates=True
        )
    else:
        all_sentence_dfs_concat = pd.DataFrame()

    trials_summary = None
    subjects_summary = None
    trials_quick_meta_df = pd.DataFrame()

    if not all_fix_dfs_concat.empty:
        savestring = "-".join(
            [pl.Path(c.name).stem if hasattr(c, "name") else pl.Path(str(c)).stem for c in csv_files]
        )[:100]
        correction_summary_df_all_multi, cleaning_summary_df_all_multi, trials_quick_meta_df = (
            get_summaries_from_trials(all_trials_by_subj)
        )
        correction_summary_df_all_multi = correction_summary_df_all_multi.merge(
            cleaning_summary_df_all_multi, on=["subject", "trial_id"]
        )
        if "question_correct" in all_words_dfs_concat.columns:
            all_words_dfs_concat["question_correct"] = all_words_dfs_concat["question_correct"].astype("boolean")
        trials_summary = pf.aggregate_trials(
            all_fix_dfs_concat, all_words_dfs_concat, all_trials_by_subj, algo_choice_multi_csv
        )
        trials_summary = trials_summary.drop(columns="subject_trialID")
        trials_summary = correction_summary_df_all_multi.merge(trials_summary, on=["subject", "trial_id"])
        trials_summary = reorder_columns(trials_summary, ["subject", "trial_id", "item", "condition"])
        trials_summary.to_csv(RESULTS_FOLDER / f"{savestring}_trials_summary.csv")
        subjects_summary = pf.aggregate_subjects(trials_summary, algo_choice_multi_csv)
        subjects_summary.to_csv(RESULTS_FOLDER / f"{savestring}_subjects_summary.csv")
        ut.save_trial_to_json(
            {
                k_outer: {
                    k: {
                        prop: val
                        for prop, val in v.items()
                        if isinstance(val, (int, float, str, list, tuple, bool, dict))
                    }
                    for k, v in v_outer.items()
                }
                for k_outer, v_outer in all_trials_by_subj.items()
            },
            RESULTS_FOLDER / f"{savestring}_comb_metadata.json",
        )
        if "msg" in all_fix_dfs_concat.columns:
            all_fix_dfs_concat = all_fix_dfs_concat.drop(columns="msg")
        if "subject_trialID" in all_fix_dfs_concat.columns:
            all_fix_dfs_concat = all_fix_dfs_concat.drop(columns="subject_trialID")
        all_fix_dfs_concat = reorder_columns(
            all_fix_dfs_concat,
            [
                "subject",
                "trial_id",
                "item",
                "condition",
                "fixation_number",
                "duration",
                "start_uncorrected",
                "stop_uncorrected",
                "start_time",
                "stop_time",
                "corrected_start_time",
                "corrected_end_time",
            ],
        )
        all_fix_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_fixations.csv")
        if not all_sacc_dfs_concat.empty:
            if "msg" in all_sacc_dfs_concat.columns:
                all_sacc_dfs_concat = all_sacc_dfs_concat.drop(columns="msg")
            all_sacc_dfs_concat = reorder_columns(
                all_sacc_dfs_concat, ["subject", "trial_id", "item", "condition", "num"]
            )
            all_sacc_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_saccades.csv")
        all_chars_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_chars.csv")
        if not all_words_dfs_concat.empty:
            if "subject_trialID" in all_words_dfs_concat.columns:
                all_words_dfs_concat = all_words_dfs_concat.drop(columns="subject_trialID")
            all_words_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_words.csv")
        if not all_sentence_dfs_concat.empty:
            if "subject_trialID" in all_sentence_dfs_concat.columns:
                all_sentence_dfs_concat = all_sentence_dfs_concat.drop(columns="subject_trialID")
            all_sentence_dfs_concat.to_csv(RESULTS_FOLDER / f"{savestring}_comb_sentences.csv")

        # Create master zipfile with organized subfolders
        date_str = datetime.now().strftime("%Y%m%d")
        master_zip_name = f"all_customfile_results_{date_str}.zip"
        master_zip_path = create_master_results_zipfile(
            RESULTS_FOLDER,
            zipfiles_with_results,
            master_zip_name,
            file_type_prefix="csv"
        )
        # Add master zip to the list for user download
        zipfiles_with_results.append(master_zip_path)

    return (
        list_of_trial_lists,
        results_keys,
        zipfiles_with_results,
        all_fix_dfs_concat,
        all_sacc_dfs_concat,
        all_chars_dfs_concat,
        all_words_dfs_concat,
        all_sentence_dfs_concat,
        all_trials_by_subj,
        trials_summary,
        subjects_summary,
        trials_quick_meta_df,
    )


def pop_large_trial_entries(all_trials_by_subj, subject, trial_id, trial):
    trial_for_comb = copy.deepcopy(trial)
    trial_for_comb["line_heights"] = list(np.unique(trial_for_comb["line_heights"]))
    if "dffix_no_clean" in trial_for_comb:
        trial_for_comb.pop("dffix_no_clean")
    if "chars_list" in trial_for_comb:
        trial_for_comb.pop("chars_list")
    if "trial_lines" in trial_for_comb:
        trial_for_comb.pop("trial_lines")
    if "dffix" in trial_for_comb:
        trial_for_comb.pop("dffix")
    if "gaze_df" in trial_for_comb:
        trial_for_comb.pop("gaze_df")
    if "chars_df" in trial_for_comb:
        trial_for_comb.pop("chars_df")
    if "saccade_df" in trial_for_comb:
        trial_for_comb.pop("saccade_df")
    if "combined_df" in trial_for_comb:
        trial_for_comb.pop("combined_df")
    if "own_sentence_measures_dfs_for_algo" in trial_for_comb:
        trial_for_comb.pop("own_sentence_measures_dfs_for_algo")
    if "own_word_measures_dfs_for_algo" in trial_for_comb:
        trial_for_comb.pop("own_word_measures_dfs_for_algo")
    all_trials_by_subj[subject][trial_id] = trial_for_comb
    return trial_for_comb


def add_cols_from_trial_info(
    asc_file_stem, trial_id, trial, dffix, saccade_df, chars_df, words_df, sent_measures_multi
):
    subject_for_key = trial.get("subject") or asc_file_stem
    subject_for_key = str(subject_for_key)
    trial_id_for_key = str(trial.get("trial_id", trial_id))
    subject_trial_key = f"{subject_for_key}_{trial_id_for_key}"

    if "item" not in dffix.columns and "item" in trial:
        dffix.insert(loc=0, column="item", value=trial["item"])
    if "condition" not in dffix.columns and "condition" in trial:
        dffix.insert(loc=0, column="condition", value=trial["condition"])
    if "trial_id" not in dffix.columns and "trial_id" in trial:
        dffix.insert(loc=0, column="trial_id", value=trial["trial_id"])
    if "subject" not in dffix.columns and "subject" in trial:
        dffix.insert(loc=0, column="subject", value=trial["subject"])
    if "subject_trialID" not in dffix.columns:
        dffix.insert(loc=0, column="subject_trialID", value=subject_trial_key)
    if "item" not in saccade_df.columns:
        saccade_df.insert(loc=0, column="item", value=trial["item"])
    if "condition" not in saccade_df.columns:
        saccade_df.insert(loc=0, column="condition", value=trial["condition"])
    if "trial_id" not in saccade_df.columns:
        saccade_df.insert(loc=0, column="trial_id", value=trial["trial_id"])
    if "subject" not in saccade_df.columns:
        saccade_df.insert(loc=0, column="subject", value=trial["subject"])
    if "item" not in chars_df.columns:
        chars_df.insert(loc=0, column="item", value=trial["item"])
    if "condition" not in chars_df.columns:
        chars_df.insert(loc=0, column="condition", value=trial["condition"])
    if "trial_id" not in chars_df.columns:
        chars_df.insert(loc=0, column="trial_id", value=trial["trial_id"])
    if "subject" not in chars_df.columns:
        chars_df.insert(loc=0, column="subject", value=trial["subject"])
    if words_df is not None:
        if "item" not in words_df.columns:
            words_df.insert(loc=0, column="item", value=trial["item"])
        if "condition" not in words_df.columns:
            words_df.insert(loc=0, column="condition", value=trial["condition"])
        if "trial_id" not in words_df.columns:
            words_df.insert(loc=0, column="trial_id", value=trial["trial_id"])
        if "subject" not in words_df.columns:
            words_df.insert(loc=0, column="subject", value=trial["subject"])
        if "subject_trialID" not in words_df.columns:
            words_df.insert(loc=0, column="subject_trialID", value=subject_trial_key)
    if sent_measures_multi is not None:
        add_cols_from_trial(trial, sent_measures_multi, cols=["item", "condition", "trial_id", "subject"])


def get_asc_filelist(asc_files):
    files_in_unzipped_folder = UNZIPPED_FOLDER.rglob("*")

    for file_path in (path_object for path_object in files_in_unzipped_folder if path_object.is_file()):
        try:
            file_path.unlink()
        except PermissionError as e:
            st.session_state["logger"].warning(f"Failed to delete file from unzipped folder: {file_path}")
            st.session_state["logger"].warning(e)
    asc_files_to_do = []
    for filename_full in asc_files:
        if hasattr(filename_full, "name") and not isinstance(filename_full, pl.Path):
            file = filename_full.name
            st.session_state["logger"].info(f"Filename is {file}")
        else:
            file = filename_full
        if not isinstance(file, str):
            file_stem = pl.Path(file.name).stem
        else:
            file_stem = pl.Path(file).stem
        savefolder = UNZIPPED_FOLDER.joinpath(file_stem)
        st.session_state["logger"].info(f"Operating on file {file}")
        if ".zip" in file:
            with zipfile.ZipFile(filename_full, "r") as z:
                z.extractall(str(savefolder))
        elif ".tar" in file:
            shutil.unpack_archive(file, savefolder, "tar")
        elif ".asc" in file:
            asc_files_to_do.append(filename_full)
        else:
            st.session_state["logger"].warning(f"Unsopported file format found in files")
        newfiles = [str(x) for x in savefolder.glob(f"*.asc")]
        asc_files_to_do += newfiles
    return asc_files_to_do


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def make_trial_from_stimulus_df(
    stim_plot_df,
    filename,
    trial_id,
    close_gaps_between_words:bool,
    close_gaps_between_lines:bool,
):
    chars_list = []
    words_list = []
    for idx, row in stim_plot_df.reset_index(drop=True).iterrows():
        char_dict = row.to_dict()
        chars_list.append(char_dict)

    words_list, chars_list = ut.add_words(chars_list)
    if close_gaps_between_words:
        words_list = ut.close_gaps_in_words_list(words_list)
    if close_gaps_between_lines:
        chars_list = ut.close_gaps_between_lines(chars_list,prefix='char') 
        words_list = ut.close_gaps_between_lines(words_list,prefix='word') 

    letter_width_avg = np.mean([x["char_xmax"] - x["char_xmin"] for x in chars_list if x["char_xmax"] > x["char_xmin"]])
    line_heights = [x["char_ymax"] - x["char_ymin"] for x in chars_list]
    line_xcoords_all = [x["char_x_center"] for x in chars_list]
    line_xcoords_no_pad = np.unique(line_xcoords_all)

    line_ycoords_all = [x["char_y_center"] for x in chars_list]
    line_ycoords_no_pad = np.unique(line_ycoords_all)

    trial = dict(
        filename=filename,
        y_midline=[float(x) for x in list(stim_plot_df["char_y_center"].unique())],
        num_char_lines=len(stim_plot_df["char_y_center"].unique()),
        y_diff=[float(x) for x in list(np.round(np.unique(np.diff(stim_plot_df["char_ymin"])), decimals=2))],
        trial_id=trial_id,
        chars_list=chars_list,
        words_list=words_list,
        trial_is="paragraph",
        text="".join([x["char"] for x in chars_list]),
    )

    trial["x_char_unique"] = [float(x) for x in list(line_xcoords_no_pad)]
    trial["y_char_unique"] = list(map(float, list(line_ycoords_no_pad)))
    x_diff, y_diff = ut.calc_xdiff_ydiff(
        line_xcoords_no_pad, line_ycoords_no_pad, line_heights, allow_multiple_values=False
    )
    trial["x_diff"] = float(x_diff)
    trial["y_diff"] = float(y_diff)
    trial["num_char_lines"] = len(line_ycoords_no_pad)
    trial["line_heights"] = list(map(float, line_heights))
    trial["letter_width_avg"] = letter_width_avg
    trial["chars_list"] = chars_list

    return trial


def get_fixations_file_trials_list(dffix, stimulus):
    if isinstance(stimulus, pd.DataFrame):
        mapper = {
            k: v
            for k, v in {
                st.session_state["x_col_name_fix_stim"]: "char_x_center",
                st.session_state["x_start_col_name_fix_stim"]: "char_xmin",
                st.session_state["x_end_col_name_fix_stim"]: "char_xmax",
                st.session_state["y_col_name_fix_stim"]: "char_y_center",
                st.session_state["y_start_col_name_fix_stim"]: "char_ymin",
                st.session_state["y_end_col_name_fix_stim"]: "char_ymax",
                st.session_state["char_col_name_fix_stim"]: "char",
                st.session_state["trial_id_col_name_stim"]: "trial_id",
                st.session_state["line_num_col_name_stim"]: "assigned_line",
            }.items()
            if v not in stimulus.columns
        }
        stimulus.rename(
            mapper=mapper,
            axis=1,
            inplace=True,
        )
        stimulus["assigned_line"] -= stimulus["assigned_line"].min()
    mapper = {
        k: v
        for k, v in {
            st.session_state["x_col_name_fix"]: "x",
            st.session_state["y_col_name_fix"]: "y",
            st.session_state["time_start_col_name_fix"]: "corrected_start_time",
            st.session_state["time_stop_col_name_fix"]: "corrected_end_time",
            st.session_state["trial_id_col_name_fix"]: "trial_id",
            st.session_state["subject_col_name_fix"]: "subject",
        }.items()
        if v not in dffix.columns
    }
    dffix.rename(
        mapper=mapper,
        axis=1,
        inplace=True,
    )
    dffix["duration"] = dffix.corrected_end_time - dffix.corrected_start_time
    if "trial_id" in stimulus and "trial_id" not in dffix.columns:
        dffix["trial_id"] = stimulus["trial_id"]
    if "trial_id" in dffix:
        if "subject" in dffix.columns and len(dffix["subject"].unique()) > 1:
            dffix["subject_trialID"] = [f"{id}_{num}" for id, num in zip(dffix["subject"], dffix["trial_id"])]
            enum = dffix.groupby("subject_trialID")
            if "subject" in stimulus.columns:
                stimulus["subject_trialID"] = [
                    f"{id}_{num}" for id, num in zip(stimulus["subject"], stimulus["trial_id"])
                ]
            else:
                stimulus["subject_trialID"] = stimulus["trial_id"]
            trial_keys = list(dffix["subject_trialID"].unique())
        else:
            enum = dffix.groupby("trial_id")
            trial_keys = list(dffix["trial_id"].unique())
        st.session_state["logger"].info(f"Found keys {trial_keys} for {st.session_state['single_csv_file'].name}")
    else:
        enum = dffix.groupby("trial_id")
        st.session_state["logger"].warning(f"trial id column not found assigning trial id trial_0.")
        st.warning(f"trial id column not found assigning trial id trial_0.")
        dffix["trial_id"] = "trial_0"
    st.session_state["fixations_df"] = dffix
    trials_by_ids = {}
    for trial_id, subdf in stqdm(enum, desc="Creating trials"):
        if isinstance(stimulus, pd.DataFrame):
            stim_df = stimulus[stimulus.trial_id == subdf["trial_id"].iloc[0]]
            if stim_df.empty:
                st.session_state["logger"].warning(f"stim_df dataframe is empty because trial_id {trial_id} not in stimulus trial ids:\n{stimulus.trial_id.unique()}")

            stim_df = stim_df.dropna(axis=0, how="all")
            subdf = subdf.dropna(axis=0, how="all")
            stim_df = stim_df.dropna(axis=1, how="all")
            subdf = subdf.dropna(axis=1, how="all")
            if subdf.empty:
                st.session_state["logger"].warning(f"Sub dataframe is empty for trial_id {trial_id}")
                continue
            subdf = subdf.reset_index(drop=True).copy()
            stim_df = stim_df.reset_index(drop=True).copy()
            assert not stim_df.empty, "stimulus df is empty"
            trial = make_trial_from_stimulus_df(
                stim_df,
                st.session_state["single_csv_file_stim"].name,
                trial_id,
                close_gaps_between_words=st.session_state["close_gap_between_words_single_csv"],
                close_gaps_between_lines=st.session_state["close_gap_between_lines_single_csv"],
            )
        else:
            if "trial_id" in stimulus.keys() and (
                isinstance(stimulus["trial_id"], dict) or isinstance(stimulus["trial_id"], pd.DataFrame)
            ):
                trial = stimulus["trial_id"]
            else:
                trial = stimulus
        chars_df = pd.DataFrame(trial["chars_list"])  # TODO look into making this more flexible if words are provided
        subdf["fixation_number"] = np.arange(subdf.shape[0], dtype=int)
        subdf["trial_id"] = trial_id
        trial["dffix"] = subdf
        if "filename" not in trial:
            trial["filename"] = f"{trial_id}"
        if "subject" not in trial:
            trial["subject"] = pl.Path(trial["filename"]).stem
        if "subject" not in dffix.columns:
            dffix["subject"] = trial["subject"]
        trial["letter_width_avg"] = (chars_df["char_xmax"] - chars_df["char_xmin"]).mean()
        trial["plot_file"] = str(PLOTS_FOLDER.joinpath(f"{trial_id}_2ndInput_chars_channel_sep.png"))
        trials_by_ids[trial_id] = trial

    return trials_by_ids, trial_keys

def make_ints_float(df):
    for col in df.columns:
        if 'int' in str(df[col].dtype).lower():
            df[col] = pd.to_numeric(df[col], downcast='float') 
    return df

def load_csv_delim_agnostic(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.shape[1] > 1:
            return make_ints_float(df)
        else:
            dec_file = get_decoded_input_from_file(file_path)
            df = pd.read_csv(StringIO(dec_file.replace(";", ",").replace("\t", ",")))
            return make_ints_float(df)
    except Exception as e:
        dec_file = get_decoded_input_from_file(file_path)
        df = pd.read_csv(StringIO(dec_file.replace(";", ",").replace("\t", ",")))
        return make_ints_float(df)


def stringify_column_keep_decimal_if_needed(series: pd.Series) -> pd.Series:
    try:
        floats = series.astype(float)
        ints = floats.astype(int)
        if (floats == ints).all():
            return ints.astype(str)
        return floats.astype(str)
    except Exception:
        return series.astype(str)


def _get_session_state_value(key: str, default: str):
    if key in st.session_state and st.session_state[key] not in (None, ""):
        return st.session_state[key]
    return default


def prepare_fixation_dataframe_for_multi_csv(dffix: pd.DataFrame) -> pd.DataFrame:
    dffix = dffix.copy()
    rename_map = {}
    for key, target in [
        ("x_col_name_fix", "x"),
        ("y_col_name_fix", "y"),
        ("time_start_col_name_fix", "corrected_start_time"),
        ("time_stop_col_name_fix", "corrected_end_time"),
        ("trial_id_col_name_fix", "trial_id"),
        ("subject_col_name_fix", "subject"),
    ]:
        default_value = COLNAME_CANDIDATES_CUSTOM_CSV_FIX_DEFAULT.get(key)
        if default_value is None:
            continue
        source = _get_session_state_value(key, default_value)
        if source in dffix.columns and target not in dffix.columns:
            rename_map[source] = target
    if rename_map:
        dffix = dffix.rename(columns=rename_map)

    if "trial_id" not in dffix.columns:
        raise ValueError("No trial_id column found after applying column mappings.")

    dffix["trial_id"] = stringify_column_keep_decimal_if_needed(dffix["trial_id"])

    if "subject" in dffix.columns:
        dffix["subject"] = stringify_column_keep_decimal_if_needed(dffix["subject"])

    if "corrected_start_time" not in dffix.columns:
        if "start_time" in dffix.columns:
            dffix["corrected_start_time"] = dffix["start_time"]
        elif "start" in dffix.columns:
            dffix["corrected_start_time"] = dffix["start"]

    if "corrected_end_time" not in dffix.columns:
        if "end_time" in dffix.columns:
            dffix["corrected_end_time"] = dffix["end_time"]
        elif "stop" in dffix.columns:
            dffix["corrected_end_time"] = dffix["stop"]

    if "start_uncorrected" not in dffix.columns and "corrected_start_time" in dffix.columns:
        dffix["start_uncorrected"] = dffix["corrected_start_time"]
    if "stop_uncorrected" not in dffix.columns and "corrected_end_time" in dffix.columns:
        dffix["stop_uncorrected"] = dffix["corrected_end_time"]

    if "corrected_start_time" not in dffix.columns or "corrected_end_time" not in dffix.columns:
        raise ValueError("Fixation dataframe requires start and end time columns after mapping.")

    if "duration" not in dffix.columns:
        dffix["duration"] = dffix["corrected_end_time"] - dffix["corrected_start_time"]

    return dffix


def prepare_stimulus_dataframe_for_multi_csv(stim_df: pd.DataFrame, trial_id: str) -> pd.DataFrame:
    stim_df = stim_df.copy()
    rename_map = {}
    for key, target in [
        ("x_col_name_fix_stim", "char_x_center"),
        ("y_col_name_fix_stim", "char_y_center"),
        ("x_start_col_name_fix_stim", "char_xmin"),
        ("x_end_col_name_fix_stim", "char_xmax"),
        ("y_start_col_name_fix_stim", "char_ymin"),
        ("y_end_col_name_fix_stim", "char_ymax"),
        ("char_col_name_fix_stim", "char"),
        ("trial_id_col_name_stim", "trial_id"),
        ("line_num_col_name_stim", "assigned_line"),
    ]:
        default_value = COLNAMES_CUSTOM_CSV_STIM_DEFAULT.get(key)
        if default_value is None:
            continue
        source = _get_session_state_value(key, default_value)
        if source in stim_df.columns and target not in stim_df.columns:
            rename_map[source] = target
    if rename_map:
        stim_df = stim_df.rename(columns=rename_map)

    if "trial_id" not in stim_df.columns:
        stim_df["trial_id"] = str(trial_id)
    else:
        stim_df["trial_id"] = stringify_column_keep_decimal_if_needed(stim_df["trial_id"])

    if "assigned_line" in stim_df.columns:
        stim_df["assigned_line"] = stim_df["assigned_line"] - stim_df["assigned_line"].min()

    required_cols = {"char_xmin", "char_xmax", "char_ymin", "char_ymax", "char"}
    missing_cols = required_cols - set(stim_df.columns)
    if missing_cols:
        raise ValueError(f"Stimulus dataframe missing required columns: {', '.join(sorted(missing_cols))}")

    return stim_df.dropna(how="all", axis=0)


def find_col_name_suggestions(cols, candidates_dict):
    scores_lists = []
    for k, v in candidates_dict.items():
        for word in cols:
            for candidate in v:
                resdict = {
                    "category": k,
                    "word_in_df": word,
                    "candidate": candidate,
                    "score": jf.levenshtein_distance(candidate, word),
                }
                scores_lists.append(resdict)
    scores_df = pd.DataFrame(scores_lists)
    scores_df.groupby(["category", "candidate"])["score"].min()
    mappings = {}
    for _, row in scores_df.loc[scores_df.groupby(["category"])["score"].idxmin(), :].iterrows():
        mappings[row["category"]] = row["word_in_df"]

    return mappings


def get_decoded_input_from_file(file):
    for enc in ["ISO-8859-1", "utf-8"]:
        try:
            decoded_input = file.getvalue().decode(enc)
        except Exception as e:
            st.session_state["logger"].warning(e)
            st.session_state["logger"].warning(f"File decoding failed using {enc}")
    return decoded_input


def get_eyekit_measures(_txt, _seq, trial, get_char_measures=False):
    return ekm.get_eyekit_measures(_txt, _seq, trial, get_char_measures=get_char_measures)


get_all_measures = st.cache_data(get_all_measures)

compute_sentence_measures = st.cache_data(pf.compute_sentence_measures)
get_fix_seq_and_text_block = st.cache_data(ekm.get_fix_seq_and_text_block)
eyekit_plot = st.cache_data(ekm.eyekit_plot)


def filter_trial_for_export(trial):
    trial = copy.deepcopy(trial)
    _ = [trial.pop(k) for k in list(trial.keys()) if isinstance(trial[k], (pd.DataFrame, np.ndarray))]
    _ = [
        trial.pop(k)
        for k in list(trial.keys())
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
            "gaze_df",
            "dffix",
        ]
    ]
    if "line_heights" in trial:
        trial["line_heights"] = list(np.unique(trial["line_heights"]))
    return trial


def check_for_32bit_dtypes(x):
    if np.issubdtype(type(x), np.number) and int(x) == x and not isinstance(x, bool):
        return int(x)
    if np.issubdtype(type(x), np.number) and float(x) == x and not isinstance(x, bool):
        return float(x)
    return x


def process_trial_choice_single_csv(trial, algo_choice, models_dict, file=None):
    words_df = pd.DataFrame(trial["words_list"])
    words_df["word_number"] = np.arange(words_df.shape[0])
    trial["words_list"] = words_df.to_dict(orient="records")
    if "subject" not in trial:
        if "filename" in trial:
            trial["subject"] = pl.Path(trial["filename"]).stem
        else:
            trial["subject"] = ""
    if "item" not in trial:
        trial["item"] = None
    if "condition" not in trial:
        trial["condition"] = None
    trial_id = trial["trial_id"]
    if "dffix" in trial:
        dffix = trial["dffix"]
    else:
        fname = pl.Path(str(file.name)).stem
        trial["plot_file"] = str(PLOTS_FOLDER.joinpath(f"{fname}_{trial_id}_2ndInput_chars_channel_sep.png"))
        trial["filename"] = fname
        dffix = trial["dffix"] = st.session_state["trials_by_ids_single_csv"][trial_id]["dffix"]
    if "item" not in dffix.columns and "item" in trial:
        dffix.insert(loc=0, column="item", value=trial["item"])
    if "condition" not in dffix.columns and "condition" in trial:
        dffix.insert(loc=0, column="condition", value=trial["condition"])
    if "subject" not in dffix.columns and "subject" in trial:
        dffix.insert(loc=0, column="subject", value=trial["subject"])
    if "blink" not in dffix.columns:
        dffix["blink"] = False
    font, font_size, dpi, screen_res = get_plot_props(trial, AVAILABLE_FONTS)
    trial["font"] = font
    trial["font_size"] = font_size
    trial["dpi"] = dpi
    trial["screen_res"] = screen_res
    if "chars_list" in trial:
        words_list, chars_list_reconstructed = ut.add_words(trial["chars_list"])
        chars_df = pd.DataFrame(chars_list_reconstructed)
        chars_df = add_popEye_cols_to_chars_df(chars_df)
        trial["chars_df"] = chars_df
        trial["chars_list"] = chars_df.to_dict("records")
        trial["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())

    if algo_choice is not None:
        dffix = correct_df(
            dffix,
            algo_choice,
            trial,
            for_multi=False,
            is_outside_of_streamlit=False,
            classic_algos_cfg=CLASSIC_ALGOS_CFGS,
            models_dict=models_dict,
        )
    return dffix, trial, dpi, screen_res, font, font_size


def main():
    if "models_dict" not in st.session_state:
        st.session_state["models_dict"] = get_cached_models(DIST_MODELS_FOLDER)

    st.title("Fixation data processing and analysis")
    st.markdown(
        f"[Contact Us](mailto:{CONTACT_EMAIL})  &emsp;  [Read about DIST model](https://doi.org/10.1109/TPAMI.2024.3411938)"
    )

    single_file_tab, multi_file_tab = st.tabs(["Single File 📁", "Multiple Files 📁 📁"])

    single_file_tab_asc_tab, single_file_tab_csv_tab = single_file_tab.tabs([".asc files", "custom files"])

    settings_to_save = {
        k.replace("_single_asc", ""): check_for_32bit_dtypes(v)
        for (k, v) in st.session_state.items()
        if k
        in [
            "trial_start_keyword_single_asc",
            "trial_end_keyword_single_asc",
            "close_gap_between_words_single_asc",
            "close_gap_between_lines_single_asc",
            "paragraph_trials_only_single_asc",
            "discard_fixations_without_sfix_single_asc",
            "discard_far_out_of_text_fix_single_asc",
            "discard_blinks_fix_single_asc",
            "outlier_crit_x_threshold_single_asc",
            "outlier_crit_y_threshold_single_asc",
            "discard_long_fix_single_asc",
            "discard_long_fix_threshold_single_asc",
            "choice_handle_short_and_close_fix_single_asc",
            "merge_distance_threshold_single_asc",
            "algo_choice_single_asc",
            "measures_to_calculate_single_asc",
            "font_face_for_eyekit_single_asc",
            "y_txt_start_for_eyekit_single_asc",
            "x_txt_start_for_eyekit_single_asc",
            "font_size_for_eyekit_single_asc",
            "include_word_coords_in_output_single_asc",
            "fix_cols_to_add_single_asc",
            "sent_measures_to_calculate_single_asc",
        ]
    }
    if len(settings_to_save) > 0:
        single_file_tab_asc_tab.download_button(
            "⏬ Download all single .asc file settings as JSON",
            json.dumps(settings_to_save, cls=ut.NpEncoder),
            "settings_to_save.json",
            "json",
            key="download_settings_to_save",
            help="Can be used to reload settings later or to use them for multi .asc file processing.",
        )
    with single_file_tab_asc_tab.expander("Load config file"):
        with st.form("single_file_tab_asc_tab_load_settings_from_file_form"):
            st.file_uploader(
                "Select .json config file to reload a previous processing configuration",
                accept_multiple_files=False,
                key="single_asc_file_settings_file_uploaded",
                type=["json"],
                help="Load in a configuration file as .json to reproduce previous processing and analysis.",
            )
            cfg_load_btn_single_asc = st.form_submit_button("Load in config")
    if cfg_load_btn_single_asc and in_st_nn("single_asc_file_settings_file_uploaded"):
        if "saccade_df" in st.session_state:
            del st.session_state["saccade_df"]
        if "dffix_single_asc" in st.session_state:
            del st.session_state["dffix_single_asc"]
        if "own_word_measures_single_asc" in st.session_state:
            del st.session_state["own_word_measures_single_asc"]
        if "dffix_cleaned_single_asc" in st.session_state:
            del st.session_state["dffix_cleaned_single_asc"]
        json_string = st.session_state["single_asc_file_settings_file_uploaded"].getvalue().decode("utf-8")
        st.session_state["loaded_settings_single_asc"] = {
            f"{k}_single_asc": v for k, v in json.loads(json_string).items()
        }
        st.session_state["_loaded_settings_single_asc"] = {
            f"_{k}_single_asc": v for k, v in json.loads(json_string).items()
        }
        st.session_state.update(st.session_state["loaded_settings_single_asc"])
        st.session_state.update(st.session_state["_loaded_settings_single_asc"])

    with single_file_tab_asc_tab.form("single_file_tab_asc_tab_load_example_form"):
        st.markdown("### File selection")
        file_upload_col_single_asc, ex_file_sel_col_single_asc = st.columns(2)
        with file_upload_col_single_asc:
            st.file_uploader(
                "Upload a single .asc file",
                accept_multiple_files=False,
                key="single_asc_uploaded_file",
                type=["asc"],
                help="Drag and drop or select a single .asc file that you wish to process. This can be left blank if you chose to use the examples.",
            )
            st.file_uploader(
                "Upload all .ias files associated with the .asc file. Leave empty if you don't use .ias files.",
                accept_multiple_files=True,
                key="single_asc_file_ias_files_uploaded",
                type=["ias"],
                help="If the stimulus information is not part of the .asc file then all .ias files associated with your file should be put here. This will allow the program to align each trial found in the .asc file with the correct stimulus text by finding the .ias filename in the .asc file (Needs to be flagged with 'IAREA FILE').",
            )
        with ex_file_sel_col_single_asc:
            if len(EXAMPLE_ASC_FILES) > 0 and os.path.isfile(EXAMPLE_ASC_FILES[0]):
                st.selectbox(
                    "Select which example file to use",
                    options=EXAMPLE_ASC_FILES,
                    key="single_file_tab_asc_tab_example_file_choice",
                    help="If the 'Example File' option is selected below, the file that gets selected here will be used for processing.",
                )
            else:
                st.session_state["single_file_tab_asc_tab_example_use_example_or_uploaded_file_choice"] = (
                    "Uploaded File"
                )
                st.session_state["single_file_tab_asc_tab_example_file_choice"] = None

        if len(EXAMPLE_ASC_FILES) > 0 and os.path.isfile(EXAMPLE_ASC_FILES[0]):

            with st.columns(3)[1]:
                use_example_or_uploaded_file_choice = st.radio(
                    "Should the uploaded file be used or the selected example file?",
                    index=1,
                    options=["Uploaded File", "Example File"],
                    key="single_file_tab_asc_tab_example_use_example_or_uploaded_file_choice",
                    horizontal=True,
                    help="This selection determines if the uploaded .asc file on the top left or the selected example file on the top right will be used for processing.",
                )
        show_file_parsing_settings("_single_asc")

        upload_file_button = st.form_submit_button(label="Load selected data.")
    if upload_file_button:
        if "dffix_single_asc" in st.session_state:
            del st.session_state["dffix_single_asc"]
        if "trial_single_asc" in st.session_state:
            del st.session_state["trial_single_asc"]
        if st.session_state["single_file_tab_asc_tab_example_use_example_or_uploaded_file_choice"] == "Example File":
            st.session_state["single_asc_file_asc"] = st.session_state["single_file_tab_asc_tab_example_file_choice"]
            st.session_state["single_asc_file_ias_files"] = []
        else:
            st.session_state["single_asc_file_asc"] = st.session_state["single_asc_uploaded_file"]
            st.session_state["single_asc_file_ias_files"] = st.session_state["single_asc_file_ias_files_uploaded"]
        if "events_df" in st.session_state:
            del st.session_state["events_df"]
        if "trial_single_asc" in st.session_state:
            del st.session_state["trial_single_asc"]
        if in_st_nn("single_asc_file_asc"):
            if st.session_state["trial_start_keyword_single_asc"] == "custom":
                trial_start_keyword = st.session_state["trial_custom_start_keyword_single_asc"]
            else:
                trial_start_keyword = st.session_state["trial_start_keyword_single_asc"]
            if st.session_state["trial_end_keyword_single_asc"] == "custom":
                trial_end_keyword = st.session_state["trial_custom_end_keyword_single_asc"]
            else:
                trial_end_keyword = st.session_state["trial_end_keyword_single_asc"]
            trial_choices_single_asc, trials_by_ids, lines, asc_file, trials_dict = ut.get_trials_list(
                st.session_state["single_asc_file_asc"],
                close_gap_between_words=st.session_state["close_gap_between_words_single_asc"],
                close_gap_between_lines=st.session_state["close_gap_between_lines_single_asc"],
                paragraph_trials_only=st.session_state["paragraph_trials_only_single_asc"],
                ias_files=st.session_state["single_asc_file_ias_files"],
                trial_start_keyword=trial_start_keyword,
                end_trial_at_keyword=trial_end_keyword,
            )
            asc_file_stem = pl.Path(str(st.session_state["single_asc_file_asc"])).stem
            st.session_state["logger"].info(f"Found {len(trials_by_ids)} trials in {asc_file_stem}.")
            st.session_state["trials_dict_single_asc"] = trials_dict
            st.session_state["trials_by_ids"] = trials_by_ids
            st.session_state["trial_choices_single_asc"] = trial_choices_single_asc
            st.session_state["lines"] = lines
            st.session_state["asc_file"] = asc_file
        else:
            st.warning("Please select a file to run")

    if in_st_nn("single_asc_file_asc") and in_st_nn("trials_dict_single_asc"):
        single_file_tab_asc_tab.markdown("### Metadata found in .asc file")
        trials_dict_for_showing = copy.deepcopy(
            {
                k: {
                    k1: v1
                    for k1, v1 in v.items()
                    if k1
                    not in [
                        "chars_list",
                        "line_heights",
                        "x_char_unique",
                    ]
                }
                for k, v in st.session_state["trials_dict_single_asc"].items()
                if isinstance(v, dict)
            }
        )
        for k, v in st.session_state["trials_dict_single_asc"].items():
            if not isinstance(v, dict):
                trials_dict_for_showing[k] = v
        single_file_tab_asc_tab.json(ut.make_json_serializable(trials_dict_for_showing), expanded=False)
    if "trial_choices_single_asc" in st.session_state:
        single_file_tab_asc_tab.markdown("### Trial and algorithm selection")
        with single_file_tab_asc_tab.form(key="single_file_tab_asc_tab_trial_select_form"):
            trial_choice = st.selectbox(
                "Which trial should be cleaned and corrected?",
                st.session_state["trial_choices_single_asc"],
                key="trial_id",
                index=0,
                help="This is a list of the trial ids found in the ASC, please choose which one should used for further processing.",
            )
            ensure_state_option(
                "discard_fixations_without_sfix_single_asc",
                True,
                options=[True, False],
            )
            discard_fixations_without_sfix = st.checkbox(
                "Should fixations that start before trial start but end after be discarded?",
                key="discard_fixations_without_sfix_single_asc",
                help="In cases where the trigger flag for the start of the trial occurs during a fixation, this setting determines wether that fixation is to be discarded or kept.",
            )
            load_trial_btn = st.form_submit_button("Load trial")
        if load_trial_btn:
            cp2st("discard_fixations_without_sfix_single_asc")
            if "dffix_cleaned_single_asc" in st.session_state:
                del st.session_state["dffix_cleaned_single_asc"]
            if "dffix_single_asc" in st.session_state:
                del st.session_state["dffix_single_asc"]

            single_file_tab_asc_tab.write(f'You selected: {st.session_state["trial_id"]}')
            trial = st.session_state["trials_by_ids"][st.session_state["trial_id"]]
            trial_lines = st.session_state["lines"][trial["trial_start_idx"] + 1 : trial["trial_end_idx"]]
            trial["trial_lines"] = trial_lines
            events_df, trial = get_raw_events_df_and_trial(
                trial, st.session_state["discard_fixations_without_sfix_single_asc"]
            )
            st.session_state["events_df"] = events_df
            st.session_state["trial_single_asc"] = trial
        if "events_df" in st.session_state:

            if "trial_single_asc" in st.session_state:
                filtered_trial = filter_trial_for_export(copy.deepcopy(st.session_state["trial_single_asc"]))
                single_file_tab_asc_tab.markdown(
                    f'### Result dataframes for trial {st.session_state["trial_single_asc"]["trial_id"]}'
                )
                trial_expander_single = single_file_tab_asc_tab.expander("Show Trial Information", False)
                trial_expander_single.json(ut.make_json_serializable(filtered_trial), expanded=False)
            events_df_expander_single = single_file_tab_asc_tab.expander("Show fixations and saccades before cleaning")
            events_df = st.session_state["events_df"].set_index("num").copy()
            events_df_expander_single.markdown("## Events before cleaning")
            events_df_expander_single.markdown("### Fixations")
            events_df_expander_single.dataframe(
                events_df[events_df["msg"] == "FIX"].dropna(how="all", axis=1).copy(),
                width='stretch',
                height=200,
            )
            events_df_expander_single.markdown("### Saccades")
            events_df_expander_single.dataframe(
                events_df[events_df["msg"] == "SAC"].dropna(how="all", axis=1).copy(),
                width='stretch',
                height=200,
            )
            if not events_df[events_df["msg"] == "BLINK"].empty:
                events_df_expander_single.markdown("### Blinks")
                blinksdf = events_df[events_df["msg"] == "BLINK"].dropna(how="all", axis=1).copy()
                blinksdf = blinksdf.drop(
                    columns=[c for c in blinksdf.columns if c in ["blink", "blink_after", "blink_before"]]
                )
                events_df_expander_single.dataframe(blinksdf, width='stretch', height=200)
            show_cleaning_options(single_file_tab_asc_tab, events_df[events_df["msg"] == "FIX"], "single_asc")

        if "dffix_cleaned_single_asc" in st.session_state and "trial_single_asc" in st.session_state:
            show_cleaning_results(
                single_file_tab_asc_tab,
                trial=st.session_state["trial_single_asc"],
                expander_text="Show Cleaned Fixations Dataframe",
                dffix_cleaned=st.session_state["dffix_cleaned_single_asc"],
                dffix_no_clean_name="dffix_no_clean",
                expander_open=True,
                key_str="single_asc",
            )

            with single_file_tab_asc_tab.form(key="correction_options_form_single_asc"):
                algo_choice_single_asc = st.multiselect(
                    "Choose line-assignment algorithm",
                    ALGO_CHOICES,
                    key="algo_choice_single_asc",
                    default=get_def_val_w_underscore("algo_choice_single_asc", DEFAULT_ALGO_CHOICE, ALGO_CHOICES),
                    help="This selection determines which of the available line assignment algorithms should be used to assign each fixation to their most appropriate line of text. The rest of the analysis is dependent on this line assignment. It is recommended to try out multiple different assignment approaches to make sure it performs well for on your data.",
                )

                with st.popover("Fixation features definitions"):
                    fix_colnames_markdown = get_fix_colnames_markdown()
                    st.markdown(fix_colnames_markdown)
                fix_cols_to_add_single_asc = st.multiselect(
                    "Select what fixation measures to calculate.",
                    options=ALL_FIX_MEASURES,
                    key="fix_cols_to_add_single_asc",
                    default=get_def_val_w_underscore(
                        "fix_cols_to_add_single_asc", DEFAULT_FIX_MEASURES, ALL_FIX_MEASURES
                    ),
                    help="This selection determines what fixation-level measures will be calculated. If you are in doubt about which ones you might need for your analysis, you can select all of them since it only slightly adds to the processing time.",
                )
                cp2st("fix_cols_to_add_single_asc")

                process_trial_btn = st.form_submit_button("Correct fixations for trial")
            if process_trial_btn:
                process_single_dffix_and_add_to_state("_single_asc")
                high_fix_count_dfs = check_for_large_number_of_fixations_on_word(
                    st.session_state["dffix_single_asc"],
                    single_file_tab_asc_tab,
                    st.session_state["algo_choice_single_asc"],
                )

        if "dffix_single_asc" in st.session_state and st.session_state["dffix_single_asc"].empty:
            st.warning("Fixations dataframe empty")
            del st.session_state["dffix_single_asc"]
        if "dffix_single_asc" in st.session_state and "trial_single_asc" in st.session_state:
            trial = st.session_state["trial_single_asc"]
            filtered_trial = filter_trial_for_export(copy.deepcopy(trial))
            trial_expander_single = single_file_tab_asc_tab.expander("Show Trial Information", False)
            trial_expander_single.markdown(f'### Metadata for trial {trial["trial_id"]}')
            trial_expander_single.json(ut.make_json_serializable(filtered_trial), expanded=False)
            if "saccade_df" not in st.session_state:
                if st.session_state["dffix_single_asc"].shape[0] > 1:
                    saccade_df = get_saccade_df(
                        st.session_state["dffix_single_asc"],
                        trial,
                        st.session_state["algo_choice_single_asc"],
                        st.session_state["events_df"],
                    )
                    saccade_df = reorder_columns(saccade_df)
                    st.session_state["saccade_df"] = saccade_df
                    trial["saccade_df"] = saccade_df
                    fig = plot_saccade_df(st.session_state["dffix_single_asc"], saccade_df, trial, True, False)
                    fig.savefig(RESULTS_FOLDER / f"{trial['subject']}_{trial['trial_id']}_saccades.png")
                else:
                    st.warning(
                        f"🚨 Only {st.session_state['dffix'].shape[0]} fixation left after processing. saccade_df not created for trial {st.session_state['trial']['trial_id']} 🚨"
                    )
            dffix_expander_single = single_file_tab_asc_tab.expander("Show Fixations Dataframe", False)
            with dffix_expander_single.popover("Column name definitions"):
                fix_colnames_markdown = get_fix_colnames_markdown()
                st.markdown(fix_colnames_markdown)
            if "saccade_df" in st.session_state:
                saccade_df_expander_single = single_file_tab_asc_tab.expander("Show Saccade Dataframe", False)
                with saccade_df_expander_single.popover("Column name definitions"):
                    sac_colnames_markdown = get_sac_colnames_markdown()
                    st.markdown(sac_colnames_markdown)
                saccade_df_expander_single.dataframe(st.session_state["saccade_df"], height=200)
            if "chars_list" in trial or "words_list" in trial:
                df_stim_expander_single = single_file_tab_asc_tab.expander("Show Stimulus Dataframes", False)
                df_stim_expander_single.markdown("### Characters dataframe")
                with df_stim_expander_single.popover(
                    "Column names definitions", help="Show column names and their definitions."
                ):
                    chars_colnames_markdown = read_chars_col_names()
                    st.markdown(chars_colnames_markdown)
                df_stim_expander_single.dataframe(
                    pd.DataFrame(trial["chars_list"]), width='stretch', height=200
                )
                if "words_list" in trial:
                    df_stim_expander_single.markdown("### Words dataframe")
                    df_stim_expander_single.dataframe(
                        pd.DataFrame(trial["words_list"]), width='stretch', height=200
                    )
            else:
                st.warning("🚨 No stimulus information in session state")
            single_file_tab_asc_tab.markdown(f'### Fixation related plots for trial {trial["trial_id"]}')
            plot_expander_single = single_file_tab_asc_tab.expander("Show Plots", True, icon="📈")

            fix_cols_to_keep = [
                c
                for c in st.session_state["dffix_single_asc"].columns
                if (
                    (
                        any([lname in c for lname in ALL_FIX_MEASURES])
                        and any([lname in c for lname in st.session_state["fix_cols_to_add_single_asc"]])
                    )
                    or (not any([lname in c for lname in ALL_FIX_MEASURES]))
                )
            ]

            dffix_for_display_and_save = st.session_state["dffix_single_asc"].loc[:, fix_cols_to_keep].copy()
            dffix_expander_single.dataframe(dffix_for_display_and_save, height=200)
            csv = convert_df(dffix_for_display_and_save)
            dffix_expander_single.download_button(
                "⏬ Download fixation dataframe",
                csv,
                f'{filtered_trial["subject"]}_{filtered_trial["trial_id"]}.csv',
                "text/csv",
                key="download-csv_single_asc",
                help="This downloads the corrected fixations dataframe as a .csv file with the filename containing the subject name and trial id.",
            )
            trial_expander_single.download_button(
                "⏬ Download trial info as JSON",
                json.dumps(filtered_trial, cls=ut.NpEncoder),
                f'{filtered_trial["subject"]}_{filtered_trial["trial_id"]}.json',
                "json",
                key="download-trial_single_asc",
                help="This downloads the extracted trial information as a .json file with the filename containing the subject name and trial id.",
            )
            plot_expander_single_options_c1, plot_expander_single_options_c2 = plot_expander_single.columns([0.6, 0.3])
            ensure_state_option(
                "plotting_checkboxes_single_asc",
                ["Uncorrected Fixations", "Corrected Fixations", "Characters", "Word boxes"],
                options=STIM_FIX_PLOT_OPTIONS,
            )
            plotting_checkboxes_single = plot_expander_single_options_c1.multiselect(
                "Select what gets plotted",
                STIM_FIX_PLOT_OPTIONS,
                key="plotting_checkboxes_single_asc",
                help="This selection determines what information is plotted. The Corrected Fixations are the fixations after being snapped to their assigned line of text. The Word and Character boxes are the bounding boxes for the stimulus.",
            )
            ensure_state_option(
                "scale_factor_single_asc",
                0.5,
                validator=lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 3.0,
            )
            scale_factor_single_asc = plot_expander_single_options_c2.number_input(
                label="Scale factor for stimulus image",
                min_value=0.01,
                max_value=3.0,
                step=0.1,
                key="scale_factor_single_asc",
                help="This can be used to simply make the plot larger or smaller.",
            )
            ensure_state_option(
                "lines_in_plot_single_asc",
                "Uncorrected",
                options=["Uncorrected", "Corrected", "Both", "Neither"],
            )
            lines_in_plot_single_asc = plot_expander_single_options_c1.radio(
                "Lines between fixations for:",
                ["Uncorrected", "Corrected", "Both", "Neither"],
                key="lines_in_plot_single_asc",
                help="This selection determines which of the fixations in the plot will be connected by lines rather than a simple scatter plot of fixation points.",
            )

            dffix = st.session_state["dffix_single_asc"]
            saccade_df = st.session_state["saccade_df"]
            plot_expander_single.markdown("#### Fixations before and after line assignment")

            ensure_state_option(
                "show_fix_sacc_plots_single_asc",
                True,
                options=[True, False],
            )
            show_fix_sacc_plots_single_asc = plot_expander_single.checkbox(
                "Show plots",
                key="show_fix_sacc_plots_single_asc",
            )
            if show_fix_sacc_plots_single_asc:
                selected_plotting_font_single_asc = plot_expander_single_options_c2.selectbox(
                    "Font to use for plotting",
                    AVAILABLE_FONTS,
                    index=FONT_INDEX,
                    key="selected_plotting_font_single_asc",
                    help="This selects which font is used to display the words or characters making up the stimulus. This selection only affects the plot and has no effect on the analysis as everything else is based on the bounding boxes of the words and characters.",
                )
                plot_expander_single.plotly_chart(
                    plotly_plot_with_image(
                        dffix,
                        trial,
                        to_plot_list=plotting_checkboxes_single,
                        algo_choice=st.session_state["algo_choice_single_asc"],
                        scale_factor=scale_factor_single_asc,
                        font=selected_plotting_font_single_asc,
                        lines_in_plot=lines_in_plot_single_asc,
                    ),
                    width='content',
                )
                plot_expander_single.markdown("#### Saccades")

                saccade_plot_options = [
                    "Saccades",
                    "Saccades snapped to line",
                    "Uncorrected Fixations",
                    "Corrected Fixations",
                    "Word boxes",
                    "Characters",
                    "Character boxes",
                ]
                ensure_state_option(
                    "plotting_checkboxes_sacc_single_asc",
                    ["Saccades", "Characters", "Word boxes"],
                    options=saccade_plot_options,
                )
                plotting_checkboxes_sacc_single_asc = plot_expander_single.multiselect(
                    "Select what gets plotted",
                    saccade_plot_options,
                    key="plotting_checkboxes_sacc_single_asc",
                    help="This selection determines what information is plotted. The Corrected Fixations are the fixations after being snapped to their assigned line of text. The saccades snapped to line follow the same logic. The Word and Character boxes are the bounding boxes for the stimulus.",
                )
                plot_expander_single.plotly_chart(
                    plotly_plot_with_image(
                        dffix,
                        trial,
                        saccade_df=saccade_df,
                        to_plot_list=plotting_checkboxes_sacc_single_asc,
                        algo_choice=st.session_state["algo_choice_single_asc"],
                        scale_factor=scale_factor_single_asc,
                        font=selected_plotting_font_single_asc,
                        lines_in_plot=lines_in_plot_single_asc,
                    ),
                    width='content',
                )
                plot_expander_single.markdown("#### Y-coordinate correction due to line-assignment")
                plot_expander_single.plotly_chart(
                    plot_y_corr(dffix, st.session_state["algo_choice_single_asc"]), width='stretch'
                )
            if "average_y_corrections" in trial:
                plot_expander_single.markdown(
                    "Average y-correction:",
                    help="Average difference between raw y position of a fixation and the center of the line to which it was assigned in pixels",
                )
                plot_expander_single.dataframe(pd.DataFrame(trial["average_y_corrections"]), hide_index=True)

            if show_fix_sacc_plots_single_asc:
                select_and_show_fix_sacc_feature_plots(
                    dffix,
                    saccade_df,
                    plot_expander_single,
                    plot_choice_fix_feature_name="plot_choice_fix_features",
                    plot_choice_sacc_feature_name="plot_choice_sacc_features",
                    feature_plot_selection="feature_plot_selection_single_asc",
                    plot_choice_fix_sac_feature_x_axis_name="feature_plot_x_selection_single_asc",
                )
            if "chars_list" in st.session_state["trial_single_asc"]:
                single_file_tab_asc_tab.markdown(
                    f'### Analysis for trial {st.session_state["trial_single_asc"]["trial_id"]}'
                )
                analysis_expander_single_asc = single_file_tab_asc_tab.expander("Show Analysis results", True)
                with analysis_expander_single_asc.form("run_show_analysis_single_asc_form"):
                    algo_choice_single_asc_eyekit = st.selectbox(
                        "Select which corrected fixations should be used for the analysis.",
                        st.session_state["algo_choice_single_asc"],
                        index=get_default_index(
                            "_algo_choice_single_asc_eyekit", st.session_state["algo_choice_single_asc"], 0
                        ),
                        key="algo_choice_single_asc_eyekit",
                        help="If more than one line assignment algorithm was selected above, this selection determines which of the resulting line assignments should be used for the analysis.",
                    )
                    ensure_state_option(
                        "measures_to_calculate_single_asc",
                        DEFAULT_WORD_MEASURES,
                        options=ALL_MEASURES_OWN,
                    )
                    measures_to_calculate_single_asc = st.multiselect(
                        "Select what word measures to calculate.",
                        options=ALL_MEASURES_OWN,
                        key="measures_to_calculate_single_asc",
                        help="This selection determines which of the supported word-level measures should be calculated.",
                    )
                    ensure_state_option(
                        "sent_measures_to_calculate_single_asc",
                        DEFAULT_SENT_MEASURES,
                        options=ALL_SENT_MEASURES,
                    )
                    sent_measures_to_calculate_single_asc = st.multiselect(
                        "Select what sentence measures to calculate.",
                        options=ALL_SENT_MEASURES,
                        key="sent_measures_to_calculate_single_asc",
                        help="This selection determines which of the supported sentence-level measures should be calculated.",
                    )

                    ensure_state_option(
                        "include_word_coords_in_output_single_asc",
                        False,
                        options=[True, False],
                    )
                    include_word_coords_in_output_single_asc = st.checkbox(
                        "Should word bounding box coordinates be included in the measures table?",
                        key="include_word_coords_in_output_single_asc",
                        help="Determines if the bounding box coordinates should be included in the word measures dataframe.",
                    )
                    run_show_analysis_single_asc_button = st.form_submit_button("Run and show analysis")
                if run_show_analysis_single_asc_button and len(algo_choice_single_asc_eyekit) > 0:
                    cp2st("sent_measures_to_calculate_single_asc")
                    cp2st("measures_to_calculate_single_asc")
                    cp2st("algo_choice_single_asc_eyekit")
                    cp2st("include_word_coords_in_output_single_asc")
                    if len(measures_to_calculate_single_asc) > 0:
                        own_word_measures = get_all_measures(
                            st.session_state["trial_single_asc"],
                            st.session_state["dffix_single_asc"],
                            prefix="word",
                            use_corrected_fixations=True,
                            correction_algo=st.session_state["algo_choice_single_asc_eyekit"],
                            measures_to_calculate=st.session_state["measures_to_calculate_single_asc"],
                            include_coords=st.session_state["include_word_coords_in_output_single_asc"],
                            save_to_csv=True,
                        )
                        st.session_state["own_word_measures_single_asc"] = own_word_measures
                        sent_measures = compute_sentence_measures(
                            st.session_state["dffix_single_asc"],
                            ensure_dataframe(st.session_state["trial_single_asc"]["chars_df"]),
                            st.session_state["algo_choice_single_asc_eyekit"],
                            st.session_state["sent_measures_to_calculate_single_asc"],
                            save_to_csv=True,
                        )
                        st.session_state["own_sent_measures_single_asc"] = sent_measures
                    else:
                        st.warning("Please select one or more word measures to continue.")
                        if "own_word_measures_single_asc" in st.session_state:
                            del st.session_state["own_word_measures_single_asc"]
                if "own_word_measures_single_asc" in st.session_state:
                    own_word_measures = st.session_state["own_word_measures_single_asc"]

                    own_analysis_tab, eyekit_tab = analysis_expander_single_asc.tabs(
                        ["Analysis without eyekit", "Analysis using eyekit"]
                    )
                    with own_analysis_tab:
                        st.markdown(
                            "This analysis method does not require manual alignment and works when the stimulus coordinates are correctly identified."
                        )
                        st.markdown("### Word measures")
                        with st.popover("Column names definitions", help="Show column names and their definitions."):
                            with open("word_measures.md", "r") as f:
                                word_measure_colnames_markdown = "\n".join(f.readlines())
                            st.markdown(word_measure_colnames_markdown)
                        st.dataframe(own_word_measures, width='stretch', hide_index=True, height=200)
                        own_word_measures_csv = convert_df(own_word_measures)
                        subject = st.session_state["trial_single_asc"]["subject"]
                        trial_id = st.session_state["trial_single_asc"]["trial_id"]
                        st.download_button(
                            "⏬ Download word measures data",
                            own_word_measures_csv,
                            f"{subject}_{trial_id}_own_word_measures_df.csv",
                            "text/csv",
                            key="own_word_measures_df_download_btn_single_asc",
                            help="Download word level measures as a .csv file with the filename containing the trial id.",
                        )
                        ensure_state_option(
                            "show_plot_analysis_single_asc",
                            True,
                            options=[True, False],
                        )
                        show_plot = st.checkbox(
                            "Show Plot",
                            key="show_plot_analysis_single_asc",
                            help="If unticked, the plots in this section will be hidden. This can speed up using the interface if the plots are not required.",
                        )
                        if show_plot:
                            measure_words_own = st.selectbox(
                                "Select measure to visualize",
                                list(own_word_measures.columns),
                                key="measure_words_own_single_asc",
                                help="Selection determines which of the calculated word-level measures gets plotted. Where the measure is dependent on the line assignment, the name of the algorithm used to carry out those line assignments is included in the name of the measure.",
                                index=own_word_measures.shape[1] - 1,
                            )
                            fix_to_plot = ["Corrected Fixations"]
                            own_word_measures_fig, desired_width_in_pixels, desired_height_in_pixels = (
                                matplotlib_plot_df(
                                    st.session_state["dffix_single_asc"],
                                    st.session_state["trial_single_asc"],
                                    [st.session_state["algo_choice_single_asc_eyekit"]],
                                    None,
                                    box_annotations=own_word_measures[measure_words_own],
                                    fix_to_plot=fix_to_plot,
                                    stim_info_to_plot=["Characters", "Word boxes"],
                                )
                            )
                            st.pyplot(own_word_measures_fig)
                        st.markdown("### Sentence measures")
                        with st.popover("Column names definitions", help="Show column names and their definitions."):
                            with open("sentence_measures.md", "r") as f:
                                sentence_measure_colnames_markdown = "\n".join(f.readlines())
                            st.markdown(sentence_measure_colnames_markdown)
                        st.dataframe(
                            st.session_state["own_sent_measures_single_asc"],
                            width='stretch',
                            hide_index=True,
                            height=200,
                        )

                        own_sent_measures_csv = convert_df(st.session_state["own_sent_measures_single_asc"])
                        st.download_button(
                            "⏬ Download sentence measures data",
                            own_sent_measures_csv,
                            f"{subject}_{trial_id}_own_sentence_measures_df.csv",
                            "text/csv",
                            key="own_sent_measures_df_download_btn_single_asc",
                            help="Download sentence level measures as a .csv file with the filename containing the trial id.",
                        )
                    with eyekit_tab:
                        eyekit_input("_single_asc")

                        fixations_tuples, textblock_input_dict, screen_size = get_fix_seq_and_text_block(
                            st.session_state["dffix_single_asc"],
                            st.session_state["trial_single_asc"],
                            x_txt_start=st.session_state["x_txt_start_for_eyekit_single_asc"],
                            y_txt_start=st.session_state["y_txt_start_for_eyekit_single_asc"],
                            font_face=st.session_state["font_face_for_eyekit_single_asc"],
                            font_size=st.session_state["font_size_for_eyekit_single_asc"],
                            line_height=st.session_state["line_height_for_eyekit_single_asc"],
                            use_corrected_fixations=True,
                            correction_algo=st.session_state["algo_choice_single_asc_eyekit"],
                        )

                        eyekitplot_img = ekm.eyekit_plot(fixations_tuples, textblock_input_dict, screen_size)
                        st.image(eyekitplot_img, "Fixations and stimulus as used for anaylsis")

                        eyekit_run_analysis_button_single_asc = st.button(
                            "Run Eyekit powered analysis",
                            key="eyekit_run_analysis_button_single_asc",
                            help="Click to run analysis using Eyekit with the input as displayed above",
                        )
                        if eyekit_run_analysis_button_single_asc:
                            st.session_state["show_eyekit_analysis_single_asc"] = True
                        if (
                            "show_eyekit_analysis_single_asc" in st.session_state
                            and st.session_state["show_eyekit_analysis_single_asc"]
                            and textblock_input_dict is not None
                        ):

                            subject = st.session_state["trial_single_asc"]["subject"]
                            trial_id = st.session_state["trial_single_asc"]["trial_id"]
                            with open(
                                f"results/fixation_sequence_eyekit_{subject}_{trial_id}.json",
                                "r",
                            ) as f:
                                fixation_sequence_json = json.load(f)
                            fixation_sequence_json_str = json.dumps(fixation_sequence_json, cls=ut.NpEncoder)

                            st.download_button(
                                "⏬ Download fixations in eyekits format",
                                fixation_sequence_json_str,
                                f"fixation_sequence_eyekit_{subject}_{trial_id}.json",
                                "json",
                                key="download_eyekit_fix_json_single_asc",
                                help="This downloads the extracted fixation information as a .json file in the eyekit format with the filename containing the subject name and trial id.",
                            )

                            with open(f"results/textblock_eyekit_{subject}_{trial_id}.json", "r") as f:
                                textblock_json = json.load(f)
                            textblock_json_str = json.dumps(textblock_json, cls=ut.NpEncoder)

                            st.download_button(
                                "⏬ Download stimulus in eyekits format",
                                textblock_json_str,
                                f"textblock_eyekit_{subject}_{trial_id}.json",
                                "json",
                                key="download_eyekit_text_json_single_asc",
                                help="This downloads the extracted stimulus information as a .json file in the eyekit format with the filename containing the subject name and trial id.",
                            )

                            word_measures_df, character_measures_df = get_eyekit_measures(
                                fixations_tuples,
                                textblock_input_dict,
                                trial=st.session_state["trial_single_asc"],
                                get_char_measures=False,
                            )

                            st.dataframe(word_measures_df, width='stretch', hide_index=True, height=200)
                            word_measures_df_csv = convert_df(word_measures_df)

                            st.download_button(
                                "⏬ Download word measures data",
                                word_measures_df_csv,
                                f"{subject}_{trial_id}_word_measures_df.csv",
                                "text/csv",
                                key="word_measures_df_download_btn_single_asc",
                            )
                            measure_words = st.selectbox(
                                "Select measure to visualize",
                                list(ekm.MEASURES_DICT.keys()),
                                key="measure_words_single_asc",
                                index=0,
                            )
                            st.image(
                                ekm.plot_with_measure(
                                    fixations_tuples, textblock_input_dict, screen_size, measure_words
                                )
                            )

                            if character_measures_df is not None:
                                st.dataframe(
                                    character_measures_df, width='stretch', hide_index=True, height=200
                                )
            else:
                single_file_tab_asc_tab.warning("🚨 Stimulus information needed for analysis 🚨")

    single_file_tab_csv_tab.markdown(
        "#### Upload one .csv file for the fixations and one .json or .csv file for the stimulus information and select a trial. Then select a line-assignment algorithm and plot/download the results"
    )

    def change_which_file_is_used_and_clear_results_for_custom():
        if st.session_state["single_file_tab_csv_tab_example_use_example_or_uploaded_file_choice"] == "Example Files":
            st.session_state["single_csv_file"] = EXAMPLE_CUSTOM_CSV_FILE
            st.session_state["single_csv_file_stim"] = EXAMPLE_CUSTOM_JSON_FILE
        else:
            st.session_state["single_csv_file"] = st.session_state["single_csv_file_uploaded"]
            st.session_state["single_csv_file_stim"] = st.session_state["single_csv_file_stim_uploaded"]

    with single_file_tab_csv_tab.form("single_file_tab_csv_tab_load_example_form"):
        csv_upl_col1, csv_upl_col2 = st.columns(2)
        single_csv_file = csv_upl_col1.file_uploader(
            "Select .csv file containing the fixation data",
            accept_multiple_files=False,
            key="single_csv_file_uploaded",
            type={"csv", "txt", "dat"},
            help="Drag and drop or select a single .csv, .txt or .dat file that you wish to process. This can be left blank if you chose to use the examples.",
        )
        single_csv_stim_file = csv_upl_col2.file_uploader(
            "Select .csv or .json file containing the stimulus data",
            accept_multiple_files=False,
            key="single_csv_file_stim_uploaded",
            type={"json", "csv", "txt", "dat","jpeg","png"},
            help="Drag and drop or select a single .json, .csv, .txt, .dat, jpeg or png file that you wish to process as the stimulus file for the uploaded fixation data. If an image is uploaded OCR will be attempted to extract the character bounding boxes, note the filename MUST match the trial name for images(so for trial t1 the image MUST be called t1.png). This can be left blank if you chose to use the examples.",
        )

        st.checkbox(
            label="Should spaces between words be included in word bounding box?",
            value=get_default_val("close_gap_between_words_csv", True),
            key="close_gap_between_words_single_csv",
            help="If this is selected, each word bounding box will include half the spaces between adjacent words. If not, the word bounding boxes will simply be the combined bounding boxes of the letters making up the word.",  # TODO check if this affects analysis
        )
        st.checkbox(
            label="Should spaces between lines be included in word and character bounding boxes?",
            value=get_default_val("close_gap_between_lines_single_csv", True),
            key="close_gap_between_lines_single_csv",
            help="If this is selected, each word and char bounding box will include half the spaces between adjacent lines.",  # TODO check if this affects analysis
        )
        use_example_or_uploaded_file_choice = st.radio(
            "Should the uploaded files be used or some example files?",
            index=1,
            options=["Uploaded Files", "Example Files"],
            key="single_file_tab_csv_tab_example_use_example_or_uploaded_file_choice",
            help="This selection determines if the uploaded file on the top left or the included example files will be used for processing.",
        )
        upload_custom_file_button = st.form_submit_button(
            label="Load selected data.", on_click=change_which_file_is_used_and_clear_results_for_custom
        )

    if upload_custom_file_button:
        for k in [
            "trial_keys_single_csv",
            "trial_single_csv",
            "dffix_single_csv",
            "dffix_cleaned_single_csv",
            "stimdf_single_csv",
            "dffix_cleaned_corrected_single_csv",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        if use_example_or_uploaded_file_choice != "Example Files":
            st.session_state["dffix_single_csv"] = load_csv_delim_agnostic(single_csv_file)
            st.session_state["dffix_col_mappings_guess_single_csv"] = find_col_name_suggestions(
                list(st.session_state["dffix_single_csv"].columns), COLNAME_CANDIDATES_CUSTOM_CSV_FIX
            )
        else:
            st.session_state["dffix_single_csv"] = pd.read_csv(EXAMPLE_CUSTOM_CSV_FILE)
            st.session_state["dffix_col_mappings_guess_single_csv"] = COLNAME_CANDIDATES_CUSTOM_CSV_FIX_DEFAULT
        st.session_state.update(st.session_state["dffix_col_mappings_guess_single_csv"])

        if use_example_or_uploaded_file_choice != "Example Files":
            if ".json" in single_csv_stim_file.name:
                decoded_input = get_decoded_input_from_file(single_csv_stim_file)
                trial = json.loads(decoded_input)
                st.session_state["stimdf_single_csv"] = trial
                colnames_stim = list(st.session_state["stimdf_single_csv"].keys())
            elif any([".png" in single_csv_stim_file.name, ".jpeg" in single_csv_stim_file.name]):
                stimdf_single_csv = get_cached_or_run_ocr(
                    single_csv_stim_file,
                    image_name=single_csv_stim_file.name,
                    trial_label=f"single_csv:{single_csv_stim_file.name}",
                )
                stimdf_single_csv.to_csv(RESULTS_FOLDER / f"{single_csv_stim_file.name}_stimdf_single_from_OCR.csv")
                if 'trial_id' in stimdf_single_csv.columns:
                    stimdf_single_csv['trial_id'] = stimdf_single_csv['trial_id'].astype(str)
                st.session_state["stimdf_single_csv"] = stimdf_single_csv
                colnames_stim = st.session_state["stimdf_single_csv"].columns
            else:
                st.session_state["stimdf_single_csv"] = load_csv_delim_agnostic(single_csv_stim_file)
                colnames_stim = st.session_state["stimdf_single_csv"].columns
            st.session_state["chars_df_col_mappings_guess_single_csv"] = find_col_name_suggestions(
                list(colnames_stim), COLNAMES_CUSTOM_CSV_STIM
            )
            for colname in COLNAME_CANDIDATES_CUSTOM_CSV_FIX['trial_id_col_name_fix']:
                if colname in st.session_state["dffix_single_csv"].columns:
                    st.session_state["dffix_single_csv"][colname] = stringify_column_keep_decimal_if_needed(st.session_state["dffix_single_csv"][colname])

        else:
            with open(EXAMPLE_CUSTOM_JSON_FILE, "r") as json_file:
                json_string = json_file.read()
            st.session_state["stimdf_single_csv"] = json.loads(json_string)
            colnames_stim = list(st.session_state["stimdf_single_csv"].keys())
            st.session_state["chars_df_col_mappings_guess_single_csv"] = COLNAMES_CUSTOM_CSV_STIM_DEFAULT
        st.session_state.update(st.session_state["chars_df_col_mappings_guess_single_csv"])

        if "algo_choice_analysis_single_csv" in st.session_state:
            del st.session_state["algo_choice_analysis_single_csv"]
    if in_st_nn("dffix_single_csv"):
        with single_file_tab_csv_tab.expander("Preview loaded files"):
            if in_st_nn("dffix_single_csv"):
                st.dataframe(
                    st.session_state["dffix_single_csv"],
                    width='stretch',
                    hide_index=True,
                    on_select="ignore",
                    height=200,
                )
            if in_st_nn("stimdf_single_csv"):
                if ".json" in st.session_state["single_csv_file_stim"].name:
                    st.json(ut.make_json_serializable(st.session_state["stimdf_single_csv"]), expanded=False)
                else:
                    st.dataframe(
                        st.session_state["stimdf_single_csv"],
                        width='stretch',
                        hide_index=True,
                        on_select="ignore",
                        height=200,
                    )
    if in_st_nn("single_csv_file") and in_st_nn("single_csv_file_stim"):
        with single_file_tab_csv_tab.expander("Column names for csv files", expanded=True):
            with st.form("Column names for csv files"):
                st.markdown("### Please set column/key names for csv/json files")
                st.markdown("#### Fixation file column names:")
                c1, c2, c3 = st.columns(3)
                x_col_name_fix = c1.text_input(
                    "x coordinate",
                    key="x_col_name_fix",
                    value=get_default_val(
                        "x_col_name_fix", st.session_state["dffix_col_mappings_guess_single_csv"]["x_col_name_fix"]
                    ),
                    help="This should be a column that contains the horizontal position (usually in pixels) of where fixations were detected.",
                )
                y_col_name_fix = c2.text_input(
                    "y coordinate",
                    key="y_col_name_fix",
                    value=get_default_val(
                        "y_col_name_fix", st.session_state["dffix_col_mappings_guess_single_csv"]["y_col_name_fix"]
                    ),
                    help="This should be a column that contains the vertical position (usually in pixels) of where fixations were detected.",
                )
                subject_col_name_fix = c1.text_input(
                    "subject id",
                    key="subject_col_name_fix",
                    value=get_default_val(
                        "subject_col_name_fix",
                        st.session_state["dffix_col_mappings_guess_single_csv"]["subject_col_name_fix"],
                    ),
                    help="This should be a column that contains the unique identifier for each subject.",
                )
                trial_id_col_name_fix = c3.text_input(
                    "trial id",
                    key="trial_id_col_name_fix",
                    value=get_default_val(
                        "trial_id_col_name_fix",
                        st.session_state["dffix_col_mappings_guess_single_csv"]["trial_id_col_name_fix"],
                    ),
                    help="A column that contains identifiers or numbers corresponding to specific trials of an experiment.",
                )
                time_start_col_name_fix = c2.text_input(
                    "fixation start time",
                    key="time_start_col_name_fix",
                    value=get_default_val(
                        "time_start_col_name_fix",
                        st.session_state["dffix_col_mappings_guess_single_csv"]["time_start_col_name_fix"],
                    ),
                    help="This should be a column that contains the timestamp when fixations start.",
                )
                time_stop_col_name_fix = c3.text_input(
                    "fixation end time",
                    key="time_stop_col_name_fix",
                    value=get_default_val(
                        "time_stop_col_name_fix",
                        st.session_state["dffix_col_mappings_guess_single_csv"]["time_stop_col_name_fix"],
                    ),
                    help="This should be a column that contains the timestamp when fixations ended.",
                )
                st.markdown("#### Stimulus file column/key names:")
                c1, c2, c3 = st.columns(3)
                x_col_name_fix_stim = c1.text_input(
                    "x coordinate",
                    key="x_col_name_fix_stim",
                    value=get_default_val("x_col_name_fix_stim", "char_x_center"),
                    help="This should be a column that contains the horizontal position (usually in pixels) of the center of the characters.",
                )
                y_col_name_fix_stim = c2.text_input(
                    "y coordinate",
                    key="y_col_name_fix_stim",
                    value=get_default_val("y_col_name_fix_stim", "char_y_center"),
                    help="This should be a column that contains the vertical position (usually in pixels) of the center of the characters",
                )
                x_start_col_name_fix_stim = c3.text_input(
                    "x min of interest areas",
                    key="x_start_col_name_fix_stim",
                    value=get_default_val("x_start_col_name_fix_stim", "char_xmin"),
                    help="This should be a column that contains the minimum horizontal position (in pixels) for each interest area.",
                )
                x_end_col_name_fix_stim = c1.text_input(
                    "x max of interest areas",
                    key="x_end_col_name_fix_stim",
                    value=get_default_val("x_end_col_name_fix_stim", "char_xmax"),
                    help="This should be a column that contains the maximum horizontal position (in pixels) for each interest area.",
                )
                y_start_col_name_fix_stim = c2.text_input(
                    "y min of interest areas",
                    key="y_start_col_name_fix_stim",
                    value=get_default_val("y_start_col_name_fix_stim", "char_ymin"),
                    help="This should be a column that contains the minimum vertical position (in pixels) for each interest area.",
                )
                y_end_col_name_fix_stim = c3.text_input(
                    "x max of interest areas",
                    key="y_end_col_name_fix_stim",
                    value=get_default_val("y_end_col_name_fix_stim", "char_ymax"),
                    help="This should be a column that contains the maximum vertical position (in pixels) for each interest area.",
                )
                char_col_name_fix_stim = c1.text_input(
                    "content of interest area",
                    key="char_col_name_fix_stim",
                    value=get_default_val("char_col_name_fix_stim", "char"),
                    help="This should be a column that contains the content associated with each interest area.",
                )
                line_num_col_name_stim = c3.text_input(
                    "line number for interest areas",
                    key="line_num_col_name_stim",
                    value=get_default_val("line_num_col_name_stim", "assigned_line"),
                    help="This should be a column that contains the unique identifier assigned to each line.",
                )
                # TODO Change to item rather than trial id?
                trial_id_col_name_stim = c2.text_input(
                    "trial id",
                    key="trial_id_col_name_stim",
                    value=get_default_val("trial_id_col_name_stim", "trial_id"),
                    help="This should be a column that contains the unique identifier for each stimulus.",
                )
                form_submitted = st.form_submit_button("Confirm column/key names")

    if (
        in_st_nn("single_csv_file")
        and in_st_nn("single_csv_file_stim")
        and in_st_nn("dffix_single_csv")
        and form_submitted
    ):
        if "trial_keys_single_csv" in st.session_state:
            del st.session_state["trial_keys_single_csv"]
        if "trial_single_csv" in st.session_state:
            del st.session_state["trial_single_csv"]
        if "trial_id_selected_single_csv" in st.session_state:
            del st.session_state["trial_id_selected_single_csv"]
        if "algo_choice_analysis_single_csv" in st.session_state:
            del st.session_state["algo_choice_analysis_single_csv"]
        if "dffix_cleaned_single_csv" in st.session_state:
            del st.session_state["dffix_cleaned_single_csv"]
        if "dffix_cleaned_corrected_single_csv" in st.session_state:
            del st.session_state["dffix_cleaned_corrected_single_csv"]

        try:
            trials_by_ids, trial_keys = get_fixations_file_trials_list(
                st.session_state["dffix_single_csv"], st.session_state["stimdf_single_csv"]
            )

            st.session_state["trials_by_ids_single_csv"] = trials_by_ids
            st.session_state["trial_keys_single_csv"] = trial_keys
        except Exception as e:
            st.session_state["logger"].warning(e)
            st.session_state["logger"].warning("get_fixations_file_trials_list failed")
            st.warning("Getting dataframes failed. Please make sure the column names are correct.")
    if "trial_keys_single_csv" in st.session_state:
        with single_file_tab_csv_tab.form(key="trial_selection_form_single_csv"):
            trial_choice = st.selectbox(
                "Which trial should be corrected?",
                st.session_state["trial_keys_single_csv"],
                key="trial_id_selected_single_csv",
                index=0,
                help="Choose one of the available trials from the list displayed.",
            )
            select_trial_btn = st.form_submit_button("Select trial")
    if "trial_keys_single_csv" in st.session_state and select_trial_btn:
        if "dffix_cleaned_single_csv" in st.session_state:
            del st.session_state["dffix_cleaned_single_csv"]
        if "dffix_cleaned_corrected_single_csv" in st.session_state:
            del st.session_state["dffix_cleaned_corrected_single_csv"]
        st.session_state["trial_single_csv"] = st.session_state["trials_by_ids_single_csv"][trial_choice]
        st.session_state["trial_single_csv"]["dffix_no_clean"] = st.session_state["trial_single_csv"]["dffix"].copy()
    if "trial_id_selected_single_csv" in st.session_state and "trial_single_csv" in st.session_state:
        trial = st.session_state["trial_single_csv"]
        show_cleaning_options(single_file_tab_csv_tab, trial["dffix"], "single_csv")
    if "dffix_cleaned_single_csv" in st.session_state:
        show_cleaning_results(
            single_file_tab_csv_tab,
            st.session_state["trials_by_ids_single_csv"][trial_choice],
            "Show Clean results",
            st.session_state["dffix_cleaned_single_csv"],
            "dffix_no_clean",
            True,
            key_str="single_csv",
        )
    if "dffix_cleaned_single_csv" in st.session_state:
        with single_file_tab_csv_tab.form(key="algo_selection_form_single_csv"):
            algo_choice_single_csv = st.multiselect(
                "Choose line-assignment algorithms",
                ALGO_CHOICES,
                key="algo_choice_single_csv",
                default=get_def_val_w_underscore("algo_choice_single_csv", DEFAULT_ALGO_CHOICE, ALGO_CHOICES),
                help="This selection determines which of the available line assignment algorithms should be used to assign each fixation to their most appropriate line of text. The rest of the analysis is dependent on this line assignment. It is recommended to try out multiple different assignment approaches to make sure it performs well for on your data.",
            )
            process_trial_btn = st.form_submit_button("Correct fixations")
    if "dffix_cleaned_single_csv" in st.session_state and process_trial_btn:
        cp2st("algo_choice_single_csv")
        if "algo_choice_analysis_single_csv" in st.session_state:
            del st.session_state["algo_choice_analysis_single_csv"]

        trial["dffix"] = st.session_state["dffix_cleaned_single_csv"]
        dffix, trial, dpi, screen_res, font, font_size = process_trial_choice_single_csv(
            trial, algo_choice_single_csv, st.session_state["models_dict"]
        )
        st.session_state["trial_single_csv"] = trial
        st.session_state["dffix_cleaned_corrected_single_csv"] = dffix
    if "dffix_cleaned_corrected_single_csv" in st.session_state:
        trial = st.session_state["trial_single_csv"]
        dffix = st.session_state["dffix_cleaned_corrected_single_csv"]
        csv = convert_df(dffix)

        single_file_tab_csv_tab.download_button(
            "⏬ Download corrected fixation data",
            csv,
            f'{trial["trial_id"]}.csv',
            "text/csv",
            key="download-csv-single_csv",
            help="This downloads the corrected fixations dataframe as a .csv file with the filename containing the trial id.",
        )
        with single_file_tab_csv_tab.expander("Show corrected fixation data", expanded=True):
            st.dataframe(dffix, width='stretch', hide_index=True, height=200)
        with single_file_tab_csv_tab.expander("Show fixation plots", expanded=True):

            plotting_checkboxes_single_single_csv = st.multiselect(
                "Select what gets plotted",
                STIM_FIX_PLOT_OPTIONS,
                default=["Uncorrected Fixations", "Corrected Fixations", "Characters", "Word boxes"],
                key="plotting_checkboxes_single_single_csv",
                help="This selection determines what information is plotted. The Corrected Fixations are the fixations after being snapped to their assigned line of text. The Word and Character boxes are the bounding boxes for the stimulus.",
            )

            st.plotly_chart(
                plotly_plot_with_image(
                    dffix,
                    trial,
                    to_plot_list=plotting_checkboxes_single_single_csv,
                    algo_choice=st.session_state["algo_choice_single_csv"],
                ),
                width='stretch',
            )
            st.plotly_chart(plot_y_corr(dffix, st.session_state["algo_choice_single_csv"]), width='stretch')
            plotlist = [x for x in dffix.columns if "Unnamed" not in str(x)]
            plot_choice = st.multiselect(
                "Which measures should be visualized?",
                plotlist,
                key="plot_choice_fix_measure",
                default=plotlist[-1],
            )
            st.plotly_chart(plot_fix_measure(dffix, plot_choice, "Index"), width='stretch')

        if "chars_list" in trial:
            analysis_expander_custom = single_file_tab_csv_tab.expander("Show Analysis results", True)
            with analysis_expander_custom.form("run_analysis_single_csv"):
                algo_choice_custom_eyekit = st.selectbox(
                    "Algorithm", st.session_state["algo_choice_single_csv"], index=None, key="algo_choice_custom_eyekit"
                )
                run_analysis_btn_custom_csv = st.form_submit_button("Run Analysis")
            if run_analysis_btn_custom_csv:
                if handle_single_csv_analysis_selection(algo_choice_custom_eyekit):
                    (
                        y_diff,
                        x_txt_start,
                        y_txt_start,
                        font_face,
                        font_size,
                        line_height,
                    ) = add_default_font_and_character_props_to_state(trial)
                    font_size = set_font_from_chars_list(trial)
                    st.session_state["from_trial_y_diff_for_eyekit_single_csv"] = y_diff
                    st.session_state["from_trial_x_txt_start_for_eyekit_single_csv"] = x_txt_start
                    st.session_state["from_trial_y_txt_start_for_eyekit_single_csv"] = y_txt_start
                    st.session_state["from_trial_font_face_for_eyekit_single_csv"] = font_face
                    st.session_state["from_trial_font_size_for_eyekit_single_csv"] = font_size
                    st.session_state["from_trial_line_height_for_eyekit_single_csv"] = line_height
        selected_algo_single_csv = st.session_state.get("algo_choice_analysis_single_csv")
        if selected_algo_single_csv:
            own_analysis_tab_custom, eyekit_tab_custom = analysis_expander_custom.tabs(
                ["Analysis without eyekit", "Analysis using eyekit"]
            )
            with eyekit_tab_custom:
                eyekit_input(ending_str="_single_csv")

                fixations_tuples, textblock_input_dict, screen_size = ekm.get_fix_seq_and_text_block(
                    dffix,
                    trial,
                    x_txt_start=st.session_state["x_txt_start_for_eyekit_single_csv"],
                    y_txt_start=st.session_state["y_txt_start_for_eyekit_single_csv"],
                    font_face=st.session_state["font_face_for_eyekit_single_csv"],
                    font_size=st.session_state["font_size_for_eyekit_single_csv"],
                    line_height=st.session_state["line_height_for_eyekit_single_csv"],
                    use_corrected_fixations=True,
                    correction_algo=selected_algo_single_csv,
                )
                eyekitplot_img = ekm.eyekit_plot(fixations_tuples, textblock_input_dict, screen_size)
                st.image(eyekitplot_img, "Fixations and stimulus as used for anaylsis")

                with open(f'results/fixation_sequence_eyekit_{trial["trial_id"]}.json', "r") as f:
                    fixation_sequence_json = json.load(f)
                fixation_sequence_json_str = json.dumps(fixation_sequence_json, cls=ut.NpEncoder)

                st.download_button(
                    "⏬ Download fixations in eyekits format",
                    fixation_sequence_json_str,
                    f'fixation_sequence_eyekit_{trial["trial_id"]}.json',
                    "json",
                    key="download_eyekit_fix_json_single_csv",
                    help="This downloads the extracted fixation information as a .json file in the eyekit format with the filename containing the subject name and trial id.",
                )

                with open(f'results/textblock_eyekit_{trial["trial_id"]}.json', "r") as f:
                    textblock_json = json.load(f)
                textblock_json_str = json.dumps(textblock_json, cls=ut.NpEncoder)

                st.download_button(
                    "⏬ Download stimulus in eyekits format",
                    textblock_json_str,
                    f'textblock_eyekit_{trial["trial_id"]}.json',
                    "json",
                    key="download_eyekit_text_json_single_csv",
                    help="This downloads the extracted stimulus information as a .json file in the eyekit format with the filename containing the subject name and trial id.",
                )

                word_measures_df, character_measures_df = get_eyekit_measures(
                    fixations_tuples, textblock_input_dict, trial=trial, get_char_measures=False
                )

                st.dataframe(word_measures_df, width='stretch', hide_index=True, height=200)
                word_measures_df_csv = convert_df(word_measures_df)

                st.download_button(
                    "⏬ Download word measures data",
                    word_measures_df_csv,
                    f'{trial["trial_id"]}_word_measures_df.csv',
                    "text/csv",
                    key="word_measures_df_download_btn_single_csv",
                )
                measure_words = st.selectbox(
                    "Select measure to visualize", list(ekm.MEASURES_DICT.keys()), key="measure_words_single_csv"
                )
                st.image(ekm.plot_with_measure(fixations_tuples, textblock_input_dict, screen_size, measure_words))

                if character_measures_df is not None:
                    st.dataframe(character_measures_df, width='stretch', hide_index=True, height=200)

            with own_analysis_tab_custom:
                st.markdown(
                    "This analysis method does not require manual alignment and works when the automated stimulus coordinates are correct."
                )
                own_word_measures = get_all_measures(
                    trial,
                    dffix,
                    prefix="word",
                    use_corrected_fixations=True,
                    correction_algo=selected_algo_single_csv,
                    save_to_csv=True,
                    measures_to_calculate=ALL_MEASURES_OWN,
                )
                st.dataframe(own_word_measures, width='stretch', hide_index=True, height=200)
                own_word_measures_csv = convert_df(own_word_measures)

                sent_measures_single_csv = compute_sentence_measures(
                    dffix,
                    ensure_dataframe(trial["chars_df"]),
                    selected_algo_single_csv,
                    ALL_SENT_MEASURES,
                    save_to_csv=True,
                )
                st.session_state["own_sent_measures_single_csv"] = sent_measures_single_csv
                st.dataframe(sent_measures_single_csv, width='stretch', hide_index=True, height=200)

                st.download_button(
                    "⏬ Download word measures data",
                    own_word_measures_csv,
                    f'{trial["trial_id"]}_own_word_measures_df.csv',
                    "text/csv",
                    key="own_word_measures_df_download_btn",
                )
                measure_words_own = st.selectbox(
                    "Select measure to visualize",
                    list(own_word_measures.columns),
                    key="measure_words_own_single_csv",
                    index=own_word_measures.shape[1] - 1,
                )
                fix_to_plot = ["Corrected Fixations"]
                own_word_measures_fig, _, _ = matplotlib_plot_df(
                    dffix,
                    trial,
                    [selected_algo_single_csv],
                    None,
                    box_annotations=own_word_measures[measure_words_own],
                    fix_to_plot=fix_to_plot,
                )
                st.pyplot(own_word_measures_fig)
    with multi_file_tab:
        st.subheader(
            "Upload one or more .asc files (Can be compressed). Then load configuration file or manually select desired options."
        )
        settings_to_save = {
            k.replace("_multi_asc", ""): v
            for (k, v) in st.session_state.items()
            if k
            in [
                "trial_start_keyword_multi_asc",
                "trial_end_keyword_multi_asc",
                "close_gap_between_words_multi_asc",
                "paragraph_trials_only_multi_asc",
                "discard_fixations_without_sfix_multi_asc",
                "discard_far_out_of_text_fix_multi_asc",
                "discard_blinks_fix_multi_asc",
                "outlier_crit_x_threshold_multi_asc",
                "outlier_crit_y_threshold_multi_asc",
                "discard_long_fix_multi_asc",
                "discard_long_fix_threshold_multi_asc",
                "choice_handle_short_and_close_fix_multi_asc",
                "merge_distance_threshold_multi_asc",
                "algo_choice_multi_asc",
                "use_multiprocessing_multi_asc",
                "fix_cols_to_add_multi_asc",
                "measures_to_calculate_multi_asc",
                "include_word_coords_in_output_multi_asc",
                "sent_measures_to_calculate_multi_asc",
                "save_files_for_each_trial_individually_multi_asc",
            ]
        }
        if len(settings_to_save) > 0:
            st.download_button(
                "⏬ Download all multi .asc file settings as JSON",
                json.dumps(settings_to_save, cls=ut.NpEncoder),
                "settings_to_save_multi_asc.json",
                "json",
                key="download_settings_to_save_multi_asc",
                help="This downloads the configuration as a .json file and can be used to reload the settings later.",
            )
        with st.expander("Load config file."):
            with st.form("multi_asc_file_tab_asc_tab_load_settings_from_file_form"):
                st.file_uploader(
                    "Select .json config file to reload a previous processing configuration",
                    accept_multiple_files=False,
                    key="multi_asc_file_settings_file_uploaded",
                    type=["json"],
                    help="Load in a configuration file as .json to set the parameters below to the previously used configuration.",
                )
                cfg_load_btn_multi_asc = st.form_submit_button("Load in config")
        if cfg_load_btn_multi_asc and in_st_nn("multi_asc_file_settings_file_uploaded"):
            json_string = st.session_state["multi_asc_file_settings_file_uploaded"].getvalue().decode("utf-8")
            st.session_state["loaded_settings_multi_asc"] = {
                f"{k}_multi_asc": v for k, v in json.loads(json_string).items()
            }
            st.session_state.update(st.session_state["loaded_settings_multi_asc"])
    with multi_file_tab.expander("Upload files and choose configuration options.", True):
        with st.form("upload_and_config_form_multiu_asc"):
            multifile_col, multi_algo_col = st.columns((1, 1))

            with multifile_col:
                st.markdown("## File selection")
                multi_asc_filelist = st.file_uploader(
                    "Upload .asc Files",
                    accept_multiple_files=True,
                    key="multi_asc_filelist",
                    type=["asc", "tar", "zip"],
                    help="Drag and drop or select a one or multiple .asc files that you wish to process. For efficient uploading it is also supported that the .asc files are compressed into a .zip or .tar file.",
                )
                multi_asc_file_ias_files_uploaded = st.file_uploader(
                    "Upload all .ias files associated with the .asc files. Leave empty if you don't use .ias files.",
                    accept_multiple_files=True,
                    key="multi_asc_file_ias_files_uploaded",
                    type=["ias"],
                    help="If the stimulus information is not part of the .asc file then all .ias files associated with your .asc files should be put here. This will allow the program to align each trial found in the .asc files with the correct stimulus text by finding the .ias filename in the .asc file (Needs to be flagged with the 'IAREA FILE').",
                )

            with multi_algo_col:
                st.markdown("## Configuration")
                show_file_parsing_settings("_multi_asc")
                st.markdown("### Trial cleaning settings")
                ensure_state_option(
                    "discard_fixations_without_sfix_multi_asc",
                    get_default_val("discard_fixations_without_sfix_multi_asc", True),
                    options=[True, False],
                )
                discard_fixations_without_sfix = st.checkbox(
                    "Should fixations that start before trial start but end after be discarded?",
                    key="discard_fixations_without_sfix_multi_asc",
                    help="In cases where the trigger flag for the start of the trial occurs during a fixation, this setting determines wether that fixation is to be discarded or kept.",
                )
                ensure_state_option(
                    "discard_blinks_fix_multi_asc",
                    get_default_val("discard_blinks_fix_multi_asc", True),
                    options=[True, False],
                )
                discard_blinks_fix_multi_asc = st.checkbox(
                    "Should fixations that happen just before or after a blink event be discarded?",
                    key="discard_blinks_fix_multi_asc",
                    help="This determines if fixations that occur just after or just before a detected blink are discarded and therefore excluded from analysis.",
                )
                ensure_state_option(
                    "discard_far_out_of_text_fix_multi_asc",
                    get_default_val("discard_far_out_of_text_fix_multi_asc", True),
                    options=[True, False],
                )
                discard_far_out_of_text_fix_multi_asc = st.checkbox(
                    "Should fixations that are far outside the text be discarded? (set margins below)",
                    key="discard_far_out_of_text_fix_multi_asc",
                    help="Using the thresholds set below this option determines whether fixations that are further outside the text lines in both horizontal and vertical direction should be discarded.",
                )
                ensure_state_option(
                    "outlier_crit_x_threshold_multi_asc",
                    get_default_val("outlier_crit_x_threshold_multi_asc", 2.0),
                    validator=lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 20.0,
                )
                outlier_crit_x_threshold_multi_asc = st.number_input(
                    "Maximum horizontal distance from first/last character on line (in character widths)",
                    min_value=0.0,
                    max_value=20.0,
                    step=0.25,
                    key="outlier_crit_x_threshold_multi_asc",
                    help=r"This option is used to set the maximum horizontal distance a fixation can have from the edges of a line of text before it will be considered to be far outside the text. This distance uses the average character width found in the stimulus text as a unit with the smallest increment being 25 % of this width.",
                )
                ensure_state_option(
                    "outlier_crit_y_threshold_multi_asc",
                    get_default_val("outlier_crit_y_threshold_multi_asc", 0.5),
                    validator=lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 5.0,
                )
                outlier_crit_y_threshold_multi_asc = st.number_input(
                    "Maximum vertical distance from top/bottom of line (in line heights)",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.05,
                    key="outlier_crit_y_threshold_multi_asc",
                    help=r"This option is used to set the maximum vertical distance a fixation can have from the top and bottom edges of a line of text before it will be considered to be far outside the text. This distance uses the unit of average line height and the smallest increment is 5 % of this height.",
                )

                ensure_state_option(
                    "discard_long_fix_multi_asc",
                    get_default_val("discard_long_fix_multi_asc", True),
                    options=[True, False],
                )
                discard_long_fix_multi_asc = st.checkbox(
                    "Should long fixations be discarded? (set threshold below)",
                    key="discard_long_fix_multi_asc",
                    help="If this option is selected, overly long fixations will be discarded. What is considered an overly long fixation is determined by the duration threshold set below.",
                )
                ensure_state_option(
                    "discard_long_fix_threshold_multi_asc",
                    get_default_val("discard_long_fix_threshold_multi_asc", DEFAULT_LONG_FIX_THRESHOLD),
                    validator=lambda v: isinstance(v, (int, float)) and 20 <= v <= 3000,
                )
                discard_long_fix_threshold_multi_asc = st.number_input(
                    "Maximum duration allowed for fixations (ms)",
                    min_value=20,
                    max_value=3000,
                    step=5,
                    key="discard_long_fix_threshold_multi_asc",
                    help="Fixations longer than this duration will be considered overly long fixations.",
                )

                short_fix_default_index = get_default_index(
                    "choice_handle_short_and_close_fix_multi_asc", SHORT_FIX_CLEAN_OPTIONS, 1
                )
                ensure_state_option(
                    "choice_handle_short_and_close_fix_multi_asc",
                    SHORT_FIX_CLEAN_OPTIONS[short_fix_default_index],
                    options=SHORT_FIX_CLEAN_OPTIONS,
                )
                choice_handle_short_and_close_fix_multi_asc = st.radio(
                    "How should short fixations be handled?",
                    SHORT_FIX_CLEAN_OPTIONS,
                    key="choice_handle_short_and_close_fix_multi_asc",
                    help="Merge: merges with either previous or next fixation and discards it if it is the last fixation and below the threshold. Merge then discard first tries to merge short fixations and then discards any short fixations that could not be merged. Discard simply discards all short fixations.",
                )
                ensure_state_option(
                    "short_fix_threshold_multi_asc",
                    get_default_val("short_fix_threshold_multi_asc", 80),
                    validator=lambda v: isinstance(v, (int, float)) and 1 <= v <= 500,
                )
                short_fix_threshold_multi_asc = st.number_input(
                    "Minimum fixation duration (ms)",
                    min_value=1,
                    max_value=500,
                    key="short_fix_threshold_multi_asc",
                    help="Fixations shorter than this duration will be considered short fixations.",
                )
                ensure_state_option(
                    "merge_distance_threshold_multi_asc",
                    get_default_val("merge_distance_threshold_multi_asc", DEFAULT_MERGE_DISTANCE_THRESHOLD),
                    validator=lambda v: isinstance(v, (int, float)) and 1 <= v <= 20,
                )
                merge_distance_threshold_multi_asc = st.number_input(
                    "Maximum distance between fixations when merging (in character widths)",
                    min_value=1,
                    max_value=20,
                    key="merge_distance_threshold_multi_asc",
                    help="When merging short fixations this is the maximum allowed distance between them.",
                )
                st.markdown("### Line assignment settings")
                ensure_state_option(
                    "algo_choice_multi_asc",
                    get_default_val("algo_choice_multi_asc", DEFAULT_ALGO_CHOICE),
                    options=ALGO_CHOICES,
                )
                algo_choice_multi_asc = st.multiselect(
                    "Choose line-assignment algorithms",
                    ALGO_CHOICES,
                    key="algo_choice_multi_asc",
                    help="This selection determines which of the available line assignment algorithms should be used to assign each fixation to their most appropriate line of text. The rest of the analysis is dependent on this line assignment. It is recommended to try out multiple different assignment approaches to make sure it performs well for on your data.",
                )
                st.markdown("### Analysis settings")
                ensure_state_option(
                    "fix_cols_to_add_multi_asc",
                    get_default_val("fix_cols_to_add_multi_asc", DEFAULT_FIX_MEASURES),
                    options=ALL_FIX_MEASURES,
                )
                fix_cols_to_add_multi_asc = st.multiselect(
                    "Select what fixation measures to calculate.",
                    options=ALL_FIX_MEASURES,
                    key="fix_cols_to_add_multi_asc",
                    help="This selection determines what fixation-level measures will be calculated. If you are in doubt about which ones you might need for your analysis, you can select all of them since it only slightly adds to the processing time.",
                )
                ensure_state_option(
                    "measures_to_calculate_multi_asc",
                    get_default_val("measures_to_calculate_multi_asc", DEFAULT_WORD_MEASURES),
                    options=ALL_MEASURES_OWN,
                )
                measures_to_calculate_multi_asc = st.multiselect(
                    "Select what word measures to calculate.",
                    options=ALL_MEASURES_OWN,
                    key="measures_to_calculate_multi_asc",
                    help="This selection determines which of the supported word-level measures should be calculated.",
                )
                ensure_state_option(
                    "include_word_coords_in_output_multi_asc",
                    get_default_val("include_word_coords_in_output_multi_asc", False),
                    options=[True, False],
                )
                include_word_coords_in_output_multi_asc = st.checkbox(
                    "Should word bounding box coordinates be included in the measures table?",
                    key="include_word_coords_in_output_multi_asc",
                    help="Determines if the bounding box coordinates should be included in the word measures dataframe.",
                )

                ensure_state_option(
                    "sent_measures_to_calculate_multi_asc",
                    get_default_val("sent_measures_to_calculate_multi_asc", DEFAULT_SENT_MEASURES),
                    options=ALL_SENT_MEASURES,
                )
                sent_measures_to_calculate_multi_asc = st.multiselect(
                    "Select what sentence measures to calculate.",
                    options=ALL_SENT_MEASURES,
                    key="sent_measures_to_calculate_multi_asc",
                    help="This selection determines which of the supported sentence-level measures should be calculated.",
                )
                st.markdown("### Multiprocessing setting")
                ensure_state_option(
                    "use_multiprocessing_multi_asc",
                    get_default_val("use_multiprocessing_multi_asc", True),
                    options=[True, False],
                )
                use_multiprocessing_multi_asc = st.checkbox(
                    "Process trials in parallel (fast but experimental)",
                    key="use_multiprocessing_multi_asc",
                    help="This determines whether multiprocessing is used for processing the trials in an .asc file in parallel. This can significantly speed up processing but will not show a progress bar for each trial. If it fails the program will fall back to a single process.",
                )
                ensure_state_option(
                    "save_files_for_each_trial_individually_multi_asc",
                    get_default_val("save_files_for_each_trial_individually_multi_asc", False),
                    options=[True, False],
                )
                save_files_for_each_trial_individually_multi_asc = st.checkbox(
                    "Save fixations, saccades, stimulus and metadata for each trial to a seperate file.",
                    key="save_files_for_each_trial_individually_multi_asc",
                    help="This setting determines if the results for each trial will be saved as an individual file or just be recorded as part of the overall output dataframes.",
                )
            st.markdown("### Click to run")
            process_trial_btn_multi = st.form_submit_button(
                "🚀 Process files",
                help="Using the configuration set above this button will start the processing of all trials in all .asc files. The results will be displayed below once completed. Depending on the number of trials, this can take several minutes.",
            )
        if process_trial_btn_multi and not (
            "multi_asc_filelist" in st.session_state and len(st.session_state["multi_asc_filelist"]) > 0
        ):
            st.warning("Please upload files to run processing.")
        if (
            process_trial_btn_multi
            and "multi_asc_filelist" in st.session_state
            and len(st.session_state["multi_asc_filelist"]) > 0
        ):
            if "dffix_multi_asc" in st.session_state:
                del st.session_state["dffix_multi_asc"]

            if "results" in st.session_state:
                st.session_state["results"] = {}

            if st.session_state["trial_start_keyword_multi_asc"] == "custom":
                trial_start_keyword_multi_asc = st.session_state["trial_custom_start_keyword_multi_asc"]
            else:
                trial_start_keyword_multi_asc = st.session_state["trial_start_keyword_multi_asc"]
            if st.session_state["trial_end_keyword_multi_asc"] == "custom":
                end_trial_at_keyword_multi_asc = st.session_state["trial_custom_end_keyword_multi_asc"]
            else:
                end_trial_at_keyword_multi_asc = st.session_state["trial_end_keyword_multi_asc"]
            (
                list_of_trial_lists,
                _,
                results_keys,
                zipfiles_with_results,
                all_fix_dfs_concat,
                all_sacc_dfs_concat,
                all_chars_dfs_concat,
                all_words_dfs_concat,
                all_sentence_dfs_concat,
                all_trials_by_subj,
                trials_summary,
                subjects_summary,
                trials_quick_meta_df,
            ) = process_all_asc_files(
                asc_files=multi_asc_filelist,
                algo_choice_multi_asc=algo_choice_multi_asc,
                ias_files=multi_asc_file_ias_files_uploaded,
                close_gap_between_words=st.session_state["close_gap_between_words_multi_asc"],
                close_gap_between_lines=st.session_state["close_gap_between_lines_multi_asc"],
                trial_start_keyword=trial_start_keyword_multi_asc,
                end_trial_at_keyword=end_trial_at_keyword_multi_asc,
                paragraph_trials_only=st.session_state["paragraph_trials_only_multi_asc"],
                choice_handle_short_and_close_fix=choice_handle_short_and_close_fix_multi_asc,
                discard_fixations_without_sfix=discard_fixations_without_sfix,
                discard_far_out_of_text_fix=discard_far_out_of_text_fix_multi_asc,
                x_thres_in_chars=outlier_crit_x_threshold_multi_asc,
                y_thresh_in_heights=outlier_crit_y_threshold_multi_asc,
                short_fix_threshold=short_fix_threshold_multi_asc,
                merge_distance_threshold=merge_distance_threshold_multi_asc,
                discard_long_fix=discard_long_fix_multi_asc,
                discard_long_fix_threshold=discard_long_fix_threshold_multi_asc,
                discard_blinks=discard_blinks_fix_multi_asc,
                measures_to_calculate_multi_asc=measures_to_calculate_multi_asc,
                include_coords_multi_asc=include_word_coords_in_output_multi_asc,
                sent_measures_to_calculate_multi_asc=sent_measures_to_calculate_multi_asc,
                use_multiprocessing=use_multiprocessing_multi_asc,
                fix_cols_to_add_multi_asc=fix_cols_to_add_multi_asc,
                save_files_for_each_trial_individually=save_files_for_each_trial_individually_multi_asc,
            )
            if trials_summary is not None:
                st.session_state["trials_summary_df_multi_asc"] = trials_summary
            if subjects_summary is not None:
                st.session_state["subjects_summary_df_multi_asc"] = subjects_summary

            st.session_state["list_of_trial_lists"] = list_of_trial_lists
            st.session_state["trial_choices_multi_asc"] = results_keys
            st.session_state["zipfiles_with_results"] = zipfiles_with_results
            st.session_state["all_fix_dfs_concat_multi_asc"] = all_fix_dfs_concat
            st.session_state["all_sacc_dfs_concat_multi_asc"] = all_sacc_dfs_concat
            st.session_state["all_chars_dfs_concat_multi_asc"] = all_chars_dfs_concat
            st.session_state["all_words_dfs_concat_multi_asc"] = all_words_dfs_concat
            st.session_state["all_sentence_dfs_concat_multi_asc"] = all_sentence_dfs_concat
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
            st.session_state["all_trials_by_subj"] = {
                k_outer: {
                    k: {prop: val for prop, val in v.items() if prop not in offload_list} for k, v in v_outer.items()
                }
                for k_outer, v_outer in all_trials_by_subj.items()
            }
            subs_str = "-".join([s for s in all_trials_by_subj.keys()])
            if len(subs_str) > 100:
                subs_str = subs_str[:100] + "_etc"
            st.session_state["trials_df"] = trials_quick_meta_df.drop_duplicates().dropna(subset="text", axis=0)
            st.session_state["trials_df"].to_csv(RESULTS_FOLDER / f"{subs_str}_comb_items_lines_text.csv")
            if "text_with_newlines" in st.session_state["trials_df"].columns:
                st.session_state["trials_df"] = (
                    st.session_state["trials_df"].drop(columns=["text_with_newlines"]).copy()
                )
            st.session_state["all_own_word_measures_concat"] = all_words_dfs_concat

    if in_st_nn("all_fix_dfs_concat_multi_asc"):
        if "all_trials_by_subj" in st.session_state:
            multi_file_tab.markdown("### All meta data by subject and trial")
            multi_file_tab.json(ut.make_json_serializable(st.session_state["all_trials_by_subj"]), expanded=False)
        multi_file_tab.markdown("### Item level stimulus overview")
        with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
            item_colnames_markdown = read_item_col_names()
            st.markdown(item_colnames_markdown)
        multi_file_tab.dataframe(st.session_state["trials_df"], width='stretch', height=200)
        if in_st_nn("subjects_summary_df_multi_asc"):
            multi_file_tab.markdown("### Subject level summary statistics")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                subject_measure_colnames_markdown = read_subject_meas_col_names()
                st.markdown(subject_measure_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["subjects_summary_df_multi_asc"], width='stretch', height=200
            )
        if in_st_nn("trials_summary_df_multi_asc"):
            multi_file_tab.markdown("### Trial level summary statistics")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                trials_colnames_markdown = read_trial_col_names()
                st.markdown(trials_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["trials_summary_df_multi_asc"], width='stretch', height=200
            )

        multi_file_tab.markdown("### Combined fixations dataframe and fixation level features")
        with multi_file_tab.popover("Column name definitions"):
            fix_colnames_markdown = get_fix_colnames_markdown()
            st.markdown(fix_colnames_markdown)
        multi_file_tab.dataframe(st.session_state["all_fix_dfs_concat_multi_asc"], width='stretch', height=200)

        high_fix_count_dfs = []
        for algo_choice in st.session_state["algo_choice_multi_asc"]:
            fixation_counts = (
                st.session_state["all_fix_dfs_concat_multi_asc"]
                .loc[:, ["subject", "trial_id", f"on_word_number_{algo_choice}", f"on_word_{algo_choice}"]]
                .value_counts()
                .sort_values(ascending=False)
            )
            high_fixation_words = fixation_counts[fixation_counts >= 7].index
            high_fix_count_dfs.append(
                fixation_counts[high_fixation_words]
                .reset_index(name=f"assigned_fixations_{algo_choice}")
                .rename({f"on_word_number_{algo_choice}": "word_number", f"on_word_{algo_choice}": "word"}, axis=1)
            )
        if len(high_fix_count_dfs) > 1:
            merged_df = high_fix_count_dfs[0]
            for df in high_fix_count_dfs[1:]:
                merged_df = pd.merge(merged_df, df, how="outer", on=["subject", "trial_id", "word_number", "word"])
            high_fix_count_dfs_cat = merged_df
        else:
            high_fix_count_dfs_cat = high_fix_count_dfs[0]
        if not high_fix_count_dfs_cat.empty:
            multi_file_tab.warning(
                "Some words had a large number of fixations assigned to them.  If this seems incorrect please adjust the correction algorithm."
            )
            multi_file_tab.markdown(
                "### Words that had a large number of fixations assigned to them and may need to be investigated"
            )
            multi_file_tab.dataframe(high_fix_count_dfs_cat, width='stretch', height=200)
            subs_str = "-".join([s for s in st.session_state["all_trials_by_subj"].keys()])
            if len(subs_str) > 100:
                subs_str = subs_str[:100] + "_etc"
            high_fix_count_dfs_cat.to_csv(RESULTS_FOLDER / f"{subs_str}_words_with_many_fixations.csv")

        if "all_correction_stats" in st.session_state:
            multi_file_tab.markdown("### Correction statistics")
            multi_file_tab.dataframe(st.session_state["all_correction_stats"], width='stretch', height=200)
        multi_file_tab.markdown("### Combined saccades dataframe and saccade level features")
        with multi_file_tab.popover("Column name definitions"):
            sac_colnames_markdown = get_sac_colnames_markdown()
            st.markdown(sac_colnames_markdown)
        multi_file_tab.dataframe(
            st.session_state["all_sacc_dfs_concat_multi_asc"], width='stretch', height=200
        )
        multi_file_tab.markdown("### Combined characters dataframe")
        with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
            chars_colnames_markdown = read_chars_col_names()
            st.markdown(chars_colnames_markdown)
        multi_file_tab.dataframe(
            st.session_state["all_chars_dfs_concat_multi_asc"], width='stretch', height=200
        )

        if not st.session_state["all_own_word_measures_concat"].empty:
            multi_file_tab.markdown("### Combined words dataframe and word level features")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                word_measure_colnames_markdown = read_word_meas_col_names()
                st.markdown(word_measure_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["all_own_word_measures_concat"], width='stretch', height=200
            )
        if not st.session_state["all_sentence_dfs_concat_multi_asc"].empty:
            multi_file_tab.markdown("### Combined sentence dataframe and sentence level features")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                sentence_measure_colnames_markdown = read_sent_meas_col_names()
                st.markdown(sentence_measure_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["all_sentence_dfs_concat_multi_asc"], width='stretch', height=200
            )
    if "zipfiles_with_results" in st.session_state:
        multi_res_col1, multi_res_col2 = multi_file_tab.columns(2)

        chosen_zip = multi_res_col1.selectbox("Choose results to download", st.session_state["zipfiles_with_results"])
        zipnamestem = pl.Path(chosen_zip).stem
        with open(chosen_zip, "rb") as f:
            multi_res_col2.download_button(f"⏬ Download {zipnamestem}.zip", f, file_name=f"results_{zipnamestem}.zip")

    # CSV bulk processing expander
    with multi_file_tab.expander("Upload multiple CSV files and image files for bulk processing.", expanded=False):
        with st.form("upload_and_config_form_multi_csv"):
            csv_file_col, csv_algo_col = st.columns((1, 1))

            with csv_file_col:
                st.markdown("## File selection")
                multi_csv_filelist = st.file_uploader(
                    "Upload .csv files containing fixation data",
                    accept_multiple_files=True,
                    key="multi_csv_filelist",
                    type=["csv", "txt", "dat"],
                    help="Drag and drop or select multiple .csv, .txt, or .dat files that contain the fixation data. Each file should contain fixations for one or more trials.",
                )
                multi_image_filelist = st.file_uploader(
                    "Upload image files for stimulus recognition",
                    accept_multiple_files=True,
                    key="multi_image_filelist",
                    type=["png", "jpg", "jpeg"],
                    help="Drag and drop or select image files. The filename (without extension) should match the trial_id in the CSV files.",
                )
                st.checkbox(
                    label="Should spaces between words be included in word bounding box?",
                    value=get_default_val("close_gap_between_words_multi_csv", True),
                    key="close_gap_between_words_multi_csv",
                    help="If this is selected, each word bounding box will include half the spaces between adjacent words.",
                )
                st.checkbox(
                    label="Should spaces between lines be included in word and character bounding boxes?",
                    value=get_default_val("close_gap_between_lines_multi_csv", True),
                    key="close_gap_between_lines_multi_csv",
                    help="If this is selected, each word and char bounding box will include half the spaces between adjacent lines.",
                )

            with csv_algo_col:
                st.markdown("## Configuration")
                st.markdown("### Trial cleaning settings")
                discard_blinks_fix_multi_csv = st.checkbox(
                    "Should fixations that happen just before or after a blink event be discarded?",
                    value=get_def_val_w_underscore("discard_blinks_fix_multi_csv", True, [True, False]),
                    key="discard_blinks_fix_multi_csv",
                    help="This determines if fixations that occur just after or just before a detected blink are discarded.",
                )
                discard_far_out_of_text_fix_multi_csv = st.checkbox(
                    "Should fixations that are far outside the text be discarded? (set margins below)",
                    value=get_default_val("discard_far_out_of_text_fix_multi_csv", True),
                    key="discard_far_out_of_text_fix_multi_csv",
                    help="Using the thresholds set below this option determines whether fixations that are further outside the text lines should be discarded.",
                )
                outlier_crit_x_threshold_multi_csv = st.number_input(
                    "Maximum horizontal distance from first/last character on line (in character widths)",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.0,
                    step=0.25,
                    key="outlier_crit_x_threshold_multi_csv",
                    help="Maximum horizontal distance a fixation can have from the edges of a line of text.",
                )
                outlier_crit_y_threshold_multi_csv = st.number_input(
                    "Maximum vertical distance from top/bottom of line (in line heights)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.05,
                    key="outlier_crit_y_threshold_multi_csv",
                    help="Maximum vertical distance a fixation can have from the top and bottom edges of a line of text.",
                )

                discard_long_fix_multi_csv = st.checkbox(
                    "Should long fixations be discarded? (set threshold below)",
                    value=get_default_val("discard_long_fix_multi_csv", True),
                    key="discard_long_fix_multi_csv",
                    help="If this option is selected, overly long fixations will be discarded.",
                )
                discard_long_fix_threshold_multi_csv = st.number_input(
                    "Maximum duration allowed for fixations (ms)",
                    min_value=20,
                    max_value=3000,
                    value=DEFAULT_LONG_FIX_THRESHOLD,
                    step=5,
                    key="discard_long_fix_threshold_multi_csv",
                    help="Fixations longer than this duration will be considered overly long fixations.",
                )

                choice_handle_short_and_close_fix_multi_csv = st.radio(
                    "How should short fixations be handled?",
                    SHORT_FIX_CLEAN_OPTIONS,
                    index=get_default_index("choice_handle_short_and_close_fix_multi_csv", SHORT_FIX_CLEAN_OPTIONS, 1),
                    key="choice_handle_short_and_close_fix_multi_csv",
                    help="Merge: merges with either previous or next fixation. Merge then discard first tries to merge short fixations and then discards any that could not be merged. Discard simply discards all short fixations.",
                )
                short_fix_threshold_multi_csv = st.number_input(
                    "Minimum fixation duration (ms)",
                    min_value=1,
                    max_value=500,
                    value=get_default_val("short_fix_threshold_multi_csv", 80),
                    key="short_fix_threshold_multi_csv",
                    help="Fixations shorter than this duration will be considered short fixations.",
                )
                merge_distance_threshold_multi_csv = st.number_input(
                    "Maximum distance between fixations when merging (in character widths)",
                    min_value=1,
                    max_value=20,
                    value=get_default_val("merge_distance_threshold_multi_csv", DEFAULT_MERGE_DISTANCE_THRESHOLD),
                    key="merge_distance_threshold_multi_csv",
                    help="When merging short fixations this is the maximum allowed distance between them.",
                )
                st.markdown("### Line assignment settings")

                algo_choice_multi_csv = st.multiselect(
                    "Choose line-assignment algorithms",
                    ALGO_CHOICES,
                    key="algo_choice_multi_csv",
                    default=get_default_val("algo_choice_multi_csv", DEFAULT_ALGO_CHOICE),
                    help="This selection determines which line assignment algorithms should be used.",
                )
                st.markdown("### Analysis settings")
                fix_cols_to_add_multi_csv = st.multiselect(
                    "Select what fixation measures to calculate.",
                    options=ALL_FIX_MEASURES,
                    key="fix_cols_to_add_multi_csv",
                    default=get_default_val("fix_cols_to_add_multi_csv", DEFAULT_FIX_MEASURES),
                    help="This selection determines what fixation-level measures will be calculated.",
                )
                measures_to_calculate_multi_csv = st.multiselect(
                    "Select what word measures to calculate.",
                    options=ALL_MEASURES_OWN,
                    key="measures_to_calculate_multi_csv",
                    default=get_default_val("measures_to_calculate_multi_csv", DEFAULT_WORD_MEASURES),
                    help="This selection determines which word-level measures should be calculated.",
                )
                include_word_coords_in_output_multi_csv = st.checkbox(
                    "Should word bounding box coordinates be included in the measures table?",
                    value=get_default_val("include_word_coords_in_output_multi_csv", False),
                    key="include_word_coords_in_output_multi_csv",
                    help="Determines if the bounding box coordinates should be included in the word measures dataframe.",
                )

                sent_measures_to_calculate_multi_csv = st.multiselect(
                    "Select what sentence measures to calculate.",
                    options=ALL_SENT_MEASURES,
                    key="sent_measures_to_calculate_multi_csv",
                    default=get_default_val("sent_measures_to_calculate_multi_csv", DEFAULT_SENT_MEASURES),
                    help="This selection determines which sentence-level measures should be calculated.",
                )
                save_files_for_each_trial_individually_multi_csv = st.checkbox(
                    "Save fixations, saccades, stimulus and metadata for each trial to a seperate file.",
                    value=get_default_val("save_files_for_each_trial_individually_multi_csv", False),
                    key="save_files_for_each_trial_individually_multi_csv",
                    help="This setting determines if the results for each trial will be saved as an individual file.",
                )
            st.markdown("### Click to run")
            process_trial_btn_multi_csv = st.form_submit_button(
                "🚀 Process CSV files",
                help="Using the configuration set above this button will start the processing of all trials in all CSV files.",
            )
        if process_trial_btn_multi_csv and not (
            "multi_csv_filelist" in st.session_state and len(st.session_state["multi_csv_filelist"]) > 0
        ):
            st.warning("Please upload CSV files to run processing.")
        if (
            process_trial_btn_multi_csv
            and "multi_csv_filelist" in st.session_state
            and len(st.session_state["multi_csv_filelist"]) > 0
        ):
            if "dffix_multi_csv" in st.session_state:
                del st.session_state["dffix_multi_csv"]

            if "results" in st.session_state:
                st.session_state["results"] = {}

            # Process CSV files
            (
                list_of_trial_lists_csv,
                results_keys_csv,
                zipfiles_with_results_csv,
                all_fix_dfs_concat_csv,
                all_sacc_dfs_concat_csv,
                all_chars_dfs_concat_csv,
                all_words_dfs_concat_csv,
                all_sentence_dfs_concat_csv,
                all_trials_by_subj_csv,
                trials_summary_csv,
                subjects_summary_csv,
                trials_quick_meta_df_csv,
            ) = process_all_csv_files(
                csv_files=multi_csv_filelist,
                image_files=multi_image_filelist,
                algo_choice_multi_csv=algo_choice_multi_csv,
                close_gap_between_words=st.session_state["close_gap_between_words_multi_csv"],
                close_gap_between_lines=st.session_state["close_gap_between_lines_multi_csv"],
                choice_handle_short_and_close_fix=choice_handle_short_and_close_fix_multi_csv,
                discard_fixations_without_sfix=False,  # Not applicable for CSV
                discard_far_out_of_text_fix=discard_far_out_of_text_fix_multi_csv,
                x_thres_in_chars=outlier_crit_x_threshold_multi_csv,
                y_thresh_in_heights=outlier_crit_y_threshold_multi_csv,
                short_fix_threshold=short_fix_threshold_multi_csv,
                merge_distance_threshold=merge_distance_threshold_multi_csv,
                discard_long_fix=discard_long_fix_multi_csv,
                discard_long_fix_threshold=discard_long_fix_threshold_multi_csv,
                discard_blinks=discard_blinks_fix_multi_csv,
                measures_to_calculate_multi_csv=measures_to_calculate_multi_csv,
                include_coords_multi_csv=include_word_coords_in_output_multi_csv,
                sent_measures_to_calculate_multi_csv=sent_measures_to_calculate_multi_csv,
                fix_cols_to_add_multi_csv=fix_cols_to_add_multi_csv,
                save_files_for_each_trial_individually=save_files_for_each_trial_individually_multi_csv,
            )
            if trials_summary_csv is not None:
                st.session_state["trials_summary_df_multi_csv"] = trials_summary_csv
            if subjects_summary_csv is not None:
                st.session_state["subjects_summary_df_multi_csv"] = subjects_summary_csv

            st.session_state["trial_choices_multi_csv"] = results_keys_csv
            st.session_state["zipfiles_with_results_csv"] = zipfiles_with_results_csv
            st.session_state["all_fix_dfs_concat_multi_csv"] = all_fix_dfs_concat_csv
            st.session_state["all_sacc_dfs_concat_multi_csv"] = all_sacc_dfs_concat_csv
            st.session_state["all_chars_dfs_concat_multi_csv"] = all_chars_dfs_concat_csv
            st.session_state["all_words_dfs_concat_multi_csv"] = all_words_dfs_concat_csv
            st.session_state["all_sentence_dfs_concat_multi_csv"] = all_sentence_dfs_concat_csv
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
            st.session_state["all_trials_by_subj_csv"] = {
                k_outer: {
                    k: {prop: val for prop, val in v.items() if prop not in offload_list} for k, v in v_outer.items()
                }
                for k_outer, v_outer in all_trials_by_subj_csv.items()
            }
            subs_str_csv = ("-".join([s for s in all_trials_by_subj_csv.keys()]))
            if len(subs_str_csv) > 100:
                subs_str_csv = subs_str_csv[:100] + "_etc"
            st.session_state["trials_df_csv"] = trials_quick_meta_df_csv.drop_duplicates().dropna(subset="text", axis=0)
            st.session_state["trials_df_csv"].to_csv(RESULTS_FOLDER / f"{subs_str_csv}_comb_items_lines_text.csv")
            if "text_with_newlines" in st.session_state["trials_df_csv"].columns:
                st.session_state["trials_df_csv"] = (
                    st.session_state["trials_df_csv"].drop(columns=["text_with_newlines"]).copy()
                )
            st.session_state["all_own_word_measures_concat_csv"] = all_words_dfs_concat_csv

    if in_st_nn("all_fix_dfs_concat_multi_csv"):
        if "all_trials_by_subj_csv" in st.session_state:
            multi_file_tab.markdown("### All meta data by subject and trial (CSV)")
            multi_file_tab.json(ut.make_json_serializable(st.session_state["all_trials_by_subj_csv"]), expanded=False)
        multi_file_tab.markdown("### Item level stimulus overview (CSV)")
        with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
            item_colnames_markdown = read_item_col_names()
            st.markdown(item_colnames_markdown)
        multi_file_tab.dataframe(st.session_state["trials_df_csv"], width='stretch', height=200)
        if in_st_nn("subjects_summary_df_multi_csv"):
            multi_file_tab.markdown("### Subject level summary statistics (CSV)")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                subject_measure_colnames_markdown = read_subject_meas_col_names()
                st.markdown(subject_measure_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["subjects_summary_df_multi_csv"], width='stretch', height=200
            )
        if in_st_nn("trials_summary_df_multi_csv"):
            multi_file_tab.markdown("### Trial level summary statistics (CSV)")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                trials_colnames_markdown = read_trial_col_names()
                st.markdown(trials_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["trials_summary_df_multi_csv"], width='stretch', height=200
            )

        multi_file_tab.markdown("### Combined fixations dataframe and fixation level features (CSV)")
        with multi_file_tab.popover("Column name definitions"):
            fix_colnames_markdown = get_fix_colnames_markdown()
            st.markdown(fix_colnames_markdown)
        multi_file_tab.dataframe(st.session_state["all_fix_dfs_concat_multi_csv"], width='stretch', height=200)

        # Similar processing for high fixation counts, etc.
        # ... (omitted for brevity, similar to ASC processing)

        if not st.session_state["all_own_word_measures_concat_csv"].empty:
            multi_file_tab.markdown("### Combined words dataframe and word level features (CSV)")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                word_measure_colnames_markdown = read_word_meas_col_names()
                st.markdown(word_measure_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["all_own_word_measures_concat_csv"], width='stretch', height=200
            )
        if not st.session_state["all_sentence_dfs_concat_multi_csv"].empty:
            multi_file_tab.markdown("### Combined sentence dataframe and sentence level features (CSV)")
            with multi_file_tab.popover("Column names definitions", help="Show column names and their definitions."):
                sentence_measure_colnames_markdown = read_sent_meas_col_names()
                st.markdown(sentence_measure_colnames_markdown)
            multi_file_tab.dataframe(
                st.session_state["all_sentence_dfs_concat_multi_csv"], width='stretch', height=200
            )
    if "zipfiles_with_results_csv" in st.session_state:
        multi_res_col1_csv, multi_res_col2_csv = multi_file_tab.columns(2)

        chosen_zip_csv = multi_res_col1_csv.selectbox("Choose CSV results to download", st.session_state["zipfiles_with_results_csv"])
        zipnamestem_csv = pl.Path(chosen_zip_csv).stem
        with open(chosen_zip_csv, "rb") as f:
            multi_res_col2_csv.download_button(f"⏬ Download {zipnamestem_csv}.zip", f, file_name=f"results_{zipnamestem_csv}.zip")

    render_multi_trial_section(
        multi_file_tab,
        suffix="multi_asc",
        algo_choices_key="algo_choice_multi_asc",
        section_label=" (ASC)",
    )
    render_multi_trial_section(
        multi_file_tab,
        suffix="multi_csv",
        algo_choices_key="algo_choice_multi_csv",
        section_label=" (CSV)",
    )
    if "rerun_done" not in st.session_state:
        st.session_state["rerun_done"] = True
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()


def check_for_large_number_of_fixations_on_word(dffix, single_file_tab_asc_tab, algo_choices):
    high_fix_count_dfs = []
    if "dffix_single_asc" in st.session_state:
        for algo_choice in algo_choices:
            fixation_counts = (
                dffix.loc[:, [f"on_word_number_{algo_choice}", f"on_word_{algo_choice}"]]
                .value_counts()
                .sort_values(ascending=False)
            )
            high_fixation_words = fixation_counts[fixation_counts >= 7].index
            high_fix_count_dfs.append(
                fixation_counts[high_fixation_words].reset_index(name=f"assigned_fixations_{algo_choice}")
            )
            for word, count in zip(high_fixation_words, fixation_counts[high_fixation_words]):
                single_file_tab_asc_tab.warning(
                    f'For algorithm {algo_choice} the word "{word[1]}" (number {int(word[0])}) has had {count} fixations assigned to it. If this seems incorrect please adjust the correction algorithm.'
                )
    return pd.concat(high_fix_count_dfs, axis=0).reset_index(drop=True)


@st.cache_data
def read_sent_meas_col_names():
    with open("sentence_measures.md", "r") as f:
        sentence_measure_colnames_markdown = "\n".join(f.readlines())
    return sentence_measure_colnames_markdown


@st.cache_data
def read_subject_meas_col_names():
    with open("subject_measures.md", "r") as f:
        subject_measures_colnames_markdown = "\n".join(f.readlines())
    return subject_measures_colnames_markdown


@st.cache_data
def read_word_meas_col_names():
    with open("word_measures.md", "r") as f:
        word_measure_colnames_markdown = "\n".join(f.readlines())
    return word_measure_colnames_markdown


@st.cache_data
def read_chars_col_names():
    with open("chars_df_columns.md", "r") as f:
        chars_colnames_markdown = "\n".join(f.readlines())
    return chars_colnames_markdown


@st.cache_data
def read_item_col_names():
    with open("item_df_columns.md", "r") as f:
        item_colnames_markdown = "\n".join(f.readlines())
    return item_colnames_markdown


@st.cache_data
def read_trial_col_names():
    with open("trials_df_columns.md", "r") as f:
        trial_colnames_markdown = "\n".join(f.readlines())
    return trial_colnames_markdown


@st.cache_data
def get_fix_colnames_markdown():
    with open("fixations_df_columns.md", "r") as f:
        fix_colnames_markdown = "\n".join(f.readlines())
    return fix_colnames_markdown


@st.cache_data
def get_sac_colnames_markdown():
    with open("saccades_df_columns.md", "r") as f:
        sac_colnames_markdown = "\n".join(f.readlines())
    return sac_colnames_markdown


def show_file_parsing_settings(suffix: str):
    st.markdown("### File parsing settings")
    st.selectbox(
        label="Keyword in .asc file indicating start of a trial.",
        options=START_KEYWORD_OPTIONS,
        index=0,
        key=f"trial_start_keyword{suffix}",
        help="This list contains the most common keywords used in .asc files to indicate the start of a trial. If you are unsure which one to use, open an .asc file and check when these keywords occur in relation to your text stimulus presentation. It is recommendable to use a keyword that occurs directly before the text stimulus appears. You can add a custom keyword by selecting 'custom' and entering it in the field below.",
    )
    st.text_input(
        "Custom trial start keyword",
        key=f"trial_custom_start_keyword{suffix}",
        help="If the 'custom' option is selected above, this keyword will be used to find the start timestamp of the trials in the .asc file. If keyword is not found it will default to 'START'",
    )
    st.selectbox(
        label="Keyword in .asc file indicating end of a trial.",
        options=END_KEYWORD_OPTIONS,
        index=0,
        key=f"trial_end_keyword{suffix}",
        help="This list contains the most common keywords used in .asc files to indicate the end of a trial. If you are unsure which one to use, open an .asc file and check when these keywords occur in relation to your text stimulus presentation. It is recommendable to use a keyword that occurs directly after the text stimulus disappears. You can add a custom keyword by selecting 'custom' and entering it in the field below.",
    )
    st.text_input(
        "Custom trial end keyword",
        key=f"trial_custom_end_keyword{suffix}",
        help="If the 'custom' option is selected above, this keyword will be used to find the end timestamp of the trials in the .asc file. If keyword is not found it will default to 'TRIAL_RESULT'",
    )
    st.checkbox(
        label="Should spaces between words be included in word bounding box?",
        value=get_default_val(f"close_gap_between_words{suffix}", True),
        key=f"close_gap_between_words{suffix}",
        help="If this is selected, each word bounding box will include half the spaces between adjacent words. If not, the word bounding boxes will simply be the combined bounding boxes of the letters making up the word.",  # TODO check if this affects analysis
    )
    st.checkbox(
        label="Should spaces between lines be included in word and character bounding boxes?",
        value=get_default_val(f"close_gap_between_lines{suffix}", True),
        key=f"close_gap_between_lines{suffix}",
        help="If this is selected, each word and char bounding box will include half the spaces between adjacent lines.",  # TODO check if this affects analysis
    )
    st.markdown("### Trial filtering settings")

    st.checkbox(
        label="Should Practice and question trials be excluded if possible?",
        value=get_default_val(f"paragraph_trials_only{suffix}", True),
        key=f"paragraph_trials_only{suffix}",
        help="This option will restrict the trials that are used for processing to the 'paragraph' trials and therefore exclude practice and question trials. This relies on either the trial id following the convention of question trials starting with the letter 'F' and practice trials starting with the letter 'P' or by trials being marked as practice or paragraph in the lines of the .asc file marked with 'TRIAL_VAR'.",
    )


def get_summaries_from_trials(all_trials_by_subj):
    keep_list = ["condition", "item", "text"]
    correction_summary_list_all_multi = []
    cleaning_summary_list_all_multi = []
    trials_quick_meta_list = []
    for subj, v_subj in all_trials_by_subj.items():
        if not isinstance(v_subj, dict):
            continue
        for trial_id, v_trials in v_subj.items():
            # Skip non-trial entries (e.g., summary dicts) - check if it's a dict with trial data
            if not isinstance(v_trials, dict) or "Fixation Cleaning Stats" not in v_trials:
                continue
            record = {}
            for k, v in v_trials.items():
                if k in keep_list:
                    record[k] = v
                if k == "line_list":
                    record["text_with_newlines"] = "\n".join(v)
                if k == "Fixation Cleaning Stats":
                    clean_rec = {"subject": subj, "trial_id": trial_id}
                    clean_rec.update(v)
                    cleaning_summary_list_all_multi.append(clean_rec)
                if k == "average_y_corrections":
                    if isinstance(v, pd.DataFrame):
                        v_dict = v.to_dict("records")
                    else:
                        v_dict = v
                    correction_info_dict = {
                        "subject": subj,
                        "trial_id": trial_id,
                    }
                    for v_sub in v_dict:
                        correction_info_dict.update(
                            {f"average_y_correction_{v_sub['Algorithm']}": v_sub["average_y_correction"]}
                        )
                    correction_summary_list_all_multi.append(correction_info_dict)
            trials_quick_meta_list.append(record)
    return (
        pd.DataFrame(correction_summary_list_all_multi),
        pd.DataFrame(cleaning_summary_list_all_multi),
        pd.DataFrame(trials_quick_meta_list),
    )


def process_single_dffix_and_add_to_state(ending_str: str):
    cp2st(f"algo_choice{ending_str}")
    if "saccade_df" in st.session_state:
        del st.session_state["saccade_df"]
    if f"dffix{ending_str}" in st.session_state:
        del st.session_state[f"dffix{ending_str}"]
    if f"own_word_measures{ending_str}" in st.session_state:
        del st.session_state[f"own_word_measures{ending_str}"]
    dffix = st.session_state[f"dffix_cleaned{ending_str}"].copy()
    chars_df = ensure_dataframe(st.session_state[f"trial{ending_str}"]["chars_df"])
    dffix = reorder_columns(dffix)
    st.session_state[f"trial{ending_str}"]["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())
    st.session_state[f"trial{ending_str}"]["chars_df"] = chars_df
    dffix = correct_df(
        dffix,
        st.session_state[f"algo_choice{ending_str}"],
        st.session_state[f"trial{ending_str}"],
        for_multi=False,
        is_outside_of_streamlit=False,
        classic_algos_cfg=CLASSIC_ALGOS_CFGS,
        models_dict=st.session_state["models_dict"],
        fix_cols_to_add=st.session_state[f"fix_cols_to_add{ending_str}"],
    )
    st.session_state[f"dffix{ending_str}"] = dffix


def render_multi_trial_section(
    container,
    suffix: str,
    algo_choices_key: str,
    section_label: str = "",
    results_state_key: str = "results",
):
    trial_choices_key = f"trial_choices_{suffix}"
    trial_choices = st.session_state.get(trial_choices_key)
    if not trial_choices:
        return

    label_suffix = section_label if section_label else ""
    trial_select_label = (
        f"Which trial should be plotted{label_suffix}?" if label_suffix else "Which trial should be plotted?"
    )

    with container.form(key=f"multi_file_tab_trial_select_form_{suffix}"):
        multi_plotting_options_col1, multi_plotting_options_col2 = st.columns(2)

        trial_choice = multi_plotting_options_col1.selectbox(
            trial_select_label,
            trial_choices,
            key=f"trial_id_{suffix}",
            placeholder="Select trial to display and plot",
            help="Choose one of the available trials from the list displayed.",
        )

        plotting_checkbox_state_key = f"plotting_checkboxes_{suffix}"
        default_plots = st.session_state.get(
            plotting_checkbox_state_key,
            ["Uncorrected Fixations", "Corrected Fixations", "Characters", "Word boxes"],
        )
        multi_plotting_options_col2.multiselect(
            "Select what gets plotted",
            STIM_FIX_PLOT_OPTIONS,
            default=default_plots,
            key=plotting_checkbox_state_key,
            help=(
                "This selection determines what information is plotted. The Corrected Fixations are the fixations "
                "after being snapped to their assigned line of text. The Word and Character boxes are the bounding "
                "boxes for the stimulus."
            ),
        )
        process_trial_btn = st.form_submit_button("Plot and analyse trial")

    if process_trial_btn and trial_choice:
        results_map = st.session_state.get(results_state_key, {})
        if trial_choice in results_map:
            result_record = results_map[trial_choice]
            st.session_state[f"dffix_{suffix}"] = result_record["dffix"]
            st.session_state[f"trial_{suffix}"] = result_record["trial"]
            if "words_df" in result_record:
                st.session_state[f"own_word_measures_{suffix}"] = result_record["words_df"]
            if "sent_measures_multi" in result_record:
                st.session_state[f"sentence_measures_{suffix}"] = result_record["sent_measures_multi"]
            st.session_state[f"selected_trial_key_{suffix}"] = trial_choice
        else:
            container.warning("Selected trial is no longer available. Please rerun the processing step.")

    dffix_state_key = f"dffix_{suffix}"
    trial_state_key = f"trial_{suffix}"
    if dffix_state_key not in st.session_state or trial_state_key not in st.session_state:
        return

    dffix = st.session_state[dffix_state_key]
    trial = st.session_state[trial_state_key]

    saccade_df_dict = trial.get("saccade_df", {})
    saccade_df = pd.DataFrame(saccade_df_dict) if len(saccade_df_dict) > 0 else pd.DataFrame()

    info_suffix = label_suffix
    trial_info_expander = container.expander(f"Show Trial Information{info_suffix}", False)

    show_cleaning_results(
        container,
        trial=trial,
        expander_text=f"Show Cleaned Fixations Dataframe{info_suffix}",
        dffix_cleaned=dffix,
        dffix_no_clean_name="dffix_no_clean",
        expander_open=False,
        key_str=suffix,
    )

    dffix_expander = container.expander(f"Show Fixations Dataframe{info_suffix}", False)
    with dffix_expander.popover("Column name definitions"):
        st.markdown(get_fix_colnames_markdown())
    dffix_expander.dataframe(dffix, height=200)

    saccade_df_expander = container.expander(f"Show Saccade Dataframe{info_suffix}", False)
    if not saccade_df.empty:
        saccade_df_expander.dataframe(saccade_df, height=200)
    else:
        saccade_df_expander.info("No saccade data available for this trial.")

    df_stim_expander = container.expander(f"Show Stimulus Dataframe{info_suffix}", False)
    if "chars_list" in trial:
        df_stim_expander.dataframe(pd.DataFrame(trial["chars_list"]), height=200)
    else:
        df_stim_expander.info("No stimulus information available for this trial.")

    filtered_trial = filter_trial_for_export(trial)
    trial_info_expander.json(ut.make_json_serializable(filtered_trial))

    plot_expander = container.expander(f"Show corrected fixation plots{info_suffix}", True)
    show_plots_key = f"show_fix_sacc_plots_{suffix}"
    show_plots_default = st.session_state.get(show_plots_key, True)
    show_plots = plot_expander.checkbox("Show plots", show_plots_default, show_plots_key)
    if show_plots:
        font_selection_key = f"selected_plotting_font_{suffix}_single_plot"
        selected_font_index = FONT_INDEX if 0 <= FONT_INDEX < len(AVAILABLE_FONTS) else 0
        selected_font = plot_expander.selectbox(
            "Font to use for plotting",
            AVAILABLE_FONTS,
            index=selected_font_index,
            key=font_selection_key,
            help=(
                "This selects which font is used to display the words or characters making up the stimulus. "
                "This selection only affects the plot and has no effect on the analysis as everything else is based "
                "on the bounding boxes of the words and characters."
            ),
        )

        plotting_checkbox_state_key = f"plotting_checkboxes_{suffix}"
        plotting_checkboxes = st.session_state.get(
            plotting_checkbox_state_key,
            ["Uncorrected Fixations", "Corrected Fixations", "Characters", "Word boxes"],
        )
        algo_choices = st.session_state.get(algo_choices_key, [])

        if algo_choices:
            plot_expander.plotly_chart(
                plotly_plot_with_image(
                    dffix,
                    trial,
                    algo_choice=algo_choices,
                    to_plot_list=plotting_checkboxes,
                    font=selected_font,
                ),
                width='stretch',
            )
            plot_expander.plotly_chart(plot_y_corr(dffix, algo_choices), width='stretch')
        else:
            plot_expander.info("Select at least one line-assignment algorithm to see the plots.")

        if algo_choices and not saccade_df.empty and not dffix.empty:
            select_and_show_fix_sacc_feature_plots(
                dffix,
                saccade_df,
                plot_expander,
                plot_choice_fix_feature_name=f"plot_choice_fix_features_{suffix}",
                plot_choice_sacc_feature_name=f"plot_choice_sacc_features_{suffix}",
                feature_plot_selection=f"feature_plot_selection_{suffix}",
                plot_choice_fix_sac_feature_x_axis_name=f"feature_plot_x_selection_{suffix}",
            )

    if "chars_list" not in trial:
        container.warning("🚨 Stimulus information needed for analysis 🚨")
        return

    analysis_expander = container.expander(f"Show Analysis results{info_suffix}", True)
    algo_choices = st.session_state.get(algo_choices_key, [])
    if not algo_choices:
        analysis_expander.info("Select at least one line-assignment algorithm above to unlock the analysis section.")
        return

    analysis_algo_key = f"algo_choice_{suffix}_eyekit"
    selected_algo_index = 0 if algo_choices else None
    analysis_expander.selectbox(
        "Algorithm",
        algo_choices,
        index=selected_algo_index,
        key=analysis_algo_key,
        help=(
            "If more than one line assignment algorithm was selected above, this selection determines which of the "
            "resulting line assignments should be used for the analysis."
        ),
    )

    selected_algo = st.session_state.get(analysis_algo_key)
    if not selected_algo:
        analysis_expander.info("Select an algorithm to continue with the analysis.")
        return

    own_analysis_tab, eyekit_tab = analysis_expander.tabs(["Analysis without eyekit", "Analysis using eyekit"])

    with eyekit_tab:
        ending_str = f"_{suffix}"
        eyekit_input(ending_str=ending_str)

        eyekit_kwargs_missing = [
            key
            for key in [
                f"x_txt_start_for_eyekit{ending_str}",
                f"y_txt_start_for_eyekit{ending_str}",
                f"font_face_for_eyekit{ending_str}",
                f"font_size_for_eyekit{ending_str}",
                f"line_height_for_eyekit{ending_str}",
            ]
            if key not in st.session_state
        ]
        if eyekit_kwargs_missing:
            st.warning("Please provide the missing eyekit parameters above to run the eyekit-based analysis.")
        else:
            fixations_tuples, textblock_input_dict, screen_size = ekm.get_fix_seq_and_text_block(
                dffix,
                trial,
                x_txt_start=st.session_state[f"x_txt_start_for_eyekit{ending_str}"],
                y_txt_start=st.session_state[f"y_txt_start_for_eyekit{ending_str}"],
                font_face=st.session_state[f"font_face_for_eyekit{ending_str}"],
                font_size=st.session_state[f"font_size_for_eyekit{ending_str}"],
                line_height=st.session_state[f"line_height_for_eyekit{ending_str}"],
                use_corrected_fixations=True,
                correction_algo=selected_algo,
            )
            eyekitplot_img = ekm.eyekit_plot(fixations_tuples, textblock_input_dict, screen_size)
            st.image(eyekitplot_img, "Fixations and stimulus as used for anaylsis")

            with open(f'results/fixation_sequence_eyekit_{trial["trial_id"]}.json', "r") as f:
                fixation_sequence_json = json.load(f)
            fixation_sequence_json_str = json.dumps(fixation_sequence_json, cls=ut.NpEncoder)

            st.download_button(
                "⏬ Download fixations in eyekits format",
                fixation_sequence_json_str,
                f'fixation_sequence_eyekit_{trial["trial_id"]}.json',
                "json",
                key=f"download_eyekit_fix_json_{suffix}",
                help="This downloads the extracted fixation information as a .json file in the eyekit format with the filename containing the subject name and trial id.",
            )

            with open(f'results/textblock_eyekit_{trial["trial_id"]}.json', "r") as f:
                textblock_json = json.load(f)
            textblock_json_str = json.dumps(textblock_json, cls=ut.NpEncoder)

            st.download_button(
                "⏬ Download stimulus in eyekits format",
                textblock_json_str,
                f'textblock_eyekit_{trial["trial_id"]}.json',
                "json",
                key=f"download_eyekit_text_json_{suffix}",
                help="This downloads the extracted stimulus information as a .json file in the eyekit format with the filename containing the subject name and trial id.",
            )

            word_measures_df, character_measures_df = get_eyekit_measures(
                fixations_tuples,
                textblock_input_dict,
                trial=trial,
                get_char_measures=False,
            )

            st.dataframe(word_measures_df, width='stretch', hide_index=True, height=200)
            word_measures_df_csv = convert_df(word_measures_df)

            st.download_button(
                "⏬ Download word measures data",
                word_measures_df_csv,
                f'{trial["trial_id"]}_word_measures_df.csv',
                "text/csv",
                key=f"word_measures_df_download_btn_{suffix}",
                help="This downloads the word-level measures as a .csv file with the filename containing the trial id.",
            )

            measure_words = st.selectbox(
                "Select measure to visualize",
                list(ekm.MEASURES_DICT.keys()),
                key=f"measure_words_{suffix}",
            )
            st.image(ekm.plot_with_measure(fixations_tuples, textblock_input_dict, screen_size, measure_words))

            if character_measures_df is not None:
                st.dataframe(character_measures_df, width='stretch', hide_index=True, height=200)

    with own_analysis_tab:
        st.markdown(
            "This analysis method does not require manual alignment and works when the automated stimulus coordinates are correct."
        )

        own_word_state_key = f"own_word_measures_{suffix}"
        sentence_state_key = f"sentence_measures_{suffix}"

        if own_word_state_key in st.session_state:
            own_word_measures = st.session_state[own_word_state_key]
        else:
            own_word_measures = get_all_measures(
                trial,
                dffix,
                prefix="word",
                use_corrected_fixations=True,
                correction_algo=selected_algo,
                save_to_csv=True,
                measures_to_calculate=ALL_MEASURES_OWN,
            )
            st.session_state[own_word_state_key] = own_word_measures

        if sentence_state_key in st.session_state:
            sent_measures = st.session_state[sentence_state_key]
        else:
            sent_measures = compute_sentence_measures(
                dffix,
                ensure_dataframe(trial["chars_df"]),
                selected_algo,
                DEFAULT_SENT_MEASURES,
                save_to_csv=True,
            )
            st.session_state[sentence_state_key] = sent_measures

        st.markdown("Word measures")
        own_word_measures = reorder_columns(own_word_measures)
        if "question_correct" in own_word_measures.columns:
            own_word_measures = own_word_measures.drop(columns=["question_correct"])
        st.dataframe(own_word_measures, width='stretch', hide_index=True, height=200)
        own_word_measures_csv = convert_df(own_word_measures)
        st.download_button(
            "⏬ Download word measures data",
            own_word_measures_csv,
            f'{trial["trial_id"]}_own_word_measures_df.csv',
            "text/csv",
            key=f"own_word_measures_df_download_btn_{suffix}",
            help="This downloads the word-level measures as a .csv file with the filename containing the trial id.",
        )

        measure_words_own = st.selectbox(
            "Select measure to visualize",
            list(own_word_measures.columns),
            key=f"measure_words_own_{suffix}",
            index=max(len(own_word_measures.columns) - 1, 0),
            help="This selection determines which of the calculated word-level features should be visualized by displaying the value to the corresponding word bounding box.",
        )
        fix_to_plot = ["Corrected Fixations"]
        own_word_measures_fig, _, _ = matplotlib_plot_df(
            dffix,
            trial,
            [selected_algo],
            None,
            box_annotations=own_word_measures[measure_words_own],
            fix_to_plot=fix_to_plot,
        )
        st.pyplot(own_word_measures_fig)
        st.markdown("Sentence measures")
        st.dataframe(sent_measures, width='stretch', hide_index=True, height=200)


def eyekit_input(ending_str: str):
    st.markdown("Analysis powered by [eyekit](https://jwcarr.github.io/eyekit/)")
    st.markdown(
        "Please adjust parameters below to align fixations with stimulus using the sliders. Eyekit analysis is based on this alignment."
    )
    sliders_on = st.radio(
        "Input method for eyekit parameters",
        ["Sliders", "Direct input"],
        index=0,
        key=f"sliders_on{ending_str}",
        help="This selection determines if the fixation to stimulus alignment parameters can be set via sliders or via directly inputting the desired number.",
    )

    def set_state_to_false():
        st.session_state[f"show_eyekit_analysis{ending_str}"] = False

    if f"font_size_for_eyekit_from_trial{ending_str}" not in st.session_state:
        (
            y_diff,
            x_txt_start,
            y_txt_start,
            font_face,
            font_size,
            line_height,
        ) = add_default_font_and_character_props_to_state(st.session_state[f"trial{ending_str}"])
        font_size = set_font_from_chars_list(st.session_state[f"trial{ending_str}"])
        st.session_state[f"y_diff_for_eyekit_from_trial{ending_str}"] = y_diff
        st.session_state[f"x_txt_start_for_eyekit_from_trial{ending_str}"] = x_txt_start
        st.session_state[f"y_txt_start_for_eyekit_from_trial{ending_str}"] = y_txt_start
        st.session_state[f"font_size_for_eyekit_from_trial{ending_str}"] = font_size
        st.session_state[f"line_height_for_eyekit_from_trial{ending_str}"] = line_height
    with st.form(f"form_eyekit_input{ending_str}"):
        a_c1, a_c2, a_c3, a_c4, a_c5 = st.columns(5)

        a_c1.selectbox(
            label="Select Font",
            options=AVAILABLE_FONTS,
            index=FONT_INDEX,
            key=f"font_face_for_eyekit{ending_str}",
        )
        if sliders_on == "Sliders":
            default_val = float(st.session_state[f"font_size_for_eyekit_from_trial{ending_str}"])
            font_size = a_c2.select_slider(
                "Font Size",
                np.arange(min(5, default_val), max(36, default_val + 0.25), 0.25, dtype=float),
                st.session_state[f"font_size_for_eyekit_from_trial{ending_str}"],
                key=f"font_size_for_eyekit{ending_str}",
                help="This sets the font size for aligning the fixations with the stimulus as reconstructed by eyekit.",
            )
            default_val = int(round(st.session_state[f"x_txt_start_for_eyekit_from_trial{ending_str}"]))
            x_txt_start = a_c3.select_slider(
                "x",
                np.arange(min(300, default_val), max(601, default_val + 1), 1, dtype=int),
                default_val,
                key=f"x_txt_start_for_eyekit{ending_str}",
                help="This sets the x coordinate of first character",
            )
            default_val = int(round(st.session_state[f"y_txt_start_for_eyekit_from_trial{ending_str}"]))
            y_txt_start = a_c4.select_slider(
                "y",
                np.arange(min(100, default_val), max(501, default_val + 1), 1, dtype=int),
                default_val,
                key=f"y_txt_start_for_eyekit{ending_str}",
                help="This sets the y coordinate of first character",
            )
            default_val = int(round(st.session_state[f"line_height_for_eyekit_from_trial{ending_str}"]))
            line_height = a_c5.select_slider(
                "Line height",
                np.arange(min(0, default_val), max(151, default_val + 1), 1, dtype=int),
                default_val,
                key=f"line_height_for_eyekit{ending_str}",
                help="This sets the line height for aligning the fixations with the stimulus as reconstructed by eyekit.",
            )
        else:
            default_val = float(st.session_state[f"font_size_for_eyekit_from_trial{ending_str}"])
            font_size = a_c2.number_input(
                "Font Size",
                None,
                None,
                default_val,
                key=f"font_size_for_eyekit{ending_str}",
                help="This sets the font size for aligning the fixations with the stimulus as reconstructed by eyekit.",
            )
            default_val = int(round(st.session_state[f"x_txt_start_for_eyekit_from_trial{ending_str}"]))
            x_txt_start = a_c3.number_input(
                "x",
                None,
                None,
                default_val,
                key=f"x_txt_start_for_eyekit{ending_str}",
                help="This sets the x coordinate of first character",
            )
            default_val = int(round(st.session_state[f"y_txt_start_for_eyekit_from_trial{ending_str}"]))
            y_txt_start = a_c4.number_input(
                "y",
                None,
                None,
                default_val,
                key=f"y_txt_start_for_eyekit{ending_str}",
                help="This sets the y coordinate of first character",
            )
            default_val = int(round(st.session_state[f"line_height_for_eyekit_from_trial{ending_str}"]))
            line_height = a_c5.number_input(
                "Line height",
                None,
                None,
                default_val,
                key=f"line_height_for_eyekit{ending_str}",
                help="This sets the line height for aligning the fixations with the stimulus as reconstructed by eyekit.",
            )
        st.form_submit_button(
            "Apply selected parameters",
            help="Uses selected parameters for Eyekit Analysis.",
            on_click=set_state_to_false,
        )
    return 0


def cp2st(key: str):
    st.session_state[f"_{key}"] = st.session_state[key]


def ensure_state_option(key: str, default, options=None, validator=None):
    def _sanitize(value):
        if options is None:
            return value
        if isinstance(value, list):
            sanitized = [item for item in value if item in options]
            return sanitized if sanitized else default
        return value if value in options else default

    if key in st.session_state:
        candidate = st.session_state[key]
    elif f"_{key}" in st.session_state:
        candidate = st.session_state[f"_{key}"]
    else:
        candidate = default

    candidate = _sanitize(candidate)
    if validator is not None and not validator(candidate):
        candidate = default

    st.session_state[key] = candidate
    return candidate


def get_default_val(k, v):
    if k not in st.session_state:
        return v
    else:
        return st.session_state[k]


def get_def_val_w_underscore(k, v, options):
    is_list = isinstance(v, list)
    if k in st.session_state:
        if is_list:
            is_in_options = all([v1 in options for v1 in st.session_state[k]])
        else:
            is_in_options = st.session_state[k] in options
        if is_in_options:
            return st.session_state[k]
        else:
            return v
    elif f"_{k}" in st.session_state:
        if is_list:
            is_in_options = all([v1 in options for v1 in st.session_state[f"_{k}"]])
        else:
            is_in_options = st.session_state[f"_{k}"] in options
        if is_in_options:
            return st.session_state[f"_{k}"]
        else:
            return v
    else:
        return v


def get_default_index(k, options, v):
    if k in st.session_state and st.session_state[k] in options:
        return options.index(st.session_state[k])
    else:
        return v


def show_cleaning_options(single_file_tab_asc_tab, dffix, key_ending_string):
    form_key = f"cleaning_options_form_{key_ending_string}"
    discard_blinks_fix_single_asc_key = f"discard_blinks_fix_{key_ending_string}"
    discard_far_out_of_text_fix_single_asc_key = f"discard_far_out_of_text_fix_{key_ending_string}"
    outlier_crit_x_threshold_single_asc_key = f"outlier_crit_x_threshold_{key_ending_string}"
    # TODO Finish abstracting all keys
    with single_file_tab_asc_tab.form(key=form_key):
        st.markdown("### Cleaning options")
        st.checkbox(
            "Should fixations that happen just before or after a blink event be discarded?",
            value=get_def_val_w_underscore(f"{discard_blinks_fix_single_asc_key}", True, [True, False]),
            key=discard_blinks_fix_single_asc_key,
            help="This determines if fixations that occur just after or just before a detected blink are discarded and therefore excluded from analysis.",
        )
        st.checkbox(
            "Should fixations that are far outside the text be discarded? (set margins below)",
            value=get_def_val_w_underscore(f"{discard_far_out_of_text_fix_single_asc_key}", True, [True, False]),
            key=discard_far_out_of_text_fix_single_asc_key,
            help="Using the thresholds set below this option determines whether fixations that are further outside the text lines in both horizontal and vertical direction should be discarded.",
        )
        st.number_input(
            "Maximum horizontal distance from first/last character on line (in character widths)",
            min_value=0.0,
            max_value=20.0,
            value=get_def_val_w_underscore(
                f"{outlier_crit_x_threshold_single_asc_key}", 2.0, list(np.arange(0.0, 20.0, 0.25))
            ),
            step=0.25,
            key=outlier_crit_x_threshold_single_asc_key,
            help=r"This option is used to set the maximum horizontal distance a fixation can have from the edges of a line of text before it will be considered to be far outside the text. This distance uses the average character width found in the stimulus text as a unit with the smallest increment being 25 % of this width.",
        )
        outlier_crit_y_threshold_single_asc_key = f"outlier_crit_y_threshold_{key_ending_string}"
        st.number_input(
            "Maximum vertical distance from top/bottom of line (in line heights)",
            min_value=0.0,
            max_value=5.0,
            value=get_def_val_w_underscore(
                f"{outlier_crit_y_threshold_single_asc_key}", 0.5, list(np.arange(0.0, 6.0, 0.05))
            ),
            step=0.05,
            key=outlier_crit_y_threshold_single_asc_key,
            help=r"This option is used to set the maximum vertical distance a fixation can have from the top and bottom edges of a line of text before it will be considered to be far outside the text. This distance uses the unit of average line height and the smallest increment is 5 % of this height.",
        )

        discard_long_fix_single_asc_key = f"discard_long_fix_{key_ending_string}"
        st.checkbox(
            "Should long fixations be discarded? (set threshold below)",
            value=get_def_val_w_underscore(f"{discard_long_fix_single_asc_key}", True, [True, False]),
            key=discard_long_fix_single_asc_key,
            help="If this option is selected, overly long fixations will be discarded. What is considered an overly long fixation is determined by the duration threshold set below.",
        )
        discard_long_fix_threshold_single_asc_key = f"discard_long_fix_threshold_{key_ending_string}"
        st.number_input(
            "Maximum duration allowed for fixations (ms)",
            min_value=20,
            max_value=3000,
            value=get_def_val_w_underscore(
                f"{discard_long_fix_threshold_single_asc_key}", DEFAULT_LONG_FIX_THRESHOLD, list(range(3001))
            ),
            step=5,
            key=discard_long_fix_threshold_single_asc_key,
            help="Fixations longer than this duration will be considered overly long fixations.",
        )

        choice_handle_short_and_close_fix_single_asc_key = f"choice_handle_short_and_close_fix_{key_ending_string}"
        st.radio(
            "How should short fixations be handled?",
            SHORT_FIX_CLEAN_OPTIONS,
            index=get_default_index(f"_{choice_handle_short_and_close_fix_single_asc_key}", SHORT_FIX_CLEAN_OPTIONS, 1),
            key=choice_handle_short_and_close_fix_single_asc_key,
            help="Merge: merges with either previous or next fixation and discards it if it is the last fixation and below the threshold. Merge then discard first tries to merge short fixations and then discards any short fixations that could not be merged. Discard simply discards all short fixations.",
        )
        short_fix_threshold_single_asc_key = f"short_fix_threshold_{key_ending_string}"
        st.number_input(
            "Minimum fixation duration (ms)",
            min_value=1,
            max_value=500,
            value=get_def_val_w_underscore(f"{short_fix_threshold_single_asc_key}", 80, list(range(501))),
            key=short_fix_threshold_single_asc_key,
            help="Fixations shorter than this duration will be considered short fixations.",
        )
        merge_distance_threshold_single_asc_key = f"merge_distance_threshold_{key_ending_string}"
        st.number_input(
            "Maximum distance between fixations when merging (in character widths)",
            min_value=1,
            max_value=20,
            value=get_def_val_w_underscore(
                f"{merge_distance_threshold_single_asc_key}", DEFAULT_MERGE_DISTANCE_THRESHOLD, list(range(25))
            ),
            key=merge_distance_threshold_single_asc_key,
            help="When merging short fixations this is the maximum allowed distance between them.",
        )
        if "chars_list" not in st.session_state[f"trial_{key_ending_string}"]:
            st.warning("Stimulus information not present for trial, cleaning will be limited")
        clean_button_single_asc = st.form_submit_button(label="Apply cleaning")
    if clean_button_single_asc:
        cp2st(discard_blinks_fix_single_asc_key)
        cp2st(discard_far_out_of_text_fix_single_asc_key)
        cp2st(outlier_crit_x_threshold_single_asc_key)
        cp2st(outlier_crit_y_threshold_single_asc_key)
        cp2st(discard_long_fix_single_asc_key)
        cp2st(discard_long_fix_threshold_single_asc_key)
        cp2st(choice_handle_short_and_close_fix_single_asc_key)
        cp2st(short_fix_threshold_single_asc_key)
        cp2st(merge_distance_threshold_single_asc_key)
        if f"dffix_{key_ending_string}" in st.session_state:
            del st.session_state[f"dffix_{key_ending_string}"]
        if f"own_word_measures_{key_ending_string}" in st.session_state:
            del st.session_state[f"own_word_measures_{key_ending_string}"]
        dffix_cleaned, trial = clean_dffix_own(
            st.session_state[f"trial_{key_ending_string}"],
            choice_handle_short_and_close_fix=st.session_state[
                f"choice_handle_short_and_close_fix_{key_ending_string}"
            ],
            discard_far_out_of_text_fix=st.session_state[f"discard_far_out_of_text_fix_{key_ending_string}"],
            x_thres_in_chars=st.session_state[f"outlier_crit_x_threshold_{key_ending_string}"],
            y_thresh_in_heights=st.session_state[f"outlier_crit_y_threshold_{key_ending_string}"],
            short_fix_threshold=st.session_state[f"short_fix_threshold_{key_ending_string}"],
            merge_distance_threshold=st.session_state[f"merge_distance_threshold_{key_ending_string}"],
            discard_long_fix=st.session_state[f"discard_long_fix_{key_ending_string}"],
            discard_long_fix_threshold=st.session_state[f"discard_long_fix_threshold_{key_ending_string}"],
            discard_blinks=st.session_state[discard_blinks_fix_single_asc_key],
            dffix=dffix.copy(),
        )
        if dffix_cleaned.empty:
            st.session_state["logger"].warning("Empty fixation dataframe")
            single_file_tab_asc_tab.warning("Empty fixation dataframe")
        else:
            st.session_state[f"dffix_cleaned_{key_ending_string}"] = reorder_columns(
                dffix_cleaned.dropna(how="all", axis=1).copy()
            )
            st.session_state[f"trial_{key_ending_string}"] = trial


def select_and_show_fix_sacc_feature_plots(
    dffix,
    saccade_df,
    plot_expander_single,
    plot_choice_fix_feature_name,
    plot_choice_sacc_feature_name,
    feature_plot_selection,
    plot_choice_fix_sac_feature_x_axis_name,
):
    with plot_expander_single.form(feature_plot_selection):
        default_val = ["duration"] if "duration" in dffix.columns else [dffix.columns[-1]]
        st.multiselect(
            "Which fixation feature should be visualized?",
            dffix.columns,
            key=plot_choice_fix_feature_name,
            default=get_def_val_w_underscore(f"{plot_choice_fix_feature_name}", default_val, dffix.columns),
            help="From this list of fixation features choose which ones should be visualized below.",
        )
        default_val = ["duration"] if "duration" in saccade_df.columns else [saccade_df.columns[-1]]
        st.multiselect(
            "Which saccade feature should be visualized?",
            saccade_df.columns,
            key=plot_choice_sacc_feature_name,
            default=get_def_val_w_underscore(f"{plot_choice_sacc_feature_name}", default_val, saccade_df.columns),
            help="From this list of saccade features choose which ones should be visualized below.",
        )
        st.radio(
            "X-Axis",
            options=["Index", "Start Time"],
            index=get_default_index(plot_choice_fix_sac_feature_x_axis_name, ["Index", "Start Time"], 0),
            key=plot_choice_fix_sac_feature_x_axis_name,
            help="This selection determines whether to use the index of the fixation/saccade as the x-axis or the timestamp.",
        )
        feature_plot_selection_button_single_asc = st.form_submit_button("📈 Plot selected features!")
    if feature_plot_selection_button_single_asc:
        cp2st(plot_choice_fix_feature_name)
        cp2st(plot_choice_sacc_feature_name)
        cp2st(plot_choice_fix_sac_feature_x_axis_name)
    if plot_choice_fix_feature_name in st.session_state:
        fix_feature_plot_col_single_asc, sacc_feature_plot_col_single_asc = plot_expander_single.columns(2)
        fix_feature_plot_col_single_asc.plotly_chart(
            plot_fix_measure(
                dffix,
                st.session_state[plot_choice_fix_feature_name],
                x_axis_selection=st.session_state[plot_choice_fix_sac_feature_x_axis_name],
                label_start="Fixation",
            ),
            width='stretch',
        )
        sacc_feature_plot_col_single_asc.plotly_chart(
            plot_fix_measure(
                saccade_df,
                st.session_state[plot_choice_sacc_feature_name],
                x_axis_selection=st.session_state[plot_choice_fix_sac_feature_x_axis_name],
                label_start="Saccade",
            ),
            width='stretch',
        )


def show_cleaning_results(
    single_file_tab_asc_tab, trial, expander_text, dffix_cleaned, dffix_no_clean_name, expander_open, key_str
):
    with single_file_tab_asc_tab.expander(expander_text, expander_open):
        st.markdown("### Cleaning results")
        show_plot = st.checkbox(
            "Show Plot",
            True,
            f"show_plot_check_{key_str}",
            help="If unticked, the plots in this section will be hidden. This can speed up using the interface if the plots are not required.",
        )
        if dffix_no_clean_name in trial:
            if show_plot:
                dffix_no_clean_fig, _, _ = matplotlib_plot_df(
                    dffix_cleaned,
                    trial,
                    None,
                    trial[dffix_no_clean_name],
                    box_annotations=None,
                    fix_to_plot=["Uncorrected Fixations"],
                    stim_info_to_plot=["Characters", "Word boxes"],
                )
                st.markdown("#### Fixations before cleaning")
                st.pyplot(dffix_no_clean_fig)
                dffix_clean_fig, _, _ = matplotlib_plot_df(
                    dffix_cleaned,
                    trial,
                    None,
                    None,
                    box_annotations=None,
                    fix_to_plot=["Uncorrected Fixations"],
                    stim_info_to_plot=["Characters", "Word boxes"],
                    use_duration_arrow_sizes=False,
                )
                st.markdown("#### Fixations after cleaning")
                st.pyplot(dffix_clean_fig)
            st.markdown("#### Fixations comparison before and after cleaning")
            if "Fixation Cleaning Stats" in trial:
                st.json(ut.make_json_serializable(trial["Fixation Cleaning Stats"]))
        st.markdown("#### Cleaned fixations dataframe")

        st.dataframe(dffix_cleaned, height=200)


if __name__ == "__main__":
    main()
