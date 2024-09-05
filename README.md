# GazeGenie

A versatile tool for parsing, cleaning, aligning and analysing fixations from eye-tracking reading experiments

## Use via huggingface spaces

In Browser navigate to : https://huggingface.co/spaces/bugroup/GazeGenie

## Run via Docker

mkdir results

docker run --name gazegenie_app -p 8501:8501 -v $pwd/results:/app/results dockinthehubbing/gaze_genie:latest

In Browser navigate to : http://localhost:8501 

To restart container later:

docker start -a gazegenie_app

## Local installation

#### Install conda to get python

https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe

#### Package installation in Terminal

mamba create -n eye python=3.11 -y

mamba activate eye

mamba install conda-forge::cairo

pip install -r requirements.txt

#### Run program from Terminal

conda activate eye 

streamlit run app.py

In Browser navigate to : http://localhost:8501 