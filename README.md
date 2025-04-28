# FunSearch Implementation

This repository implements the following publication:

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)

## <span id="colab">Run FunSearch Demo on Colab</span>

Upload the jupyter notebook in `TSP_notebook` to google colab to run.
Please note that do not run jupyter notebook locally, as the jupyter notebook backend does not support multiprocess running.

## Project Structure

There are some independent directories in this project:

- `TSP_notebook` contains an example jupyter notebook for the TSP task.
- `TSP_algorithm` contains different initial algorithm for solving TSP
- `implementation` contains an implementation of the evolutionary algorithm, code manipulation routines, and a single-threaded implementation of the FunSearch pipeline. 
- `llm-server` contains the implementations of an LLM server that gets the prompt by monitoring requests from FunSearch and response to the inference results to the FunSearch algorithm. 

## Files in `funsearch/implementation`

There are some files in `funsearch/implementation`. They are as follows:

- `code_manipulatoin.py` provides functions to modify the code in the specification.
- `config.py` includes configs of funsearch.
- `evaluator.py` trims the sample results from LLM, and evaluates the sampled functions.
- `evaluator_accelerate.py` accelerates the evaluation using the 'numba' library.
- `funsearch.py` implements funsearch pipeline. 
- `profile.py` records the score of the sampled functions.
- `programs_database.py` evolves the sampled functions.
- `sampler.py` sends prompts to LLM and gets results.


## Issue

If you encounter any difficulty using the code, please do not hesitate to submit an issue!
