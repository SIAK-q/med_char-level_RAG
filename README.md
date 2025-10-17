# med_char-level_RAG

An exploration of character-level Retrieval-Augmented Generation (RAG) for the medical domain.

## Core Components

* **RAG Pipeline**: The core implementation for retrieval and generation on medical texts, located in the `RAG/` directory.
* **Custom Tokenizers**: Includes experimental character-level tokenizers (e.g., stroke-based, pinyin-based) in the `customs_tokenizers/` directory.
* **Main Entrypoint**: The primary workflow and experiments are detailed in `main.ipynb`.

## Usage

1.  Set up the required environment and dependencies.
2.  Prepare the necessary datasets (note: datasets and models are ignored by Git via `.gitignore`).
3.  Run the experiments and pipeline through `main.ipynb`.
