# OCR Marimo Notebooks

This repository includes a few [Marimo](https://marimo.io) notebooks that contain Python scripts that aim to showcase how to quickly run and test some of the OCR tools locally that utilize lightweight CPU-based models.

# How to Run Marimo Notebooks

You need to make sure that uv, the Python package and project manager, is installed on your system. Then, to run the notebook for a certain OCR tool, you need to navigate to the associated directory in the terminal and then follow these steps:

Run the following command, which will create a virtual environment within the same directory and will install the required packages listed in `pyproject.toml`.

```sh
uv sync
```

Then, you need to activate the virtual environment:

Windows:

```sh
./.venv/Scripts/activate
```

Linux:

```sh
source venv/bin/activate
```

Now, it's time to run the notebook with one of the following modes:

Edit mode, where you can modify and tinker with the code:

```sh
marimo edit {notebook_name}
```

Application mode, where you can only view and interact with the notebook.

```sh
marimo run {notebook_name}
```