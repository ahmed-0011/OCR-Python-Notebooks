# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", app_title="EasyOCR Notebook")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # EasyOCR Notebook
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Upload Image File(s)
    ---
    """
    )
    return


@app.cell
def _():
    import json
    import marimo as mo
    import pyperclip
    import easyocr
    import PIL
    from PIL import Image
    from PIL import ImageDraw
    from os import makedirs


    file_browser = mo.ui.file_browser(
        label="Upload an image with one of the following supported formats: .jpg, jpeg, .png, and webp!",
        filetypes=[".jpg", ".jpeg", ".png", ".webp"],
        multiple=True,
    )
    mo.vstack([file_browser])
    return ImageDraw, PIL, easyocr, file_browser, makedirs, mo, pyperclip


@app.cell
def _(mo):
    def get_error_as_html(message) -> mo.Html:
        return mo.Html(message).style(
            {
                "font-size": "1.35rem",
                "color": "#ff1111",
                "text-decoration": "underline",
            }
        )
    return (get_error_as_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Image File(s) to Processs""")
    return


@app.cell
def _(file_browser, get_error_as_html, mo):
    # This will stop the execution of the notebook starting from this cell when no file is selected
    if not file_browser.path():
        mo.stop(
            True,
            get_error_as_html(
                "<strong>NOTE: You need to select an image file for this cell to execute!</strong>"
            ),
        )

    num_of_images = len(file_browser.value)
    file_paths = [file_browser.path(i) for i in range(num_of_images)]
    dict_path_obj_to_file_data = {}
    for file_path in file_paths:
        dict_path_obj_to_file_data[file_path] = file_path.name.rsplit(".", 1)

    mo.ui.array(
        [mo.image(src=file_path).batch() for file_path in file_paths],
        label="Input Images",
    )
    return dict_path_obj_to_file_data, file_paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Select The Langauge of The Text in The Image""")
    return


@app.cell
def _(mo):
    lang_dropdown = mo.ui.dropdown(
        options=["en", "ar", "ch_tra", "ja"], label="Select language:", value="en"
    )
    mo.vstack([lang_dropdown])
    return (lang_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## EasyOCR Reader Class Documentation
    ---
    """
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(easyocr, mo):
    mo.doc(obj=easyocr.Reader)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use an Instance of EasyOCR  to Process The Selected File
    ---
    """
    )
    return


@app.cell
def _(makedirs):
    # Create a directory where the OCR results will be saved
    makedirs("./output", exist_ok=True)
    return


@app.cell
def _(
    ImageDraw,
    PIL,
    dict_path_obj_to_file_data,
    easyocr,
    file_paths,
    get_error_as_html,
    lang_dropdown,
    mo,
):
    def save_ocr_result(bboxes, str_file_path, str_output_img_file_path) -> None:
        img = PIL.Image.open(str_file_path)
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            p0, p1, p2, p3 = bbox
            draw.line([*p0, *p1, *p2, *p3, *p0], fill="green", width=2)
        img.save(str_output_img_file_path)


    results = []
    list_bounding_box = []
    list_score = []
    list_text = []
    list_text_combined = []
    output_img_str_paths = []
    lang = lang_dropdown.value
    reader = easyocr.Reader([lang, "en"], gpu=False)

    for _file_path, file_data in dict_path_obj_to_file_data.items():
        str_file_path = str(_file_path)
        file_name_without_extension, extension = file_data

        str_output_img_file_path = (
            f"./output/{file_name_without_extension}_ocr_result_img.{extension}"
        )
        output_img_str_paths.append(str_output_img_file_path)
        try:
            result = reader.readtext(
                str_file_path,
                batch_size=3,
                output_format="standard",
            )
            results.append(result)

            current_bboxes = [bbox for bbox, _, _ in result]
            current_texts = [text for _, text, _ in result]

            list_text_combined.append(" ".join(current_texts))

            list_bounding_box.extend(current_bboxes)
            list_score.extend([score for _, _, score in result])
            list_text.extend(current_texts)

            save_ocr_result(
                current_bboxes, str_file_path, str_output_img_file_path
            )
        except Exception as e:
            mo.stop(True, get_error_as_html(f"<strong>ERROR: {e}</strong>"))

    html_model_storage_path = mo.md(
        f"""`EasyOCR models are stored in the following path: {
            reader.model_storage_directory
        }`"""
    )

    original_file_names = ", ".join(map(lambda path: path.name, file_paths))
    html_file_names = mo.md(f"`Original file name(s): [{original_file_names}]`")

    mo.vstack([html_model_storage_path, html_file_names])
    return (
        list_bounding_box,
        list_score,
        list_text,
        list_text_combined,
        original_file_names,
        output_img_str_paths,
        results,
    )


@app.cell(disabled=True, hide_code=True)
def _(easyocr, mo):
    mo.doc(obj=easyocr.Reader)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Output Image(s)""")
    return


@app.cell
def _(mo, output_img_str_paths):
    output_imgs = [
        mo.image(src=output_file_path).batch()
        for output_file_path in output_img_str_paths
    ]

    mo.ui.array(output_imgs, label="Output Images")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## OCR Results
    ---
    """
    )
    return


@app.cell
def _(
    easyocr,
    list_bounding_box,
    list_score,
    list_text,
    mo,
    original_file_names,
):
    dict_table_data = {
        "Bounding Box": easyocr.utils.np.asarray(list_bounding_box).tolist(),
        "Score": easyocr.utils.np.asarray(list_score).tolist(),
        "Text": list_text,
    }

    mo.ui.table(
        data=dict_table_data, label=f"`[{original_file_names}] OCR results`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## File(s) Content""")
    return


@app.cell
def _(file_paths, list_text_combined, mo, results):
    output_html = None
    buttons = []
    get_text, set_text = mo.state("")


    def change_text_state(txt):
        set_text(txt)


    if len(results) == 1:
        text = list_text_combined[0]
        button = mo.ui.button(
            label="Copy",
            kind="info",
            on_change=lambda _, text=text: change_text_state(text),
        )
        container = mo.vstack([button.right(), mo.md(text)])
        output_html = container.style(max_width="80%").callout(kind="info")
    else:
        dict_page_number_to_text = {}
        for i, _result in enumerate(results):
            text = list_text_combined[i]

            button = mo.ui.button(
                label="Copy",
                kind="info",
                on_change=lambda _, text=text: change_text_state(text),
            )
            buttons.append(button)
            dict_page_number_to_text[f"{file_paths[i]}"] = mo.vstack(
                [button.right(), text]
            )

        output_html = mo.accordion(dict_page_number_to_text).callout(kind="info")

    output_html
    return (get_text,)


@app.cell
def _(get_text, pyperclip):
    # Copy text to clipboard
    pyperclip.copy(get_text())
    return


if __name__ == "__main__":
    app.run()
