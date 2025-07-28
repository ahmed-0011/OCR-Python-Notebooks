# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium", app_title="CnOCR Notebook")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CnOCR Notebook
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""![CnOCR OCR](https://raw.githubusercontent.com/breezedeus/CnOCR/master/docs/figs/cnocr-logo.jpg)"""
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
    from cnocr import CnOcr
    import PIL
    from PIL import Image
    from PIL import ImageDraw


    file_browser = mo.ui.file_browser(
        label="Upload an image with one of the following supported formats: .jpg, jpeg, .png, and webp!",
        filetypes=[".jpg", ".jpeg", ".png", ".webp"],
        multiple=True,
    )
    mo.vstack([file_browser])
    return CnOcr, ImageDraw, PIL, file_browser, mo, pyperclip


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
        options=["en", "ch", "jp", "ar"], label="Select language:", value="en"
    )
    mo.vstack([lang_dropdown])
    return (lang_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## CnOCR Model Parameters
    ---
    """
    )
    return


@app.cell
def _(lang_dropdown, mo):
    lang = lang_dropdown.value
    lang_to_rec_models = {
        "en": "densenet_lite_136-gru",
        "ch": "densenet_lite_136-gru",
        "jp": "japan_PP-OCRv3",
        "ar": "arabic_PP-OCRv3",
    }

    box_score_threshould_slider = mo.md("Box score threshold: {slider_el}").batch(
        slider_el=mo.ui.slider(
            start=0,
            stop=1,
            step=0.02,
            value=0.3,
            debounce=True,
            show_value=True,
        )
    )

    mo.vstack([box_score_threshould_slider], gap=3).callout(kind="neutral")
    return box_score_threshould_slider, lang, lang_to_rec_models


@app.cell
def _(box_score_threshould_slider, mo):
    box_score_threshould = box_score_threshould_slider.value["slider_el"]

    mo.md(f"**Parameters:** \n`{box_score_threshould = }`")
    return (box_score_threshould,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## CnOCR Class Documentation
    ---
    """
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(CnOcr, mo):
    mo.doc(obj=CnOcr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use an Instance of CnOCR  to Process The Selected File
    ---
    """
    )
    return


@app.cell
def _(
    CnOcr,
    ImageDraw,
    PIL,
    box_score_threshould,
    dict_path_obj_to_file_data,
    file_paths,
    get_error_as_html,
    lang,
    lang_to_rec_models,
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

    for _file_path, file_data in dict_path_obj_to_file_data.items():
        str_file_path = str(_file_path)
        file_name_without_extension, extension = file_data

        str_output_img_file_path = (
            f"./output/{file_name_without_extension}_ocr_result_img.{extension}"
        )
        output_img_str_paths.append(str_output_img_file_path)
        try:
            ocr_engine = CnOcr(
                det_model_name="db_shufflenet_v2",
                rec_model_name=lang_to_rec_models[lang],
                context="cpu",
                det_model_backend="pytorch",  # or onnx
                rec_model_backend="onnx",  # or pytorch
            )
            result = ocr_engine.ocr(
                str_file_path, box_score_thresh=box_score_threshould
            )

            current_bboxes = [rec_data["position"] for rec_data in result]
            current_texts = [rec_data["text"] for rec_data in result]

            list_text_combined.append(" ".join(current_texts))

            results.append(result)
            list_bounding_box.extend(current_bboxes)
            list_score.extend([rec_data["score"] for rec_data in result])
            list_text.extend(current_texts)
            save_ocr_result(
                current_bboxes, str_file_path, str_output_img_file_path
            )
        except Exception as e:
            mo.stop(True, get_error_as_html(f"<strong>ERROR: {e}</strong>"))

    orignal_file_names = ", ".join(map(lambda path: path.name, file_paths))

    mo.md(f"`Original file name(s): [{orignal_file_names}]`")
    return (
        list_bounding_box,
        list_score,
        list_text,
        list_text_combined,
        orignal_file_names,
        output_img_str_paths,
        results,
    )


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
def _(list_bounding_box, list_score, list_text, mo, orignal_file_names):
    dict_table_data = {
        "Bounding Box": list_bounding_box,
        "Score": list_score,
        "Text": list_text,
    }

    mo.ui.table(
        data=dict_table_data, label=f"`[{orignal_file_names}] OCR results`"
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
