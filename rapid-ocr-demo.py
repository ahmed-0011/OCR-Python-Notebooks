# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium", app_title="RapidOCR Notebook")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # RapidOCR Notebook
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![PaddleOCR 3.0](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/RapidOCR_LOGO.png)""")
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
    from rapidocr import (
        EngineType,
        LangDet,
        ModelType,
        OCRVersion,
        RapidOCR,
        LangRec,
    )


    file_browser = mo.ui.file_browser(
        label="Upload an image with one of the following supported formats: .jpg, jpeg, .png, and webp!",
        filetypes=[".jpg", ".jpeg", ".png", ".webp"],
        multiple=True,
    )
    mo.vstack([file_browser])
    return (
        EngineType,
        LangDet,
        LangRec,
        ModelType,
        OCRVersion,
        RapidOCR,
        file_browser,
        mo,
        pyperclip,
    )


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
                "<strong>NOTE: You need to select an image or a PDF file for this cell to execute!</strong>"
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
    return dict_path_obj_to_file_data, file_path, file_paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Select The Langauge of The Text in The Image""")
    return


@app.cell
def _(mo):
    lang_dropdown = mo.ui.dropdown(
        options=["en", "ch", "jp"], label="Select language:", value="en"
    )
    mo.vstack([lang_dropdown])
    return (lang_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## RapidOCR Model Parameters
    ---
    """
    )
    return


@app.cell
def _(LangRec, lang_dropdown, mo):
    lang = lang_dropdown.value
    lang_to_enum_lang = {"en": LangRec.EN, "ch": LangRec.CH, "jp": LangRec.JAPAN}
    params_label_to_bool = {
        "Doc & Textline Orientation Classify": True,
        "Single Char Box": True,
        "Word Box": True,
    }
    switches = [
        mo.ui.switch(label=label, value=params_label_to_bool[label])
        for label in params_label_to_bool.keys()
    ]

    switches_container = mo.hstack([*switches], widths=[1, 1, 1])
    text_score_threshould_slider = mo.md(
        "Text score threshold: {slider_el}"
    ).batch(
        slider_el=mo.ui.slider(
            start=0,
            stop=1,
            step=0.02,
            value=0.5,
            debounce=True,
            show_value=True,
        )
    )

    mo.vstack([switches_container, text_score_threshould_slider], gap=3).callout(
        kind="neutral"
    )
    return lang, lang_to_enum_lang, switches, text_score_threshould_slider


@app.cell
def _(mo, switches, text_score_threshould_slider):
    doc_textline_orientation_classify, single_char_box, word_box = map(
        lambda switch: switch.value, switches
    )

    text_score_threshould = text_score_threshould_slider.value["slider_el"]

    mo.md(
        f"**Parameters:** \n`{doc_textline_orientation_classify = }, {single_char_box = }, {word_box = }, {text_score_threshould = }`"
    )
    return (
        doc_textline_orientation_classify,
        single_char_box,
        text_score_threshould,
        word_box,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RapidOCR Class Documentation
    ---
    """
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(RapidOCR, mo):
    mo.doc(obj=RapidOCR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use an Instance of RapidOCR  to Process The Selected File
    ---
    """
    )
    return


@app.cell
def _(
    EngineType,
    LangDet,
    ModelType,
    OCRVersion,
    RapidOCR,
    dict_path_obj_to_file_data,
    doc_textline_orientation_classify,
    file_paths,
    get_error_as_html,
    lang,
    lang_to_enum_lang,
    mo,
    single_char_box,
    text_score_threshould,
    word_box,
):
    def save_ocr_result(result, str_output_img_file_path) -> None:
        result.vis(str_output_img_file_path)


    results = []
    list_bounding_box = []
    list_score = []
    list_text = []
    output_img_str_paths = []
    engine = RapidOCR(
        params={
            "Rec.engine_type": EngineType.ONNXRUNTIME,  # or EngineType.OPENVINO or EngineType.PADDLE or EngineType.TORCH
            "Det.engine_type": EngineType.ONNXRUNTIME,  # or EngineType.OPENVINO or EngineType.PADDLE or EngineType.TORCH
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Det.model_type": ModelType.MOBILE,
            "Rec.lang_type": lang_to_enum_lang[lang],
            "Det.lang_type": LangDet.EN,
        }
    )


    for _file_path, file_data in dict_path_obj_to_file_data.items():
        str_file_path = str(_file_path)
        file_name_without_extension, extension = file_data

        str_output_img_file_path = (
            f"./output/{file_name_without_extension}_ocr_result_img.{extension}"
        )
        output_img_str_paths.append(str_output_img_file_path)
        try:
            result = engine(
                str_file_path,
                use_cls=doc_textline_orientation_classify,
                return_single_char_box=single_char_box,
                return_word_box=word_box,
                text_score=text_score_threshould,
            )

            results.append(result)
            list_bounding_box.extend(result.boxes.tolist())
            list_score = (*list_score, *result.scores)
            list_text = (*list_text, *result.txts)
            save_ocr_result(result, str_output_img_file_path)
        except Exception as e:
            mo.stop(True, get_error_as_html(f"<strong>ERROR: {e}</strong>"))

    mo.md(
        f"`Original file name(s): [{', '.join(map(lambda path: path.name, file_paths))}]`"
    )
    return (
        list_bounding_box,
        list_score,
        list_text,
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
def _(file_path, list_bounding_box, list_score, list_text, mo):
    dict_table_data = {
        "Bounding Box": list_bounding_box,
        "Score": list_score,
        "Text": list_text,
    }

    mo.ui.table(data=dict_table_data, label=f"`{file_path.name} OCR results`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## File(s) Content""")
    return


@app.cell
def _(file_paths, mo, results):
    output_html = None
    buttons = []
    get_text, set_text = mo.state("")


    def change_text_state(txt):
        set_text(txt)


    if len(results) == 1:
        text = results[0].to_markdown()
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
            text = _result.to_markdown()

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
