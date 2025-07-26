# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium", app_title="PaddleOCR 3.0 Notebook")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # PaddleOCR 3.0 Notebook
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![PaddleOCR 3.0](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/docs/images/Banner.png)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Upload an Image or A PDF File
    ---
    """
    )
    return


@app.cell
def _():
    import json
    import marimo as mo
    import pyperclip
    from paddleocr import PaddleOCR
    from pypdfium2 import PdfiumError


    file_browser = mo.ui.file_browser(
        label="Upload an image with one of the following supported formats: .pdf, .jpg, jpeg, and .png!",
        filetypes=[".pdf", ".jpg", ".jpeg", ".png"],
        multiple=False,
    )
    mo.vstack([file_browser])
    return PaddleOCR, file_browser, mo, pyperclip


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
    mo.md(r"""## Image/PDF File to Processs""")
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

    file_path = file_browser.path()
    file_name_without_extension, extension = file_path.name.rsplit(".")

    file_is_pdf = extension.lower() == "pdf"

    mo.pdf(src=file_path) if file_is_pdf else mo.image(src=file_path)
    return extension, file_is_pdf, file_name_without_extension, file_path


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
    ## PaddleOCR Model Parameters
    ---
    """
    )
    return


@app.cell
def _(lang_dropdown, mo):
    lang = lang_dropdown.value
    params_label_to_bool = {
        "Doc Orientation Classify": True,
        "Doc Unwarping": False,
        "Textline Orientation": True,
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
    return lang, switches, text_score_threshould_slider


@app.cell
def _(mo, switches, text_score_threshould_slider):
    doc_orientation_classify, doc_unwarping, textline_orientation = map(
        lambda switch: switch.value, switches
    )

    text_score_threshould = text_score_threshould_slider.value["slider_el"]

    mo.md(
        f"**Parameters:** \n`{doc_orientation_classify = }, {doc_unwarping = }, {textline_orientation = }, {text_score_threshould = }`"
    )
    return (
        doc_orientation_classify,
        doc_unwarping,
        text_score_threshould,
        textline_orientation,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## PaddleOCR Class Documentation
    ---
    """
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(PaddleOCR, mo):
    mo.doc(obj=PaddleOCR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use an Instance of PaddleOCR  to Process The Selected File
    ---
    """
    )
    return


@app.cell
def _(
    PaddleOCR,
    doc_orientation_classify,
    doc_unwarping,
    file_is_pdf,
    file_path,
    get_error_as_html,
    lang,
    mo,
    text_score_threshould,
    textline_orientation,
):
    def save_ocr_result(result) -> None:
        result.save_all("output")


    ocr = PaddleOCR(
        use_doc_orientation_classify=doc_orientation_classify,
        use_doc_unwarping=doc_unwarping,
        use_textline_orientation=textline_orientation,
        text_rec_score_thresh=text_score_threshould,
        lang=lang,
        ocr_version="PP-OCRv4",  # or PP-OCRv3 or PP-OCRv5
        # device="cpu",
        # cpu_threads=8,
    )

    str_file_path = str(file_path)

    try:
        results = ocr.predict(str_file_path)
    except Exception as e:
        mo.stop(True, get_error_as_html(f"<strong>ERROR: {e}</strong>"))


    if file_is_pdf:
        for result in results:
            save_ocr_result(result)
    else:
        save_ocr_result(results[0])

    mo.md(f"`Original file name: {file_path.name}`")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Output Image(s)""")
    return


@app.cell
def _(extension, file_is_pdf, file_name_without_extension, mo, results):
    images = None

    if file_is_pdf:
        str_output_image_path = "./output/{0}_{1}_ocr_res_img.png"
        images = [
            mo.image(
                str_output_image_path.format(file_name_without_extension, i)
            ).batch()
            for i in range(len(results))
        ]
    else:
        str_output_image_path = "./output/{0}_ocr_res_img.{1}"
        images = [
            mo.image(
                str_output_image_path.format(
                    file_name_without_extension, extension
                )
            ).batch()
        ]

    mo.ui.array(images or [], label="Output Images")
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
def _(file_is_pdf, file_path, mo, results):
    list_bounding_box = [] if file_is_pdf else results[0].json["res"]["rec_polys"]
    list_score = [] if file_is_pdf else results[0].json["res"]["rec_scores"]
    list_text = [] if file_is_pdf else results[0].json["res"]["rec_texts"]

    if file_is_pdf:
        for _result in results:
            dict_ocr_data = _result.json["res"]
            list_bounding_box.extend(dict_ocr_data["rec_polys"])
            list_score.extend(dict_ocr_data["rec_scores"])
            list_text.extend(dict_ocr_data["rec_texts"])

    dict_table_data = dict(
        [
            ("Bounding Box", list_bounding_box),
            ("Score", list_score),
            ("Text", list_text),
        ]
    )

    mo.ui.table(data=dict_table_data, label=f"`{file_path.name} OCR results`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## File Content""")
    return


@app.cell
def _(file_is_pdf, mo, results):
    output_html = None
    buttons = []
    get_text, set_text = mo.state("")


    def change_text_state(txt):
        set_text(txt)


    if file_is_pdf:
        dict_page_number_to_text = {}
        for i, _result in enumerate(results):
            text = " ".join(_result.json["res"]["rec_texts"])
            button = mo.ui.button(
                label="Copy",
                kind="info",
                on_change=lambda _, text=text: change_text_state(text),
            )
            buttons.append(button)
            dict_page_number_to_text[f"Page {i}"] = mo.vstack(
                [button.right(), text]
            )

        output_html = mo.accordion(dict_page_number_to_text).callout(kind="info")
    else:
        text = " ".join(results[0].json["res"]["rec_texts"])
        button = mo.ui.button(
            label="Copy",
            kind="info",
            on_change=lambda _, text=text: change_text_state(text),
        )
        container = mo.vstack([button.right(), mo.md(text)])
        output_html = container.style(max_width="80%").callout(kind="info")

    output_html
    return (get_text,)


@app.cell
def _(get_text, pyperclip):
    # Copy text to clipboard
    pyperclip.copy(get_text())
    return


if __name__ == "__main__":
    app.run()
