# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", app_title="SuryaOCR Notebook")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # SuryaOCR Notebook
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
    from PIL import Image, ImageDraw
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from os import makedirs

    file_browser = mo.ui.file_browser(
        label="Upload an image with one of the following supported formats: .jpg, jpeg, .png, and webp!",
        filetypes=[".jpg", ".jpeg", ".png", ".webp"],
        multiple=True,
    )
    mo.vstack([file_browser])
    return (
        DetectionPredictor,
        FoundationPredictor,
        Image,
        ImageDraw,
        RecognitionPredictor,
        file_browser,
        makedirs,
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
                "<strong>NOTE: You need to select image file(s) for this cell to execute!</strong>"
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
    mo.md(
        """
    ## SuryaOCR Model Parameters
    ---
    """
    )
    return


@app.cell
def _(mo):
    params_label_to_bool = {
        "Math Mode": False,
        "Return Words": True,
        "Sort Lines": False,
    }

    switches = [
        mo.ui.switch(label=label, value=params_label_to_bool[label])
        for label in params_label_to_bool.keys()
    ]

    switches_container = mo.hstack([*switches], widths=[1, 1, 1])
    confidence_threshould_slider = mo.md(
        "Confidence threshold: {slider_el}"
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

    mo.vstack([switches_container, confidence_threshould_slider], gap=3).callout(
        kind="neutral"
    )
    return confidence_threshould_slider, switches


@app.cell
def _(confidence_threshould_slider, mo, switches):
    math_mode, return_words, sort_lines = map(
        lambda switch: switch.value, switches
    )

    confidence_threshould = confidence_threshould_slider.value["slider_el"]

    mo.md(
        f"**Parameters:** \n`{math_mode = }, {return_words = }, {sort_lines = }, {confidence_threshould = }`"
    )
    return confidence_threshould, math_mode, return_words, sort_lines


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## SuryaOCR Class Documentation
    ---
    """
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(RecognitionPredictor, mo):
    mo.doc(obj=RecognitionPredictor)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use an Instance of SuryaOCR  to Process The Selected File
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
    DetectionPredictor,
    FoundationPredictor,
    Image,
    ImageDraw,
    RecognitionPredictor,
    confidence_threshould,
    dict_path_obj_to_file_data,
    file_paths,
    get_error_as_html,
    math_mode,
    mo,
    return_words,
    sort_lines,
):
    def save_ocr_result(image, polygons, str_output_img_file_path) -> None:
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            p0, p1, p2, p3 = polygon
            draw.line([*p0, *p1, *p2, *p3, *p0], fill="green", width=2)
        image.save(str_output_img_file_path)


    images = []
    list_bounding_box = []
    list_confidence = []
    list_text = []
    list_text_combined = []
    output_img_str_paths = []

    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()


    for _file_path, file_data in dict_path_obj_to_file_data.items():
        images.append(Image.open(_file_path))
        file_name_without_extension, extension = file_data

        str_output_img_file_path = (
            f"./output/{file_name_without_extension}_ocr_result_img.{extension}"
        )
        output_img_str_paths.append(str_output_img_file_path)
    try:
        results = recognition_predictor(
            images,
            det_predictor=detection_predictor,
            math_mode=math_mode,
            return_words=return_words,
            sort_lines=sort_lines,
        )

    except Exception as e:
        mo.stop(True, get_error_as_html(f"<strong>ERROR: {e}</strong>"))

    for i, result in enumerate(results):
        image = images[i].copy()
        output_path = output_img_str_paths[i]
        txts = []

        for text_line in result.text_lines:
            polygons = []

            if text_line.confidence > confidence_threshould:
                list_bounding_box.append(text_line.bbox)
                list_confidence.append(text_line.confidence)
                list_text.append(text_line.text)
                txts.append(text_line.text)
                polygons.append(text_line.polygon)
            save_ocr_result(image, polygons, str_output_img_file_path)
        list_text_combined.append(" ".join(txts))


    orignal_file_names = ", ".join(map(lambda path: path.name, file_paths))

    mo.md(f"`Original file name(s): [{orignal_file_names}]`")
    return (
        list_bounding_box,
        list_confidence,
        list_text,
        list_text_combined,
        orignal_file_names,
        output_img_str_paths,
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
def _(list_bounding_box, list_confidence, list_text, mo, orignal_file_names):
    dict_table_data = {
        "Bounding Box": list_bounding_box,
        "Confidence": list_confidence,
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
def _(file_paths, list_text_combined, mo):
    output_html = None
    buttons = []
    get_text, set_text = mo.state("")


    def change_text_state(txt):
        set_text(txt)


    if len(list_text_combined) == 1:
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
        for idx, text in zip(range(len(file_paths)), list_text_combined):
            button = mo.ui.button(
                label="Copy",
                kind="info",
                on_change=lambda _, text=text: change_text_state(text),
            )
            buttons.append(button)
            dict_page_number_to_text[f"{file_paths[idx]}"] = mo.vstack(
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
