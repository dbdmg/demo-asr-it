import logging
import sys
import gradio as gr
from transformers import pipeline, AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DICT_MODELS = {
    "robust-300m": {"model_id": "dbdmg/wav2vec2-xls-r-300m-italian-robust", "has_lm": True},
    "robust-1b": {"model_id": "dbdmg/wav2vec2-xls-r-1b-italian-robust", "has_lm": True},
    "300m": {"model_id": "dbdmg/wav2vec2-xls-r-300m-italian", "has_lm": True},
}


# LANGUAGES = sorted(LARGE_MODEL_BY_LANGUAGE.keys())

# the container given by HF has 16GB of RAM, so we need to limit the number of models to load
MODELS = sorted(DICT_MODELS.keys())
CACHED_MODELS_BY_ID = {}

def build_html(history):
    html_output = "<div class='result'>"
    for item in history:
        if item["error_message"] is not None:
            html_output += f"<div class='result_item result_item_error'>{item['error_message']}</div>"
        else:
            url_suffix = " + Guided by Language Model" if item["decoding_type"] == "Guided by Language Model" else ""
            html_output += "<div class='result_item result_item_success'>"
            html_output += f'<strong><a target="_blank" href="https://huggingface.co/{item["model_id"]}">{item["model_id"]}{url_suffix}</a></strong><br/><br/>'
            html_output += f'{item["transcription"]}<br/>'
            html_output += "</div>"
    html_output += "</div>"
    return html_output

def run(uploaded_file, input_file, model_name, decoding_type, history):
    
    model = DICT_MODELS.get(model_name)
    history = history or []
    
    if uploaded_file is None and input_file is None:
        history.append({
            "model_id": model["model_id"],
            "decoding_type": decoding_type,
            "transcription": "",
            "error_message": "No input provided."
        })
    else:

        if input_file is None:
            input_file = uploaded_file

        logger.info(f"Running ASR {model_name}-{decoding_type} for {input_file}")

        history = history or []

        if model is None:
            history.append({
                "error_message": f"Model size {model_size} not found for {language} language :("
            })
        elif decoding_type == "Guided by Language Model" and not model["has_lm"]:
            history.append({
                "error_message": f"LM not available for {language} language :("
            })
        else:

            # model_instance = AutoModelForCTC.from_pretrained(model["model_id"])
            model_instance = CACHED_MODELS_BY_ID.get(model["model_id"], None)
            if model_instance is None:
                model_instance = AutoModelForCTC.from_pretrained(model["model_id"])
                CACHED_MODELS_BY_ID[model["model_id"]] = model_instance

            if decoding_type == "Guided by Language Model":
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(model["model_id"])
                asr = pipeline("automatic-speech-recognition", model=model_instance, tokenizer=processor.tokenizer, 
                            feature_extractor=processor.feature_extractor, decoder=processor.decoder)
            else:
                processor = Wav2Vec2Processor.from_pretrained(model["model_id"])
                asr = pipeline("automatic-speech-recognition", model=model_instance, tokenizer=processor.tokenizer, 
                            feature_extractor=processor.feature_extractor, decoder=None)

            transcription = asr(input_file, chunk_length_s=5, stride_length_s=1)["text"]

            logger.info(f"Transcription for {input_file}: {transcription}")

            history.append({
                "model_id": model["model_id"],
                "decoding_type": decoding_type,
                "transcription": transcription,
                "error_message": None
            })

    html_output = build_html(history)

    return html_output, history


gr.Interface(
    run,
    inputs=[
        gr.inputs.Audio(source="upload", type='filepath', optional=True),
        gr.inputs.Audio(source="microphone", type="filepath", label="Record something...", optional=True),
        gr.inputs.Radio(label="Model", choices=MODELS),
        gr.inputs.Radio(label="Decoding type", choices=["Standard", "Guided by Language Model"]),
        "state"
    ],
    outputs=[
        gr.outputs.HTML(label="Outputs"),
        "state"
    ],
    title="Italian Robust ASR",
    description="",
    css="""
    .result {display:flex;flex-direction:column}
    .result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
    .result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
    .result_item_error {background-color:#ff7070;color:white;align-self:start}
    """,
    allow_screenshot=False,
    allow_flagging="never",
    theme="huggingface",
    examples = [
        ['demo_example_1.mp3', 'demo_example_1.mp3', 'robust-300m', 'Guided by Language Model']
    ]
).launch(enable_queue=True)