import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from info_extraction.demo_ie_spacy import spacy_info_ext
from info_extraction.demo_ie_rules import rules_info_ext
from info_extraction.demo_ie_stanford import stanford_info_ext
from topic_modelling.demo_lda_sklearn import lda_s_topic_model
from topic_modelling.demo_lda_gensim import lda_g_topic_model
from topic_modelling.demo_lsa_sklearn import lsa_s_topic_model
from topic_modelling.demo_nmf_sklearn import nmf_s_topic_model
from sentiment_analysis.demo_sentiment_analysis import load_sentiment_analysis_demo

RULES_BASED = "Rules-Based"
CNN_RNN = "CNN & RNN (SpaCy)"
CRF = "CRF (Stanford NER)"

######## Info Ext ########

def update_radio(option):
    visible = {
        RULES_BASED: False,
        CNN_RNN: False,
        CRF: False
    }
    if option == RULES_BASED:
        visible[RULES_BASED] = True
    elif option == CNN_RNN:
        visible[CNN_RNN] = True
    elif option == CRF:
        visible[CRF] = True
    
    return {
        text_rules: gr.Textbox(visible=visible[RULES_BASED]),
        text_spacy: gr.Textbox(visible=visible[CNN_RNN]),
        text_stanford: gr.Textbox(visible=visible[CRF]),
        html_rules: gr.HTML(visible=visible[RULES_BASED]),
        html_spacy: gr.HTML(visible=visible[CNN_RNN]),
        html_stanford: gr.HTML(visible=visible[CRF])
    }

with gr.Blocks() as info_ext:
    with gr.Row():
        with gr.Column():
            ie_file = gr.File(label="File Upload", file_count='single', file_types=['.csv'], type="filepath")
            with gr.Row():
                run_rules = gr.Button(variant="primary", value=RULES_BASED)
                run_spacy = gr.Button(variant="primary", value=CNN_RNN)
                run_stanford = gr.Button(variant="primary", value=CRF)
        with gr.Column():
            ie_radio = gr.Radio([RULES_BASED, CNN_RNN, CRF], label="Model Type", value=RULES_BASED)
            text_rules = gr.JSON(label=RULES_BASED, visible=True)
            text_spacy = gr.JSON(label=CNN_RNN, visible=False)
            text_stanford = gr.JSON(label=CRF, visible=False)
    with gr.Row():
        html_rules = gr.HTML(label="NER Output", visible=True)
        html_spacy = gr.HTML(label="NER Output", visible=False)
        html_stanford = gr.HTML(label="NER Output", visible=False)

    ie_radio.change(update_radio, ie_radio, [text_rules, text_spacy, text_stanford, html_rules, html_spacy, html_stanford])
    rules_event = run_rules.click(rules_info_ext, ie_file, [html_rules, text_rules])
    spacy_event = run_spacy.click(spacy_info_ext, ie_file, [html_spacy, text_spacy])
    stanford_event = run_stanford.click(stanford_info_ext, ie_file, [html_stanford, text_stanford])

######## Topic Modelling ########
    
LDA_SKLEARN = "LDA (SkLearn)"
LDA_GENSIM = "LDA (Gensim)"
LSA_SKLEARN = "LSA (SkLearn)"
NMF_SKLEARN = "NMF (SkLearn)"

max_textboxes = 10

def variable_outputs(path, model_type):
    imgs = []
    if model_type == LDA_SKLEARN:
        imgs = lda_s_topic_model(path)
    elif model_type == LDA_GENSIM:
        imgs = lda_g_topic_model(path)
    elif model_type == LSA_SKLEARN:
        imgs = lsa_s_topic_model(path)
    elif model_type == NMF_SKLEARN:
        imgs = nmf_s_topic_model(path)

    result = [gr.Markdown(f"# {model_type}")]
    for img in imgs:
        result.append(gr.Image(value=img, visible=True))
    result.extend([gr.Image(visible=False)]*(max_textboxes-len(imgs)))

    return result

with gr.Blocks() as topic_mod:
    with gr.Row():
        with gr.Column():
            ie_file = gr.File(label="File Upload", file_count='single', file_types=['.csv'], type="filepath")
            with gr.Row():
                run_lda_s = gr.Button(variant="primary", value=LDA_SKLEARN)
                run_lda_g = gr.Button(variant="primary", value=LDA_GENSIM)
                run_lsa_s = gr.Button(variant="primary", value=LSA_SKLEARN)
                run_nmf_s = gr.Button(variant="primary", value=NMF_SKLEARN)
        with gr.Column():
            title = gr.Markdown("# Topic Modelling")
            imgs = [title]
            for i in range(max_textboxes):
                img = gr.Image(visible=False)
                imgs.append(img)

    lds_s_event = run_lda_s.click(lambda path: variable_outputs(path, LDA_SKLEARN), ie_file, imgs)
    lds_g_event = run_lda_g.click(lambda path: variable_outputs(path, LDA_GENSIM), ie_file, imgs)
    lsa_s_event = run_lsa_s.click(lambda path: variable_outputs(path, LSA_SKLEARN), ie_file, imgs)
    nmf_s_event = run_nmf_s.click(lambda path: variable_outputs(path, NMF_SKLEARN), ie_file, imgs)

######## Sentiment Analysis ########

sent_analy = load_sentiment_analysis_demo()

######## Integration ########

demo = gr.TabbedInterface([sent_analy, topic_mod, info_ext], ["Sentiment Analysis", "Topic Modeling", "Information Extraction"])

if __name__ == "__main__":
    demo.queue().launch()