import gradio as gr
from agent import TranscriberAgent
agent = TranscriberAgent()

def transcribe(audio):
    print(audio)
    return agent.create_conversation(audio_path=audio)

demo = gr.Interface(
    transcribe,
    gr.Audio(sources=["upload"], type = 'filepath' ),
    "text",
)

demo.launch()