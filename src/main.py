import gradio as gr
from tabs.scenario_tab import create_scenario_tab
from tabs.vocab_tab import create_vocab_tab
from utils.logger import LOG

def main():
    with gr.Blocks(title="LangCoach 英语私教") as language_coach_app:
        create_scenario_tab()
        create_vocab_tab()
    
    # 启动应用
    language_coach_app.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
