import streamlit as st

from dialog_step_debug_ui import render_dialog_step_debug_page


st.set_page_config(
    layout="wide",
    page_title="对话逐条观察台",
    page_icon="🧪",
    initial_sidebar_state="expanded",
)

render_dialog_step_debug_page(
    state_prefix="dialog_step_debug_standalone",
    show_page_title=True,
    embedded=False,
)
