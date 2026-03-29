import json

import streamlit as st

from config import get_settings
from pipeline import MindBridgePipeline


st.set_page_config(page_title="MindBridge Demo", page_icon="MB", layout="wide")
st.title("MindBridge Supportive Dialogue Demo")
st.caption("Minimal multi-agent demo for empathy, strategy, safety, and reflection.")

user_input = st.text_area(
    "User message",
    placeholder="I feel overwhelmed with school and I do not know where to start.",
    height=160,
)

if st.button("Run pipeline", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a user message first.")
    else:
        settings = get_settings()
        settings.validate()
        pipeline = MindBridgePipeline(settings)
        result = pipeline.run(user_input.strip())

        st.subheader("Final response")
        st.write(result.final_response)

        st.subheader("Intermediate outputs")
        st.json(result.to_dict(), expanded=False)

st.divider()
st.markdown(
    "This demo is a project scaffold, not a production mental-health support system."
)
