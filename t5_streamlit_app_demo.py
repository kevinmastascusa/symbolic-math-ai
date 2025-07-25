import streamlit as st
from transformers.pipelines import pipeline
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import List, Any
from pydantic import Field

class T5LLM(LLM):
    pipeline: Any = Field(default=None, exclude=True)

    def __init__(self, pipeline, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        output = self.pipeline(prompt, max_length=100, clean_up_tokenization_spaces=True)
        return output[0]["generated_text"]

    @property
    def _llm_type(self) -> str:
        return "t5-small"

    class Config:
        arbitrary_types_allowed = True

st.set_page_config(page_title="T5 Math Solver")
st.title("T5-small Math Solver")

@st.cache_resource
def load_model():
    t5_pipe = pipeline("text2text-generation", model="t5-small")
    return T5LLM(pipeline=t5_pipe)

llm = load_model()

prompt = PromptTemplate(
    input_variables=["question"],
    template="solve: {question}"
)

chain = prompt | llm

with st.form("math_form"):
    user_input = st.text_area(
        "🔢 Enter a math problem:",
        "If a pencil costs 2 dollars and you buy 4, how much do you pay?",
    )
    submitted = st.form_submit_button("Submit")

    if submitted and user_input.strip():
        result = chain.invoke({"question": user_input})
        st.success("✅ Answer:")
        st.markdown(f"**{result}**")