import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import List, Any
from pydantic import Field
import torch

class T5LLM(LLM):
    tokenizer: AutoTokenizer = Field(default=None, exclude=True)
    model: AutoModelForSeq2SeqLM = Field(default=None, exclude=True)
    
    def __init__(self, tokenizer, model, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.model = model

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            do_sample=False,
            num_beams=2,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "flan-t5-small"

    class Config:
        arbitrary_types_allowed = True

st.set_page_config(page_title="Flan-T5 Math Solver")
st.title("Flan-T5-small Math Solver")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return T5LLM(tokenizer=tokenizer, model=model)


llm = load_model()

prompt = PromptTemplate(
    input_variables=["question"],
    template="solve: {question}"
)

chain = prompt | llm

with st.form("math_form"):
    user_input = st.text_area(
        "Enter a math problem:",
        "If a pencil costs 2 dollars and you buy 4, how much do you pay?",
    )
    submitted = st.form_submit_button("Submit")

    if submitted and user_input.strip():
        result = chain.invoke({"question": user_input})
        st.success("Answer:")
        st.markdown(f"**{result}**")