import torch
from transformers import pipeline, AutoTokenizer,AutoModelForCausalLM

from config import LLM_MODEL_NAME

device = 0 if torch.cuda.is_available() else -1

# Load LLM model once

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def generate_answer(context, query_text):
    """Generate an answer using Falcon-11B."""
    
    prompt = f"""
    **User Query:** {query_text}

    **Retrieved Context:** {context}

    **Task:** Provide a concise, accurate response using only the retrieved context.

    **Instructions:** 
    - Summarize key insights.
    - Avoid redundancy.
    - List source links for all referenced documents.

    **Response Format:**
    - **Summary**: Key insights from sources.
    - **Sources**: [Document #] [Title]: [Link]
    """

    response = llm_pipeline(
        prompt,
        max_new_tokens=50,  # Number of tokens to generate
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )[0]["generated_text"]


    return response
