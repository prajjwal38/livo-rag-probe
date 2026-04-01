# %%
from src.answer_generation import answer_query, render_answer

# %%
# Edit these inputs and run the cell in VS Code's Interactive Window.
question = "Explain self-attention in simple terms."
target_language = None  # Example: "Hindi"
top_k = 10

# %%
result = answer_query(
    query=question,
    top_k=top_k,
    target_language=target_language,
)

# %%
print(render_answer(result))

# %%
# Inspect the raw retrieved chunks if you want to debug grounding.
result["raw_retrieval"]
