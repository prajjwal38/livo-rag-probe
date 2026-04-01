import json
with open('d:/OG projects/rag_for_LIVO.ai/llm_chunks/video_3_transcript.json', 'r', encoding='utf-8') as f:
    text = f.read()

try:
    json.loads(text)
    print("Loaded successfully")
except json.decoder.JSONDecodeError as e:
    print(f"Error at line {e.lineno}, col {e.colno}, char {e.pos} : {e.msg}")
    start = max(0, e.pos - 50)
    end = min(len(text), e.pos + 50)
    print("--- CONTEXT ---")
    print(repr(text[start:end]))
    print("-------")
