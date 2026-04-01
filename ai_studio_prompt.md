# 🎙️ Google AI Studio: Semantic Audio Chunking Prompt

**Instructions:** 
1. Open [Google AI Studio](https://aistudio.google.com/).
2. Select **Gemini 1.5 Pro** (it is best at long-context audio and strict JSON).
3. Upload your `.mp3` file.
4. Copy and paste the entire prompt below into the text box.
5. Fill in the specific `[VARS]` at the top of the prompt.
6. Run it!

---
## Copy everything below this line
---

**Audio File Provided:** [Uploaded MP3]

**Video Metadata**
- `video_id`: [ENTER_ID_HERE, e.g., "video_1"]
- `title`: [ENTER_TITLE_HERE]
- `channel`: [ENTER_CHANNEL_HERE]
- `language`: [ENTER_LANGUAGE_HERE, e.g., "English"]

**Task:**
You are an expert transcriptionist and data engineer. I have provided an audio recording of a technical educational video. Your job is to listen to the audio, transcribe exactly what is spoken, and divide the transcript into **Semantic Chunks**.

**What is a Semantic Chunk?**
A semantic chunk is a continuous block of text that covers a single, cohesive topic, idea, or explanation. 
- Do NOT just split blindly every 30 seconds.
- Wait for a natural transition in the speaker's topic (e.g., moving from "What is backpropagation" to "How to calculate the loss function").
- Aim for chunks between roughly 100 to 300 words. Too short, and it lacks context for search. Too long, and it dilutes the specific topic.

**Output Format Requirements:**
You must output ONLY a valid JSON array of objects. Do not include any markdown formatting blocks like ```json or any conversational filler. Just the raw JSON array.

Strictly adhere to this exact schema for every object in the array:
[
  {
    "text": "<The exact spoken transcript of this chunk. Fix obvious verbal stumbles or 'um's, but preserve all technical terminology accurately.>",
    "video_id": "<Copy the value exactly from the Video Metadata above>",
    "title": "<Copy the value exactly from the Video Metadata above>",
    "channel": "<Copy the value exactly from the Video Metadata above>",
    "language": "<Copy the value exactly from the Video Metadata above>",
    "topic_summary": "<A very brief 3-5 word summary of the concept discussed in this chunk>",
    "key_entities": ["<List of technical keywords or concepts>", "<Keyword 2>"],
    "timestamp": "<The starting time as human-readable MM:SS>",
    "start_ms": "<The starting time converted exactly into milliseconds (e.g., 1:30 = 90000)>",
    "end_ms": "<The ending time of the chunk converted into milliseconds>"
  }
]

**Critical Rules:**
1. **No Data Loss:** Ensure 100% of the spoken educational content is transcribed. Do not summarize; transcribe verbatim, just organized into topic blocks.
2. **Timestamps:** Ensure your timestamps accurately reflect the audio timeline.
3. **JSON Validity:** Double check that braces, brackets, and quotes are correctly formatted so it can be parsed immediately by Python's `json.loads()`.
