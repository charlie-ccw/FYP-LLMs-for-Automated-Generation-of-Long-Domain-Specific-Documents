RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_SYSTEM = "You are a useful assistant."

RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_PROMPT = """Below, I will provide you with an EXAMPLE TEXT and a DRAFT TEXT. Your task is to optimize and enrich the draft text by imitating the style, format, and writing approach of the example text.
Note:
1. Do not modify the core ideas of the draft text, and do not add unrealistic content.
2. Only imitate the style, format, and writing approach of the example text; do not incorporate the core content of the target text into the draft text.
3. Think independently and optimize the draft text as much as possible.

EXAMPLE TEXT:
{example_text}

DRAFT TEXT:
{draft_text}

respond in JSON format:
{{
    "optimised_text": "the text after optimisation. In string format"
}}"""

