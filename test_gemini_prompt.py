import os
from backend.gemini_wrapper import GeminiLLM

def test_gemini():
    llm = GeminiLLM(api_key=os.environ.get("GEMINI_API_KEY"))
    prompt = """[INST] You are a university regulation assistant. Answer ONLY using the document excerpts below.

RULES:
1. Answer ONLY if explicitly stated.
2. If NOT found -> respond ONLY: "The information is not available..."

FORMAT:
Answer:
<answer here>

Source:
<source here>

DOCUMENT EXCERPTS:
Some document text about attendance being 75%.

QUESTION: What is the attendance requirement?
[/INST]"""
    
    print("Testing Gemini directly...")
    try:
        res = llm(
            prompt, 
            max_tokens=150, 
            temperature=0.0,
            stop=["[QUESTION]", "---", "[/INST]", "USER QUESTION"]
        )
        print("Response:", repr(res))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_gemini()
