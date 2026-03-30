import os
import google.generativeai as genai

def test_truncation():
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = """You are a university regulation assistant. Answer ONLY using the document excerpts below.

RULES:
1. Answer ONLY if explicitly stated.
2. If NOT found -> respond ONLY: "The information is not available..."

FORMAT:
Answer:
<1-3 sentence factual answer here>

Source:
<cite source here>

DOCUMENT EXCERPTS:
Inter-college transfer is permitted for students who have scored above 8.5 CGPA in their first year.

QUESTION: What are the eligibility criteria for inter-college transfer?"""
    
    print("Prompt:")
    print(prompt)
    print("-" * 50)
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.1,
                stop_sequences=["[QUESTION]", "---", "[/INST]", "USER QUESTION"]
            )
        )
        print("Response Text:", repr(response.text))
        print("Finish Reason:", response.candidates[0].finish_reason)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_truncation()
