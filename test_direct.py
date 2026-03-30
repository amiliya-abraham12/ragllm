import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = """You are a university regulation assistant. Answer ONLY using the document excerpts below.

RULES:
1. Answer ONLY if explicitly stated in the excerpts.
2. If NOT found -> respond ONLY: "The information is not available in the provided documents."
3. NEVER guess, infer, assume, or use external knowledge.
4. NEVER invent course codes, percentages, dates, or regulation numbers.
5. Provide a clear 1-3 sentence factual answer. Do not add any extra conversational filler, notes, or disclaimers.

DOCUMENT EXCERPTS:
B. Tech. Regulation, 2024 APJ Abdul Kalam Technological University 22 ii. The SFA/FA shall ensure that all relevant information is communicated to the students to facilitate the timely completion of all academic activities as per the schedule published by the college and University. 6. Attendance R 6.1 i. Students are expected to attain 100% attendance for all courses. However, under unavoidable circumstances, they are permitted to take leave, provided that the total leave of absence does not exceed 25% of the academic contact hours for a course. ii. A minimum of 75% attendance is mandatory to be eligible to appear for the end semester examination. iii. Menstrual Leave Attendance Relaxation: A 2% relaxation in attendance shall be granted to students as menstrual leave. iv. PWD Attendance Relaxation: A 5% relaxation in attendance shall be granted to students with disabilities (PWD). v. The students shall be informed about their attendance status periodically by the colleges so that the students shall be cautioned to make up the shortage. R 6.2 i. Attendance Requirement Relaxation: i. Eligibility for Relaxation: In exceptional cases, such as medical reasons or personal emergencies, the Principal may grant permission for condonation of attendance for students if their attendance is less than 75% but greater than or equal to 60%. ii...

QUESTION: What is the attendance policy?"""

res = model.generate_content(prompt, generation_config={"temperature": 0.1, "max_output_tokens": 256})
print("TEXT:", repr(res.text))
print("FINISH REASON:", res.candidates[0].finish_reason)
