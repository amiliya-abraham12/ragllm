"""Test the full ask() pipeline with the mode of communication query."""
import json, sys
sys.path.insert(0, '.')
from backend.chat import ask  # type: ignore[import]

query = "what is the mode of communication for the application of PhD admission"
result = ask(query)
print("ANSWER:", result)

with open("debug_answer.txt", "w", encoding="utf-8") as f:
    f.write(f"Query: {query}\n\nAnswer: {result}\n")
print("DONE")
