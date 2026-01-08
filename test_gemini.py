
import google.generativeai as genai
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
print(f"API Key present: {bool(key)}")

if key:
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Model initialized.")
        response = model.generate_content("Hello, can you reply with valid JSON: {\"status\": \"ok\"}")
        print("Response received.")
        print(response.text)
    except Exception:
        traceback.print_exc()
else:
    print("Skipping test, no key.")
