import os
import random
import json
import chromadb
from groq import Groq

# 1. Setup - Connect to your unzipped textbook data
client = chromadb.PersistentClient(path="./em_vectors")
collection = client.get_collection(name="emergency_medicine_textbook")

# 2. Connect to Groq
ai_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_thread():
    # A list of broad topics to keep the AI focused but varied
    topics = ["Pediatric Resuscitation", "Traumatic Brain Injury", "Toxicology Pearls", 
              "EKG Interpretation", "Ultrasound in Shock", "OBGYN Emergencies", "Airway Management"]
    
    selected_topic = random.choice(topics)

    # Search textbook for 3 relevant snippets
    results = collection.query(query_texts=[selected_topic], n_results=3)
    context = " ".join(results['documents'][0])

    prompt = f"""
    You are a medical education bot. Based on this textbook info: {context}
    
    Create a Reddit-style thread.
    Return ONLY a JSON object with this exact structure:
    {{
      "title": "Title here",
      "body": "Main post content here",
      "category": "Clinical Category",
      "comments": [
        {{"user": "Senior_Attending", "text": "...", "color": "purple"}},
        {{"user": "Resident_Life", "text": "...", "color": "green"}},
        {{"user": "Mnemonic_Master", "text": "...", "color": "yellow"}}
      ]
    }}
    Keep the tone conversational but highly educational. Use medical jargon.
    """

    response = ai_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

# Generate and print the JSON
print(generate_thread())
