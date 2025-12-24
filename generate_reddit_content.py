import os
import random
import json
import requests
import chromadb
from groq import Groq

# --- 1. CONFIGURATION & KEYS ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GCS_API_KEY = os.environ.get("GCS_API_KEY")
GCS_CX = os.environ.get("GCS_CX")

ai_client = Groq(api_key=GROQ_API_KEY)
client = chromadb.PersistentClient(path="./em_vectors")
collection = client.get_collection(name="emergency_medicine_textbook")

# --- 2. THE 20 CANADIAN PERSONAS ---
PERSONAS = [
    {"name": "Dr. Aris Thorne", "city": "Toronto, ON", "role": "Staff Physician", "style": "Wise and kind attending"},
    {"name": "Dr. Victor Sterling", "city": "Calgary, AB", "role": "Chief of Surgery", "style": "Condescending and arrogant attending"},
    {"name": "Dr. 'Ziggy' Zalesky", "city": "Vancouver, BC", "role": "EP", "style": "Funny community emergency physician"},
    {"name": "Dr. Barb Miller", "city": "Winnipeg, MB", "role": "EP", "style": "Jaded community emergency physician"},
    {"name": "Leo Chen", "city": "Montreal, QC", "role": "MS1", "style": "Naive first year medical student"},
    {"name": "Sarah Jenkins", "city": "Ottawa, ON", "role": "MS3", "style": "Know-it-all third year medical student"},
    {"name": "Dr. Kevin Park", "city": "Edmonton, AB", "role": "PGY1", "style": "Nervous first year medical resident"},
    {"name": "Dr. Maya Gupta", "city": "Mississauga, ON", "role": "PGY2", "style": "Confident second year medical resident"},
    {"name": "Dr. Henry Oldman", "city": "Victoria, BC", "role": "CCFP", "style": "Wise family physician"},
    {"name": "Dr. Brock Stone", "city": "Hamilton, ON", "role": "FRCSC", "style": "Suspicious surgeon"},
    {"name": "Dr. Elena Varkov", "city": "Quebec City, QC", "role": "FRCPC", "style": "Tired internist"},
    {"name": "Turbo Tom", "city": "London, ON", "role": "RN", "style": "Hilarious emergency nurse"},
    {"name": "Martha 'The Hammer'", "city": "Halifax, NS", "role": "RN", "style": "Jaded emergency nurse"},
    {"name": "Chad Broson", "city": "Brampton, ON", "role": "Pre-med", "style": "Funny and a bit dumb pre-med student"},
    {"name": "Barnaby Finch", "city": "Surrey, BC", "role": "Patient", "style": "Quixotic and poetic patient"},
    {"name": "Dr. Sarah Sharp", "city": "Laval, QC", "role": "EP", "style": "Industrious and efficient emergency physician"},
    {"name": "Dr. Archibald Thorne", "city": "Saskatoon, SK", "role": "Physician", "style": "Physician from the 1930s using archaic medical terms"},
    {"name": "Dr. Rick 'The Rage'", "city": "Windsor, ON", "role": "EP", "style": "Incredibly angry emergency physician"},
    {"name": "Dr. Phil Muddle", "city": "Regina, SK", "role": "EP", "style": "Confused emergency physician"},
    {"name": "Sparkle-Eyes Sue", "city": "St. John's, NL", "role": "LPN", "style": "Entertaining and cheerful nurse"}
]

# --- 3. IMAGE SEARCH FUNCTION ---
def get_google_image(query):
    if not GCS_API_KEY or not GCS_CX:
        return None
    try:
        search_query = f"{query} medical emergency"
        url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&searchType=image&key={GCS_API_KEY}&cx={GCS_CX}&safe=active"
        r = requests.get(url, timeout=5).json()
        return r['items'][0]['link']
    except:
        return None

# --- 4. MAIN GENERATION ENGINE ---
def generate_full_site():
    # Tintinalli Master Topic List
    possible_topics = [
        "Sudden Cardiac Death", "Anaphylaxis Management", "Acid-Base Disorders", 
        "Electrolyte Emergencies", "Cardiac Rhythm Disturbances", "Noninvasive Airway Management", 
        "Tracheal Intubation", "Surgical Airway Management", "Acute Pain Management", 
        "Procedural Sedation", "Chest Pain", "Acute Coronary Syndromes", "Cardiogenic Shock", 
        "Syncope", "Acute Heart Failure", "Venous Thromboembolism", "Pulmonary Embolism", 
        "Aortic Dissection", "Hemoptysis", "Acute Asthma", "COPD Exacerbation",
        "Community-Acquired Pneumonia", "Pneumothorax", "Acute Abdominal Pain", 
        "Upper GI Bleeding", "Lower GI Bleeding", "Appendicitis", "Bowel Obstruction", 
        "Acute Cholecystitis", "Acute Pancreatitis", "Kidney Stones", "Ectopic Pregnancy", 
        "Emergency Delivery", "Pediatric Resuscitation", "Neonatal Emergencies", 
        "Pediatric Fever", "Sepsis and Septic Shock", "Meningitis", "Ischemic Stroke", 
        "Intracranial Hemorrhage", "Status Epilepticus", "Opioid Overdose", 
        "Acetaminophen Toxicity", "Diabetic Ketoacidosis", "Thyroid Storm", 
        "Acute Angle-Closure Glaucoma", "Epistaxis", "Head Trauma", "Spine Trauma", 
        "Burn Injuries", "Compartment Syndrome"
    ]
    
    selected_topics = random.sample(possible_topics, 7)
    all_posts = []

    for i, topic in enumerate(selected_topics):
        print(f"Generating post {i+1}/7: {topic}...")
        results = collection.query(query_texts=[topic], n_results=3)
        context = " ".join(results['documents'][0])

        prompt = f"""
        CONTEXT: {context}
        TOPIC: {topic}
        TASK: Create a Reddit-style thread for r/EmergencyMedicine.
        REQUIREMENTS:
        1. TITLE: Catchy and professional.
        2. BODY: Detailed explanation based on the context.
        3. COMMENTS: Exactly 20 comments using these personas: {json.dumps(PERSONAS)}
        RULES: Only MDs/Residents get "Dr.". Use specific persona styles.
        OUTPUT: Return ONLY a JSON object:
        {{
          "title": "title",
          "body": "body",
          "category": "CAT",
          "comments": [{{ "user": "Name", "role": "Role", "city": "City", "text": "text", "color": "purple" }}]
        }}
        """

        try:
            response = ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            post_data = json.loads(response.choices[0].message.content)
            if i % 3 == 0:
                post_data['image'] = get_google_image(topic)
            else:
                post_data['image'] = None
            all_posts.append(post_data)
        except Exception as e:
            print(f"Error: {e}")

    with open("new_post.json", "w") as f:
        json.dump(all_posts, f, indent=2)

if __name__ == "__main__":
    generate_full_site()
