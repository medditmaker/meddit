from datetime import datetime, timezone
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
    {"name": "Dr. Aris Thorne", "city": "Toronto, ON", "role": "Staff Physician", "style": "Wise and kind attending. Focuses on patient-centered care and high-level pearls."},
    {"name": "Dr. Victor Sterling", "city": "Calgary, AB", "role": "Chief of Surgery", "style": "Condescending and arrogant. Always thinks the ER is over-diagnosing. Uses 'Amateurs'."},
    {"name": "Dr. 'Ziggy' Zalesky", "city": "Vancouver, BC", "role": "EP", "style": "Funny community physician. Quick-witted, uses ER slang, very practical."},
    {"name": "Dr. Barb Miller", "city": "Winnipeg, MB", "role": "EP", "style": "Jaded community physician. Cynical about the system, but knows her medicine."},
    {"name": "Leo Chen", "city": "Montreal, QC", "role": "MS1", "style": "Naive first year med student. Asks basic textbook questions. Overly enthusiastic."},
    {"name": "Sarah Jenkins", "city": "Ottawa, ON", "role": "MS3", "style": "Know-it-all third year med student. Cites specific papers but lacks clinical 'feel'."},
    {"name": "Dr. Kevin Park", "city": "Edmonton, AB", "role": "PGY1", "style": "Nervous intern. Worried about missing the 'big one'. Very formal."},
    {"name": "Dr. Maya Gupta", "city": "Mississauga, ON", "role": "PGY2", "style": "Confident and fast resident. Loves procedures and quick dispos."},
    {"name": "Dr. Henry Oldman", "city": "Victoria, BC", "role": "CCFP", "style": "Wise family physician. Focuses on the long-term history and 'old school' physical exam."},
    {"name": "Dr. Brock Stone", "city": "Hamilton, ON", "role": "FRCSC", "style": "Suspicious surgeon. Always looking for reasons NOT to admit a patient."},
    {"name": "Dr. Elena Varkov", "city": "Quebec City, QC", "role": "FRCPC", "style": "Exhausted internist. Wants to know the 'Sodium and the story' before admitting."},
    {"name": "Turbo Tom", "city": "London, ON", "role": "RN", "style": "Hilarious ER nurse. Sees through the doctor's ego. Very efficient."},
    {"name": "Martha 'The Hammer'", "city": "Halifax, NS", "role": "RN", "style": "Jaded ER nurse. No-nonsense, 30 years experience, scares the residents."},
    {"name": "Chad Broson", "city": "Brampton, ON", "role": "Pre-med", "style": "Funny, a bit dumb, obsessed with getting into med school."},
    {"name": "Barnaby Finch", "city": "Surrey, BC", "role": "Patient", "style": "Quixotic, poetic, and slightly confused about his symptoms."},
    {"name": "Dr. Sarah Sharp", "city": "Laval, QC", "role": "EP", "style": "Industrious, evidence-based, efficient, and direct."},
    {"name": "Dr. Archibald Thorne", "city": "Saskatoon, SK", "role": "Physician", "style": "Physician from the 1930s. Uses terms like 'apoplexy' and 'liniment'."},
    {"name": "Dr. Rick 'The Rage'", "city": "Windsor, ON", "role": "EP", "style": "Angry, burnt out, hates hospital admin."},
    {"name": "Dr. Phil Muddle", "city": "Regina, SK", "role": "EP", "style": "Generally confused, asks what day it is, but lucky with diagnoses."},
    {"name": "Sparkle-Eyes Sue", "city": "St. John's, NL", "role": "LPN", "style": "Extremely cheerful, talks in Newfoundland slang, very kind."}
]

# --- 3. MASTER TINTINALLI TOPIC LIST (100+) ---
POSSIBLE_TOPICS = [
    "Emergency Medical Services Systems", "Mass Gathering Medical Care", "Disaster Preparedness", "Bioterrorism",
    "Sudden Cardiac Death", "Approach to Nontraumatic Shock", "Approach to Traumatic Shock", "Anaphylaxis Management",
    "Acid-Base Disorders", "Electrolyte Emergencies", "Cardiac Rhythm Disturbances", "Vasopressors and Inotropes",
    "Noninvasive Airway Management", "Tracheal Intubation", "Mechanical Ventilation in the ED", "Surgical Airway Management",
    "Acute Pain Management", "Procedural Sedation", "Local and Regional Anesthesia", "Wound Preparation and Closure",
    "Face and Scalp Lacerations", "Hand and Wrist Lacerations", "Soft Tissue Foreign Bodies", "Puncture Wounds and Bites",
    "Chest Pain", "Acute Coronary Syndromes", "Cardiogenic Shock", "Low-Probability ACS", "Syncope", 
    "Acute Heart Failure", "Valvular Emergencies", "Cardiomyopathies and Pericardial Disease", 
    "Venous Thromboembolism", "Pulmonary Embolism", "Systemic Hypertension", "Aortic Dissection", "Aneurysms",
    "Respiratory Distress", "Hemoptysis", "Acute Asthma and Status Asthmaticus", "COPD Exacerbation",
    "Community-Acquired Pneumonia", "Pneumothorax", "Acute Respiratory Distress Syndrome",
    "Acute Abdominal Pain", "Nausea and Vomiting", "Upper GI Bleeding", "Lower GI Bleeding", "Esophageal Emergencies",
    "Peptic Ulcer Disease", "Appendicitis", "Diverticulitis", "Bowel Obstruction", "Hernias", "Anorectal Disorders",
    "Jaundice", "Acute Cholecystitis", "Acute Pancreatitis", "Complications of Cirrhosis",
    "Acute Urinary Retention", "Urinary Tract Infections", "Kidney Stones", "Male Genital Emergencies",
    "Ectopic Pregnancy", "Vaginal Bleeding in Early Pregnancy", "Complications of Late Pregnancy", "Emergency Delivery",
    "Postpartum Emergencies", "Pelvic Inflammatory Disease", "Sexual Assault Management",
    "Pediatric Airway Management", "Pediatric Resuscitation", "Neonatal Emergencies", "Pediatric Fever",
    "Pediatric Respiratory Distress", "Pediatric Seizures", "Pediatric GI Disorders", "Pediatric Orthopedics",
    "Child Abuse and Neglect", "Sepsis and Septic Shock", "Soft Tissue Infections", "Sexually Transmitted Infections",
    "Meningitis", "Tick-Borne Illnesses", "Tuberculosis", "Headache", "Ischemic Stroke", "Intracranial Hemorrhage",
    "Seizures and Status Epilepticus", "Vertigo and Dizziness", "Altered Mental Status and Coma", "Spinal Cord Compression",
    "General Approach to the Poisoned Patient", "Anticholinergic Toxicity", "Opioid Overdose", "Acetaminophen Toxicity",
    "Salicylate Toxicity", "Toxic Alcohols", "Beta Blocker and Calcium Channel Blocker Toxicity", "Cocaine and Stimulants",
    "Frostbite and Hypothermia", "Heat-Related Illness", "Bites and Stings", "Drowning", "Electrical and Lightning Injuries",
    "Carbon Monoxide Poisoning", "Diabetic Ketoacidosis", "Hyperosmolar Hyperglycemic State", "Hypoglycemia", "Thyroid Storm",
    "Acute Angle-Closure Glaucoma", "Retinal Detachment", "Epistaxis", "Otitis Media and Externa", "Pharyngitis",
    "Trauma in Adults", "Trauma in Pregnancy", "Head Trauma", "Spine Trauma", "Thoracic Trauma", "Abdominal Trauma",
    "Genitourinary Trauma", "Burn Injuries", "Fractures and Dislocations", "Pelvic Fractures", "Compartment Syndrome"
]

# --- 4. IMAGE SEARCH ---
def get_image(query):
    if not GCS_API_KEY or not GCS_CX: return None
    url = f"https://www.googleapis.com/customsearch/v1?q={query}+emergency+medicine&searchType=image&key={GCS_API_KEY}&cx={GCS_CX}&safe=active"
    try:
        r = requests.get(url, timeout=5).json()
        return r['items'][0]['link']
    except: return None

# --- 5. GENERATOR ---
def generate_full_site():
    selected_topics = random.sample(POSSIBLE_TOPICS, 10)
    all_posts = []

    for i, topic in enumerate(selected_topics):
        print(f"Generating post {i+1}/10: {topic}...")
        results = collection.query(query_texts=[topic], n_results=3)
        context = " ".join(results['documents'][0])

        prompt = f"""
        TEXTBOOK CONTEXT: {context}
        TOPIC: {topic}
        CAST OF 20 PERSONAS: {json.dumps(PERSONAS)}

        TASK: Create a highly realistic Reddit thread.
        1. THE POST: Write as a clinical ANECDOTE, CASE STUDY, or a frustrated "Today I Learned" or other typical reddit post style. 
           Cab start with things like "Just saw a patient with...", "Had a miss last night...", or "Rant about...".  Can also just post generally discussing a topic.
           Are written in a high level targetted at practicing emergency physicians.
        2. THE COMMENTS: 10 - 30 comments. 
           - MUST BE THREADED: Use the "id" and "parent_id" fields. 
           - Some comments reply to the post (parent_id: null).
           - A parent comment can have multiple responses.
        3. CONTENT: The content is intended for experienced emergency physicians as a CME review.  Use high level medical language.  Comments should be long enough (ie. 200-500 words) to explore a topic in high level detail.  The parent comments are typically longer.  Child comments ask questions related to the parent comment to clarify detail.  
        The topic should be covered in full detail by all of the comments combined.  Any acronyms are clarified once.  They can use internet jargon (ie. lol, TLDR:, etc), write in a semi casual tone, be inappropriate sometimes, professional tone more often.  

        JSON FORMAT:
        {{
          "id": {i},
          "title": "...",
          "body": "...",
          "category": "...",
          "comments": [
             {{ "id": 1, "parent_id": null, "user": "Name", "role": "Suffix", "city": "City", "text": "...", "color": "purple" }},
             {{ "id": 2, "parent_id": 1, "user": "Name", "role": "Suffix", "city": "City", "text": "...", "color": "green" }}
          ]
        }}
        """

        try:
            response = ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            if i % 3 == 0: data['image'] = get_image(topic)
            all_posts.append(data)
        except Exception as e: print(f"Error: {e}")
    
    # Create a wrapper object that includes the timestamp
    final_data = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "posts": all_posts
    }

    with open("new_post.json", "w") as f:
        json.dump(final_data, f, indent=2)
    with open("new_post.json", "w") as f:
        json.dump(all_posts, f, indent=2)

if __name__ == "__main__":
    generate_full_site()
