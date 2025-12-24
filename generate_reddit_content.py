import os
import random
import json
import requests
import chromadb
from groq import Groq

# --- 1. CONFIGURATION & KEYS ---
# These are pulled from your GitHub Secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GCS_API_KEY = os.environ.get("GCS_API_KEY")
GCS_CX = os.environ.get("GCS_CX")

# Setup AI Clients
ai_client = Groq(api_key=GROQ_API_KEY)
# Connect to your unzipped textbook database
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
    # Topics to choose from (7 posts to fill the screen)
possible_topics = [
        # SECTION 1 & 2: Prehospital & Disaster
        "Emergency Medical Services Systems", "Mass Gathering Medical Care", "Disaster Preparedness", "Bioterrorism",
        
        # SECTION 3 & 4: Resuscitation & Procedures
        "Sudden Cardiac Death", "Approach to Nontraumatic Shock", "Approach to Traumatic Shock", "Anaphylaxis Management",
        "Acid-Base Disorders", "Electrolyte Emergencies", "Cardiac Rhythm Disturbances", "Vasopressors and Inotropes",
        "Noninvasive Airway Management", "Tracheal Intubation", "Mechanical Ventilation in the ED", "Surgical Airway Management",
        "Hemodynamic Monitoring", "Cardiac Pacing", "Defibrillation and Cardioversion",
        
        # SECTION 5 & 6: Analgesia & Wounds
        "Acute Pain Management", "Procedural Sedation", "Local and Regional Anesthesia", "Wound Preparation and Closure",
        "Face and Scalp Lacerations", "Hand and Wrist Lacerations", "Soft Tissue Foreign Bodies", "Puncture Wounds and Bites",
        
        # SECTION 7: Cardiovascular
        "Chest Pain", "Acute Coronary Syndromes", "Cardiogenic Shock", "Low-Probability ACS", "Syncope", 
        "Acute Heart Failure", "Valvular Emergencies", "Cardiomyopathies and Pericardial Disease", 
        "Venous Thromboembolism", "Pulmonary Embolism", "Systemic Hypertension", "Aortic Dissection", "Aneurysms",
        
        # SECTION 8: Pulmonary
        "Respiratory Distress", "Hemoptysis", "Acute Asthma and Status Asthmaticus", "COPD Exacerbation",
        "Community-Acquired Pneumonia", "Pneumothorax", "Acute Respiratory Distress Syndrome",
        
        # SECTION 9: GI
        "Acute Abdominal Pain", "Nausea and Vomiting", "Upper GI Bleeding", "Lower GI Bleeding", "Esophageal Emergencies",
        "Peptic Ulcer Disease", "Appendicitis", "Diverticulitis", "Bowel Obstruction", "Hernias", "Anorectal Disorders",
        "Jaundice", "Acute Cholecystitis", "Acute Pancreatitis", "Complications of Cirrhosis",
        
        # SECTION 10: GU
        "Acute Urinary Retention", "Urinary Tract Infections", "Kidney Stones", "Male Genital Emergencies",
        
        # SECTION 11: OBGYN
        "Ectopic Pregnancy", "Vaginal Bleeding in Early Pregnancy", "Complications of Late Pregnancy", "Emergency Delivery",
        "Postpartum Emergencies", "Pelvic Inflammatory Disease", "Sexual Assault Management",
        
        # SECTION 12: Pediatrics
        "Pediatric Airway Management", "Pediatric Resuscitation", "Neonatal Emergencies", "Pediatric Fever",
        "Pediatric Respiratory Distress", "Pediatric Seizures", "Pediatric GI Disorders", "Pediatric Orthopedics",
        "Child Abuse and Neglect",
        
        # SECTION 13: Infectious Disease
        "Sepsis and Septic Shock", "Soft Tissue Infections", "Sexually Transmitted Infections", "HIV Emergencies",
        "Meningitis", "Tick-Borne Illnesses", "Tuberculosis",
        
        # SECTION 14: Neurology
        "Headache", "Ischemic Stroke", "Intracranial Hemorrhage", "Seizures and Status Epilepticus", "Vertigo and Dizziness",
        "Altered Mental Status and Coma", "Spinal Cord Compression",
        
        # SECTION 15: Toxicology
        "General Approach to the Poisoned Patient", "Anticholinergic Toxicity", "Opioid Overdose", "Acetaminophen Toxicity",
        "Salicylate Toxicity", "Toxic Alcohols", "Beta Blocker and Calcium Channel Blocker Toxicity", "Cocaine and Stimulants",
        
        # SECTION 16: Environmental
        "Frostbite and Hypothermia", "Heat-Related Illness", "Bites and Stings", "Drowning", "Electrical and Lightning Injuries",
        "High-Altitude Medicine", "Carbon Monoxide Poisoning",
        
        # SECTION 17: Endocrine
        "Diabetic Ketoacidosis", "Hyperosmolar Hyperglycemic State", "Hypoglycemia", "Thyroid Storm", "Adrenal Insufficiency",
        
        # SECTION 19: ENT/EYE
        "Acute Angle-Closure Glaucoma", "Retinal Detachment", "Epistaxis", "Otitis Media and Externa", "Pharyngitis",
        
        # SECTION 21 & 22: Trauma & Ortho
        "Trauma in Adults", "Trauma in Pregnancy", "Head Trauma", "Spine Trauma", "Thoracic Trauma", "Abdominal Trauma",
        "Genitourinary Trauma", "Burn Injuries", "Fractures and Dislocations", "Pelvic Fractures", "Compartment Syndrome"
    ]
    # We pick 7 unique topics from the 100+ items above
    selected_topics = random.sample(possible_topics, 7)
    
    all_posts = []

    for i, topic in enumerate(selected_topics):
        print(f"Generating post {i+1}/7: {topic}...")
               
        # 1. Get Context from Textbook
        results = collection.query(query_texts=[topic], n_results=3)
        context = " ".join(results['documents'][0])

        # 2. Build the AI Prompt
        # We pass the entire PERSONAS list so the AI knows who to cast
        prompt = f"""
        CONTEXT FROM TEXTBOOK: {context}
        TOPIC: {topic}
        
        TASK: Create a Reddit-style thread for r/EmergencyMedicine.
        
        REQUIREMENTS:
        1. TITLE: Catchy and professional.
        2. BODY: Detailed explanation of the topic based on the context.
        3. COMMENTS: Exactly 20 comments. 
        4. CAST: Use the following 20 characters. Each must speak in their specific style:
        {json.dumps(PERSONAS)}

        RULES:
        - Only MDs/Residents get "Dr." prefix. 
        - The MS1 should be naive, the 1930s doc should mention "leeches" or "bloodletting", the surgeon should be suspicious.
        - Some comments should be replies to other comments (indented logic).
        
        OUTPUT: Return ONLY a JSON object:
        {{
          "title": "post title",
          "body": "post body text",
          "category": "TRAUMA/CARDIAC/etc",
          "comments": [
            {{ "user": "Name", "role": "Suffix", "city": "City", "text": "comment content", "color": "purple/green/yellow/red/blue/orange" }}
          ]
        }}
        """

        try:
            response = ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            
            post_data = json.loads(response.choices[0].message.content)

            # 3. Add Image every 3rd or 4th post
            if i % 3 == 0:
                post_data['image'] = get_google_image(topic)
            else:
                post_data['image'] = None

            all_posts.append(post_data)
        except Exception as e:
            print(f"Error generating post {topic}: {e}")

    # --- 5. SAVE TO FILE ---
    # This writes the final file that your index.html reads
    with open("new_post.json", "w") as f:
        json.dump(all_posts, f, indent=2)
    print("Successfully generated new_post.json with 7 threads.")

if __name__ == "__main__":
    generate_full_site()
