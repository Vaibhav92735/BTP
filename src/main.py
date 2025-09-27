import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv(dotenv_path='../.env') 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- DATASET STRUCTURE DEFINITION ---
LANGUAGES = ["English", "Mandarin Chinese", "Hindi", "Spanish", "French", "Hinglish", "Spanglish", "Franglais"]
TEXT_LENGTHS = {
    "Microcopy/Label": "1-2 words",
    "Headline/Title": "3-6 words",
    "Tagline/CTA": "7-12 words",
    "Short Caption/Quote": "13-25 words"
}
TEXT_QUANTITIES = [1, 2, 3, 4, 5]
SCENARIOS = [
    "Signboards & Billboards", "Shop Signage & Display", "Full-Scene Messages",
    "Documents & Print Media", "Digital Screens", "Product Packaging",
    "Public Spaces", "Official Documents", "Creative/Artistic"
]
TEXT_VARIATIONS = [
    "Correct Spelling", "Misspelled", "Gibberish/Non-words",
    "Special Characters & Numbers", "Case Variations", "Rare/Long Words"
]
BACKGROUNDS = ["Complex Background", "Isolated/Clear Background"]
LAYOUTS = ["Uniform Font and Style", "Multiple Fonts/Styles"]

PROMPTS_PER_COMBINATION = 20

import json

def generate_complete_prompt(language, scenario, length_category, length_desc, quantity, variation, background, layout):
    print(f"  > Generating 20 prompts for: {language}, {scenario}, {variation}...")

    meta_prompt = f"""
    You are a creative prompt engineer for an advanced text-to-image AI.
    Your mission is to generate exactly 20 imaginative prompts based on a set of rules. Start each prompt with an action word like Create, Generate, Imagine, etc.

    RULES FOR THIS PROMPT:
    -   Language for Text: `{language}`
    -   Scenario Idea: `{scenario}`
    -   Background Style: `{background}`
    -   Text Layout Style: `{layout}`
    -   Number of Text Snippets: `{quantity}`
    -   Length of Each Snippet: `{length_category}` ({length_desc})
    -   Text Variation/Style: `{variation}`

    OUTPUT FORMAT:
    You MUST respond with a single, valid JSON object. This object should contain one key, "prompts", whose value is a JSON array (list) of 20 strings.

    **Correct Example Output Structure:**
    {{
      "prompts": [
        "Create an image of a neon sign that says 'Open 24/7' in a rainy, futuristic city.",
        "Generate a photo of a wooden crate stamped with the words 'TOP SECRET' in bold, red letters.",
        "Imagine a birthday cake with 'Happy Birthday, Alex' written in elegant cursive frosting."
      ]
    }}

    Now, generate the 20 prompts based on the rules provided.
    """
    try:
        response = model.generate_content(meta_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_response)
        return data.get("prompts")
        
    except (json.JSONDecodeError, AttributeError, Exception) as e:
        print(f"  ! API Error or JSON parsing failed: {e}. Returning None.")
        print(f"  ! Faulty Response Text: {response.text}")
        return None

# --- MAIN GENERATION LOOP ---
def generate_dataset():
    """Main function to generate the entire dataset."""
    total_prompts_generated = 0

    for lang in LANGUAGES[:2]:
        lang_prompts = []
        filename = f"{lang.replace(' ', '_').lower()}_prompts.json"
        print(f"\n--- Starting Language: {lang} ---")

        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    lang_prompts = json.load(f)
                print(f"  > Loaded {len(lang_prompts)} existing prompt combinations from {filename}.")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"  ! Warning: Could not read or parse {filename}. Starting fresh for this language.")
                lang_prompts = []

        processed_combos = {
            (p['language'], p['text_length_category'], p['text_quantity'], p['scenario'],
             p['text_variation'], p['background_type'], p['layout_style'])
            for p in lang_prompts
        }

        for length_cat, length_desc in list(TEXT_LENGTHS.items())[:1]:
            for quantity in TEXT_QUANTITIES[:1]:
                for scenario in SCENARIOS[:1]:
                    for variation in TEXT_VARIATIONS[:2]:
                        for background in BACKGROUNDS:
                            for layout in LAYOUTS:
                                current_combo = (lang, length_cat, quantity, scenario, variation, background, layout)

                                if current_combo in processed_combos:
                                    print(f"  > Skipping already processed combo: {variation}, {background}, {layout}")
                                    continue

                                print(f"\nProcessing new combo: {lang}, {length_cat}, Qty {quantity}, {scenario}, {variation}, {background}, {layout}")

                                prompt_texts_list = generate_complete_prompt(lang, scenario, length_cat, length_desc, quantity, variation, background, layout)

                                if prompt_texts_list:
                                    prompt_object = {
                                        "language": lang,
                                        "text_length_category": length_cat,
                                        "text_quantity": quantity,
                                        "scenario": scenario,
                                        "text_variation": variation,
                                        "background_type": background,
                                        "layout_style": layout,
                                        "prompt_text": prompt_texts_list
                                    }
                                    lang_prompts.append(prompt_object)
                                    total_prompts_generated += len(prompt_texts_list)

                                    with open(filename, 'w', encoding='utf-8') as f:
                                        json.dump(lang_prompts, f, indent=2, ensure_ascii=False)
                                    print(f"  ✔ Progress saved. Total prompt objects for {lang}: {len(lang_prompts)}.")
                                else:
                                    print("    ✗ Prompt generation failed for this combo. Waiting to retry...")
                                    time.sleep(5)

                                time.sleep(1)

        print(f"\n--- Finished language: {lang}. Total prompt objects: {len(lang_prompts)} ---")

    print(f"\n--- DONE! Total new prompts generated across all files: {total_prompts_generated} ---")

if __name__ == "__main__":
    generate_dataset()