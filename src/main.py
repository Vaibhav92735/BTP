import google.generativeai as genai
import json
import os
import time
import itertools
import logging
from dotenv import load_dotenv
from groq import Groq

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(dotenv_path='../.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GEMINI_API_KEY or not GROQ_API_KEY:
    raise ValueError("One or more API keys not found. Please set GEMINI_API_KEY and GROQ_API_KEY environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# Use GenerationConfig for JSON mode in Gemini
generation_config = genai.GenerationConfig(
    response_mime_type="application/json"
)
gemini_model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config=generation_config
)

groq_client = Groq(api_key=GROQ_API_KEY)
groq_model = "llama-3.3-70b-versatile"  # Using Llama3-70B via Groq
qwen_model = "qwen/qwen3-32b"  # Using Qwen via Groq

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

# --- API CALL FUNCTIONS FOR EACH LLM ---
def call_api_with_retry(call_func, retries=3, delay=5):
    """Generic retry mechanism for API calls."""
    for attempt in range(retries):
        try:
            data = call_func()
            if "prompts" in data and "inscriptions" in data:
                return data
            else:
                logging.warning(f"Attempt {attempt + 1}: JSON response missing required keys.")
        except (ValueError, json.JSONDecodeError) as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}.")
        
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))  # Exponential backoff
            
    logging.error(f"All {retries} attempts failed.")
    return None

def generate_with_gemini(prompt):
    def call():
        response = gemini_model.generate_content(prompt)
        return json.loads(response.text)
    return call_api_with_retry(call)

def generate_with_groq(prompt, model=groq_model):
    def call():
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    return call_api_with_retry(call)

def generate_with_qwen(prompt):
    return generate_with_groq(prompt, model=qwen_model)

# --- PROMPT CREATION FUNCTIONS ---
def create_meta_prompt(language, scenario, length_category, length_desc, quantity, variation, background, layout):
    """Constructs the initial prompt for the generative model."""
    return f"""
    You are a creative prompt engineer for an advanced text-to-image AI.
    Your mission is to generate exactly 20 imaginative prompts based on a set of rules.
    
    RULES FOR THIS PROMPT:
    - Language for Text: `{language}`
    - Scenario Idea: `{scenario}`
    - Background Style: `{background}`
    - Text Layout Style: `{layout}`
    - Number of Text Snippets: `{quantity}`
    - Length of Each Snippet: `{length_category}` ({length_desc})
    - Text Variation/Style: `{variation}`
    
    OUTPUT FORMAT:
    You MUST respond with a single, valid JSON object with two keys: "prompts" and "inscriptions". 
    Also make sure the prompt is in english only but the text should be in the given language.
    - "prompts": A JSON array of 20 strings (the full image generation prompts).
    - "inscriptions": A JSON array of 20 strings (the exact text content to be rendered in the image).
    
    Example:
    {{
      "prompts": [
        "Create an image of a neon sign that says 'Open 24/7' in a rainy, futuristic city.",
        "Generate a photo of a wooden crate stamped with the words 'TOP SECRET' in bold, red letters."
      ],
      "inscriptions": [
        "Open 24/7",
        "TOP SECRET"
      ]
    }}
    
    Now, generate the 20 prompts and their corresponding inscriptions based on the rules.
    """

def create_judge_prompt(previous_output, rules_prompt):
    """Constructs the prompt for judge LLMs."""
    return f"""
    You are an LLM acting as a judge for generated prompts and inscriptions.
    Review the following output and ensure it strictly follows the original rules.
    Correct any errors, improve creativity and quality, ensure prompts are in English only,
    inscriptions are in the specified language, match the required quantity, length, variation, etc.
    Do not change the number of items (must be exactly 20).
    
    Original Rules:
    {rules_prompt}
    
    Previous Output to Judge:
    {previous_output}
    
    OUTPUT FORMAT:
    Respond with a single, valid JSON object with two keys: "prompts" and "inscriptions", 
    containing the modified/corrected versions.
    """

# --- MAIN GENERATION FUNCTION ---
def iterative_generate_prompts(meta_prompt, retries=3):
    """Uses three LLMs iteratively: Generate with Gemini, Judge/Modify with Groq Llama, Final Judge/Modify with Groq Qwen."""
    # Step 1: Initial generation with Gemini
    initial_data = generate_with_gemini(meta_prompt)
    if not initial_data:
        return None, None
    previous_output = json.dumps(initial_data)

    # Step 2: First judge with Groq Llama
    judge1_prompt = create_judge_prompt(previous_output, meta_prompt)
    judged1_data = generate_with_groq(judge1_prompt)
    if not judged1_data:
        logging.warning("Groq Llama judge failed; falling back to initial.")
        judged1_data = initial_data
    previous_output = json.dumps(judged1_data)

    # Step 3: Second judge with Groq Qwen
    judge2_prompt = create_judge_prompt(previous_output, meta_prompt)
    final_data = generate_with_qwen(judge2_prompt)
    if not final_data:
        logging.warning("Groq Qwen judge failed; falling back to Groq Llama version.")
        final_data = judged1_data

    if "prompts" in final_data and "inscriptions" in final_data:
        return final_data["prompts"], final_data["inscriptions"]
    return None, None

# --- MAIN GENERATION LOOP ---
def generate_dataset():
    """Main function to generate the entire dataset."""
    total_new_prompts = 0

    for lang in LANGUAGES:
        lang_prompts = []
        filename = f"{lang.replace(' ', '_').lower()}_prompts.json"
        logging.info(f"--- Starting Language: {lang} ---")

        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    lang_prompts = json.load(f)
                logging.info(f"Loaded {len(lang_prompts)} existing prompt combinations from {filename}.")
            except (json.JSONDecodeError, FileNotFoundError):
                logging.warning(f"Could not read or parse {filename}. Starting fresh.")
                lang_prompts = []
        
        processed_combos = {
            (p['language'], p['text_length_category'], p['text_quantity'], p['scenario'],
             p['text_variation'], p['background_type'], p['layout_style'])
            for p in lang_prompts
        }

        # Use itertools.product for cleaner iteration
        combinations = itertools.product(
            TEXT_LENGTHS.items(), TEXT_QUANTITIES, SCENARIOS,
            TEXT_VARIATIONS, BACKGROUNDS, LAYOUTS
        )

        for (length_cat, length_desc), quantity, scenario, variation, background, layout in combinations:
            current_combo = (lang, length_cat, quantity, scenario, variation, background, layout)
            
            if current_combo in processed_combos:
                continue

            logging.info(f"Processing new combo: {lang}, {length_cat}, Qty {quantity}, {variation}...")
            
            meta_prompt = create_meta_prompt(lang, scenario, length_cat, length_desc, quantity, variation, background, layout)
            prompt_texts_list, inscriptions_list = iterative_generate_prompts(meta_prompt)

            if prompt_texts_list and inscriptions_list:
                prompt_object = {
                    "language": lang,
                    "text_length_category": length_cat,
                    "text_quantity": quantity,
                    "scenario": scenario,
                    "text_variation": variation,
                    "background_type": background,
                    "layout_style": layout,
                    "prompt_text": prompt_texts_list,
                    "inscriptions": inscriptions_list
                }
                lang_prompts.append(prompt_object)
                total_new_prompts += len(prompt_texts_list)

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(lang_prompts, f, indent=2, ensure_ascii=False)
                logging.info(f"✔ Progress saved. Total prompt objects for {lang}: {len(lang_prompts)}.")
            else:
                logging.error("✗ Prompt generation failed for this combo after retries.")
            
            time.sleep(1)  # Rate limit between successful calls

    logging.info(f"\n--- DONE! Total new prompts generated across all files: {total_new_prompts} ---")

if __name__ == "__main__":
    generate_dataset()