import os
import google.generativeai as genai
import random
import re 

limit=5
memory = []
emotions = [
    "Shock and Disbelief",
    "Fear and Anxiety",
    "Anger and Resentment",
    "Sadness and Grief",
    "Guilt and Shame",
    "Processing and Healing",
    "Confusion and Disorientation",
    "Numbness and Detachment",
    "Resentment and Bitterness",
    "Relief and Gratitude",
    "Hope and Optimism",
    "Strength and Resilience",
    "Forgiveness and Compassion"
]

def extract_prompt(text):
    pattern = r'<prompt>(.*?)</prompt>' 
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
default_api_key="AIzaSyCH_FnUm53e_9nxGuLY_0inlAksB8a9jnk"
default_model="gemini-1.5-flash-8b-exp-0827"
default_prompt="Generate a prompt for image generator. I want to edit this image to make it represent the emotion of {emotion}"
class MagifactoryPromptGenerator:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": default_api_key, "multiline": False}),
                "model": ("STRING", {"default": default_model, "multiline": False}),
                "template": ("STRING", {"default": default_prompt, "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "text/generation"
    
    def generate_prompt(self, api_key=default_api_key, model=default_model, template=default_prompt,seed=None):
        # Use the seed to initialize the random number generator
        random.seed(seed)
        memory_str = ", ".join(memory)
        emotion=random.choice(emotions)
        input_str=template.format(emotion=emotion)
        input_prompt = f"{input_str} You already created those prompts: {memory_str}. Make sure to make an original prompt. Think about it step by step and make some internal critique. Final prompt is encapsulated in <prompt> tags"
        print(f"Input: {input_str}\nMemory: {memory_str}\nGenerating...")
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(input_prompt)
            generated_prompt = response.text.strip()
            print("-------------")
            print(generated_prompt)
            memory.append(generated_prompt)
            if len(memory) > limit:
                del memory[0]
            display_text = f"Seed: {seed}\n\nGenerated Prompt:\n{generated_prompt}"
            extracted_prompt = extract_prompt(generated_prompt)
            print(extracted_prompt)
            print("-------------")

            return {
                "prompt":extracted_prompt,
                "emotion":emotion,
                "input":input_prompt
                }
        except Exception as e:
            error_message = f"Error: Failed to generate prompt. Please check your API key and model name. Details: {str(e)}"
            print(error_message)
            return {
                "error":error_message,
                "emotion":emotion,
                "input": input_prompt
            }

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MagifactoryPromptGenerator": MagifactoryPromptGenerator
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagifactoryPromptGenerator": "Magifactory Prompt Generator"
}

