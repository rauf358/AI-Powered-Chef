import streamlit as st
import os
import requests
import io
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()  # Load environment variables from a .env file

# RecipeBot class to interact with OpenAI API for generating recipes
class RecipeBot:
    def __init__(self, model="gpt-3.5-turbo", token_budget=500):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("api_key"))  # Initialize OpenAI client
        self.token_budget = token_budget
        self.sys = ('As a MasterChef AI, you provide with recipes that include instructions for preparing delicious dishes, alongside the list of ingredients with precise measurements')
        self.convo_history = [{"role": "system", "content": self.sys}]

    def token_calculate(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)  # Get encoding for the model
        except KeyError:
            encoding = tiktoken.encoding_for_model('cl100k_base')
        tokens = encoding.encode(text)
        return len(tokens)

    def total_token_used(self):
        return sum(self.token_calculate(d["content"]) for d in self.convo_history)

    def enforce_token_limit(self):
        while self.total_token_used() > self.token_budget:
            if len(self.convo_history) <= 1:
                break
            self.convo_history.pop(1)  # Remove oldest user message if token limit exceeded

    def prompt(self, prompt, temp=0.2, maxt=200):
        self.enforce_token_limit()  # Ensure token limit is not exceeded
        self.convo_history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.convo_history,
            temperature=temp,
            max_tokens=maxt
        )
        ai_response = response.choices[0].message.content
        self.convo_history.append({"role": "assistant", "content": ai_response})


# GenImages class to interact with Hugging Face API for generating images
class GenImages:
    def __init__(self, url="https://api-inference.huggingface.co/models/Corcelio/mobius"):
        self.API_URL = url
        self.headers = {"Authorization": f"Bearer {os.getenv('hugging_face_Api')}"}

    def query(self, payload):
        try:
            response = requests.post(self.API_URL, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print("Error fetching image:", e)
            return None

    def get_image(self, ai_response):
        image_bytes = self.query({"inputs": ai_response})
        if image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                return image
            except Exception as e:
                print("Error opening image with PIL:", e)
                return None
        else:
            print("No image data returned from the API.")

# Function to get nutritional analysis of a recipe using Edamam API
def get_recipe_analysis(ingredients):
    app_id = os.getenv("edamam_app_id")
    app_key = os.getenv("edamam_api_key")
    api_endpoint = 'https://api.edamam.com/api/nutrition-details'

    params = {
        'app_id': app_id,
        'app_key': app_key,
    }

    data = {
        'title': 'Sample Recipe',
        'ingr': ingredients
    }

    response = requests.post(api_endpoint, params=params, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        return None


# Streamlit code to create the web interface


st.title("Rauf's AI-Powered Chef:“🤖👩‍🍳”")

# Initialize RecipeBot
recipe_bot = RecipeBot()

# Initial welcome message from the AI
with st.chat_message('ai'):
    st.write("Hello! I'm your MasterChef AI, here to help you explore delicious recipes from any cuisine or even your favorite anime. Just let me know what you're in the mood for, and I'll whip up the perfect recipe for you. Whether it's a traditional dish or something inspired by your favorite anime series, I've got you covered. Let's get cooking!? ")

# Prompt user for input
user_input = st.chat_input("What are we making today?")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = recipe_bot.convo_history

# Retrieve conversation history from session state
recipe_bot.convo_history = st.session_state['conversation_history']

# Generate and display AI response based on user input
if user_input is not None:
    recipe_bot.prompt(user_input)

ai_response = recipe_bot.convo_history[-1]["content"]

# Display conversation history
for d in recipe_bot.convo_history:   
    if d['role'] == "system":
        continue
    with st.chat_message(d['role']):        
        st.write(d['content'])

# Sidebar options
st.sidebar.header("Options")

# Button to show nutritional analysis
if st.sidebar.button("Show Nutrients"):
    # Get recipe analysis
    ingredients = ai_response.split('\n')
    recipe_analysis = get_recipe_analysis(ingredients)
    if recipe_analysis:
        st.write("Calories:", recipe_analysis['calories'])
        st.write("Total Weight:", recipe_analysis['totalWeight'])
        st.write("Total Nutrients:")
        for nutrient, value in recipe_analysis['totalNutrients'].items():
            st.write(nutrient + ":", value['label'], value['quantity'], value['unit'])
    else:
        st.write("Failed to get recipe analysis.")

# Button to display picture
if st.sidebar.button("Display Picture"):
    # Initialize GenImages
    image_generator = GenImages()

    # Get image for AI response
    image = image_generator.get_image(recipe_bot.convo_history[-1]["content"])
    
    st.image(image)
