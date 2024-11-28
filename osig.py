import base64
from openai import OpenAI
import pytesseract
from PIL import Image
import os
import streamlit as st

# Input your OpenAI API key and instantiate the client
api_key = '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'
if not api_key:
    raise ValueError("OpenAI API key not found.")
client = OpenAI(api_key=api_key)

# Function to calculate API cost
def calculate_api_cost(prompt_tokens, completion_tokens):
    """Calculate the cost of the API call."""
    cost_per_million_input = 0.150
    cost_per_million_output = 0.600

    # Calculate costs
    input_cost = prompt_tokens * (cost_per_million_input / 1_000_000)
    output_cost = completion_tokens * (cost_per_million_output / 1_000_000)
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# Function to process the image with GPT-4o
def process_image_with_gpt4o(image_base64, client):
    """Process the uploaded image using GPT-4o API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Ti si stručnjak za osiguranja. Napravi sljedeće korake: 1. Glumi da si OCR i izvuci tekst iz slike. 2. Napiši u postotku s kojom sigurnošću je tekst točan. 3. Pročitaj tekst i provjeri da li je korisnik učitao policu osiguranja, a) ako se ne radi o polici osiguranja napiši 'Polica nije valjana' i prekinu izvođenje, b) ako se radi o polici osiguranja onda prikaži najvažnije podatke u JSON formatu."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=10424,
        )
        # Extract the main response content
        content = response.choices[0].message.content

        # Extract token usage details
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        # Return response content and token usage
        return {
            "content": content,
            "token_usage": token_usage
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("Obrada police osiguranja")
    
    # Upload an image
    uploaded_file = st.file_uploader("Uplodaj sliku police osiguranja...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert image to Base64
        image_data = uploaded_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Process the image using GPT-4o
        with st.spinner("Obrada police.."):
            result = process_image_with_gpt4o(image_base64, client)

        # Display the result
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.write("Rezultat:")
            st.write(result["content"])
            
            # Display token usage
            token_usage = result["token_usage"]
            st.write("\nToken Usage:")
            st.write(f"Prompt Tokens: {token_usage['prompt_tokens']}")
            st.write(f"Completion Tokens: {token_usage['completion_tokens']}")
            st.write(f"Total Tokens: {token_usage['total_tokens']}")
            
            # Calculate and display cost
            cost = calculate_api_cost(token_usage["prompt_tokens"], token_usage["completion_tokens"])
            st.write("\nTroškovi OpenAI API-a:")
            st.write(f"Obrada slike: ${cost['input_cost']:.6f}")
            st.write(f"Ispis rezultata: ${cost['output_cost']:.6f}")
            st.write(f"Ukupni trošak: ${cost['total_cost']:.6f}")

if __name__ == "__main__":
    main()
