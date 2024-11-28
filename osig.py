import base64
from openai import OpenAI
import pytesseract
from PIL import Image
import os
import streamlit as st

# Read the API key from Streamlit secrets
api_key = st.secrets["api_key"]

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
def process_image_with_gpt4o(image_base64, client, model):
    """Process the uploaded image using GPT-4o API."""
    try:
        # First request: Extract text from image
        response1 = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Izvuci tekst iz slike i ispiši ih."},
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
        extracted_text = response1.choices[0].message.content

        # Display extracted text immediately
        st.session_state.extracted_text = extracted_text

        # Second request: Validate the extracted text
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Je li ovo valjana polica osiguranja? {extracted_text}. Ako je valjana polica osiguranja, ispiši 'Dokument je valjan'. i nastavi s obradom podataka. ako nije valjana polica osiguranja, ispiši 'Polica nije valjana'. I prekini bradu podataka"
                }
            ],
            max_tokens=10424,
        )
        validation_result = response2.choices[0].message.content

        # Print validation result in the sidebar
        st.sidebar.write("Validacija rezultata:")
        st.sidebar.write(validation_result)

        if "Polica nije valjana" in validation_result:
            return {"content": validation_result}

        # Third request: Extract important data in JSON format
        response3 = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Izvuci najvažnije podatke u JSON formatu iz ovog teksta: {extracted_text}"
                }
            ],
            max_tokens=10424,
        )
        important_data = response3.choices[0].message.content

        # Extract token usage details
        token_usage = {
            "prompt_tokens": response1.usage.prompt_tokens + response2.usage.prompt_tokens + response3.usage.prompt_tokens,
            "completion_tokens": response1.usage.completion_tokens + response2.usage.completion_tokens + response3.usage.completion_tokens,
            "total_tokens": response1.usage.total_tokens + response2.usage.total_tokens + response3.usage.total_tokens
        }

        # Return response content and token usage
        return {
            "important_data": important_data,
            "token_usage": token_usage
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 90%;
        }
        .sidebar .sidebar-content {
            width: 300px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.title("Obrada police osiguranja")
    
    # Sidebar for model selection and file upload
    st.sidebar.title("Postavke")
    model = st.sidebar.selectbox("Odaberi model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.sidebar.title("Uplodaj sliku")
    uploaded_file = st.sidebar.file_uploader("Uplodaj sliku police osiguranja...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert image to Base64
        image_data = uploaded_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Split the main space into two columns
        col1, col2 = st.columns(2)

        # Process the image using GPT-4o
        with st.spinner("Obrada police.."):
            result = process_image_with_gpt4o(image_base64, client, model)

        # Display the extracted text immediately
        if "extracted_text" in st.session_state:
            with col1:
                st.write("Izvučeni tekst:")
                st.write(st.session_state.extracted_text)

        # Display the result
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            if "important_data" in result:
                with col2:
                    st.write("Podaci iz police u JSON formatu:")
                    st.write(result["important_data"])
        
        if "error" not in result and "token_usage" in result:
            # Calculate and display cost
            token_usage = result["token_usage"]
            cost = calculate_api_cost(token_usage["prompt_tokens"], token_usage["completion_tokens"])
            st.sidebar.metric(label="Troškovi OpenAI API-a", value=f"${cost['total_cost']:.6f}")
            st.sidebar.metric(label="Troškovi ulaznih tokena", value=f"${cost['input_cost']:.6f}")
            st.sidebar.metric(label="Troškovi izlaznih tokena", value=f"${cost['output_cost']:.6f}")

if __name__ == "__main__":
    main()
