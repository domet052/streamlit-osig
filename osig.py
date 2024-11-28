import base64
from openai import OpenAI
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Read the API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
google_api_key = st.secrets["google_api_key"]

if not openai_api_key:
    raise ValueError("OpenAI API key not found.")
if not google_api_key:
    raise ValueError("Google API key not found.")

# Initialize OpenAI client and configure Gemini
openai_client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=google_api_key)

# Function to calculate API cost
def calculate_api_cost(prompt_tokens, completion_tokens, model):
    """Calculate the cost of the API call based on the model used."""
    # Model-specific costs per million tokens
    costs = {
        "gpt-4o-mini": {
            "input": 0.150,
            "output": 0.600
        },
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00
        },
        "gemini-1.5-flash": {
            "input": 0.35,
            "output": 1.05
        },
        "gemini-1.5-pro": {
            "input": 3.50,
            "output": 10.50
        }
    }
    
    model_costs = costs[model]
    input_cost = prompt_tokens * (model_costs["input"] / 1_000_000)
    output_cost = completion_tokens * (model_costs["output"] / 1_000_000)
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# Function to process the image with the selected model
def process_image(image_base64, model):
    """Process the uploaded image using the selected model."""
    try:
        if model in ["gpt-4o-mini", "gpt-4o"]:
            client = openai_client
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
                        "content": f"Je li ovo polica osiguranja? {extracted_text}. Ako je korisnik uplodao policu osiguranja, ispiši SAMO 'Dokument je valjan'. i nastavi s obradom podataka. ako nije valjana polica osiguranja, ispiši SAMO 'Polica nije valjana'. I prekini bradu podataka"
                    }
                ],
                max_tokens=10424,
            )
            validation_result = response2.choices[0].message.content

            # Print validation result in the sidebar with larger font
            st.sidebar.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>{validation_result}</h2>", unsafe_allow_html=True)

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
        elif model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
            # Google model processing logic
            model_client = genai.GenerativeModel(model)
            
            # First request: Extract text from image
            response1 = model_client.generate_content([
                {"mime_type": "image/jpeg", "data": image_base64},
                "Extract all text from this image and only return the extracted text without any additional commentary."
            ])
            extracted_text = response1.text

            # Second request: Validate the extracted text
            response2 = model_client.generate_content(
                f"Je li ovo valjana polica osiguranja? {extracted_text}. "
                "Ako je korisnik poslao polica osiguranja, a ne neki drugi dokument ispiši SAMO 'Ovo je polica osiguranja'. "
                "Ako korisnik nije uplodao polica osiguranja, nego neki drugi dokument ili neku sliku ispiši SAMO 'Ovo nije polica osiguranja'."
            )
            validation_result = response2.text

            # Print validation result in the sidebar with larger font
            st.sidebar.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>{validation_result}</h2>", unsafe_allow_html=True)

            # Display extracted text immediately
            st.session_state.extracted_text = extracted_text

            # Third request: Extract important data in JSON format
            response3 = model_client.generate_content(
                f"Extract and format the following information into valid JSON format: {extracted_text}. "
                "Include only the JSON output without any additional text."
            )
            important_data = response3.text

            # Calculate tokens - estimate input/output split since Gemini only gives total
            total_tokens = len(extracted_text.split()) + len(validation_result.split()) + len(important_data.split())
            estimated_input_tokens = int(total_tokens * 0.4)  # Estimate 40% input
            estimated_output_tokens = int(total_tokens * 0.6)  # Estimate 60% output

            # Calculate total tokens and return with response
            token_usage = {
                "prompt_tokens": estimated_input_tokens,
                "completion_tokens": estimated_output_tokens,
                "total_tokens": total_tokens
            }

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
    st.title("OCR i obrada police osiguranja")
    
    # Sidebar for model selection and file upload
    model = st.sidebar.selectbox("Odaberi model", ["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    uploaded_file = st.sidebar.file_uploader("Uplodaj sliku police osiguranja", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert image to Base64
        image_data = uploaded_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Split the main space into two columns
        col1, col2 = st.columns(2)

        # Process the image using the selected model
        with st.spinner("Obrada police.."):
            result = process_image(image_base64, model)

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
            # Calculate and display cost using selected model
            token_usage = result["token_usage"]
            cost = calculate_api_cost(
                token_usage["prompt_tokens"], 
                token_usage["completion_tokens"],
                model
            )
            st.sidebar.metric(label=f"Troškovi {model}", value=f"${cost['total_cost']:.6f}", delta_color="off")

            st.sidebar.metric(label="Troškovi obrade slike", value=f"${cost['input_cost']:.6f}", delta_color="off")

            st.sidebar.metric(label="Troškovi ispisa podataka", value=f"${cost['output_cost']:.6f}", delta_color="off")

if __name__ == "__main__":
    main()

