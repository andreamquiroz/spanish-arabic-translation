# app.py
import streamlit as st
import subprocess
import json
import matplotlib.pyplot as plt
import os

# Set page title and configuration
st.set_page_config(
    page_title="Spanish to Arabic Translator",
    page_icon="🌍",
    layout="wide"
)

# Initialize session state for text input
if "spanish_text" not in st.session_state:
    st.session_state.spanish_text = ""

# Function to update text in session state
def update_text(text):
    st.session_state.spanish_text = text

# App title and description
st.title("Spanish to Arabic Translation")
st.markdown("Enter Spanish text below to translate it to Arabic.")

# Example sentences dropdown
examples = [
    "¿Cómo estás hoy?",
    "Me gustaría aprender árabe.",
    "El conocimiento es poder.",
    "La traducción automática es fascinante."
]

# Example sentence selection
if st.checkbox("Show example sentences"):
    example = st.selectbox("Select an example:", examples)
    if st.button("Use this example"):
        update_text(example)

# Input text area
spanish_text = st.text_area(
    "Enter Spanish text:", 
    value=st.session_state.spanish_text, 
    height=150,
    key="text_input"
)

# Update session state when text changes
if "text_input" in st.session_state:
    st.session_state.spanish_text = st.session_state.text_input

# Clear text button
if st.button("Clear text"):
    update_text("")

# Define available models
local_model_path = "/home/maquiroz/mixtec_translation_project/models/final_model"
available_models = [
    {"name": "Pre-trained (Helsinki-NLP)", "path": "Helsinki-NLP/opus-mt-es-ar"},
    {"name": "Your Fine-tuned Model", "path": local_model_path}
]

# Model selection dropdown
model_index = st.selectbox(
    "Select model:",
    range(len(available_models)),
    format_func=lambda i: available_models[i]["name"],
    index=1  # Default to your fine-tuned model
)

selected_model = available_models[model_index]["path"]

# Function to call the translation script
def call_translator(text, model_path):
    try:
        # Run the translation script as a separate process
        process = subprocess.run(
            ["python", "translator.py", text, "--model", model_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the JSON output
        result = json.loads(process.stdout)
        return result
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"Process error: {e.stderr}"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Failed to parse translator output"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# Function to display confidence visualization
def display_confidence(word_confidence):
    # Create columns for the confidence display
    cols = st.columns(len(word_confidence))
    
    # Display each word with its confidence score
    for i, wc in enumerate(word_confidence):
        # Determine color based on confidence
        if wc["confidence"] > 75:
            color = "green"
        elif wc["confidence"] > 50:
            color = "orange"
        else:
            color = "red"
        
        # Display word with colored background based on confidence
        cols[i].markdown(
            f"""
            <div style="text-align: center; margin: 0 2px; padding: 5px; 
                        background-color: rgba({255-wc['confidence']*2.55}, {wc['confidence']*2.55}, 0, 0.3); 
                        border-radius: 5px; min-height: 90px;">
                <p style="font-size: 18px; margin-bottom: 5px;">{wc['word']}</p>
                <p style="font-size: 14px; color: {color};">{wc['confidence']}%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Translate button
if st.button("Translate"):
    if spanish_text:
        with st.spinner("Translating..."):
            result = call_translator(spanish_text, selected_model)
            
            if result.get("success"):
                st.success(f"Translation complete with {result.get('overall_confidence', 'N/A')}% overall confidence!")
                
                st.write("### Arabic Translation:")
                st.write(result["translation"])
                
                st.write("### Confidence Visualization:")
                if "word_confidence" in result:
                    display_confidence(result["word_confidence"])
                    
                    # Create a bar chart of confidence scores
                    fig, ax = plt.subplots(figsize=(10, 3))
                    words = [wc["word"] for wc in result["word_confidence"]]
                    confidence = [wc["confidence"] for wc in result["word_confidence"]]
                    
                    # Assign colors based on confidence levels
                    colors = []
                    for c in confidence:
                        if c > 75:
                            colors.append('green')
                        elif c > 50:
                            colors.append('orange')
                        else:
                            colors.append('red')
                    
                    bars = ax.bar(words, confidence, color=colors)
                    
                    ax.set_ylim(0, 100)
                    ax.set_ylabel('Confidence (%)')
                    ax.set_title('Word-by-word Translation Confidence')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            else:
                st.error(f"Translation failed: {result.get('error', 'Unknown error')}")
    else:
        st.warning("Please enter some text to translate.")

# Information section
with st.expander("About this translator"):
    st.write("""
    This app uses a neural machine translation model for Spanish to Arabic translation.
    
    Available models:
    - Pre-trained (Helsinki-NLP): A general-purpose model trained by Helsinki-NLP
    - Your Fine-tuned Model: Your custom model fine-tuned specifically for this task
    
    The confidence score indicates how sure the model is about each translated word:
    - Green (75-100%): High confidence
    - Orange (50-75%): Medium confidence
    - Red (0-50%): Low confidence
    
    Low confidence may indicate an area where the translation could be improved or where
    the model had difficulty understanding the source text.
    """)

# Footer
st.markdown("---")
st.markdown("Spanish-Arabic Translation Demo")