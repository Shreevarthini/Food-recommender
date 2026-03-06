import streamlit as st
import os
from dotenv import load_dotenv
from enhanced_rag_chatbot import generate_llm_rag_response, load_food_data
from shared_functions import create_similarity_search_collection, perform_similarity_search, populate_similarity_collection

# Page configuration
st.set_page_config(page_title="AI Food Recommender", page_icon="🥗")
st.title("AI Food Recommender")
st.markdown("Your RAG-powered personal chef & nutritionist.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load data and collection (cached so it only runs once)
@st.cache_resource
def init_db():
    food_items = load_food_data('./data/FoodDataSet.json')
    collection = create_similarity_search_collection("streamlit_food_rag")
    populate_similarity_collection(collection, food_items)
    return collection

collection = init_db()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("I'm looking for a healthy Italian dinner..."):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Search & Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching our pantry..."):
            # Retrieve from ChromaDB
            results = perform_similarity_search(collection, prompt, n_results=3)
            # Generate using Groq/Llama3
            response = generate_llm_rag_response(prompt, results)
            
            st.markdown(response)
            
            # Show sources in an expandable section
            with st.expander("View Source Recommendations"):
                for r in results:
                    st.write(f"🍴 **{r['food_name']}** ({r['cuisine_type']}) - {r['food_calories_per_serving']} cal")

    st.session_state.messages.append({"role": "assistant", "content": response})