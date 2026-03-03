### AI-Powered Food Recommendation System (RAG) ###
*(Base architecture and dataset provided by IBM Skills Network via Coursera)
(Modified and enhanced by Shreevarthini to include Groq/Llama 3 integration and local environment optimization.")*

An advanced recommendation engine that combines Vector Databases with Large Language Models (LLMs) to provide personalized food suggestions through natural language.

### Overview ###
This project implements a Retrieval-Augmented Generation (RAG) pipeline. Unlike traditional keyword searches, this system understands user intent (e.g., "spicy but healthy dinner") by searching through high-dimensional vector embeddings of food data.

### Key Features ###
Semantic Search: Uses all-MiniLM-L6-v2 via Sentence-Transformers to map food items into vector space.

High-Speed Inference: Powered by Groq and Llama 3-70B for sub-second conversational responses.

Metadata Filtering: Supports complex queries with constraints on calories and cuisine types.

AI Comparison Mode: Allows users to compare two different dietary preferences using LLM analysis.

### Tech Stack ###
LLM: Llama 3 (via Groq Cloud API)

Vector Database: ChromaDB

Embedding Model: Sentence-Transformers (all-MiniLM-L6-v2)

Language: Python 3.12

Environment Management: python-dotenv

### Installation & Setup ###
Clone the repository:

Bash
git clone https://github.com/Shreevarthini/Food-recommender.git
cd Food Recommender
Install dependencies:

pip install -r requirements.txt
Configure Environment Variables:
Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here
Run the Chatbot:
python enhanced_rag_chatbot.py

### How it Works ###
Data Ingestion: Food data is loaded from FoodDataSet.json, cleaned, and enriched with taste profiles and nutritional metadata.

Vectorization: Each food item is converted into a 384-dimensional vector and stored in ChromaDB.

Retrieval: When a user asks a question, the system finds the top 3 most relevant dishes using Cosine Similarity.

Generation: The retrieved context is fed into Llama 3 to generate a friendly, conversational recommendation.


### License & Credits ###
Dataset: Provided by IBM Skills Network via Coursera.