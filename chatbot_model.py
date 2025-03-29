import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Memory state
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

# Intent responses
intent_responses = {
    "work_experience": [
        {"keywords": ["before", "past", "previous"], "response": "Before Reach3, I ran an e-commerce brand called Nick and Jess Clothing, where I handled manufacturing and business development."},
        {"keywords": ["Reach3", "current"], "response": "I'm currently a Research Consultant at Reach3 Insights, working with clients like Coca-Cola and J&J."},
        {"keywords": [], "response": "I've worked in analytics consulting and also founded my own clothing brand."}
    ],
    "education": [
        {"keywords": ["humber"], "response": "I completed the Research Analyst Program at Humber College."},
        {"keywords": ["AI", "george brown"], "response": "I'm currently studying Applied AI at George Brown College."},
        {"keywords": [], "response": "I've studied Finance, Data Science, and Applied AI from multiple institutions."}
    ],
    "skills": [
        {"keywords": ["python"], "response": "Yes, I'm skilled in Python, SQL, SPSS, Tableau, and Power BI."},
        {"keywords": [], "response": "My strengths lie in data storytelling, visualization, and advanced analytics."}
    ]
}

# Generate artificial questions
questions = [f"What about your {intent.replace('_',' ')}?" for intent in intent_responses]
labels = list(intent_responses.keys())
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

def is_followup(query):
    followups = ["what about before", "tell me more", "and", "before that", "what else"]
    return any(f in query.lower() for f in followups)

def is_personal(query):
    return any(p in query.lower() for p in ["age", "married", "religion", "where do you live", "single"])

def select_response(intent, query):
    query = query.lower()
    for item in intent_responses[intent]:
        if any(k.lower() in query for k in item["keywords"]):
            return item["response"]
    return intent_responses[intent][0]["response"]

def get_response(query):
    if is_personal(query):
        return "I'm happy to answer questions about my professional journey!"
    if is_followup(query) and st.session_state.last_intent:
        return select_response(st.session_state.last_intent, query + " follow-up")
    
    embedding = embedder.encode(query, convert_to_tensor=True)
    hit = util.semantic_search(embedding, question_embeddings, top_k=1)[0][0]
    intent = labels[hit["corpus_id"]]
    st.session_state.last_intent = intent
    return select_response(intent, query)

# Streamlit UI
st.title("🤖 Chat with My Virtual Assistant")
query = st.text_input("Ask me about my experience, skills, or past education:")

if query:
    response = get_response(query)
    st.markdown(f"**G-Tiks:** {response}")
