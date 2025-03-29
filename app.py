import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Memory state
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Full intent_responses from chatbot
intent_responses = {
    "work_experience": [
        {"keywords": ["Reach3", "current"], "response": "I'm currently a Research Consultant at Reach3 Insights, where I lead analytics for Coca-Cola, J&J, and Hershey’s."},
        {"keywords": ["before", "past", "previous"], "response": "Before Reach3, I built and ran my own e-commerce brand called Nick and Jess Clothing, handling everything from manufacturing to business development."},
        {"keywords": ["startup", "entrepreneur", "ecommerce"], "response": "I founded Nick and Jess Clothing, an e-commerce startup where I managed vendor negotiations, manufacturing, and onboarding with retail platforms."},
        {"keywords": [], "response": "I've worked in analytics consulting and have also founded an e-commerce clothing brand. My work spans client strategy, insights, and business operations."}
    ],
    "education": [
        {"keywords": ["AI", "artificial", "George Brown"], "response": "I'm currently pursuing Applied AI at George Brown College."},
        {"keywords": ["finance", "college", "university"], "response": "I hold a Bachelor’s degree in Finance Markets from H.R. College."},
        {"keywords": ["humber", "research analyst"], "response": "I completed the Research Analyst Program at Humber College, which built my foundation in market research and insights."},
        {"keywords": ["upskill", "learn", "courses"], "response": "I'm always upskilling — from Data Science at BrainStation to Applied AI and new tools to stay ahead in analytics."},
        {"keywords": [], "response": "I've studied Finance, Data Science, and Applied AI — and I'm a lifelong learner constantly picking up new skills."}
    ],
    "skills": [
        {"keywords": ["python", "code", "programming"], "response": "Yes, I'm proficient in Python and use it for data analysis, automation, and insight generation."},
        {"keywords": ["tools", "software", "tableau", "power bi"], "response": "I work with Power BI, Tableau, SQL, and SPSS — creating dashboards, automating workflows, and driving data-driven decisions."},
        {"keywords": [], "response": "My strengths lie in analytics, data storytelling, automation, and using tech like Python, SQL, and BI tools."}
    ],
    "projects": [
        {"keywords": ["fanta", "thailand", "coca"], "response": "I led real-time consumer research at Coca-Cola’s Fanta Fest in Thailand using AI-powered methodologies."},
        {"keywords": ["syndicated", "go-to-market"], "response": "I’ve worked with syndicated data and helped shape go-to-market strategies for major brands."},
        {"keywords": [], "response": "I’ve led projects ranging from experiential field studies to syndicated research and market strategy development."}
    ],
    "career_progression": [
        {"keywords": ["promotion", "growth", "senior"], "response": "I started at Reach3 in 2022, was promoted to Senior Associate, and now serve as a Research Consultant."},
        {"keywords": ["journey", "path", "roles"], "response": "My journey spans entrepreneurship and research — I’ve grown from founding a brand to leading insights for Fortune 500 clients."},
        {"keywords": [], "response": "From launching my own brand to growing in analytics consulting, my path reflects curiosity, hustle, and evolution."}
    ],
    "entrepreneurship": [
        {"keywords": ["startup", "business", "brand"], "response": "I co-founded an e-commerce brand called Nick and Jess Clothing, which I ran end-to-end from sourcing to sales."},
        {"keywords": ["ecommerce", "amazon", "store"], "response": "Nick and Jess was a fashion brand I scaled online — handling manufacturing, vendor deals, and e-retail operations."},
        {"keywords": [], "response": "Starting my own brand taught me how to think like an operator — blending product, partnerships, and business development."}
    ],
    "international_experience": [
        {"keywords": ["thailand", "abroad", "travel"], "response": "I traveled to Thailand to lead fieldwork for Coca-Cola, collecting in-the-moment consumer insights on-ground."},
        {"keywords": ["global", "multicultural", "international"], "response": "I’ve worked on global projects and collaborated across cultures to deliver high-impact research results."},
        {"keywords": [], "response": "My international work gave me global exposure and taught me how to navigate diverse markets and consumers."}
    ],
    "leadership": [
        {"keywords": ["president", "rotaract", "club"], "response": "I was President of the Rotaract Club of H.R. College, where I led over 200 projects and won the Best President award across 140+ clubs in Mumbai."},
        {"keywords": ["team", "lead", "mentor"], "response": "I’ve led teams in both community and client work, empowering others while achieving real impact."},
        {"keywords": [], "response": "Leadership has always been a part of my DNA — whether through volunteering or professional collaboration."}
    ],
    "personality_traits": [
        {"keywords": ["strength", "mindset", "team"], "response": "I’m a curious, resilient, and collaborative problem-solver who thrives in fast-paced environments."},
        {"keywords": ["learning", "growth", "resilient"], "response": "I’m growth-driven, always upskilling, and motivated by new challenges and innovation."},
        {"keywords": [], "response": "I bring a calm, focused energy to my work — with a balance of creativity and strategic thinking."}
    ],
    "life_lessons": [
        {"keywords": ["resilience", "move", "countries"], "response": "Moving countries alone taught me adaptability, independence, and how to build from scratch."},
        {"keywords": ["juggle", "balance", "school"], "response": "Balancing school and work helped me develop strong prioritization and time management skills."},
        {"keywords": [], "response": "Facing personal and professional challenges early helped shape my resilience and hustle mindset."}
    ]
}

# Questions + labels
questions = [f"Tell me about your {intent.replace('_', ' ')}." for intent in intent_responses]
labels = list(intent_responses.keys())
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

# Helpers
import re

def clean_query(query):
    query = query.lower()
    query = re.sub(r"gotham['’]s|gothams", "your", query)
    return query.strip()
def is_followup(query):
    return any(f in query.lower() for f in ["what about before", "tell me more", "and", "before that", "what else"])

def is_personal(query):
    return any(p in query.lower() for p in ["birthday", "how old", "married", "religion", "where do you live", "single"])

def select_response(intent, query):
    query = query.lower()
    for item in intent_responses[intent]:
        if any(k.lower() in query for k in item["keywords"]):
            return item["response"]
    return intent_responses[intent][0]["response"]

def smart_predict(query, threshold=0.5):
    if is_personal(query):
        return "I'm happy to answer questions about my professional experience, skills, and education!"
    if is_followup(query) and st.session_state.last_intent:
        return select_response(st.session_state.last_intent, query + " follow-up")

    embedding = embedder.encode(clean_query(query), convert_to_tensor=True)
    hit = util.semantic_search(embedding, question_embeddings, top_k=1)[0][0]
    score = hit["score"]
    intent = labels[hit["corpus_id"]]

    if score < threshold:
        return "I'm not sure how to answer that — feel free to ask about my work, education, or projects!"

    st.session_state.last_intent = intent
    return select_response(intent, query)

# UI
st.title("🤖 Chat with G-Tiks (Your Resume Bot)")
query = st.text_input("Ask me about my experience, skills, or past education:")

if st.button("🔄 Start Over"):
    st.session_state.chat_history = []
    st.session_state.last_intent = None

if query:
    response = smart_predict(query)
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("G-Tiks", response))

for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")
