# %%
# Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from textblob import TextBlob
import nltk
import pickle
import re
from nltk.tokenize import sent_tokenize



# %%
# Set NLTK path and load data
nltk.download('punkt')
nltk.download('stopwords')

# %%
word_tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))

# %%
# Define intents with training phrases
training_phrases = {
    "work_experience": ' '.join([
        "Tell me about your work experience",
        "Where have you worked before?",
        "What’s your current role?",
        "Have you worked with major brands?",
        "Where did you start your career?",
        "How long have you been at Reach3 Insights?",
        "What companies have you worked at?",
        "Can you talk about your career journey?",
        "Where did you intern?",
        "What industries have you worked in?",
        "Do you have experience with Fortune 500 clients?",
        "Have you worked in consulting or research?",
        "Who were your clients at Reach3?"
    ]),

    "education": ' '.join([
        "What’s your educational background?",
        "Where did you go to college?",
        "What are you currently studying?",
        "Have you studied AI or data science?",
        "What degrees or certifications do you hold?",
        "Did you study at George Brown?",
        "Have you done a Research Analyst program?",
        "Where did you study finance?",
        "Have you taken any tech courses recently?",
        "Tell me about your academic background",
        "Which universities or colleges have you attended?",
        "Do you have formal education in analytics?"
    ]),

    "skills": ' '.join([
        "What are your top skills?",
        "Do you know how to use Power BI?",
        "Are you proficient in Python or SQL?",
        "What data tools can you use?",
        "Do you have SPSS experience?",
        "Are you good at data visualization?",
        "What’s your strength in analytics?",
        "Can you analyze consumer trends?",
        "Do you use AI in your work?",
        "What programming languages do you know?",
        "Are you familiar with data pipelines?",
        "Which tools do you use most often?"
    ]),

    "projects": ' '.join([
        "Tell me about your Coca-Cola work",
        "Have you worked on any global projects?",
        "What was your Fanta Fest project?",
        "Have you done any leadership projects?",
        "What kind of client work have you done?",
        "Have you worked with syndicated data?",
        "Any interesting market research examples?",
        "What’s your most exciting project?",
        "Have you worked on any go-to-market strategies?",
        "What has been your most impactful project?",
        "Tell me about a project you're proud of",
        "Have you worked on any AI-driven projects?"
    ]),

    "career_progression": ' '.join([
        "How did you grow at Reach3?",
        "What roles have you had at Reach3 Insights?",
        "When did you start your career?",
        "What was your first role in the industry?",
        "What position did you begin with?",
        "How did your responsibilities change over time?",
        "When did you become a Research Consultant?",
        "Can you walk me through your career journey?",
        "Have you been promoted in your current company?"
    ]),

    "entrepreneurship": ' '.join([
        "Have you started your own business?",
        "Tell me about your e-commerce brand",
        "Did you run an Amazon store?",
        "How did your brand perform?",
        "What was your role in your startup?",
        "Have you worked in e-commerce?",
        "What did you learn from starting your business?",
        "What was your experience running a business?",
        "Have you managed online sales or inventory?"
    ]),

    "international_experience": ' '.join([
        "Have you worked outside of Canada?",
        "Did you travel for work?",
        "Have you done any international projects?",
        "What was it like working in Thailand?",
        "Any global experience?",
        "Have you collaborated with international teams?",
        "Tell me about your global exposure"
    ]),

    "leadership": ' '.join([
        "Have you held leadership roles?",
        "Tell me about your Rotaract involvement",
        "Were you ever a team lead?",
        "What did you do as President of Rotaract?",
        "Any leadership achievements?",
        "Have you led any clubs or teams?",
        "What awards have you received in leadership?",
        "What extracurricular roles have you held?",
        "Have you managed or mentored people before?",
        "Tell me about your leadership journey"
    ]),

    "personality_traits": ' '.join([
        "What are your strengths?",
        "How do you approach problem solving?",
        "What are your biggest weaknesses?",
        "How would you describe your mindset?",
        "Are you more of a team player or independent worker?",
        "How do you handle pressure?",
        "What makes you unique?",
        "How do you approach learning?",
        "What’s your work style?",
        "How do you handle deadlines or setbacks?"
    ]),

    "life_lessons": ' '.join([
        "What did moving countries teach you?",
        "How did you balance work and school?",
        "What personal challenges have you overcome?",
        "Have you ever faced failure?",
        "What experiences shaped your career outlook?"
    ])
}

# %%

# Prepare training data
X = []
y = []
for intent, phrases in training_phrases.items():
    for phrase in re.split(r'[?.!]', phrases):
        phrase = phrase.strip()
        if phrase:
            X.append(phrase)
            y.append(intent)

# %%
# Build pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])
model.fit(X, y)

# %%
# Intent responses
intent_responses = {
    "work_experience": (
        "I currently work at Reach3 Insights, where I lead analytics initiatives for top global clients "
        "like Coca-Cola, Johnson & Johnson, and Hershey’s."
    ),

    "education": (
        "I hold a Bachelor's degree in Finance Markets and have completed multiple programs to grow my tech and research skills—"
        "including Data Science at BrainStation, a Research Analyst diploma from Humber College, and the Applied AI Solutions Development "
        "program at George Brown College."
    ),

    "skills": (
        "My skillset spans data analysis and storytelling using tools like Python, SQL, SPSS, Power BI, and Tableau. "
        "I’m passionate about applying AI and advanced analytics to uncover actionable insights in market research."
    ),

    "projects": (
        "One of my favorite projects was leading real-time consumer feedback research for Coca-Cola’s Fanta Fest in Thailand, "
        "leveraging AI-powered methodologies. I’ve also worked on syndicated data analysis and go-to-market strategy studies "
        "for several global brands."
    ),

    "career_progression": (
        "I started my journey at Reach3 Insights as a Research Associate in 2022, was promoted to Senior Research Associate, "
        "and now serve as a Research Consultant—owning end-to-end analytics for global studies and client deliverables."
    ),

    "entrepreneurship": (
        "At 17, I co-founded an e-commerce brand that reached #10 in the men’s innerwear category on Amazon India. "
        "It was an incredible learning experience in branding, operations, and resilience—especially navigating the business during the pandemic."
    ),

    "international_experience": (
        "I’ve had the opportunity to work internationally—most notably conducting in-person field research in Thailand "
        "as part of Coca-Cola’s Fanta Fest activation. It gave me firsthand experience working with diverse consumers and cultures."
    ),

    "leadership": (
        "Leadership is a core part of who I am. I served as President of the Rotaract Club of H.R. College, leading over 200 projects across community service, "
        "professional development, and international collaboration. I was honored with awards like Best President and Promising Young Leader."
    ),

    "personality_traits": (
        "I’m curious, driven, and collaborative. I thrive on solving complex problems, love learning new tools, "
        "and believe in leading with empathy. That said, I’m always working on managing my tendency to overcommit."
    ),

    "life_lessons": (
        "Moving countries alone and juggling work and school taught me grit, adaptability, and how to navigate ambiguity. "
        "These experiences have shaped my resilience and ability to thrive in challenging environments."
    )
}


# %%
# Prediction helpers
def replace_synonyms(text):
    synonyms = {
        "AI": "artificial intelligence",
        "resume": "CV",
        "education": "study",
        "college": "university",
        "job": "work",
        "company": "organization",
        "clients": "brands",
        "Coca Cola": "Coca-Cola",
        "J&J": "Johnson and Johnson",
        "Hersheys": "Hershey’s"
    }
    for word, synonym in synonyms.items():
        text = text.replace(word, synonym)
    return text


# %%
def correct_spelling(text):
    return str(TextBlob(text).correct())


# %%
# Final prediction function
def smart_predict(user_input, threshold=0.1):
    query = user_input.lower()
    query = replace_synonyms(query)

    # query = correct_spelling(query)  # Optional: can be re-enabled

    proba = model.predict_proba([query])[0]
    max_proba = max(proba)
    predicted_index = proba.argmax()
    predicted_intent = model.classes_[predicted_index]

    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Predicted Intent: {predicted_intent}")
    print(f"[DEBUG] Confidence: {max_proba:.2f}")

    if max_proba < threshold:
        return "I'm not sure what you mean. Could you rephrase that?"

    return intent_responses.get(predicted_intent, "Hmm... I don’t have a good answer for that yet.")

# %%
test_queries = [
    "Can you tell me about your education?",
    "What companies have you worked at?",
    "Do you know Python?",
    "Did you travel for work?",
    "How did you grow at Reach3?",
    "What are your strengths?",
    "Have you run a business?"
]

for query in test_queries:
    response = smart_predict(query)
    print(f"User: {query}\nBot: {response}\n")

# %%
print(smart_predict("Can you tell me about your education?"))
print(smart_predict("Do you know Python?"))
print(smart_predict("What companies have you worked at?"))


# %%


# %%



