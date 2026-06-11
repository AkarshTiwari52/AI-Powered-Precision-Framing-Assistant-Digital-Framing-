import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report

# load data
df = pd.read_csv('data\Crop_recommendation.csv')

# features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

y = df['label']

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# numerical and categorical features
num_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
cat_features = []

# preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# train model
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("First 10 predictions:", y_pred[:10])
print("\nClassification Report:\n", classification_report(y_test, y_pred))


MODEL_ACCURACY = 99.3

def predict_crop(input_data):
    """
    input_data: dict with correct feature names
    """
    import pandas as pd

    X = pd.DataFrame([{
        "N": input_data["N"],
        "P": input_data["P"],
        "K": input_data["K"],
        "temperature": input_data["temperature"],
        "humidity": input_data["humidity"],
        "ph": input_data["ph"],
        "rainfall": input_data["rainfall"]
    }])

    prediction = model.predict(X)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(X)[0]) * 100

    return prediction



def confidence_predict_crop(input_data):
    """
    input_data: dict with correct feature names
    """
    import pandas as pd

    X = pd.DataFrame([{
        "N": input_data["N"],
        "P": input_data["P"],
        "K": input_data["K"],
        "temperature": input_data["temperature"],
        "humidity": input_data["humidity"],
        "ph": input_data["ph"],
        "rainfall": input_data["rainfall"]
    }])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return round(float(proba.max()) * 100, 2)

    return None    


def generate_crop_report(n, p, k, temp, humidity, ph, rainfall):

    input_data = {
        "N": n,
        "P": p,
        "K": k,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    crop = predict_crop(input_data)

    return {
        "status": "success",
        "Recommended Crop": crop,
        "soil_n": n,
        "soil_p": p,
        "soil_k": k,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "note": "Crop recommended based on soil nutrients and climate conditions"
    }



# XAI eith Rag deep dive
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

feature_names = ['N','P','K','temperature','humidity','ph','rainfall']
class_names = model.named_steps['classifier'].classes_

explainer = LimeTabularExplainer(
    training_data=X_train.to_numpy(),
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

def model_predict_proba_for_lime(x):
    return model.predict_proba(pd.DataFrame(x, columns=feature_names))

def predict_crop_with_lime(input_data):
    X_input = pd.DataFrame([input_data])

    crop = model.predict(X_input)[0]

    exp = explainer.explain_instance(
        X_input[feature_names].to_numpy()[0],
        model_predict_proba_for_lime,
        num_features=7
    )

    lime_explanation = [
        {"feature": f, "impact": round(w, 3)}
        for f, w in exp.as_list()
    ]

    return crop, lime_explanation



# Rag + govertment knowledgee xplaination
# Embedding model


#def load_rag_components():
 #   embeddings = HuggingFaceEmbeddings(
  #      model_name="sentence-transformers/all-MiniLM-L6-v2"
   # )

 #   db = FAISS.from_documents(chunks, embeddings)
  #  retriever = db.as_retriever(search_kwargs={"k": 3})
   # return retriever


#retriever = load_rag_components()



import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- ML ----------------


def predict_crop(input_data):
    X = pd.DataFrame([input_data])
    return model.predict(X)[0]

# ---------------- RAG ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="saved_models/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "saved_models/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})



def build_rag_prompt(crop, lime_output):

    lime_text = "\n".join(
        [f"{x['feature']} impact {x['impact']}" for x in lime_output]
    )

    query = f"Why is {crop} suitable based on soil and climate?"

    docs = retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an agriculture expert.
Use ONLY government or ICAR information.

Context:
{context}

Model reasoning (LIME):
{lime_text}

Explain clearly why {crop} is recommended.
"""
    return prompt


# final llm response 
from langchain_groq import ChatGroq

llm = ChatGroq(model = "llama-3.1-8b-instant")

def final_crop_explaination(crop, lime_output):
    prompt = build_rag_prompt(crop, lime_output)
    return llm.invoke(prompt).content