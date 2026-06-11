from flask import Flask , render_template , url_for, flash ,request , jsonify , redirect
import os
import mysql.connector
import requests
from ml import generate_report , save_prediction_to_db , explain_with_lime , rag_explanation
from moblitnet_dl import generate_image_report
from crop_rec_ml import generate_crop_report , predict_crop_with_lime , final_crop_explaination , MODEL_ACCURACY , confidence_predict_crop
# LangChain / LangGraph imports
from langchain.tools import tool
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import numpy as np
import time
from flask import request, redirect, url_for, flash
from flask_bcrypt import Bcrypt



from tavily import TavilyClient
from typing import Any
import os
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_api_key_here")

app = Flask(__name__)

from flask_bcrypt import Bcrypt
from flask import session

app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key_for_dev")
bcrypt = Bcrypt(app)




@app.route("/")
def index():
    return render_template("index.html")



# wheather showcase management okay and alet system 

@app.route('/wheather')
def wheather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')


    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={os.getenv('OPENWEATHER_API_KEY', 'your_openweather_api_key_here')}"
    r = requests.get(url).json()

    return jsonify({
        "city": r["name"],
        "temp": r["main"]["temp"],
        "humidity": r["main"]["humidity"],
        "wind": round(r["wind"]["speed"] * 3.6, 1),  # m/s → km/h
        "condition": r["weather"][0]["main"]
    })








# fertilizer area management okay 

@app.route('/fer_rec', methods=['GET', 'POST'])
def fertilizer_recommendation():

    if request.method == 'POST':

        # ---------- Collect input data ----------
        input_data = {
            "Nitrogen": int(request.form['n']),
            "Phosphorous": int(request.form['p']),
            "Potassium": int(request.form['k']),
            "Moisture": float(request.form['moisture']),
            "Temparature": float(request.form['temp']),  # ML spelling preserved
            
            "Crop Type": request.form['crop'],
            "Soil Type": request.form['soil_type'],
            "Humidity": float(request.form['humidity'])
        }

        # ---------- Generate fertilizer report ----------
        result = generate_report(
            crop=input_data["Crop Type"],
            n=input_data["Nitrogen"],
            p=input_data["Phosphorous"],
            k=input_data["Potassium"],
            moisture=input_data["Moisture"],
            temp=input_data["Temparature"],
            humidity=input_data["Humidity"],
            soil_type=input_data["Soil Type"]
        )

        # ---------- ML + XAI ----------
        fert, lime_exp = explain_with_lime(input_data)

        # ---------- RAG Explanation ----------
        rag_text = rag_explanation(fert, lime_exp)

        # ---------- Save to DB ----------
        save_prediction_to_db(
            crop=input_data["Crop Type"],
            n=input_data["Nitrogen"],
            p=input_data["Phosphorous"],
            k=input_data["Potassium"],
            moisture=input_data["Moisture"],
            temp=input_data["Temparature"],
            reccomended_fertlizer=result
        )

        return render_template(
            "fertilizer_reccomendation.html",
            result=result,
            lime=lime_exp,
            rag=rag_text
        )

    return render_template("fertilizer_reccomendation.html")


@app.route('/crop_rec', methods=['GET', 'POST'])
def crop_reccomendation():
    if request.method == 'POST':

        input_data = {
            "N": int(request.form['n']),
            "P": int(request.form['p']),
            "K": int(request.form['k']),
            "temperature": int(request.form['temp']),
            "humidity": float(request.form['humidity']),
            "ph": float(request.form['ph']),
            "rainfall": int(request.form['rainfall'])
        }

        # ML prediction
        result = generate_crop_report(**{
            "n": input_data["N"],
            "p": input_data["P"],
            "k": input_data["K"],
            "temp": input_data["temperature"],
            "humidity": input_data["humidity"],
            "ph": input_data["ph"],
            "rainfall": input_data["rainfall"]
        })
        

        # metrices
        confidence = confidence_predict_crop(input_data)
        metrics = {
            "accuracy" : MODEL_ACCURACY ,
            "confidence" : confidence

        }



        # XAI
        crop, lime_exp = predict_crop_with_lime(input_data)

        # RAG + LLM explanation
        rag_explanation = final_crop_explaination(crop, lime_exp)

        return render_template(
            "crop_reccomadation.html",
            result=result,
            lime=lime_exp,
            rag=rag_explanation,
            metrics = metrics
        )

    return render_template("crop_reccomadation.html")




UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



@app.route('/crop_stress', methods=['GET', 'POST'])
def crop_stress_detection():

    if request.method == 'POST':

        if 'leaf_image' not in request.files:
            return render_template(
                "crop_stress.html",
                error="No image uploaded",
                result=None
            )

        image_file = request.files['leaf_image']

        if image_file.filename == "":
            return render_template(
                "crop_stress.html",
                error="No image selected",
                result=None
            )

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        # 🔥 DL + Grad-CAM + Action Advice
        result = generate_image_report(image_path)

        return render_template(
            "crop_stress.html",
            result=result
        )

    # ✅ SAFE GET REQUEST
    return render_template("crop_stress.html", result=None)



@app.route("/crop_details")
def crop_details():
    return render_template("crop_details.html")



## Ndvi index calculatino okay
import rasterio

# def calculate_ndvi(red_band_path, nir_band_path):
#     with rasterio.open(red_band_path) as red:
#         red_data = red.read(1).astype(float)

#     with rasterio.open(nir_band_path) as nir:
#         nir_data = nir.read(1).astype(float)

#     ndvi = (nir_data - red_data)/(nir_data - red_data)  

#     return ndvi  
import cv2
def calculate_pseudo_ndvi(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img.astype(float)

    # OpenCV = BGR
    blue, green, red = cv2.split(img)

    ndvi = (green - red) / (green + red + 1e-6)
    return float(np.mean(ndvi))


def generate_ndvi_report(ndvi_value: float):
    """
    Generates NDVI-based field health report with recommendations
    """

    # Classification logic
    if ndvi_value >= 0.6:
        status = "Healthy Vegetation"
        emoji = "✅"
        causes = ["Normal crop growth"]
        actions = [
            "Continue current irrigation schedule",
            "Maintain nutrient balance",
            "Monitor regularly"
        ]

    elif 0.3 <= ndvi_value < 0.6:
        status = "Moderate Stress"
        emoji = "⚠️"
        causes = [
            "Water stress",
            "Nutrient imbalance",
            "Heat exposure"
        ]
        actions = [
            "Start drip irrigation",
            "Apply organic compost",
            "Use mulching",
            "Recheck NDVI after 7 days"
        ]

    else:
        status = "Severe Stress"
        emoji = "🚨"
        causes = [
            "Severe water shortage",
            "High temperature stress",
            "Poor soil condition"
        ]
        actions = [
            "Immediate irrigation required",
            "Apply organic matter",
            "Reduce heat stress using mulching",
            "Consult agronomist"
        ]

    report = {
        "ndvi_value": round(ndvi_value, 2),
        "vegetation_status": f"{emoji} {status}",
        "possible_causes": causes,
        "recommended_actions": actions
    }

    return report


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



@app.route('/ndvi_index', methods=['GET', 'POST'])
def crop_ndvi_detection():

    if request.method == 'POST':

        if 'leaf_image' not in request.files:
            return render_template("ndvi.html", error="No image uploaded")

        image_file = request.files['leaf_image']

        if image_file.filename == "":
            return render_template("ndvi.html", error="No image selected")

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        ndvi_value = calculate_pseudo_ndvi(image_path)
        result = generate_ndvi_report(ndvi_value)

        return render_template("ndvi.html", result=result)

    return render_template("ndvi.html")
















# chatbot system thik hai 
# chatbot config okay 
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")
# Tavily Tool
# -------------------------------
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> dict[str, Any]:
    """Search agriculture-related information"""
    return tavily_client.search(query)

# -------------------------------
# System Prompt
# -------------------------------
Base_prompt = """
You are an expert agriculture assistant.
Give practical, region-aware, farmer-friendly advice.
Always ask clarifying questions if data is missing.
Avoid medical or chemical overdose advice.
"""

from flask import request, jsonify
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import uuid

# -------------------------------
# Memory
# -------------------------------
memory = InMemorySaver()

# -------------------------------
# LLM (Groq)
# -------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# -------------------------------
# Agent (CREATE ONCE)
# -------------------------------
agent = create_agent(
    model=llm,
    system_prompt=Base_prompt,
    tools=[web_search],   # optional
    checkpointer=memory
)

# -------------------------------
# Chat Route
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    language = data.get("language", "English")

    if not user_message:
        return jsonify({"reply": "Please ask a farming question 🌱"})

    thread_id = data.get("thread_id", str(uuid.uuid4()))
    config = {"configurable": {"thread_id": thread_id}}

    try:
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Reply in {language}. {user_message}"
                    )
                ]
            },
            config=config
        )

        bot_reply = response["messages"][-1].content
        return jsonify({"reply": bot_reply, "thread_id": thread_id})

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"reply": "⚠️ Server error. Please try again."})
if __name__ == "__main__":
    app.run(debug=True)