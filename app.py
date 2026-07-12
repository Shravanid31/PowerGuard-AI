
from flask import Flask,render_template,request
import joblib,pandas as pd
app=Flask(__name__)
model=joblib.load("model/fault_model.pkl")
encoders=joblib.load("model/encoders.pkl")
features=["Voltage (V)","Current (A)","Power Load (MW)","Temperature (°C)","Wind Speed (km/h)","Weather Condition","Maintenance Status","Component Health","Duration of Fault (hrs)","Down time (hrs)"]
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    vals=[]
    for f in features:
        v=request.form[f]
        if f in encoders:
            le=encoders[f]
            if v not in le.classes_:
                v=le.classes_[0]
            v=le.transform([v])[0]
        else:
            v=float(v)
        vals.append(v)
    pred=model.predict(pd.DataFrame([vals],columns=features))[0]
    label=encoders["Fault Type"].inverse_transform([pred])[0]
    return render_template("index.html",prediction=label)
if __name__=="__main__":
    app.run(debug=True)
