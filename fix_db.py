import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ValueHealthDB"]
col = db["PatientData"]

# AI model train aagura maari sample data
dummy_data = [
    {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigree": 0.627, "Age": 50, "Outcome": 1},
    {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigree": 0.351, "Age": 31, "Outcome": 0},
    {"Pregnancies": 8, "Glucose": 183, "BloodPressure": 64, "SkinThickness": 0, "Insulin": 0, "BMI": 23.3, "DiabetesPedigree": 0.672, "Age": 32, "Outcome": 1},
    {"Pregnancies": 1, "Glucose": 89, "BloodPressure": 66, "SkinThickness": 23, "Insulin": 94, "BMI": 28.1, "DiabetesPedigree": 0.167, "Age": 21, "Outcome": 0},
    {"Pregnancies": 0, "Glucose": 137, "BloodPressure": 40, "SkinThickness": 35, "Insulin": 168, "BMI": 43.1, "DiabetesPedigree": 2.288, "Age": 33, "Outcome": 1},
    {"Pregnancies": 5, "Glucose": 116, "BloodPressure": 74, "SkinThickness": 0, "Insulin": 0, "BMI": 25.6, "DiabetesPedigree": 0.201, "Age": 30, "Outcome": 0},
    {"Pregnancies": 3, "Glucose": 78, "BloodPressure": 50, "SkinThickness": 32, "Insulin": 88, "BMI": 31.0, "DiabetesPedigree": 0.248, "Age": 26, "Outcome": 1},
    {"Pregnancies": 10, "Glucose": 115, "BloodPressure": 0, "SkinThickness": 0, "Insulin": 0, "BMI": 35.3, "DiabetesPedigree": 0.134, "Age": 29, "Outcome": 0},
    {"Pregnancies": 2, "Glucose": 197, "BloodPressure": 70, "SkinThickness": 45, "Insulin": 543, "BMI": 30.5, "DiabetesPedigree": 0.158, "Age": 53, "Outcome": 1},
    {"Pregnancies": 8, "Glucose": 125, "BloodPressure": 96, "SkinThickness": 0, "Insulin": 0, "BMI": 0.0, "DiabetesPedigree": 0.232, "Age": 54, "Outcome": 1},
    {"Pregnancies": 4, "Glucose": 110, "BloodPressure": 92, "SkinThickness": 0, "Insulin": 0, "BMI": 37.6, "DiabetesPedigree": 0.191, "Age": 30, "Outcome": 0}
]

col.insert_many(dummy_data)
print("Data Inserted! Ippo app.py-ah restart pannu da!")