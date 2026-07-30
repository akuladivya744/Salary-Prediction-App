import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read Dataset
df = pd.read_csv(r"C:\Users\akula\Downloads\job_salary_prediction_dataset.csv")

# Label Encoding
job_ti = LabelEncoder()
edu_level = LabelEncoder()
cmp_size = LabelEncoder()

df["job_title"] = job_ti.fit_transform(df["job_title"])
df["education_level"] = edu_level.fit_transform(df["education_level"])
df["company_size"] = cmp_size.fit_transform(df["company_size"])

# Features
X = df[["job_title", "experience_years", "education_level",
        "skills_count", "company_size"]]

y = df["salary"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Salary Prediction")

role = st.selectbox("Select Job Role", job_ti.classes_)
exp = st.number_input("Experience (Years)", 0, 30)
edu = st.selectbox("Education Level", edu_level.classes_)
sc = st.number_input("Skills Count", 0, 30)
cmz = st.selectbox("Company Size", cmp_size.classes_)

if st.button("Predict Salary"):

    job = job_ti.transform([role])[0]
    edu1 = edu_level.transform([edu])[0]
    cms = cmp_size.transform([cmz])[0]

    prediction = model.predict([[job, exp, edu1, sc, cms]])

    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")