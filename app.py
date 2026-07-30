import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


data = {"exp":[1,2,3,4,5],
        "salary":[10000,15000,20000,25000,30000]
        
    }
df = pd.DataFrame(data)
#supervised
x=df[["exp"]].values #input
y=df["salary"] #output

model=LinearRegression()
model.fit(x,y)

#-------web application-----------
st.title("salary prediction")
val=st.number_input("enter the ")
predict=model.predict([[val]])
if st.button("predict salary"):
    st.success(predict)
    
#print("predicted salary for 6 years of experience is:",predict[0])
#print("predicted salary={predict}")