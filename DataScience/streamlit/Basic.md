```python
import streamlit as st
#basic ones

st.title("hello class")
st.header("hello")
st.text("jello")
st.subheader("hello")
st.markdown("**hello**")

name=st.text_input("enter the value")
password=st.number_input("enter the password")
if st.button('click here'):
    st.success(f"login is succes")


st.title("hello")
st.write("hello")
st.header("hello")
st.subheader("hello")
st.markdown("hello")

if st.button("hello"):
    st.write("welocem")

#Table and charts
import pandas as pd
df=pd.DataFrame({
    "name":["guru","hh"],
    "Age":[24,25]

})

st.dataframe(df)
st.table(df.head(1))
st.json({"number":[2,4,6]})

st.line_chart(df)
st.bar_chart(df)
st.altair_chart(df)
st.area_chart(df)
# st.camera_input("hello")
st.caption("hello")
st.map()

# Inputs
name=st.text_input("enter vakues")
age=st.number_input("enter number",min_value=0,max_value=100)
score=st.number_input("select score",0,100,50)
st.button("submit")

#Radio button
gender=st.radio("select gender",["male","Female"])

# check box
if st.checkbox("show secret messege"):
    st.write("hello there")
if st.button("click here"):
    st.snow()

colors=st.multiselect("choose colors",["red","blue","green"])

file=st.file_uploader("upload file")
st.download_button("Download Text","Heloo streamlit",file_name="hello.txt")

st.success("Prediction Completed!")
st.error("Something went wrong!")


from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Example Animation 1
lottie_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_bdlrkrqv.json")
st_lottie(lottie_animation, height=300)




# st.title("hello streamlit")
# st.write("this is your first streamlit app")

# st.header("Header")
# st.subheader("sub header")
# st.text("plain text")
# st.markdown("**bold text**")

# import pandas as pd
# import numpy as np
# data=pd.DataFrame(
#     np.random.randn(10,2),
#     columns=['colums A','Coulumns B']
# )
# st.dataframe(data)
# st.table(data.head())

# name=st.text_input("""sumary_line""")
# age=st.number_input("enter age", min_value=0,max_value=100)
# agree=st.checkbox("I agree")
# aption=st.selectbox("choice an option",["1","2","3"])
# st.button("submit")

# st.line_chart(data)
# st.area_chart(data)
# st.success("Model Trained Successfully!")
# st.write("Happy Holidays!")

# st.success("Prediction Completed!")
# st.error("Something went wrong!")

# # import time
# # progress = st.progress(0)
# # for i in range(100):
# #     time.sleep(0.05)
# #     progress.progress(i + 1)

# import streamlit as st
# from streamlit_lottie import st_lottie
# import requests

# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
# st_lottie(lottie_animation, height=300)

# import streamlit as st
# from streamlit_lottie import st_lottie
# import requests

# # Function to load animation from URL
# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# # Example animation
# lottie_url = "https://assets3.lottiefiles.com/packages/lf20_bdlrkrqv.json"
# lottie_animation = load_lottieurl(lottie_url)

# st_lottie(lottie_animation, height=300, key="coding")

# st.audio("")
# st.video("")
```
