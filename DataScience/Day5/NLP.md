# Datascience Notebook Examples

| Topic            | PDF Link                                                                                                                                     | Streamlit App                                                                                      | Colab Notebook                                                                                                                                           |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| gtts and  Transformer  | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1qdH6XOMFo5CVWwqvwo_Ib2qg1JeSJflo#scrollTo=Enp2gFGyIhnt" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Product Review and Language Detection  | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/10kAZM9wyioe7s-VCNAEmt3PVwGLUA-3G?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| NLP Basics  | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/12toh20HVv4SmgFDzcqIlB7sJqkiqduRB?usp=sharing#scrollTo=kypAw_QXzq3U" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


~~~python

Predict on sample reviews
samples = ["This phone works great, battery lasts long!",
           "waste.",
           "Average product. Not too bad, not too good."]

sample_seq = tokenizer.texts_to_sequences(samples)
sample_pad = pad_sequences(sample_seq, maxlen=padded.shape[1], padding='post')
preds = model.predict(sample_pad)

for review, p in zip(samples, preds):
    print(f"\n📝 Review: {review}")
    print(f"⭐ Predicted Rating: {np.argmax(p) + 1}")

~~~
