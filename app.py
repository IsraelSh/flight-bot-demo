import os

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# הגדרת מפתח ה-API של OpenAI (בסביבת ייצור נשמור את זה בטוח יותר)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# פונקציה שטוענת את מאגר המידע פעם אחת ושומרת בזיכרון (Cache)
@st.cache_resource
def load_knowledge_base():
    loader = TextLoader("flight_policies.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()


retriever = load_knowledge_base()

# הגדרת המודל והשרשרת
llm = ChatOpenAI(model="gpt-3.5-turbo")
system_prompt = (
    "אתה נציג שירות לקוחות מקצועי ואדיב של אפליקציית טיסות."
    "ענה אך ורק על בסיס המידע המצורף. אם המידע לא קיים, אמור שאינך יודע."
    "\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# ----------------- ממשק המשתמש (Streamlit UI) ----------------- #
st.title("✈️ עוזר אישי - טיסות")
st.write("היי! אני העוזר החכם של האפליקציה. במה אוכל לעזור?")

# שורת הקלט של המשתמש
user_question = st.text_input("שאל/י משהו (למשל: מה קורה אם הטיסה בוטלה?):")

if user_question:
    # הצגת הודעת טעינה בזמן שהמודל חושב
    with st.spinner("מחפש תשובה במאגר..."):
        response = rag_chain.invoke({"input": user_question})

        # הצגת התשובה על המסך
        st.success("תשובת העוזר:")
        st.write(response["answer"])
