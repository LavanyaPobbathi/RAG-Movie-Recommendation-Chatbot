import streamlit as st
import pandas as pd
import tiktoken
import lancedb
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import LanceDB

# Function to initialize the system
def initialize_system(openai_api_key, db_uri="sample-anime-lancedb"):
    client = OpenAI(api_key=openai_api_key)
    
    # Load and preprocess the dataset
    df = pd.read_csv('anime_with_synopsis.csv')
    df = df.dropna()
    df['combined_info'] = df.apply(lambda row: f"Title: {row['Name']}. Overview: {row['sypnopsis']} Genres: {row['Genres']}", axis=1)
    
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # encoding for text-embedding-ada-002
    max_tokens = 8000  # maximum for text-embedding-ada-002 is 8191

    encoding = tiktoken.get_encoding(embedding_encoding)

    # Omit descriptions that are too long to embed
    df["n_tokens"] = df.combined_info.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]
    
    # Function to generate embeddings
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    df["embedding"] = df.combined_info.apply(lambda x: get_embedding(x, model=embedding_model))
    df.rename(columns={'embedding': 'vector', 'combined_info': 'text'}, inplace=True)
    
    # Prepare data for LanceDB
    lancedb_data = pd.DataFrame({
        'id': df['MAL_ID'],
        'text': df['text'],
        'vector': df['vector'].tolist(),
        'metadata': df.apply(lambda row: {
            'Name': row['Name'],
            'Score': row['Score'],
            'Genres': row['Genres'],
            'sypnopsis': row['sypnopsis']
        }, axis=1).tolist()
    })

    # Connect to LanceDB
    db = lancedb.connect(db_uri)

    try:
        # Try to open the existing table
        table = db.open_table("anime")
    except Exception as e:
        # If the table doesn't exist, create it
        table = db.create_table("anime", data=lancedb_data)

    # Set up embeddings and retriever
    embeddings = OpenAIEmbeddings(
        deployment="SL-document_embedder",
        model="text-embedding-ada-002",
        show_progress_bar=True,
        openai_api_key=openai_api_key
    )
    
    docsearch = LanceDB(connection=db, embedding=embeddings, table_name="anime")

    # Initialize the language model
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        api_key=openai_api_key
    )
    
    # Define custom prompt
    template = """You are a movie recommender system that helps users find anime that match their preferences. 
    Use the following pieces of context to answer the question at the end. 
    For each question, suggest three anime, with a short description of the plot and the reason why the user might like it.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Your response:"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    
    return qa_chain


# Streamlit app
def main():
    # Title of the app, ensuring it stays on one line
    st.markdown("<h1 style='text-align: center; font-size: 40px;'>RAG Movie Recommendation Chatbot</h1>", unsafe_allow_html=True)

    
    # Set up the OpenAI API key
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    # Initialize the system (This could be done once per session)
    qa_chain = initialize_system(openai_api_key)
    
    st.markdown("<h2 style='font-size: 30px;'>Customize Your Recommendations</h2>", unsafe_allow_html=True)

    # Use columns to align sliders side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        genre_weight = st.slider("Genre Weight", 0.0, 1.0, 0.5)
    with col2:
        rating_weight = st.slider("Rating Weight", 0.0, 1.0, 0.5)
    with col3:
        popularity_weight = st.slider("Popularity Weight", 0.0, 1.0, 0.5)


    user_query = st.text_input("What kind of movie are you looking for?", "I'm looking for an action movie.")
    
    if st.button("Get Recommendations"):
        if user_query:
            with st.spinner("Generating recommendations..."):
                with get_openai_callback() as cb:
                    # Modify the query or context here based on the sliders if needed
                    result = qa_chain({"query": user_query})
                
                st.write("### Recommendations:")
                st.write(result['result'])
                
                st.write("#### Explanation:")
                st.write(f"- **Genre Weight:** {genre_weight}")
                st.write(f"- **Rating Weight:** {rating_weight}")
                st.write(f"- **Popularity Weight:** {popularity_weight}")
                
                st.write(f"Total Tokens: {cb.total_tokens}")
                st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                st.write(f"Completion Tokens: {cb.completion_tokens}")
                st.write(f"Total Cost (USD): ${cb.total_cost:.5f}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
