
from dotenv import load_dotenv
import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI



def main():
    load_dotenv()
    
    # create fron end 
    st.set_page_config(page_title="ASK TP")
    st.header("Ask me 💭")

    # upload the file 
    pdf = st.file_uploader(" Upload your Pdf",type="pdf")
    #pdf = st.file_uploader(" Upload your Pdf",type="pdf", accept_multiple_files=True)

    #extract the text of the file 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text= ""
        for page in pdf_reader.pages:
            text += page.extract_text()

    #split text into chunks  
            
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

       # create embeddings
        embeddings = OpenAIEmbeddings()

        knowledge_base = FAISS.from_texts(chunks, embeddings) 
        

        user_question = st.text_input("Ask a question about the announcement")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            #st.write(docs)

            llm = OpenAI()
            chain = load_qa_chain(llm,chain_type="stuff")
            response =chain.run(input_documents=docs, question=user_question)

            st.write(response)













if __name__ == '__main__':
    main()