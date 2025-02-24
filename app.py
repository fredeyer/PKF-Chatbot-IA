import argparse
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
cors = CORS(app)


api_key1=os.getenv("OpenAI_API_Key")


@app.route("/chatbot", methods=["POST"])
def chatbot():

    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Message manquant"}), 400
    
    try:
        # Appel à l'API GPT-4
        application_prompt1 = """Tu es un assistant du service client de l'entreprise PKF Solution TI. Tu es brancher sur un site en ligne de vente des services IA. Tu répond poliment et sans entrer dans les details avec des reponses courtes aux questions des visiteurs du site.
        QUESTION:
        {user_input} 
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--docs_dir", type=str, default="./data/")
        parser.add_argument("--persist_dir", type=str, default="data_faiss")
        args = parser.parse_args()

        print(f"Using data dir {args.docs_dir}")
        print(f"Using index path {args.persist_dir}")
        
        embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        print(f"Embedding: {embedding.model_name}")

        if os.path.exists(args.persist_dir): 
            print(f"Loading FAISS index from {args.persist_dir}")
            vectorstore = FAISS.load_local(args.persist_dir, embedding,  allow_dangerous_deserialization=True)
            print("done.")
        else:
            print(f"Building FAISS index from documents in {args.docs_dir}")
            
            loader = DirectoryLoader(args.docs_dir,
                loader_cls=Docx2txtLoader,
                recursive=True,
                silent_errors=True,
                show_progress=True,
                glob="**/*.docx"  # which files get loaded
            )
            docs = loader.load()
        
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=75
            )
            frags = text_splitter.split_documents(docs)

            print(f"Poplulating vector store with {len(docs)} docs in {len(frags)} fragments")
            vectorstore = FAISS.from_documents(frags, embedding)
            print(f"Persisting vector store to: {args.persist_dir}")
            vectorstore.save_local(args.persist_dir)
            print(f"Saved FAISS index to {args.persist_dir}")
        
        llm = ChatOpenAI(
            temperature=0.2,
            max_tokens=500,
            openai_api_key=api_key1,
            model='gpt-4o'
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=1000
        )
        memory.load_memory_variables({})

        # 
        
        application_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Tu es un assistant du service client de l'entreprise PKF Solution TI.
            Tu réponds poliment et sans entrer dans les détails, avec des réponses courtes uniquement l'essentiel aux questions des visiteurs du site.
            
            CONTEXTE :
            {context}
            
            QUESTION :
            {question}
            """
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=memory,
            retriever=vectorstore.as_retriever(),
            combine_docs_chain_kwargs={'prompt': application_prompt}
        )

        print(f"user_input: {user_input}")
        memory.chat_memory.add_user_message(user_input)
        result = qa_chain({"question": user_input})
        response = result["answer"]
        memory.chat_memory.add_ai_message(response)
        print("AI:", response)
        return jsonify({"reply": response})
    except Exception as e:
        print(f"error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
