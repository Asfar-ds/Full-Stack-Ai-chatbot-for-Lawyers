from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

# Initialize embeddings
model_docs = 'sentence-transformers/all-mpnet-base-v2'
embeddings = HuggingFaceEmbeddings(model_name=model_docs)
persist_directory = 'faissembeddings'

try:            
    vector_store = FAISS.load_local(
        folder_path=persist_directory,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
except:
    vector_store = FAISS.from_texts([""], embeddings)
    vector_store.save_local(persist_directory)

# Global variables (for demonstration; use proper session management in production)
chat_history = []
current_settings = {
    'model': 'llama3-70b-8192',
    'temperature': 0.5,
    'p_value': 0.9
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    user_input = data['message']
    
    # Update settings
    current_settings.update({
        'model': data.get('model', current_settings['model']),
        'temperature': float(data.get('temperature', current_settings['temperature'])),
        'p_value': float(data.get('p_value', current_settings['p_value']))
    })
    
    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    chat_model = ChatGroq(
        api_key=api_key,
        model=current_settings['model'],
        temperature=current_settings['temperature'],
        top_p=current_settings['p_value']
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant specialized in contracts."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {user_input}")
    ]
    
    response = chat_model(messages)
    
    # Update chat history
    chat_history.append({'user': user_input, 'assistant': response.content})
    
    return jsonify({
        'response': response.content,
        'history': chat_history
    })

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file temporarily
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    # Process file
    if file.filename.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file.filename.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # Update vector store
    global vector_store
    vector_store.add_documents(splits)
    vector_store.save_local(persist_directory)
    
    return jsonify({'success': True})

if __name__ == '__main__':

    app.run(debug=True)
