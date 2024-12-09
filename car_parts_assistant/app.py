from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# Set API Key for Groq
os.environ["GROQ_API_KEY"] = "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"

# Initialize Flask app
app = Flask(__name__)

# Configure session
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "supersecretkey"
Session(app)

# Load embeddings and vector databases
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db_dir = "content/"
retrievers = []

for sub_dir in os.listdir(vector_db_dir):
    full_path = os.path.join(vector_db_dir, sub_dir)
    if os.path.isdir(full_path):
        vectordb = Chroma(
            persist_directory=full_path,
            embedding_function=embeddings
        )
        retrievers.append(vectordb.as_retriever(search_type="similarity"))

# Define a combined retriever function
def combined_retriever(query):
    combined_results = []
    for retriever in retrievers:
        combined_results.extend(retriever.get_relevant_documents(query))
    return combined_results

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0
)

# Define chatbot prompt
chatbot_prompt = """
You are an intelligent and helpful automotive parts assistant. Your role is to assist users by providing concise, relevant, and actionable information about vehicle parts and issues. Follow these guidelines:

1. **Understand the User's Query**:
   - Identify the vehicle's make, model, and year.
   - Determine the specific part or issue the user is asking about.
   - If information is missing or unclear, ask polite clarifying questions.

2. **Provide Concise Information**:
   - Highlight the most likely causes or parts relevant to the issue.
   - Include essential details such as part names, part numbers, and prices.
   - Avoid listing too many options or excessive technical details unless requested.

3. **Suggestions and Recommendations**:
   - Provide actionable advice, such as checking specific components or trying a simple fix.
   - Offer only the most important related parts or fluids.

4. **Format the Response**:
   - Start with a brief acknowledgment of the user's query.
   - List potential causes or solutions as bullet points (limit to 3-5 items).
   - Provide additional recommendations or fluid suggestions if necessary (keep this brief).

5. **Limit Length**:
   - Keep the response under 500 characters unless the user explicitly asks for more detail.

Conversation so far:
{chat_history}

User's question:
{user_input}
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    # Retrieve stored car details from the session
    car_details = session.get("car_details", None)

    # Check if the user wants to reset car details
    if "reset car" in user_input.lower():
        session.clear()  # Clear session data
        return jsonify({"response": "Got it! The car details have been reset. What car are we working with today? Please provide the make, model, and year."})

    # Check if the input contains car-related details and save them
    if any(keyword in user_input.lower() for keyword in ["honda", "toyota", "nissan", "ford", "accord", "camry", "civic"]):
        session["car_details"] = user_input  # Save car details in session
        car_details = user_input
        return jsonify({"response": f"Got it! I'll remember you're asking about a {car_details}. How can I assist further?"})

    # If car details are missing, fallback to processing the input independently
    if not car_details:
        # Process the input independently
        combined_results = combined_retriever(user_input)

        # Chat history
        chat_history = [{"role": "user", "content": user_input}]

        # Format the chatbot prompt
        formatted_prompt = chatbot_prompt.format(
            chat_history="\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat_history]),
            user_input=user_input
        )

        try:
            response = llm.predict(formatted_prompt)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

    # If car details are stored, include them in the query
    user_input_with_context = f"{car_details}. {user_input}"
    combined_results = combined_retriever(user_input_with_context)

    # Chat history
    chat_history = [{"role": "user", "content": user_input_with_context}]

    # Format the chatbot prompt with context
    formatted_prompt = chatbot_prompt.format(
        chat_history="\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat_history]),
        user_input=user_input
    )

    try:
        response = llm.predict(formatted_prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
