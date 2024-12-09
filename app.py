import os
import streamlit as st

# Langchain modules
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# Set API Key for Groq
os.environ["GROQ_API_KEY"] = "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"

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Directory containing vector databases
vector_db_dir = r"D:\Artifical Intelligence for Design and Implementation\AIDI-1003 Noopa Jagadeesh\Model_data\Streamlit\vector_db_parts_2001\content"
embedding_function = embeddings

# Initialize a list to store retrievers
retrievers = []

# Iterate through all subdirectories in the specified directory
for sub_dir in os.listdir(vector_db_dir):
    full_path = os.path.join(vector_db_dir, sub_dir)
    if os.path.isdir(full_path):
        vectordb = Chroma(
            persist_directory=full_path,
            embedding_function=embedding_function
        )
        retrievers.append(vectordb.as_retriever())

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

# Load a question-answering chain using the "stuff" chain type
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

# Create a prompt template
chatbot_prompt_template = """
You are an automotive parts assistant. When a user asks about their vehicle, you will refer to relevant documents and provide guidance in a concise, clear manner. Your goal is to stay under 1,000 tokens for each response, including all necessary details.

Here are the steps to follow:

1. **Identify the Vehicle and Parts**: Determine what part the user is asking about based on their question. If it’s unclear, ask clarifying questions to understand the model, make, and year of the vehicle.

2. **Provide Pricing**: Always provide the price for the part requested. If you don’t have the exact year of the vehicle the user asks about, provide the price for the earliest year available in your database and inform the user that you cannot help with the exact year requested. You should always mention the earliest year available, even if the part is not available for the requested year.

3. **Include Related Subcategories**: If the part requested falls under a category that has subcategories (e.g., "engine parts" has "fuel injectors", "oil filters", etc.), list those subcategories with their prices, if available.

4. **Mention Fluids**: If the part requested is related to fluids (e.g., oil, transmission fluid), also mention the fluids associated with the part and their availability/price.

5. **Maintain Chat Context**: Keep track of previous conversations and refer to them when necessary to provide consistent follow-up answers. For example, if the user has already asked about a part and later asks about another part from the same vehicle, refer back to previous details such as model, year, or part-related info.

6. **Structure the Response**:
   - Begin by acknowledging the vehicle type and confirming details, especially if the model year or make was mentioned.
   - Provide the price for the part.
   - List any relevant subcategories and related fluids.
   - If no exact match for the year is found, state the earliest year available in the database and explain the limitation.

**Keep responses concise, with no more than 1,000 tokens. If the response exceeds the token limit, trim unnecessary details.**

Conversation so far:
{chat_history}

User's question:
{user_input}
"""

# Streamlit Chatbot Function
def chatbot():
    st.title("Automotive Parts Assistant Chatbot")
    st.write("Welcome to the Car Issue Chatbot! Ask about car parts, and get helpful information.")

    # Initialize chat history in Streamlit session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input form
    user_input = st.text_input("You: ", key="user_input")

    if st.button("Send") and user_input:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Construct chat history to use as context in the prompt
        formatted_chat_history = "\n".join(
            [f"{message['role'].capitalize()}: {message['content']}" for message in st.session_state.chat_history]
        )

        # Create prompt for the current conversation context
        prompt = chatbot_prompt_template.format(
            chat_history=formatted_chat_history,
            user_input=user_input
        )

        # Retrieve documents from vector databases
        combined_results = combined_retriever(user_input)

        # Pass the combined results to the chain
        response = qa_chain.invoke(
            {"input_documents": combined_results, "question": prompt}, return_only_outputs=True
        )

        # Append assistant's response to chat history
        response_text = response['output_text']
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    # Display conversation history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You**: {message['content']}")
        else:
            st.markdown(f"**Assistant**: {message['content']}")

    # Assistant automatically asks for more information if necessary
    if (
        st.session_state.chat_history
        and st.session_state.chat_history[-1]["role"] == "assistant"
        and "clarify" in st.session_state.chat_history[-1]["content"].lower()
    ):
        # If the assistant needs clarification, add a follow-up input
        follow_up_input = st.text_input("Assistant: Could you provide more details about the make, model, or part?", key="follow_up_input")
        if follow_up_input:
            st.session_state.chat_history.append({"role": "user", "content": follow_up_input})

            # Construct new prompt with updated history
            formatted_chat_history = "\n".join(
                [f"{message['role'].capitalize()}: {message['content']}" for message in st.session_state.chat_history]
            )

            prompt = chatbot_prompt_template.format(
                chat_history=formatted_chat_history,
                user_input=follow_up_input
            )

            # Retrieve documents and generate response
            combined_results = combined_retriever(follow_up_input)

            response = qa_chain.invoke(
                {"input_documents": combined_results, "question": prompt}, return_only_outputs=True
            )

            response_text = response['output_text']
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# Run the chatbot
if __name__ == "__main__":
    chatbot()
