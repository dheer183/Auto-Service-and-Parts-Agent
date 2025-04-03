import os
import re
import streamlit as st
import pandas as pd
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from getpass import getpass

# --------------------------
# INITIALIZATION
# --------------------------

@st.cache_resource
def initialize_system():
    """Initialize system components with Colab-friendly paths"""
    system = {}

    # Set Groq API key (replace with secure retrieval if needed)
    os.environ["GROQ_API_KEY"] = "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"

    # Initialize embeddings
    system["embeddings"] = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )

    # Initialize vector databases
    system["retrievers"] = []
    vector_db_dir = "/content/content/"

    if os.path.exists(vector_db_dir) and os.listdir(vector_db_dir):
        for db_folder in os.listdir(vector_db_dir):
            db_path = os.path.join(vector_db_dir, db_folder)
            if os.path.isdir(db_path):
                vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=system["embeddings"]
                )
                system["retrievers"].append(vectordb.as_retriever())
    else:
        st.error("No vector databases found! Please upload database ZIP files.")

    # Initialize LLM
    system["llm"] = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    # Load QA chain
    system["qa_chain"] = load_qa_chain(system["llm"], chain_type="map_reduce")

    return system
# --------------------------
# AI-PROMPT TEMPLATES
# --------------------------

VEHICLE_EXTRACTION_PROMPT = """
Extract vehicle details from this message. Return as JSON:
{{
    "make": "",
    "model": "", 
    "year": "",
    "engine": ""
}}

If any field is missing, leave it empty. Do not add any other text.

Message: {input}
"""

SERVICE_PROMPT = """Act as an expert Service Advisor. For this {make} {model} ({year}, {engine}):

**Customer Concern**
{query}

**Generate Response With:**
1. Brief diagnosis in simple terms
2. Itemized quote table with:
   - Part manufacturer brands
   - Actual part names
   - Labor breakdown
   - Total costs
3. Safety implications
4. Urgency recommendation

**Required Format:**
**🔧 Service Recommendation**

**Diagnosis:** [Plain English explanation]

**Estimated Costs:**
| Item | Manufacturer | Description | Cost (CAD) |
|------|--------------|-------------|------------|
| [Part Name] | [Brand] | [Function/Reason] | $XX.XX |
| Brake Pads | ACDelco | Premium ceramic front pads | $89.99 |
| Air Filter | Mann-Filter | Cabin air filter | $34.99 |
| Labor | Dealership | [X] hours @ $130/hr | $XX.XX |
| **Total** | | | **$XXX.XX** |

**Safety Note:** [Clear safety impact explanation]

**Recommended Action:** [Urgency level 1-5 with reasoning]

**Customer-Friendly Notes:**
- Available aftermarket alternatives
- Estimated shop time
"""

# --------------------------
# MAIN APPLICATION
# --------------------------

def main():
    st.set_page_config(page_title="AI Service Advisor", page_icon="🤖")
    
    system = initialize_system()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.vehicle_info = {}
        st.session_state.initialized = False

    # Chat interface
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Describe your vehicle issue:"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            # Extract vehicle details using AI
            if not st.session_state.initialized:
                extraction_response = system["llm"].invoke(
                    VEHICLE_EXTRACTION_PROMPT.format(input=prompt)
                )
                try:
                    vehicle_data = eval(extraction_response.content)
                    st.session_state.vehicle_info.update({
                        k: v for k, v in vehicle_data.items() if v
                    })
                except:
                    pass

            # Check for missing info
            required_fields = ["make", "model", "year", "engine"]
            missing = [f for f in required_fields if not st.session_state.vehicle_info.get(f)]

            if missing and not st.session_state.initialized:
                # Ask for missing details once
                response = f"Please provide: {', '.join(missing)}"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
                return
            else:
                st.session_state.initialized = True

                # Generate service analysis
                service_prompt = SERVICE_PROMPT.format(
                    **st.session_state.vehicle_info,
                    query=prompt
                )

                docs = []
                for retriever in system["retrievers"]:
                    docs.extend(retriever.invoke(service_prompt))

                analysis = system["qa_chain"].invoke({
                    "input_documents": docs,
                    "question": service_prompt
                })

                # Format response
                response = f"""
                **Service Analysis**
                {analysis['output_text']}

                *Ask follow-up questions or provide more details for clarification*
                """
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()
