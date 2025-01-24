# Multi-AI-Agents-QA-System-with-Langgraph-and-AstraDB

This project is a **Multi-AI-Agent Question-Answering (QA) System** built using LangGraph, AstraDB, and Streamlit. It integrates multiple AI agents to answer user queries by intelligently routing questions to the most relevant data source: a vector store (AstraDB), Wikipedia, or Arxiv. The system leverages advanced natural language processing (NLP) techniques, including HuggingFace embeddings and Groq's LLM, to provide accurate and context-aware responses.

## Key Features:
**1. Dynamic Question Routing:**
  - The system uses a LangGraph-based workflow to classify and route user questions to the appropriate data source:
    - **Vector Store:** For questions related to agents, prompt engineering, and adversarial attacks.
    - **Wikipedia:** For general knowledge questions.
    - **Arxiv:** For research-related queries.
  - The routing is powered by a Groq LLM fine-tuned for decision-making.

**2. Vector Store with AstraDB:**
  - Documents are vectorized using **HuggingFace embeddings** and stored in **AstraDB**, a scalable vector database.
  - The system retrieves relevant documents from AstraDB for questions routed to the vector store.

**3. Wikipedia and Arxiv Integration:**
  - For general knowledge questions, the system queries Wikipedia using the `WikipediaAPIWrapper`.
  - For research-related questions, it retrieves the latest papers from Arxiv using the `ArxivAPIWrapper`.

**4. Streamlit Frontend:**
  - The system features a user-friendly Streamlit interface where users can input questions and receive answers.
  - The app dynamically displays the source of the answer (vector store, Wikipedia, or Arxiv) and the retrieved content.

**5. Scalable and Modular:**
  - The project is designed to be modular, allowing easy integration of additional data sources or agents.
  - The use of LangGraph ensures a flexible and extensible workflow.

## Technologies Used:
1. LangGraph: For building the multi-agent workflow.
2. Langchain: For building and connnecting tools.
3. AstraDB: For storing and retrieving vectorized documents.
4. HuggingFace Embeddings: For document vectorization.
5. Groq LLM: For question routing and decision-making.
6. Streamlit: For the interactive web interface.
7. WikipediaAPIWrapper and ArxivAPIWrapper: For querying external knowledge sources.

## How It Works:
1. The user inputs a question via the Streamlit interface.
2. The system routes the question to the appropriate data source (vector store, Wikipedia, or Arxiv) using a LangGraph-based workflow.
3. The relevant data is retrieved and displayed to the user.

## Usage
To run this application, run app.py using Streamlit as 

```streamlit run app.py```

You need to provide a GROQ_API_KEY, ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT in an .env file. Then you can ask a question and extract the answer.

## Future Enhancements:
  - Add more data sources (e.g., news APIs, scientific journals).
  - Improve the routing logic for better accuracy.
  - Add support for conversational follow-up questions.

## Contact 
For any feedback or queries, please reach out to me at [LinkedIn](https://www.linkedin.com/in/krishnakumaragrawal/)
