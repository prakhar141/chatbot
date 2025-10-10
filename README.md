# BITS Pilani Admission Chatbot

An intelligent chatbot designed to assist prospective students with BITS Pilani admission-related queries. This application uses Google's Generative AI (Gemini) to provide accurate and helpful responses based on official BITS Pilani admission documentation.

## Features

- **AI-Powered Responses**: Utilizes Google's Gemini 1.5 Flash model for intelligent query handling
- **Document-Based Knowledge**: Trained on official BITS Pilani admission documents (PDFs) to provide accurate information
- **Interactive Chat Interface**: Built with Streamlit for a user-friendly experience
- **Persistent Chat History**: Maintains conversation context throughout the session
- **Styled UI**: Custom CSS styling for an enhanced visual experience

## Technology Stack

- **Framework**: Streamlit
- **AI Model**: Google Generative AI (Gemini 1.5 Flash)
- **Document Processing**: PyPDF2
- **Text Processing**: LangChain (RecursiveCharacterTextSplitter)
- **Embeddings**: Google Generative AI Embeddings
- **Vector Store**: FAISS (Facebook AI Similarity Search)

## Key Components

### Document Processing
- Reads PDF documents from the `docs/` directory
- Extracts and processes text content from admission-related PDFs
- Splits text into manageable chunks for efficient processing

### Vector Database
- Creates FAISS vector store for semantic search
- Enables quick retrieval of relevant information
- Uses Google Generative AI embeddings for document vectorization

### Conversational AI
- Implements a conversational retrieval chain
- Maintains chat history for context-aware responses
- Provides detailed and accurate answers to admission queries

## Setup Instructions

### Prerequisites
- Python 3.7+
- Google API Key for Generative AI

### Installation

1. Clone the repository:
```bash
git clone https://github.com/prakhar141/chatbot.git
cd chatbot
```

2. Install required dependencies:
```bash
pip install streamlit google-generativeai PyPDF2 langchain langchain-google-genai faiss-cpu python-dotenv
```

3. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

4. Create a `docs/` directory and add BITS Pilani admission PDF documents

### Running the Application

```bash
streamlit run app.py
```

The application will start and open in your default web browser.

## Usage

1. Launch the application using the command above
2. Enter your question about BITS Pilani admissions in the text input field
3. Click "Submit" to receive an AI-generated response
4. Continue the conversation - the chatbot maintains context from previous messages

## Application Structure

- `app.py`: Main application file containing:
  - PDF text extraction functions
  - Vector store creation and management
  - Conversational chain setup
  - Streamlit UI implementation
  - Chat history management

## Features in Detail

### PDF Processing
The application automatically processes all PDF files in the `docs/` directory, extracting text and creating a searchable knowledge base.

### Context-Aware Responses
The chatbot uses a conversational chain that maintains chat history, allowing it to understand follow-up questions and provide contextually relevant answers.

### User-Friendly Interface
The Streamlit-based interface provides:
- Clean, modern design with custom styling
- Easy-to-use text input for questions
- Clear display of chat history
- Responsive layout

## Environment Variables

- `GOOGLE_API_KEY`: Your Google Generative AI API key (required)

## Notes

- Ensure PDF documents are placed in the `docs/` directory before running the application
- The application creates a FAISS index on startup, which may take a moment depending on the size of your document collection
- Chat history is maintained only for the current session

## Future Enhancements

- Add support for multiple document types
- Implement persistent chat history across sessions
- Add user authentication
- Expand knowledge base with more BITS Pilani resources

## License

This project is open source and available for educational purposes.

## Author

Developed for assisting prospective students with BITS Pilani admission queries.
