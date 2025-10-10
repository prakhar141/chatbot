# BITS Pilani Admission Chatbot

An intelligent, privacy-focused chatbot designed to assist prospective students with BITS Pilani admission-related queries. This application uses OpenRouter LLMs with a Retrieval-Augmented Generation (RAG) pipeline to provide accurate responses based on official BITS Pilani admission documentation.

## Features

### Core Functionality
- **AI-Powered Responses**: Utilizes OpenRouter API with multiple LLM options (GPT-4o-mini, Claude 3.5 Sonnet, Gemini Pro, Llama 3.1, Mistral)
- **RAG Pipeline**: Implements document retrieval with FAISS vector store and OpenAI embeddings for accurate, context-aware responses
- **Document-Based Knowledge**: Trained on official BITS Pilani admission documents stored locally
- **Interactive Chat Interface**: Built with Streamlit for an intuitive user experience
- **Persistent Chat History**: Maintains conversation context using Firebase Firestore

### Security & Privacy
- **Firebase Authentication**: Secure user authentication with email/password and Google OAuth
- **PDF Privacy**: All admission documents are stored locally and never uploaded to external servers
- **Session Management**: Secure chat history per authenticated user
- **Protected Routes**: Authentication required for all chat functionality

### User Experience
- **Model Flexibility**: Users can switch between different LLM models in real-time
- **Chat History Sidebar**: Easy access to previous conversations
- **Custom Styling**: Modern UI with custom CSS for enhanced visual experience
- **Responsive Design**: Mobile-friendly interface
- **Chat Export**: Download chat history as JSON

## Technology Stack

### Framework & UI
- **Framework**: Streamlit
- **Authentication**: Firebase Admin SDK, Firebase Auth
- **Styling**: Custom CSS with Streamlit components

### AI & ML
- **LLM Provider**: OpenRouter API (supports multiple models)
- **Available Models**:
  - OpenAI GPT-4o-mini (default)
  - Anthropic Claude 3.5 Sonnet
  - Google Gemini Pro 1.5
  - Meta Llama 3.1 70B
  - Mistral 7B
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain for RAG orchestration

### Document Processing
- **PDF Reader**: PyPDF2
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter (1000 char chunks, 200 char overlap)

### Database
- **Firestore**: Chat history and user data storage
- **Local Storage**: FAISS vector index and PDF documents

## Architecture

### 1. Document Processing Pipeline
- Reads PDF documents from the `docs/` directory
- Extracts text content from admission-related PDFs
- Splits text into manageable chunks (1000 chars with 200 char overlap)
- Creates embeddings using OpenAI's text-embedding-3-small model
- Stores embeddings in FAISS vector store for efficient retrieval

### 2. RAG Pipeline
- User queries are embedded using the same OpenAI embedding model
- FAISS performs similarity search to find top 5 relevant document chunks
- Retrieved context is passed to the selected LLM via OpenRouter
- LLM generates response based on retrieved context and chat history

### 3. Authentication Flow
- Users can sign up with email/password or Google OAuth
- Firebase Admin SDK validates authentication tokens
- Session state maintains user authentication across interactions
- All chat operations require valid authentication

### 4. Chat Management
- Each chat session is stored in Firestore with unique chat ID
- Messages include role (user/assistant), content, timestamp, and model used
- Users can create new chats, view history, and continue previous conversations
- Chat history is loaded from Firestore on demand

## Setup Instructions

### Prerequisites
- Python 3.8+
- OpenRouter API key
- OpenAI API key (for embeddings)
- Firebase project with Firestore and Authentication enabled

### Environment Variables
Create a `.env` file with the following:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Firebase Configuration
1. Create a Firebase project at https://console.firebase.google.com
2. Enable Email/Password and Google authentication providers
3. Create a Firestore database
4. Download your service account key JSON
5. Save it as `chatbot-firebase-adminsdk.json` in the project root
6. Copy your Firebase web config to `firebaseConfig` in `app.py`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/prakhar141/chatbot.git
cd chatbot
```

2. Install dependencies:
```bash
pip install streamlit firebase-admin pyrebase4 python-dotenv langchain langchain-community langchain-openai pypdf2 faiss-cpu openai
```

3. Create the documents directory and add PDF files:
```bash
mkdir docs
# Add your BITS Pilani admission PDFs to the docs/ directory
```

4. Initialize the vector store:
- Run the app once to process documents and create FAISS index
- The vector store will be saved as `faiss_index/`

5. Run the application:
```bash
streamlit run app.py
```

## Usage

### First Time Setup
1. Launch the application
2. Sign up with email/password or use Google Sign-In
3. Wait for document processing (first run only)
4. Start chatting with the BITS Pilani Admission Assistant

### Chatting
1. Select your preferred LLM model from the dropdown
2. Type your admission-related query in the input box
3. View AI-generated responses based on official documentation
4. Access previous conversations from the sidebar
5. Create new chats or continue existing ones

### Managing Chats
- **New Chat**: Click "New Chat" in the sidebar
- **View History**: Previous chats appear in the sidebar with timestamps
- **Continue Chat**: Click on any previous chat to resume
- **Download History**: Export your chat as JSON using the download button

## Key Components

### Authentication (`check_authentication()`)
- Validates Firebase authentication token
- Manages user session state
- Handles login/signup UI
- Supports Google OAuth integration

### Document Processing (`get_pdf_text()`, `get_text_chunks()`, `get_vector_store()`)
- Extracts text from all PDFs in docs directory
- Splits into optimized chunks for retrieval
- Creates and persists FAISS vector index
- Uses OpenAI embeddings for semantic search

### Conversational Chain (`get_conversational_chain()`)
- Implements RAG pattern with LangChain
- Supports multiple LLM models via OpenRouter
- Maintains chat history for context
- Provides system prompt for consistent behavior

### Chat Management (`save_chat_to_firebase()`, `load_chat_history()`)
- Persists all conversations to Firestore
- Loads chat history per user
- Tracks metadata (model, timestamp, tokens)
- Enables chat continuation across sessions

## Privacy & Security

### Document Privacy
- All PDF documents remain on your local server
- Documents are never uploaded to external services
- Only text embeddings are created (not raw documents)
- FAISS index is stored locally

### User Data
- Chat history stored in Firebase Firestore
- Authentication handled by Firebase Auth
- No personal information shared with LLM providers
- User queries and responses are linked to authenticated accounts

### API Security
- API keys stored in environment variables
- Firebase credentials in separate service account file
- Authentication required for all chat operations

## Model Information

The application supports multiple LLM models through OpenRouter:

1. **GPT-4o-mini** (Default): Fast, efficient, cost-effective OpenAI model
2. **Claude 3.5 Sonnet**: Anthropic's advanced model with strong reasoning
3. **Gemini Pro 1.5**: Google's multimodal model with large context window
4. **Llama 3.1 70B**: Meta's open-source model with strong performance
5. **Mistral 7B**: Efficient open-source model for quick responses

Users can switch models during conversation to compare responses or optimize for speed/quality.

## Troubleshooting

### Vector Store Issues
- Delete `faiss_index/` folder and restart to rebuild
- Ensure OpenAI API key is valid for embeddings
- Check that PDFs are present in `docs/` directory

### Authentication Issues
- Verify Firebase config in `app.py` matches your project
- Check service account JSON file is present and valid
- Ensure authentication providers are enabled in Firebase console

### API Issues
- Verify OpenRouter API key in `.env` file
- Check API rate limits and quotas
- Ensure selected model is available on OpenRouter

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is intended for educational purposes to assist prospective BITS Pilani students.

## Acknowledgments

- BITS Pilani for admission documentation
- OpenRouter for LLM access
- OpenAI for embeddings
- LangChain for RAG framework
- Streamlit for web framework
- Firebase for authentication and storage
