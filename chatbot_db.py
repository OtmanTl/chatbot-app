import os
import chainlit as cl
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient
import logging
import psycopg2
from datetime import datetime
from urllib.parse import parse_qs, urlparse
import urllib3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432")
}

def init_db(survey_id):
    """Initialize database table for a specific survey ID"""
    # Convert survey_id to string to ensure safe table name
    table_name = f"chat_interactions_survey_{str(survey_id)}"
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(create_table_query)
        logger.info(f"Database table {table_name} initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization error for survey {survey_id}: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_interaction(survey_id, question, response):
    """Save chat interaction to survey-specific table"""
    table_name = f"chat_interactions_survey_{str(survey_id)}"
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table_name} (question, response)
                VALUES (%s, %s)
                RETURNING id;
                """,
                (question, response)
            )
            interaction_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"Saved interaction {interaction_id} for survey {survey_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving interaction for survey {survey_id}: {e}")
        return False
    finally:
        if conn:
            conn.close()

class ChatbotInterface:
    def __init__(self, survey_id):
        self.survey_id = survey_id
        self.collection_name = f"user_collection"
        self.client = None
        self.chat_engine = None
        try:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key) 
            self.chat_engine = self.initialize_chat_engine()
        except Exception as e:
            logger.error(f"Error initializing ChatbotInterface: {e}")
            raise

    def initialize_chat_engine(self):
        """Initialize the chat engine with the existing Qdrant collection"""
        try:
            vector_store = QdrantVectorStore(
                collection_name=self.collection_name,
                client=self.client,
                enable_hybrid=True,
            )
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            return index.as_chat_engine(
                chat_mode=ChatMode.CONTEXT,
                verbose=True,
                streaming=True
            )
        except Exception as e:
            logger.error(f"Error initializing chat engine: {e}")
            raise

# Initialize models
Settings.llm = Groq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)
Settings.embed_model = FastEmbedEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")

def extract_survey_id(path):
    """Extract survey ID from path with fallback to 'unknown'"""
    try:
        if not path:
            return "unknown"
        
        # Parse the URL
        parsed_url = urlparse(path)
        
        # Extract query parameters
        query_params = parse_qs(parsed_url.query)
        
        # Prioritize 'ID' parameter (case-sensitive)
        survey_id = query_params.get('ID', ['unknown'])[0]
        
        return survey_id
    except Exception as e:
        logger.warning(f"Error extracting survey ID: {e}")
        return "unknown"

@cl.on_chat_start
async def start():
    """Initialize the chatbot when a new chat session starts"""
    try:
        # Get the HTTP referer
        path = cl.user_session.get("http_referer", "")
        print("path is ", path)
        
        # Extract survey ID with fallback to 'unknown'
        survey_id = extract_survey_id(path)
        print("survey_id is ", survey_id)
        
        logger.info(f"Initializing chat with survey_id: {survey_id}")
        
        # Initialize table for this specific survey
        if not init_db(survey_id):
            raise Exception("Failed to create survey-specific table")
        
        # Initialize chatbot with survey_id
        chatbot = ChatbotInterface(survey_id)
        if not chatbot.chat_engine:
            raise Exception("Chat engine initialization failed")
            
        cl.user_session.set("chatbot", chatbot)
        
        # await cl.Message(
        #     content=f"Welcome to the Policy Document Assistant! Session ID: {survey_id}",
        #     author="Assistant"
        # ).send()
        
    except Exception as e:
        error_msg = f"Error initializing chatbot: {str(e)}"
        logger.error(error_msg)
        await cl.Message(content=error_msg, author="Error").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming chat messages"""
    try:
        chatbot = cl.user_session.get("chatbot")
        if not chatbot or not chatbot.chat_engine:
            raise Exception("Chatbot not properly initialized")
        
        # Create a message element that we'll stream the response to
        msg = cl.Message(content="", author="Assistant")
        
        # Stream the response
        response_stream = chatbot.chat_engine.stream_chat(message.content)
        
        # Stream each chunk of the response
        response_text = ""
        for chunk in response_stream.response_gen:
            response_text += chunk
            await msg.stream_token(chunk)
        
        # Save the interaction to database
        save_interaction(chatbot.survey_id, message.content, response_text)
        
        # Send the final message
        await msg.send()
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        await cl.Message(content=error_msg, author="Error").send()

if __name__ == "__main__":
    cl.run()