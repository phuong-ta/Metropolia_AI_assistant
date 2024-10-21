from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
import getpass



def HuggingFace_embedding_model():
    hg_api_key = getpass.getpass("Enter your HF Inference API Key:\n\n")
    if "HUGGINGFACE_API_KEY" not in os.environ:
        os.environ["HUGGINGFACE_API_KEY"] = getpass("Provide your HuggingFace API key here")

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hg_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings


def GoogleAI_embedding():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    return embeddings

def OpenAI_embedding():

    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Provide your OpenAI API key here")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        # dimensions=1024
    )
    return embeddings


def GoogleAI_llm():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return llm

def OpenAI_llm():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )

    return llm