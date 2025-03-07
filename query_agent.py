import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.environ.get('QDRANT_CLUSTER_URL', 'url'),
    api_key=os.environ.get('QDRANT_API_KEY', 'key')
)

# Initialize the embedding model (same as used for creating document vectors)
embedding_model = SentenceTransformer('static-retrieval-mrl-en-v1')

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.environ.get('AZURE_OPENAI_API_KEY', 'key'),
    api_version=os.environ.get('API_VERSION', '2024-02-01'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', 'endpoint')
)

# Collection name
collection_name = 'qest'

class QueryAgent:
    def __init__(self):
        """Initialize the Query Agent with Qdrant client, embedding model, and Azure OpenAI client."""
        self.client = qdrant_client
        self.model = embedding_model
        self.llm_client = client

    def generate_query_vector(self, query_text):
        """Convert the user's query into a 1024-dimensional vector."""
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        return query_embedding.tolist()  # Convert to list for Qdrant compatibility

    def fetch_relevant_documents(self, query_text, limit=3):
        """
        Fetch relevant documents from Qdrant based on the query.
        Args:
            query_text (str): The user's query.
            limit (int): Number of top relevant documents to retrieve.
        Returns:
            list: List of payloads (documents) with their scores.
        """
        # Generate vector for the query
        query_vector = self.generate_query_vector(query_text)

        # Perform vector search in Qdrant
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        # Extract payloads and scores
        relevant_documents = [
            {
                'payload': hit.payload,
                'score': hit.score
            } for hit in search_result
        ]

        return relevant_documents

    def generate_response(self, query_text):
        """
        Generate a response using LLM based on the query and retrieved documents.
        Args:
            query_text (str): The user's question.
        Returns:
            str: A concise and natural language response.
        """
        # Fetch relevant documents
        relevant_docs = self.fetch_relevant_documents(query_text)
        
        if not relevant_docs:
            return "No relevant legal information found. Would you like me to search further?"

        # Prepare context from the top-scoring document (or combine multiple if needed)
        context = relevant_docs[0]['payload']['text']  # Using the top result for now
        # If you want to use multiple documents, you can concatenate them:
        # context = "\n".join([doc['payload']['text'] for doc in relevant_docs])

        # Create a prompt for the LLM
        prompt = f"""
        You are a legal assistant. Based on the following context, answer the user's question in a clear and concise manner. 
        If the information is insufficient, suggest further assistance.

        **Context:**
        {context}

        **User Question:**
        {query_text}

        **Response:**
        """

        # Call Azure OpenAI to generate the response
        response = self.llm_client.chat.completions.create(
            model=os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract and return the generated response
        return response.choices[0].message.content.strip()

# Example usage
if __name__ == "__main__":
    query_agent = QueryAgent()
    user_question = input("Please enter your legal question: ")
    response = query_agent.generate_response(user_question)

    print("Response:")
    print(response)
