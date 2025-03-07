import os
import requests
import logging
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI configuration
client = AzureOpenAI(
    api_key=os.environ.get('AZURE_OPENAI_API_KEY', 'Key'),
    api_version=os.environ.get('API_VERSION', '2024-02-01'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', 'endpoint')
)

# Query Agent API endpoint
QUERY_AGENT_URL = "https://qestbot.azurewebsites.net/api/qestbot"

class SummarizationAgent:
    def __init__(self):
        """Initialize the Summarization Agent with Azure OpenAI client."""
        self.llm_client = client

    def fetch_query_agent_response(self, question):
        """
        Fetch the response from the Query Agent API.
        Args:
            question (str): The user's query.
        Returns:
            str: The raw response from the Query Agent.
        """
        try:
            params = {"question": question}
            response = requests.get(QUERY_AGENT_URL, params=params, timeout=10)
            #print(" Response from query agent: ", response.text)
            response.raise_for_status()  # Raise an error for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from Query Agent: {e}")
            return None

    def simplify_legal_text(self, legal_text):
        """
        Convert complex legal text into simple, easy-to-understand language.
        Args:
            legal_text (str): The raw legal text from the Query Agent.
        Returns:
            str: Simplified text in plain language.
        """
        if not legal_text:
            return "No legal information available to summarize."

        prompt = f"""
        You are a legal assistant with expertise in simplifying complex legal text. 
        Take the following legal text and convert it into clear, concise, and easy-to-understand language. 
        Use simple steps or bullet points where applicable, while preserving the accuracy of the information.

        **Legal Text:**
        {legal_text}

        **Simplified Response:**
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "You are a helpful legal assistant skilled in simplifying legal language."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            simplified_text = response.choices[0].message.content.strip()
            return simplified_text
        except Exception as e:
            logger.error(f"Error simplifying text with LLM: {e}")
            return f"Error processing the legal text: {str(e)}"

    def generate_response(self, question):
        """
        Generate a simplified response by fetching data from Query Agent and summarizing it.
        Args:
            question (str): The user's query.
        Returns:
            str: The simplified response.
        """
        # Fetch raw legal text from Query Agent
        legal_text = self.fetch_query_agent_response(question)
        if not legal_text:
            return "Failed to retrieve legal information. Please try again later."

        # Simplify the legal text
        simplified_response = self.simplify_legal_text(legal_text)
        return simplified_response

# Example usage
if __name__ == "__main__":
    summarization_agent = SummarizationAgent()
    user_question = input("Enter your legal question: ")
    response = summarization_agent.generate_response(user_question)
    print("Simplified Response:")
    print(response)
