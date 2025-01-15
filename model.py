import google.generativeai as genai
import voyageai
import numpy as np
from dotenv import load_dotenv
import os
import json
from typing import List, Dict
import logging
from pathlib import Path

class DocumentQA:
    def __init__(self):
        """Initialize the QA system with both Voyage and Gemini."""
        load_dotenv()
        
        # Initialize Voyage
        self.voyage_client = voyageai.Client()
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.embeddings_cache = None
        self.chunks_cache = None
        
    def load_embeddings(self, embeddings_file: str):
        """Load pre-computed embeddings from a file."""
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.embeddings_cache = data['embeddings']
                self.chunks_cache = data['chunks']
                self.logger.info(f"Loaded {len(self.chunks_cache)} chunks with embeddings")
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Find the most relevant document chunks for a given query."""
        try:
            # Embed the query
            query_embedding = self.voyage_client.embed(
                [query],
                model="voyage-3",
                input_type="query"
            ).embeddings[0]
            
            # Calculate similarities
            similarities = []
            for doc_embedding in self.embeddings_cache:
                similarity = np.dot(query_embedding, doc_embedding)
                similarities.append(similarity)
            
            # Get top k chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_chunks = [self.chunks_cache[i] for i in top_indices]
            
            return relevant_chunks
            
        except Exception as e:
            self.logger.error(f"Error finding relevant chunks: {str(e)}")
            raise

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question using relevant document chunks and Gemini."""
        if not self.embeddings_cache or not self.chunks_cache:
            raise ValueError("No embeddings loaded. Please load embeddings first.")
            
        try:
            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(question, top_k)
            
            # Construct prompt with context
            context = "\n\n".join(relevant_chunks)
            prompt = f"""Context from document:
---
{context}
---

Question: {question}

Please answer the question based on the context provided above. If the answer cannot be found in the context, say so."""
            
            # Generate response using Gemini
            response = self.gemini.generate_content(prompt)
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    qa_system = DocumentQA()
    
    # Load pre-computed embeddings
    qa_system.load_embeddings("embeddings_results.json")
    
    # Example questions
    questions = [
        "What parts of the EU AI act are concerned with video recording?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = qa_system.answer_question(question)
        print(f"Answer: {answer}")