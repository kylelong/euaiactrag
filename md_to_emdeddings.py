import voyageai
import time
from pathlib import Path
import logging
from typing import List, Dict
import json

class MarkdownEmbedder:
    def __init__(self, api_key: str = None):
        self.client = voyageai.Client(api_key=api_key)
        self.setup_logging()
        
    def setup_logging(self):
    
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    #      Split markdown content into meaningful chunks.
    def chunk_markdown(self, content: str, chunk_size: int = 1000) -> List[str]:
     
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # If adding this paragraph exceeds chunk_size, save current chunk
            if current_length + len(paragraph) > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(paragraph)
            current_length += len(paragraph)
        
        # Add any remaining content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def embed_chunks(self, chunks: List[str], batch_size: int = 128) -> List[List[float]]:

        # Embed chunks of text using Voyage AI.

        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                self.logger.info(f"Processing batch {i//batch_size + 1}/{-(-len(chunks)//batch_size)}")
                self.logger.info(f"Tokens in current batch: {self.client.count_tokens(batch)}")
                
                batch_embeddings = self.client.embed(
                    batch,
                    model="voyage-3",
                    input_type="document"
                ).embeddings
                
                embeddings.extend(batch_embeddings)
                self.logger.info(f"Successfully embedded batch of {len(batch)} chunks")
                
                # Rate limiting
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                raise
                
        return embeddings
    # Process a markdown file and generate embeddings.
    def process_markdown_file(self, 
                            file_path: str, 
                            output_path: str = None,
                            chunk_size: int = 1000,
                            batch_size: int = 128) -> Dict:
      
        try:
            # Read and process the file
            file_path = Path(file_path)
            self.logger.info(f"Processing file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunk the content
            chunks = self.chunk_markdown(content, chunk_size)
            self.logger.info(f"Created {len(chunks)} chunks from the document")
            
            # Generate embeddings
            embeddings = self.embed_chunks(chunks, batch_size)
            
            # Prepare results
            results = {
                'chunks': chunks,
                'embeddings': embeddings,
                'metadata': {
                    'filename': file_path.name,
                    'chunk_size': chunk_size,
                    'num_chunks': len(chunks),
                    'embedding_model': 'voyage-3'
                }
            }
            
            # Save results to .json
            if output_path:
                output_path = Path(output_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f)
                self.logger.info(f"Results saved to {output_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    embedder = MarkdownEmbedder()
    results = embedder.process_markdown_file(
        file_path="act.md",
        output_path="embeddings_results.json",
        chunk_size=1000,
        batch_size=128
    )