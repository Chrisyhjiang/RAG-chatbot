import openai
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg2
from psycopg2.extras import Json

# Ensure you have set the OPENAI_API_KEY in your environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGSystem:
    def __init__(self, knowledge_base_path='knowledge_base.json', db_host=None, db_port=None, db_name=None, db_user=None, db_password=None):
        self.knowledge_base_path = knowledge_base_path
        
        # Database connection setup
        self.db_host = db_host or os.getenv('DB_HOST', 'localhost')
        self.db_port = db_port or int(os.getenv('DB_PORT', 5432))
        self.db_name = db_name or os.getenv('DB_NAME', 'rag_system')
        self.db_user = db_user or os.getenv('DB_USER', 'user')
        self.db_password = db_password or os.getenv('DB_PASSWORD', 'password')

        self.conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password
        )
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load knowledge base and embed
        self.load_knowledge_base()
        self.embed_knowledge_base()

    def load_knowledge_base(self):
        """
        Load the knowledge base from a JSON file.
        """
        with open(self.knowledge_base_path, 'r') as kb_file:
            self.knowledge_base = json.load(kb_file)

    def embed_knowledge_base(self):
        """
        Embed the knowledge base using the SentenceTransformer model and update the embeddings in PostgreSQL.
        If the table doesn't exist, create it and insert the embeddings.
        """
        docs = [f'{doc["about"]}. {doc["text"]}' for doc in self.knowledge_base]

        embeddings = self.model.encode(docs, convert_to_tensor=False)
        
        with self.conn.cursor() as cur:
            # Ensure pgvector extension is enabled
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check if the table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'documents'
                );
            """)
            
            table_exists = cur.fetchone()[0]

            if not table_exists:
                # Create the table if it doesn't exist
                print("Table 'documents' does not exist. Creating the table and inserting data...")
                cur.execute("""
                    CREATE TABLE documents (
                        id SERIAL PRIMARY KEY,
                        doc_id VARCHAR(255) UNIQUE NOT NULL,
                        about TEXT,
                        text TEXT,
                        embedding VECTOR(384)
                    );
                """)
                
                # Insert all documents into the new table
                for i, doc in enumerate(self.knowledge_base):
                    cur.execute("""
                        INSERT INTO documents (doc_id, about, text, embedding)
                        VALUES (%s, %s, %s, %s);
                    """, (doc["id"], doc["about"], doc["text"], embeddings[i].tolist()))

            else:
                # Update existing records and insert new ones if the table exists
                print("Table 'documents' exists. Updating existing records and inserting new ones...")
                
                for i, doc in enumerate(self.knowledge_base):
                    # Check if the document already exists
                    cur.execute("""
                        SELECT doc_id FROM documents WHERE doc_id = %s;
                    """, (doc["id"],))
                    
                    existing_doc = cur.fetchone()

                    if existing_doc:
                        # Update the existing document
                        cur.execute("""
                            UPDATE documents
                            SET about = %s, text = %s, embedding = %s
                            WHERE doc_id = %s;
                        """, (doc["about"], doc["text"], embeddings[i].tolist(), doc["id"]))
                    else:
                        # Insert the new document
                        cur.execute("""
                            INSERT INTO documents (doc_id, about, text, embedding)
                            VALUES (%s, %s, %s, %s);
                        """, (doc["id"], doc["about"], doc["text"], embeddings[i].tolist()))

            # Commit the transaction to save changes
            self.conn.commit()

    def normalize_query(self, query):
        """
        Normalize the query by converting it to lowercase and stripping whitespace.
        """
        return query.lower().strip()

    def retrieve(self, query, similarity_threshold=0.7, max_docs=5):
        """
        Retrieve relevant documents using KNN search with pgvector.
        """
        # Normalize query
        normalized_query = self.normalize_query(query)
        print(f"Retrieving context for query: '{normalized_query}'")

        # Query embedding
        query_embedding = self.model.encode([normalized_query])[0].tolist()

        with self.conn.cursor() as cur:
            # Perform KNN search using cosine similarity
            cur.execute("""
                SELECT doc_id, about, text, 1 - (embedding <=> %s) AS similarity
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (query_embedding, query_embedding, max_docs))
            
            results = cur.fetchall()

        retrieved_docs = []
        for result in results:
            doc_id, about, text, similarity = result
            if similarity >= similarity_threshold:
                retrieved_docs.append(f'{about}. {text}')

        # If no documents meet the threshold, include the top result regardless
        if not retrieved_docs and results:
            best_match = results[0]
            retrieved_docs.append(f'{best_match[1]}. {best_match[2]}')

        context = "\n\n".join(retrieved_docs)
        print("Retrieved Context:\n", context)

        return context

    def generate_response(self, query, context):
        # Normalize query
        normalized_query = self.normalize_query(query)
        print(f"Generating response for query: '{normalized_query}'")

        try:
            prompt = (
                "You are a dedicated assistant designed to provide positive answers about Defang. "
                "When the user says 'you', 'your', or any pronoun, interpret it as referring to Defang with context of Defang also. "
                "If the user's question involves comparisons with or references to other services, you may use external knowledge. "
                "However, if the question is strictly about Defang, you must ignore all external knowledge and only utilize the given context. "
                "When generating the answer, please put the answer first and the justification later. "
                "Any mentions of BYOD means BRING YOUR OWN DOMAIN and NOT BRING YOUR OWN DEVICE. "
                "Your objective is to remain strictly within the confines of the given context unless comparisons to other services are explicitly mentioned. "
                "Although this rarely happens, if the prompt is not related to defang reply with prompt out of scope. If the prompt contains the word `defang` proceed with answering."
                "\n\nContext:\n" + context + "\n\n"
                "User Question: " + query + "\n\n"
                "Answer:"
            )

            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": normalized_query}
                ],
                temperature=0.05,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Extract the response
            generated_response = response['choices'][0]['message']['content'].strip()

            print("Generated Response:\n", generated_response)

            return generated_response

        except openai.error.OpenAIError as e:
            print(f"Error generating response from OpenAI: {e}")
            return "An error occurred while generating the response."

    def answer_query(self, query):
        try:
            # Normalize query before use
            normalized_query = self.normalize_query(query)
            context = self.retrieve(normalized_query)
            response = self.generate_response(normalized_query, context)
            return response
        except Exception as e:
            print(f"Error in answer_query: {e}")
            return "An error occurred while generating the response."

    def rebuild_embeddings(self):
        """
        Rebuild the embeddings for the knowledge base. This should be called whenever the knowledge base is updated.
        """
        print("Rebuilding embeddings for the knowledge base...")
        self.load_knowledge_base()  # Reload the knowledge base
        self.embed_knowledge_base()  # Rebuild the embeddings
        print("Embeddings have been rebuilt.")

# Instantiate the RAGSystem
rag_system = RAGSystem()
