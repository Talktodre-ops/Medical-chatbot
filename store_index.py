from src.helpers import load_pdf_files, split_documents, download_huggingface_embeddings
import pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_files(data="Data/")

text_chunks = split_documents(extracted_data)

embeddings = download_huggingface_embeddings()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = 'medicalchatbot'

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# After creating the index, add this verification
print("Verifying index creation...")
while True:
    try:
        index_stats = pc.describe_index(index_name)
        if index_stats.status['ready']:
            print("Index is ready")
            break
        print("Waiting for index initialization...")
        time.sleep(10)
    except Exception as e:
        print(f"Index check failed: {str(e)}")
        time.sleep(5)

# Split into batches of 100 vectors each
batch_size = 100

# Add retry decorator for failed batches
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def upload_batch_with_retry(batch, embeddings, index_name):
    PineconeVectorStore.from_documents(
        documents=batch,
        embedding=embeddings,
        index_name=index_name,
        namespace="medical-namespace"
    )

# Modify your batch upload loop to:
success_count = 0
failed_batches = []

for i in tqdm(range(0, len(text_chunks), batch_size)):
    batch = text_chunks[i:i+batch_size]
    try:
        upload_batch_with_retry(batch, embeddings, index_name)
        success_count += 1
    except Exception as e:
        print(f"Permanent failure on batch {i//batch_size}: {str(e)}")
        failed_batches.append((i, str(e)))

# Add final verification
print("\nUpload Summary:")
print(f"Successfully uploaded batches: {success_count}")
print(f"Failed batches: {len(failed_batches)}")
print(f"Total vectors uploaded: {(success_count * batch_size) + len(text_chunks)%batch_size}")

if failed_batches:
    print("\nRetrying failed batches...")
    for batch_num, error in failed_batches:
        print(f"Retrying batch {batch_num}")
        try:
            upload_batch_with_retry(text_chunks[batch_num:batch_num+batch_size], embeddings, index_name)
        except Exception as e:
            print(f"Final failure on batch {batch_num}: {str(e)}")








