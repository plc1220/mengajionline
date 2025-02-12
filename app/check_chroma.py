import chromadb

def main():
    client = chromadb.PersistentClient(path="/home/cmtest/mengajionline/chroma_db")
    collection = client.get_or_create_collection("teacher_evaluations")
    
    # Get all documents
    results = collection.get()
    
    print("\nChromaDB Contents:")
    print("==================")
    print(f"Total documents: {len(results['ids'])}")
    print("\nDocuments:")
    for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
        print(f"\n{i+1}. Document ID: {doc_id}")
        print("   Metadata:")
        for key, value in metadata.items():
            print(f"   - {key}: {value}")

if __name__ == "__main__":
    main()
