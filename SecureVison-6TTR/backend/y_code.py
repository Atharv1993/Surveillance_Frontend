import json
import os
import numpy as np
from pymongo import MongoClient
from datetime import datetime

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017" 
client = MongoClient(MONGO_URI)
db = client["Smart_Surveillance"]
embeddings_collection = db["face_embeddings"]

# Path to your existing embeddings JSON file
embeddings_file = "embeddings.json"

def migrate_embeddings_to_mongodb():
    # Check if embeddings file exists
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file '{embeddings_file}' not found!")
        return False
    
    # Load existing embeddings from JSON file
    print(f"Loading embeddings from {embeddings_file}...")
    with open(embeddings_file, "r") as f:
        try:
            embeddings_db = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{embeddings_file}'")
            return False
    
    if not embeddings_db:
        print("No embeddings found in the JSON file!")
        return False
    
    print(f"Found {len(embeddings_db)} embeddings to migrate")
    
    # Counter for successfully migrated embeddings
    migrated_count = 0
    
    # Migrate each embedding to MongoDB
    for user_key, embedding in embeddings_db.items():
        try:
            # Parse the user_key (expected format: "NAME_ID")
            name, face_id = user_key.split("_", 1)
            
            # Check if this embedding already exists in MongoDB
            existing = embeddings_collection.find_one({"face_id": face_id, "name": name})
            if existing:
                print(f"Skipping {user_key} - already exists in MongoDB")
                continue
            
            # Insert the embedding into MongoDB
            embeddings_collection.insert_one({
                "face_id": face_id,
                "name": name,
                "embedding": embedding,  # Use the embedding directly from the JSON
                "created_at": datetime.now(),
                "migrated_from_json": True  # Flag to mark migrated data
            })
            
            migrated_count += 1
            print(f"Migrated {user_key}")
            
        except Exception as e:
            print(f"Error migrating {user_key}: {str(e)}")
    
    print(f"\nMigration complete! Successfully migrated {migrated_count} out of {len(embeddings_db)} embeddings.")
    return True

if __name__ == "__main__":
    print("Starting migration of embeddings from JSON to MongoDB...")
    success = migrate_embeddings_to_mongodb()
    
    if success:
        print("\nYou can now update your application to use MongoDB for embeddings storage.")
        print("Tip: You might want to back up your original embeddings.json file before modifying your main application.")
    else:
        print("\nMigration failed. Please check the error messages above.")