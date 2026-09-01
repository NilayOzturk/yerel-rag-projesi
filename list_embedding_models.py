import sys
from foundry_local_sdk import Configuration, FoundryLocalManager

try:
    config = Configuration(app_name="list_embeddings")
    manager = FoundryLocalManager(config)
    print("Foundry Local başlatıldı.")
except Exception as e:
    print(f"Başlatma hatası: {e}")
    sys.exit(1)

print("\nKatalogdaki embedding modelleri:")
for model in manager.catalog.list_models():
    # model.task veya model.type içinde "embedding" arayalım
    if hasattr(model, 'task') and 'embedding' in model.task.lower():
        print(f"  - ID: {model.id} | Task: {model.task}")
    elif hasattr(model, 'type') and 'embedding' in model.type.lower():
        print(f"  - ID: {model.id} | Type: {model.type}")
    else:
        # Bazı modellerde embedding olduğunu anlamak için isme bakalım
        if hasattr(model, 'id') and 'embed' in model.id.lower():
            print(f"  - ID: {model.id} (muhtemelen embedding)")