import sys
from foundry_local_sdk import Configuration, FoundryLocalManager

config = Configuration(app_name="find_embedding")
manager = FoundryLocalManager(config)
print("Foundry Local başlatıldı.\n")

print("Tüm embedding modelleri:")
for model in manager.catalog.list_models():
    # Önce 'task' veya 'type' kontrol et
    is_embedding = False
    if hasattr(model, 'task') and 'embedding' in str(model.task).lower():
        is_embedding = True
    elif hasattr(model, 'type') and 'embedding' in str(model.type).lower():
        is_embedding = True
    elif hasattr(model, 'id') and 'embed' in model.id.lower():
        is_embedding = True
    
    if is_embedding:
        print(f"  ID: {model.id}")
        print(f"     Task: {getattr(model, 'task', 'yok')}")
        print(f"     Type: {getattr(model, 'type', 'yok')}")
        print("---")