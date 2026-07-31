"""Fix total_mem -> total_memory in both notebooks."""
import json

for nb_file in ["notebooks/docforge_vlm_training.ipynb", "notebooks/docforge_vlm_colab.ipynb"]:
    with open(nb_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("total_mem", "total_memory")
    
    with open(nb_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Verify valid JSON
    nb = json.loads(content)
    print(f"Fixed {nb_file} - {len(nb['cells'])} cells, valid JSON")
