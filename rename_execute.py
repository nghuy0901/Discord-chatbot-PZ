import os

base_dir = r"C:\Workspaces\chatGPT-discord-bot\knowledge\docs\pz"
list_file = r"C:\Workspaces\chatGPT-discord-bot\dry_run_renames.txt"

changed = 0
errors = 0

with open(list_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or ' -> ' not in line:
            continue
        
        parts = line.split(' -> ')
        old_rel_path = parts[0].strip()
        new_name = parts[1].strip()
        
        old_full_path = os.path.join(base_dir, old_rel_path)
        dir_name = os.path.dirname(old_full_path)
        new_full_path = os.path.join(dir_name, new_name)
        
        if os.path.exists(old_full_path):
            try:
                if old_full_path != new_full_path:
                    os.rename(old_full_path, new_full_path)
                    changed += 1
            except Exception as e:
                print(f"Error renaming {old_rel_path}: {e}")
                errors += 1
        else:
            # Maybe it was already renamed in a previous run
            if not os.path.exists(new_full_path):
                print(f"File not found: {old_full_path}")
                errors += 1

print(f"Successfully renamed {changed} files. Errors: {errors}")
