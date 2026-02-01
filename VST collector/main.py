import os
from pathlib import Path

def get_clean_plugin_names(search_folders, output_file):
    # Target extensions
    vst_exts = ['*.vst3', '*.dll', '*.component']
    ni_exts = ['*.nicnt']
    
    # 1. Expanded Blacklist for Windows system folders
    blacklist = [
        'processor', 'engine', 'libeay', 'ssleay', 'mmsi', 
        'unins', 'helper', 'setup', 'vstscanner', 'crashreporter',
        'midefunc', 'mexefunc', 'mfilebag', 'vcomp', 'msvcp', 'msvcr'
    ]
    
    plugin_names = set()

    for folder in search_folders:
        root = Path(folder)
        if not root.exists():
            print(f"Skipping: {folder} (Path not found)")
            continue
            
        print(f"Scanning: {folder}...")

        # Search for files
        search_patterns = vst_exts + ni_exts
        for pattern in search_patterns:
            for p in root.rglob(pattern):
                name = p.stem
                name_lower = name.lower()
                
                # Filtering Logic
                if any(word in name_lower for word in blacklist):
                    continue
                if len(name) < 3:
                    continue
                    
                plugin_names.add(name)

    # Sort and Save
    sorted_names = sorted(list(plugin_names))
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for name in sorted_names:
            f.write(name + '\n')
            
    print(f"\nFinished! Found {len(sorted_names)} unique items across all folders.")

# --- Configuration ---
# List all directories you want to scan here
folders_to_scan = [
    r'D:\Entertainment\Music Softwares',           # Your custom folder
    r'C:\Program Files\Common Files\VST3',        # Standard VST3
    r'C:\Program Files\VSTPlugins',               # Standard VST2 (64-bit)
    r'C:\Program Files (x86)\VSTPlugins',         # Standard VST2 (32-bit)
    r'C:\Program Files\Steinberg\VSTPlugins',     # Steinberg specific
    r'C:\Program Files\Common Files\Avid\Audio\Plug-Ins' # AAX (Pro Tools)
]

output_txt = 'vst_full_system_list.txt'

get_clean_plugin_names(folders_to_scan, output_txt)