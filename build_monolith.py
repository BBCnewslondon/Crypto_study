import re
from pathlib import Path

def build_monolith():
    files = [
        "config.py",
        "data_engineering.py",
        "stationarity.py",
        "signal_processing.py",
        "validation.py",
        "visualisation.py",
        "plot_timeseries.py",
        "network_analysis.py",
        "robustness.py",
        "run_pipeline.py",
        "run_network_analysis.py",
        "run_robustness.py"
    ]
    
    base_dir = Path("pipeline")
    out_file = Path("crypto_analysis_full.py")
    
    all_imports = set()
    all_code = []
    
    for filename in files:
        filepath = base_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Remove multi-line from pipeline... import (...)
        content = re.sub(r'from\s+pipeline[^\n]*\s+import\s+\([^)]+\)', '', content)
        # Remove single-line from pipeline... import ...
        content = re.sub(r'from\s+pipeline[^\n]*\s+import\s+[^\n]+', '', content)
        # Remove intra-module from . import ...
        content = re.sub(r'from\s+\.[^\n]*\s+import\s+\([^)]+\)', '', content)
        content = re.sub(r'from\s+\.[^\n]*\s+import\s+[^\n]+', '', content)
        
        lines = content.split('\n')
        file_code = [f"\n\n# {'='*70}\n# --- FILE: {filename} ---\n# {'='*70}\n\n"]
        
        for line in lines:
            if re.match(r'^import ', line) or re.match(r'^from [a-zA-Z0-9_]+ import ', line):
                if not 'pipeline' in line and not '__future__' in line:
                    all_imports.add(line.strip())
                    continue
                    
            if '__future__' in line:
                continue
                
            file_code.append(line)
            
        all_code.append("\n".join(file_code))
        
    with open("run_all.py", "r", encoding="utf-8") as f:
        content = f.read()
        content = re.sub(r'from\s+pipeline[^\n]*\s+import\s+\([^)]+\)', '', content)
        content = re.sub(r'from\s+pipeline[^\n]*\s+import\s+[^\n]+', '', content)
        
        lines = content.split('\n')
        run_all_code = [f"\n\n# {'='*70}\n# --- FILE: run_all.py ---\n# {'='*70}\n\n"]
        for line in lines:
            if 'sys.path.append' in line or 'root_dir =' in line:
                continue
            run_all_code.append(line)
        all_code.append("\n".join(run_all_code))
        
    with open(out_file, "w", encoding="utf-8") as f:
        f.write('from __future__ import annotations\n\n')
        for imp in sorted(list(all_imports)):
            f.write(imp + '\n')
            
        for code in all_code:
            f.write(code)

if __name__ == "__main__":
    build_monolith()
