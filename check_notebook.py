"""Validate the Kaggle notebook."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("Kaggle_PINN_Training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")
print(f"Notebook format: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"GPU enabled: {nb['metadata'].get('kaggle', {}).get('isGpuEnabled', 'N/A')}")
print(f"Internet enabled: {nb['metadata'].get('kaggle', {}).get('isInternetEnabled', 'N/A')}")
print()

errors = []

for i, cell in enumerate(cells):
    ctype = cell["cell_type"]
    src = "\n".join(cell["source"]) if cell["source"] else ""
    preview = src[:100].replace("\n", " | ")
    print(f"  Cell {i:2d} [{ctype:8s}]: {preview}...")
    
    if ctype == "code" and src.strip():
        # Check for common issues
        if '"""' in src:
            # Count triple quotes - must be even
            count = src.count('"""')
            if count % 2 != 0:
                errors.append(f"Cell {i}: Unmatched triple quotes (count={count})")
        
        # Check for writing files with content
        if "f.write(" in src and "with open(" in src:
            # This is a file-writing cell, check it looks valid
            pass
    
    if not cell["source"]:
        errors.append(f"Cell {i}: Empty source")

print()
if errors:
    print(f"ISSUES FOUND ({len(errors)}):")
    for e in errors:
        print(f"  ❌ {e}")
    sys.exit(1)
else:
    print("✅ No structural issues found.")
    
# Try to compile all code cells
print("\nSyntax-checking code cells...")
compile_errors = []
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "\n".join(cell["source"])
    # Skip cells with shell commands
    if src.strip().startswith("!") or "!pip" in src or "!python" in src:
        continue
    # Remove shell commands from mixed cells
    lines = []
    for line in src.split("\n"):
        stripped = line.strip()
        if stripped.startswith("!"):
            lines.append("# " + stripped)  # comment out shell commands
        else:
            lines.append(line)
    clean_src = "\n".join(lines)
    
    try:
        compile(clean_src, f"cell_{i}", "exec")
    except SyntaxError as e:
        compile_errors.append(f"Cell {i}: {e}")

if compile_errors:
    print(f"\n❌ SYNTAX ERRORS ({len(compile_errors)}):")
    for e in compile_errors:
        print(f"  {e}")
else:
    print("✅ All code cells pass syntax check.")
