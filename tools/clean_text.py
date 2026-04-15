import re

input_path = "data/raw/shakespeare.txt"
output_path = "data/raw/shakespeare_clean.txt"

cleaned_lines = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # skip empty lines (but keep spacing structure)
        if line == "":
            cleaned_lines.append("")
            continue

        # remove pure numbers (page numbers)
        if re.fullmatch(r"\d+", line):
            continue

        # remove "p. 1234" style lines
        if re.fullmatch(r"p\.\s*\d+", line, flags=re.IGNORECASE):
            continue

        cleaned_lines.append(line)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

print("Cleaned file written to:", output_path)