import json
import random

INPUT_FILE = "data/eval_final.jsonl"
OUTPUT_FILE = "data/eval_final_with_aspect_polarity.jsonl"
MALFORMED_FILE = "data/eval_final_malformed.jsonl"

POLARITIES = ["positive", "negative", "neutral"]

skipped = 0
processed = 0
malformed_lines = []

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            aspects = record.get("labels", {}).get("aspects", [])
            aspect_polarities = {aspect: random.choice(POLARITIES) for aspect in aspects}
            record["labels"]["aspect_polarities"] = aspect_polarities
            outfile.write(json.dumps(record) + "\n")
            processed += 1
        except json.JSONDecodeError as e:
            print(f"Skipping line {i} due to JSON error: {e}")
            malformed_lines.append(f"LINE {i}: {line}\n\n")
            skipped += 1

# Only write malformed file if there are bad records
if malformed_lines:
    with open(MALFORMED_FILE, "w", encoding="utf-8") as malformed:
        malformed.writelines(malformed_lines)
    print(f"Malformed: {skipped} → saved to {MALFORMED_FILE}")
else:
    print(f"No malformed records found!")

print(f"\nDone!")
print(f"Processed: {processed}")
print(f"Output saved to {OUTPUT_FILE}")
