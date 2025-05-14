import csv

input_file = "node_info.csv"
output_file = "node_info_clean.csv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["paper", "title", "abstract"])
    writer.writeheader()

    for row in reader:
        paper_id = row["paper"]
        title = (row.get("title") or "").replace("\n", " ").replace("\r", " ").strip()
        abstract = (row.get("abstract") or "").replace("\n", " ").replace("\r", " ").strip()

        if not title:
            continue  # 跳过无效记录

        writer.writerow({"paper": paper_id, "title": title, "abstract": abstract})

print("✅ Cleaned CSV saved to:", output_file)
