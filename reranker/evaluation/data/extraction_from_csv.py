import csv

input_csv = "evaluation_questions_garanteed_k50.csv"
output_txt = "question.txt"

with open(input_csv, "r", encoding="utf-8", newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    with open(output_txt, "w", encoding="utf-8") as txtfile:
        for row in reader:
            txtfile.write(row["question"] + "\n")

print(f"Les questions ont été enregistrées dans {output_txt}")