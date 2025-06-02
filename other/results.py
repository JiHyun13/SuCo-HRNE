import os
import re
import csv

def evaluate_accuracy_for_all_results(
    result_dir="output/results",
    threshold=69.27,
    output_path_txt="output/accuracy_report2.txt",
    output_path_csv="output/accuracy_report2.csv"
):
    print('result 계산 시작')
    csv_files = [f for f in os.listdir(result_dir) if f.startswith("scores_") and f.endswith(".csv")]

    report_lines = []
    csv_rows = [["Filename", "QueryID", "TP", "TN", "FP", "FN", "Total", "Accuracy(%)"]]
    summary_stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for filename in sorted(csv_files):
        query_id = filename.split("_")[-1].replace(".csv", "").zfill(5)
        file_path = os.path.join(result_dir, filename)

        TP = TN = FP = FN = 0
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("type") != "reference":
                    continue

                score = float(row.get("similarity", 0))
                ref_id = str(row.get("song_id", "")).zfill(5)
                is_correct = (ref_id == query_id)
                pred = score >= threshold

                if pred and is_correct:
                    TP += 1
                elif not pred and not is_correct:
                    TN += 1
                elif pred and not is_correct:
                    FP += 1
                elif not pred and is_correct:
                    FN += 1

        total = TP + TN + FP + FN
        acc = (TP + TN) / total * 100 if total > 0 else 0

        report_lines.append(f"{filename} ▶ 정확도: {acc:.2f}% (TP={TP}, TN={TN}, FP={FP}, FN={FN})")
        csv_rows.append([filename, query_id, TP, TN, FP, FN, total, round(acc, 2)])

        summary_stats["TP"] += TP
        summary_stats["TN"] += TN
        summary_stats["FP"] += FP
        summary_stats["FN"] += FN

    total_all = summary_stats["TP"] + summary_stats["TN"] + summary_stats["FP"] + summary_stats["FN"]
    overall_acc = (summary_stats["TP"] + summary_stats["TN"]) / total_all * 100 if total_all > 0 else 0
    report_lines.append(f"{total_all} 개의 데이터")
    report_lines.append("\n[전체 정확도]")
    report_lines.append(f"▶ 정확도: {overall_acc:.2f}% (TP={summary_stats['TP']}, TN={summary_stats['TN']}, FP={summary_stats['FP']}, FN={summary_stats['FN']})")

    with open(output_path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    with open(output_path_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        writer.writerow([])
        writer.writerow(["[전체 정확도]", f"{overall_acc:.2f}%", f"TP={summary_stats['TP']}, TN={summary_stats['TN']}, FP={summary_stats['FP']}, FN={summary_stats['FN']}"])

    print(f"[완료] TXT: {output_path_txt}")
    print(f"[완료] CSV: {output_path_csv}")
