import os
import re
import csv

def evaluate_accuracy_for_all_results(
    result_dir="SuCo-HRNE/output/results",
    threshold=50.0,
    output_path_txt="SuCo-HRNE/output/accuracy_report.txt",
    output_path_csv="SuCo-HRNE/output/accuracy_report.csv"
):
    print('result 계산 시작')
    pattern = re.compile(r'test_similarity_scores_eucli_(\d{5})\.txt')
    result_files = [f for f in os.listdir(result_dir) if pattern.match(f)]

    report_lines = []
    csv_rows = [["Filename", "QueryID", "TP", "TN", "FP", "FN", "Total", "Accuracy(%)"]]
    summary_stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for filename in sorted(result_files):
        query_id = pattern.match(filename).group(1)
        file_path = os.path.join(result_dir, filename)

        TP = TN = FP = FN = 0
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                match = re.search(r"\(([^)]+)\) 유사도: ([\d.]+)", line)
                if not match:
                    continue
                ref_filename, score = match.groups()
                score = float(score)

                # 정답인지 확인 (query_id가 ref_filename에 포함되어 있음)
                is_correct = query_id in ref_filename
                pred = score >= threshold
                label = is_correct

                if pred and label:
                    TP += 1
                elif not pred and not label:
                    TN += 1
                elif pred and not label:
                    FP += 1
                elif not pred and label:
                    FN += 1

        total = TP + TN + FP + FN
        acc = (TP + TN) / total * 100 if total > 0 else 0

        report_lines.append(f"{filename} ▶ 정확도: {acc:.2f}% (TP={TP}, TN={TN}, FP={FP}, FN={FN})")
        csv_rows.append([filename, query_id, TP, TN, FP, FN, total, round(acc, 2)])

        # 전체 통계 누적
        summary_stats["TP"] += TP
        summary_stats["TN"] += TN
        summary_stats["FP"] += FP
        summary_stats["FN"] += FN

    # 전체 정확도 계산
    total_all = sum(summary_stats.values())
    overall_acc = (summary_stats["TP"] + summary_stats["TN"]) / total_all * 100 if total_all > 0 else 0
    report_lines.append(f"{total_all} 개의 데이터")
    report_lines.append("\n[전체 정확도]")
    report_lines.append(f"▶ 정확도: {overall_acc:.2f}% (TP={summary_stats['TP']}, TN={summary_stats['TN']}, FP={summary_stats['FP']}, FN={summary_stats['FN']})")

    # TXT 저장
    with open(output_path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # CSV 저장
    with open(output_path_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        writer.writerow([])  # 빈 줄
        writer.writerow(["[전체 정확도]", f"{overall_acc:.2f}%", f"TP={summary_stats['TP']}, TN={summary_stats['TN']}, FP={summary_stats['FP']}, FN={summary_stats['FN']}"])

    print(f"[완료] TXT: {output_path_txt}")
    print(f"[완료] CSV: {output_path_csv}")
