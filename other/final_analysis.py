import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 폴더 경로 설정
scores_dir = "output/results/"
distribution_dir = "output/distributions/"
performance_dir = "output/performance_curves/"
output_csv_path = "output/threshold_analysis_summary.csv"
os.makedirs(distribution_dir, exist_ok=True)
os.makedirs(performance_dir, exist_ok=True)

# 기준 threshold (시각화 및 기준 비교용)
threshold = 50.0  # 0.0 ~ 100.0 범위로 설정

# 결과 저장 리스트
results = []
all_best_thresholds = []

# 📊 분포 시각화 함수
def save_distribution_plot(query_score, reference_scores, filename, threshold):
    plt.figure(figsize=(10,6))
    sns.histplot(reference_scores, bins=50, kde=True, label="Reference")
    plt.axvline(query_score, color='red', linestyle='--', label=f"Query: {query_score:.2f}")
    plt.axvline(threshold, color='green', linestyle='-', label=f"Threshold: {threshold:.2f}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.title(f"Similarity Score Distribution\n{filename}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(distribution_dir, f"distribution_{filename.replace('.csv', '.png')}")
    plt.savefig(save_path)
    plt.close()

# 📈 성능 곡선 시각화 함수
def save_performance_curve_plot(y_true, y_scores, filename):
    def evaluate(t):
        y_pred = [1 if s >= t else 0 for s in y_scores]
        return (
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)
        )

    thresholds = [i for i in range(10, 101,1)]
    accs, precs, recs, f1s = zip(*[evaluate(t) for t in thresholds])

    plt.figure(figsize=(10,6))
    plt.plot(thresholds, accs, label="Accuracy")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Performance by Threshold\n{filename}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(performance_dir, f"performance_{filename.replace('.csv', '.png')}")
    plt.savefig(save_path)
    plt.close()

# 🔁 파일 반복 분석
file_list = [f for f in os.listdir(scores_dir) if f.startswith("scores_eucli_") and f.endswith(".csv")]
total_files = len(file_list)

for i, filename in enumerate(file_list):
    print(f"🔍 [{i+1}/{total_files}] 분석 중: {filename}")

    filepath = os.path.join(scores_dir, filename)
    df = pd.read_csv(filepath)

    try:
        query_row = df[df["type"] == "query"].iloc[0]
        query_score = query_row["similarity"]
        query_id = str(query_row["song_id"]).zfill(5)
        reference_df = df[df["type"] == "reference"]
    except IndexError:
        print(f"⚠️ 데이터 이상으로 건너뜀: {filename}")
        continue

    # 정답 기준: reference filename에 query_id가 포함되면 정답
    y_true = reference_df["filename"].apply(lambda x: 1 if query_id in x else 0).tolist()
    y_scores = reference_df["similarity"].tolist()

    y_true = [1] + y_true  # query는 항상 정답
    y_scores = [query_score] + y_scores

    # ✅ 최적 threshold 탐색
    best_f1 = -1
    best_threshold = None
    for t in [i for i in range(10, 101,1)]:  # 0.0 ~ 100.0
        y_pred_tmp = [1 if s >= t else 0 for s in y_scores]
        f1_tmp = f1_score(y_true, y_pred_tmp, zero_division=0)
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_threshold = t

    print(f"   ↳ 최적 threshold: {best_threshold:.1f}, Best F1: {best_f1:.4f}")
    all_best_thresholds.append(best_threshold)

    # 기준 threshold로 평가
    y_pred = [1 if s >= best_threshold else 0 for s in y_scores]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        "Filename": filename,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "Query_Score": round(query_score, 4),
        "Best_Threshold": round(best_threshold, 2),
        "Best_F1_Score": round(best_f1, 4)
    })

    # 그래프 저장
    save_distribution_plot(query_score, reference_df["similarity"].tolist(), filename,best_threshold)
    save_performance_curve_plot(y_true, y_scores, filename)
    print(f"   📈 그래프 저장 완료: {filename.replace('.csv', '')}")

# CSV 저장
summary_df = pd.DataFrame(results)
summary_df.to_csv(output_csv_path, index=False)

# 추천 threshold 계산
recommended_threshold = round(pd.Series(all_best_thresholds).mean(), 2)

# 로그 출력
print(f"📊 전체 {len(results)}개 파일 분석 완료")
print(f"📈 평균 정확도: {summary_df['Accuracy'].mean():.4f}")
print(f"✅ 추천 임계값 (전체 평균): {recommended_threshold}")
print(f"📝 분석 요약 CSV 저장 완료: {output_csv_path}")
