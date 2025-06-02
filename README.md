# 🎵 SuCo-HRNE: Enhanced Feature Summarizing for Effective Cover Song Identification

본 프로젝트는 SuCo-HRNE 모델을 기반으로 음악 커버 유사도를 평가하는 시스템을 구현하고, 다양한 개선 실험을 수행한 작업입니다. Python 기반으로 구현되었으며, 크로마 특징 추출부터 유사도 계산까지의 전 과정을 구조화하였습니다.

> 💻 **GitHub 코드**: [JiHyun13/SuCo-HRNE](https://github.com/JiHyun13/SuCo-HRNE)  
> 🎧 **데이터 출처**: [AIHub 음악 유사성 판별 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71544)

---

## 📁 폴더 구조

SuCo-HRNE/
├── run.py # 전체 파이프라인 실행
├── output/ # 분석 결과 및 시각화 저장
│ ├── distributions/ # 정답/오답 분포 그래프
│ ├── performance_curves/ # threshold 성능 곡선
│ ├── results/ # 유사도 계산 CSV 결과
│ └── accuracy_report.csv # 성능 종합 리포트
├── cache/ # 특징 및 요약 캐시 저장
├── data/ # 원본 음원 데이터 (wav)
├── features/
│ └── extract_features.py # 크로마 특징 추출
├── modules/
│ ├── hubness.py # 허브 제거 (SNN) *사용X
│ ├── network.py # SSM 강화 *사용X
│ ├── mass.py # MASS 행렬 생성 *사용X
│ ├── sdm.py # sdm 생성 *사용X
│ └── chroma_summarizer.py # 특징 요약 (segment_mean_std)
├── similarity/
│ └── cross_similarity.py # 유사도 계산(eucli)
│ ├── qmax.py # 거리계산 *사용X
├── other/
│ ├── cache.py # 캐시 제어
│ ├── results.py # 정확도 평가 및 threshold 튜닝
│ └── final_analysis.py # 리포트 분석
└── README.md

---

## ⚙️ 실행 흐름

1. `run.py` 실행 → 모든 wav 캐싱 및 쿼리/레퍼런스 특징 요약
2. `eucli()` 기반 유사도 계산 후 `scores_eucli_*.csv`로 저장
3. 정답 기준: reference 파일명에 query ID 포함 여부
4. `evaluate_accuracy_for_all_results()` 호출
   - threshold를 0~100으로 변화시켜 F1-score 최대화
   - 최적 threshold 자동 탐색 및 저장
5. 정확도/정밀도/재현율/F1-score 계산 및 시각화

---

## 📊 최종 성능 결과

- 평균 정확도: **99.08%**
- 평가 기준: F1-score 기반 threshold 최적화(69.27)

---

## 🔍 주요 모듈 설명

| 파일 | 설명 |
|------|------|
| `extract_features.py` | Librosa로 특징 벡터 추출 |
| `chroma_summarizer.py` | segment-wise mean/std 기반 요약 |
| `cross_similarity.py` | 유클리디안 유사도 계산 |
| `results.py` | 정답/오답 분포 기반 성능 평가 및 시각화 |

---


## 📌 참고 사항

- wav 파일은 직접 다운로드 후 `/data` 경로에 저장해야 함. 1분 샘플링하여 사용.
- 캐시 디렉토리는 자동 생성되며 재사용 가능
- PyTorch 등 학습 모듈은 포함되어 있지 않으며, 유사도 평가 중심 구조임

---
