# 🎵 SuCo-HRNE: Enhanced Feature Summarizing for Effective Cover Song Identification

본 프로젝트는 SuCo-HRNE 모델을 기반으로 음악 커버 유사도를 평가하는 시스템을 구현하고, 다양한 개선 실험을 수행한 작업입니다. Python 기반으로 구현되었으며, 크로마 특징 추출부터 유사도 계산까지의 전 과정을 구조화하였습니다.

> 📄 **보고서 원본**: [Notion 링크](https://enormous-raisin-a4a.notion.site/SuCo-HRNE-1-1e933a73ea0f8014a196ff2967655bde?pvs=4)  
> 💻 **GitHub 코드**: [JiHyun13/SuCo-HRNE](https://github.com/JiHyun13/SuCo-HRNE)  
> 🎧 **데이터 출처**: [AIHub 음악 유사성 판별 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71544)

---

## 📁 폴더 구조

SuCo-HRNE/
├── run.py # 전체 실행 스크립트
├── file.py # 파일 정리 및 샘플 분리
├── cashe.py # 캐시 저장 함수
├── data/
│ ├── query/ # 원본 쿼리 WAV
│ ├── reference/ # 커버곡 WAV
│ └── results/ # 유사도 계산 결과
├── cache/ # 캐시 저장 (쿼리별 폴더)
├── features/
│ └── extract_features.py
├── modules/
│ ├── mass.py
│ ├── hubness.py
│ ├── network.py
│ └── summarize.py
├── similarity/
│ ├── cross_similarity.py
│ └── qmax.py

---

## ⚙️ 실행 흐름

1. 쿼리 곡 로딩 및 크로마 특징 추출
2. MASS 알고리즘으로 SDM 계산
3. 허브 효과 제거 (SNN 기반)
4. SDM을 SSM 형태로 강화
5. 반복 구간 탐지 또는 thumb 방식 요약
6. cosine similarity 및 Qmax 방식 유사도 계산
7. 정확도 및 결과 파일 저장

---

## 🔍 주요 모듈 설명

| 파일명 | 기능 |
|--------|------|
| `extract_features.py` | Librosa 기반 크로마 추출 |
| `mass.py` | FFT 기반 MASS 계산으로 SDM 생성 |
| `hubness.py` | SNN 기반 허브 제거 (벡터화 처리) |
| `network.py` | SDM → SSM 강화 (구조적 패턴 강조) |
| `summarize.py` | 반복(repeat) 또는 thumb 방식 요약 |
| `cross_similarity.py` | 요약 벡터 간 cosine similarity 계산 |
| `qmax.py` | similarity matrix에서 대표값 추출 |

---

## 🔁 개선점 요약

- **MASS 최적화**: 2중 for문 → FFT 내적 계산 (속도 개선)
- **허브니스 제거 벡터화**: `csr_matrix` + dot 연산 적용
- **요약 실패 대비 fallback**: repeat 실패 시 thumb 방식 대체
- **캐시 도입**: `.npy` / `.pkl` 저장 및 재활용으로 실행 시간 대폭 단축
- **진행 로그 추가**: `[로그 x/14]` 형식으로 디버깅에 도움

---

## ❗ 현재 문제 및 해결 계획

- 요약 벡터의 스케일이 너무 작아 평균 벡터 정규화 후에도 **cosine similarity가 0**으로 출력되는 문제 발생
- 해결 방안:
  - `enhance_network()` 내부의 `np.exp(-ssm/σ)` 이전에 스케일 보정 (`ssm /= std`, log, MinMax 등)
  - 요약이 아닌 원본 특징 기반 DTW/soft-DTW 방식 적용
  - 요약 제거 후 단순 평균 벡터 비교 방식 실험

---

## 🧪 향후 실험 계획(ing..)

- 다양한 요약 전략 repeat, thumb, 평균 비교 등 실험
- 대규모 원본 데이터 기반 다곡 비교 실험
- 결과 해석력을 높이기 위한 시각화 및 공통 구간 강조 기능 구현

---

## 📌 참고 사항

- 캐시 디렉토리는 쿼리 ID별로 분리 저장되어 효율적인 실험 반복 가능
- 데이터는 GitHub에 포함되지 않으며 직접 다운로드 후 사용해야 함

---

