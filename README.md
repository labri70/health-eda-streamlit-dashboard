# Health EDA Streamlit Dashboard

심장병과 당뇨 미니프로젝트 분석 결과를 정리한 Streamlit 대시보드입니다.

## 주요 기능

- 사이드바에서 `심장병 EDA`, `당뇨 EDA` 선택
- 데이터 요약 카드와 원본 데이터 미리보기
- 주요 변수별 시각화
- 모델 결과와 혼동행렬 요약
- 각 표와 그래프에 대한 간단한 결과 해석

## 폴더 구조

```text
health-eda-streamlit/
├─ service_app.py
├─ data/
│  ├─ heart.csv
│  └─ diabetes.csv
├─ requirements.txt
├─ packages.txt
└─ README.md
```

## 로컬 실행

```powershell
cd E:\2026-EST-Learn\Python\Streamlit\health-eda-streamlit
..\.venv\Scripts\Activate.ps1
python -m streamlit run service_app.py
```

## Streamlit Cloud 배포 설정

```text
Repository: health-eda-streamlit-dashboard
Branch: main
Main file path: service_app.py
```

## 주의

이 앱은 교육용 데이터 분석 대시보드입니다. 실제 의학적 진단, 치료, 건강검진 판정 목적으로 사용할 수 없습니다.
