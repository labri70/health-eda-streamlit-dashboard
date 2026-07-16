from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


st.set_page_config(
    page_title="건강 데이터 EDA 대시보드",
    page_icon="🩺",
    layout="wide",
)


def set_korean_font():
    plt.rcParams["axes.unicode_minus"] = False
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            plt.rcParams["font.sans-serif"] = [font]
            break


sns.set_theme(style="whitegrid")
set_korean_font()


@st.cache_data
def load_heart_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "heart.csv")


@st.cache_data
def load_diabetes_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "diabetes.csv")


def show_metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def explain_result(summary: str, terms: list[tuple[str, str]] | None = None):
    st.info(summary)
    if terms:
        with st.expander("용어 설명"):
            for term, description in terms:
                st.markdown(f"- **{term}**: {description}")


def pct(value: float) -> str:
    return f"{value:.1f}%"


def bar_rate_chart(data: pd.DataFrame, x: str, y: str, title: str, x_label: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x=x, y=y, ax=ax, color="#2F80ED")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("비율(%)")
    ax.set_ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)
    st.pyplot(fig, use_container_width=True)


def heatmap_from_matrix(matrix, labels, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    st.pyplot(fig, use_container_width=True)


def intro():
    st.title("건강 데이터 EDA 대시보드")
    st.caption("심장병과 당뇨 미니프로젝트 분석 결과를 Streamlit으로 정리한 포트폴리오 앱")
    st.info(
        "이 앱은 교육용 데이터 분석 대시보드입니다. 실제 의학적 진단이나 치료 판단에는 사용할 수 없습니다."
    )


def apply_sidebar_style():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
            font-size: 1.75rem;
            font-weight: 800;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 1.35rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
            gap: 0.45rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
            font-size: 1.05rem;
        }
        [data-testid="stSidebar"] .sidebar-footer {
            position: fixed;
            bottom: 1.75rem;
            left: 1.25rem;
            width: 16rem;
            color: rgba(49, 51, 63, 0.68);
            font-size: 0.83rem;
            line-height: 1.7;
        }
        [data-testid="stSidebar"] .sidebar-footer hr {
            margin: 0 0 1.25rem 0;
            border: 0;
            border-top: 1px solid rgba(49, 51, 63, 0.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def heart_page():
    df = load_heart_data()
    df_clean = df.drop_duplicates().copy()

    st.header("심장병 EDA")
    st.caption("Kaggle Heart Disease Dataset 기반 분석 요약")

    show_metric_row(
        [
            ("원본 데이터", f"{len(df):,}행", "중복 제거 전 데이터 크기"),
            ("중복 제거 후", f"{len(df_clean):,}행", "주요 해석 기준 데이터"),
            ("심장병 있음", f"{df_clean['target'].mean() * 100:.1f}%", "target=1 비율"),
            ("테스트 정확도", "80.3%", "미니프로젝트 로지스틱 회귀 결과"),
        ]
    )

    tab_overview, tab_viz, tab_model, tab_note = st.tabs(
        ["데이터 개요", "핵심 시각화", "모델 결과", "해석 및 주의사항"]
    )

    with tab_overview:
        st.subheader("데이터 미리보기")
        st.dataframe(df_clean.head(80), use_container_width=True)
        explain_result(
            f"이 표는 중복을 제거한 심장병 데이터 {len(df_clean):,}명 중 일부를 보여줍니다. 한 행은 한 사람의 검사 기록이고, 이 데이터에서는 심장병 있음(target=1)이 {pct(df_clean['target'].mean() * 100)}로 약 절반보다 조금 많습니다.",
            [
                ("target", "심장병 여부입니다. 0은 심장병 없음, 1은 심장병 있음을 의미합니다."),
                ("cp", "흉통 유형입니다. 값이 0~3처럼 숫자로 저장되어 있어 숫자 자체보다 그룹별 차이를 비교하는 용도로 봅니다."),
                ("thalach", "검사 중 측정된 최대 심박수입니다."),
                ("ca", "형광 투시 검사에서 확인된 주요 혈관 수를 뜻합니다."),
            ],
        )

        st.subheader("컬럼 요약")
        st.dataframe(df_clean.describe().T, use_container_width=True)
        explain_result(
            f"이 표는 각 변수의 평균과 범위를 요약합니다. 예를 들어 평균 나이는 {df_clean['age'].mean():.1f}세, 평균 콜레스테롤은 {df_clean['chol'].mean():.1f}이며, 이런 기초 통계를 먼저 봐야 뒤의 발병률 차이를 해석할 수 있습니다.",
            [
                ("count", "해당 컬럼에 값이 들어 있는 데이터 개수입니다."),
                ("mean", "평균값입니다."),
                ("std", "표준편차입니다. 값들이 평균 주변에 얼마나 퍼져 있는지를 보여줍니다."),
                ("25%, 50%, 75%", "데이터를 작은 값부터 정렬했을 때의 사분위수입니다. 50%는 중앙값입니다."),
            ],
        )

        target_counts = (
            df_clean["target"]
            .value_counts()
            .rename(index={0: "심장병 없음", 1: "심장병 있음"})
            .reset_index()
        )
        target_counts.columns = ["구분", "인원 수"]
        st.dataframe(target_counts, use_container_width=True)
        target_0 = int((df_clean["target"] == 0).sum())
        target_1 = int((df_clean["target"] == 1).sum())
        explain_result(
            f"이 표는 심장병 없음 {target_0}명, 심장병 있음 {target_1}명으로 구성되어 있음을 보여줍니다. 두 그룹이 완전히 같지는 않지만 한쪽으로 심하게 치우친 데이터는 아니어서 비교 분석을 진행하기에 무리가 적습니다.",
            [("인원 수", "각 구분에 해당하는 데이터 행의 개수입니다.")],
        )

    with tab_viz:
        st.subheader("흉통 유형별 심장병 발병률")
        cp_rate = (
            df_clean.groupby("cp", observed=False)["target"]
            .mean()
            .mul(100)
            .reset_index(name="발병률")
        )
        bar_rate_chart(cp_rate, "cp", "발병률", "흉통 유형(cp)별 심장병 발병률", "흉통 유형(cp)")
        cp_min = cp_rate.loc[cp_rate["발병률"].idxmin()]
        cp_max = cp_rate.loc[cp_rate["발병률"].idxmax()]
        explain_result(
            f"이 그래프는 흉통 유형에 따라 심장병 발병률이 크게 달라짐을 보여줍니다. 가장 낮은 그룹은 cp={int(cp_min['cp'])}의 {pct(cp_min['발병률'])}, 가장 높은 그룹은 cp={int(cp_max['cp'])}의 {pct(cp_max['발병률'])}로, 흉통 유형이 심장병 여부와 강하게 연결되어 보입니다.",
            [
                ("발병률", "해당 그룹 안에서 target=1인 사람의 비율입니다."),
                ("cp", "흉통 유형을 숫자로 인코딩한 값입니다. 의료적 의미는 원자료의 코드 정의와 함께 확인해야 합니다."),
            ],
        )

        c1, c2 = st.columns(2)
        with c1:
            sex_rate = (
                df_clean.groupby("sex", observed=False)["target"]
                .mean()
                .mul(100)
                .reset_index(name="발병률")
            )
            sex_rate["sex_label"] = sex_rate["sex"].map({0: "여성(0)", 1: "남성(1)"})
            bar_rate_chart(sex_rate, "sex_label", "발병률", "성별 심장병 발병률", "성별")
            sex_values = dict(zip(sex_rate["sex_label"], sex_rate["발병률"]))
            explain_result(
                f"이 그래프는 성별에 따라 심장병 발병률 차이가 있음을 보여줍니다. 이 데이터에서는 여성(0)이 {pct(sex_values.get('여성(0)', 0))}, 남성(1)이 {pct(sex_values.get('남성(1)', 0))}로 나타나며, 단일 변수 해석보다는 다른 검사 수치와 함께 보는 것이 안전합니다.",
                [("sex", "이 데이터에서는 0과 1로 저장된 성별 구분 변수입니다.")],
            )

        with c2:
            age_df = df_clean.copy()
            age_df["age_group"] = pd.cut(
                age_df["age"],
                bins=[20, 40, 50, 60, 80],
                labels=["40세 이하", "41~50세", "51~60세", "61세 이상"],
            )
            age_rate = (
                age_df.groupby("age_group", observed=False)["target"]
                .mean()
                .mul(100)
                .reset_index(name="발병률")
            )
            bar_rate_chart(age_rate, "age_group", "발병률", "연령대별 심장병 발병률", "연령대")
            age_max = age_rate.loc[age_rate["발병률"].idxmax()]
            age_min = age_rate.loc[age_rate["발병률"].idxmin()]
            explain_result(
                f"이 그래프는 연령대를 나누어 심장병 발병률을 비교합니다. 가장 높은 구간은 {age_max['age_group']}의 {pct(age_max['발병률'])}, 가장 낮은 구간은 {age_min['age_group']}의 {pct(age_min['발병률'])}입니다. 다만 연령이 높을수록 항상 단조롭게 증가하는 형태는 아니므로 다른 변수와 함께 해석해야 합니다.",
                [("연령대", "연속형 나이 값을 해석하기 쉽게 몇 개의 구간으로 묶은 것입니다.")],
            )

    with tab_model:
        st.subheader("로지스틱 회귀 모델 결과 요약")
        show_metric_row(
            [
                ("정확도", "80.3%", "중복 제거 후 테스트 데이터 기준"),
                ("실제 환자 중 맞춘 수", "28 / 33", "심장병 있음 class의 맞춘 수"),
                ("놓친 환자 수", "5명", "실제 심장병인데 정상으로 예측한 수"),
            ]
        )
        heatmap_from_matrix([[21, 7], [5, 28]], ["정상", "심장병"], "혼동행렬")
        explain_result(
            "이 혼동행렬은 모델이 정상 21명과 심장병 28명을 맞췄지만, 정상 7명을 심장병으로 잘못 예측하고 실제 심장병 5명을 정상으로 놓쳤음을 보여줍니다. 의료 선별에서는 특히 실제 환자를 놓치는 5건에 주의해야 합니다.",
            [
                ("정확도", "전체 예측 중 맞춘 비율입니다."),
                ("혼동행렬", "실제값과 예측값을 교차해서 맞고 틀린 개수를 보여주는 표입니다."),
                ("놓친 환자 수", "실제 심장병이 있는데 모델이 정상으로 예측한 경우입니다. 의료 선별에서는 특히 주의해서 봐야 합니다."),
            ],
        )
        st.markdown(
            """
            계수 절댓값 기준으로는 `cp`, `sex`, `thalach`, `ca`, `thal` 변수가 상대적으로 중요했습니다.
            의료 선별 상황에서는 단순 정확도보다 실제 환자를 놓치지 않는 recall 관점이 중요합니다.
            """
        )

    with tab_note:
        st.subheader("핵심 해석")
        st.markdown(
            """
            - 가설인 **흉통 유형(`cp`)에 따라 심장병 진단 비율이 다를 것**이라는 예상은 대체로 맞았습니다.
            - 콜레스테롤(`chol`) 평균 차이는 중복 제거 후 기준에서 통계적으로 유의하다고 보기 어려웠습니다.
            - 이 분석은 관찰 데이터 기반이므로 상관관계를 인과관계로 해석하면 안 됩니다.
            """
        )
        st.warning("교육용 분석 결과입니다. 실제 진단이나 치료 판단에는 의료 전문가의 상담이 필요합니다.")


def diabetes_page():
    df = load_diabetes_data()

    st.header("당뇨 EDA")
    st.caption("Kaggle Pima Indians Diabetes Database 기반 분석 요약")

    show_metric_row(
        [
            ("데이터", f"{len(df):,}행", "전체 데이터 크기"),
            ("당뇨 양성", f"{df['Outcome'].mean() * 100:.1f}%", "Outcome=1 비율"),
            ("테스트 정확도", "73.4%", "미니프로젝트 로지스틱 회귀 결과"),
            ("Test F1", "65.0%", "기본 임계값 0.5 기준"),
        ]
    )

    tab_overview, tab_missing, tab_features, tab_model, tab_note = st.tabs(
        ["데이터 개요", "숨은 결측", "주요 변수", "모델 결과", "해석 및 주의사항"]
    )

    with tab_overview:
        st.subheader("데이터 미리보기")
        st.dataframe(df.head(80), use_container_width=True)
        explain_result(
            f"이 표는 당뇨 데이터 {len(df):,}명 중 일부를 보여줍니다. 한 행은 한 사람의 검사 기록이며, 전체에서 당뇨 양성(Outcome=1)은 {pct(df['Outcome'].mean() * 100)}로 음성보다 적습니다.",
            [
                ("Outcome", "당뇨 여부입니다. 0은 당뇨 음성, 1은 당뇨 양성을 의미합니다."),
                ("Glucose", "혈당 수치입니다. 이 분석에서 당뇨 여부와 가장 뚜렷한 차이를 보인 변수입니다."),
                ("BMI", "체질량지수입니다. 키와 몸무게를 이용해 계산한 비만도 지표입니다."),
                ("Pregnancies", "임신 횟수입니다. 이 데이터셋의 대상 특성상 포함된 변수입니다."),
            ],
        )

        st.subheader("컬럼 요약")
        st.dataframe(df.describe().T, use_container_width=True)
        explain_result(
            f"이 표는 당뇨 데이터의 평균과 범위를 요약합니다. 평균 혈당은 {df['Glucose'].mean():.1f}, 평균 BMI는 {df['BMI'].mean():.1f}, 평균 나이는 {df['Age'].mean():.1f}세입니다. 혈당, 혈압, BMI처럼 0이 현실적으로 나오기 어려운 항목은 숨은 결측값일 수 있습니다.",
            [
                ("mean", "평균값입니다. 당뇨 양성/음성 그룹 비교의 기준으로 자주 사용됩니다."),
                ("min", "최솟값입니다. 의학적으로 불가능한 0이 있는지 확인할 때 봅니다."),
                ("max", "최댓값입니다. 지나치게 큰 이상치가 있는지 확인할 때 봅니다."),
                ("50%", "중앙값입니다. 평균보다 극단값의 영향을 덜 받습니다."),
            ],
        )

        outcome_counts = (
            df["Outcome"]
            .value_counts()
            .rename(index={0: "당뇨 음성", 1: "당뇨 양성"})
            .reset_index()
        )
        outcome_counts.columns = ["구분", "인원 수"]
        st.dataframe(outcome_counts, use_container_width=True)
        outcome_0 = int((df["Outcome"] == 0).sum())
        outcome_1 = int((df["Outcome"] == 1).sum())
        explain_result(
            f"이 표는 당뇨 음성 {outcome_0}명, 당뇨 양성 {outcome_1}명으로 구성되어 있음을 보여줍니다. 양성 데이터가 더 적기 때문에 모델 평가는 정확도뿐 아니라 Recall, F1 Score도 함께 봐야 합니다.",
            [("당뇨 양성", "데이터에서 Outcome=1로 기록된 경우입니다.")],
        )

    with tab_missing:
        st.subheader("0으로 표시된 숨은 결측")
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        zero_rate = (
            (df[zero_cols] == 0)
            .mean()
            .mul(100)
            .reset_index(name="0값 비율")
            .rename(columns={"index": "컬럼"})
        )
        bar_rate_chart(zero_rate, "컬럼", "0값 비율", "컬럼별 0값 비율", "컬럼")
        zero_max = zero_rate.loc[zero_rate["0값 비율"].idxmax()]
        explain_result(
            f"이 그래프는 측정값이 0으로 기록된 비율을 보여줍니다. 특히 {zero_max['컬럼']}의 0값 비율이 {pct(zero_max['0값 비율'])}로 가장 높아, 실제 0이라기보다 측정 누락에 가까운 값으로 보고 처리할 필요가 있습니다.",
            [
                ("숨은 결측", "비어 있음으로 표시되지 않고 0 같은 특정 숫자로 저장된 결측값입니다."),
                ("0값 비율", "전체 데이터 중 해당 컬럼 값이 0인 행의 비율입니다."),
                ("결측 처리", "비어 있거나 잘못 기록된 값을 제거하거나 평균/중앙값 등으로 대체하는 작업입니다."),
            ],
        )

    with tab_features:
        st.subheader("Outcome별 주요 변수 분포")
        feature = st.selectbox("비교할 변수", ["Glucose", "BMI", "Age", "Pregnancies"])
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="Outcome", y=feature, ax=ax, palette=["#8FB9E8", "#F28E8E"], hue="Outcome", legend=False)
        ax.set_title(f"Outcome별 {feature} 분포")
        ax.set_xlabel("Outcome (0=음성, 1=양성)")
        st.pyplot(fig, use_container_width=True)
        feature_means = df.groupby("Outcome")[feature].mean()
        direction = "높습니다" if feature_means.loc[1] > feature_means.loc[0] else "낮습니다"
        explain_result(
            f"이 박스플롯은 `{feature}` 값이 당뇨 음성과 양성 그룹에서 어떻게 다른지 보여줍니다. 현재 선택한 변수는 양성 그룹 평균이 {feature_means.loc[1]:.1f}, 음성 그룹 평균이 {feature_means.loc[0]:.1f}로 양성 그룹이 더 {direction}.",
            [
                ("박스플롯", "데이터의 중앙값, 주요 범위, 이상치를 한 번에 보여주는 그래프입니다."),
                ("중앙값", "값을 작은 순서대로 정렬했을 때 가운데에 있는 값입니다."),
                ("이상치", "다른 값들과 비교해 유난히 크거나 작은 값입니다."),
            ],
        )

        mean_table = (
            df.groupby("Outcome")[["Glucose", "BMI", "Age"]]
            .mean()
            .rename(index={0: "당뇨 음성", 1: "당뇨 양성"})
        )
        st.dataframe(mean_table, use_container_width=True)
        explain_result(
            f"이 표는 당뇨 음성과 양성 그룹의 평균 차이를 직접 비교합니다. 양성 그룹은 혈당 평균 {mean_table.loc['당뇨 양성', 'Glucose']:.1f}, BMI 평균 {mean_table.loc['당뇨 양성', 'BMI']:.1f}, 나이 평균 {mean_table.loc['당뇨 양성', 'Age']:.1f}로 음성 그룹보다 모두 높습니다.",
            [("그룹 평균", "같은 Outcome 값을 가진 사람들끼리 묶어서 계산한 평균입니다.")],
        )

        st.subheader("임신 횟수 구간별 당뇨 양성 비율")
        pregnancy_df = df.copy()
        pregnancy_df["pregnancy_group"] = pd.cut(
            pregnancy_df["Pregnancies"],
            bins=[-1, 0, 2, 5, pregnancy_df["Pregnancies"].max()],
            labels=["0회", "1~2회", "3~5회", "6회 이상"],
        )
        pregnancy_rate = (
            pregnancy_df.groupby("pregnancy_group", observed=False)["Outcome"]
            .mean()
            .mul(100)
            .reset_index(name="당뇨 양성 비율")
        )
        bar_rate_chart(
            pregnancy_rate,
            "pregnancy_group",
            "당뇨 양성 비율",
            "임신 횟수 구간별 당뇨 양성 비율",
            "임신 횟수 구간",
        )
        explain_result(
            f"이 그래프는 임신 횟수가 많은 구간일수록 당뇨 양성 비율이 높아지는 흐름을 보여줍니다. 1~2회 구간은 {pct(float(pregnancy_rate.loc[pregnancy_rate['pregnancy_group'].astype(str) == '1~2회', '당뇨 양성 비율'].iloc[0]))}로 가장 낮고, 6회 이상 구간은 {pct(float(pregnancy_rate.loc[pregnancy_rate['pregnancy_group'].astype(str) == '6회 이상', '당뇨 양성 비율'].iloc[0]))}로 가장 높습니다.",
            [("구간화", "숫자 값을 0회, 1~2회처럼 해석하기 쉬운 범위로 묶는 작업입니다.")],
        )

    with tab_model:
        st.subheader("임계값별 모델 성능")
        threshold_table = pd.DataFrame(
            {
                "임계값": [0.3, 0.5, 0.7],
                "Precision": ["54.5%", "60.3%", "61.9%"],
                "Recall": ["88.9%", "70.4%", "48.1%"],
                "F1 Score": ["67.6%", "65.0%", "54.2%"],
                "Accuracy": ["70.1%", "73.4%", "71.4%"],
            }
        )
        st.dataframe(threshold_table, use_container_width=True, hide_index=True)
        heatmap_from_matrix([[75, 25], [16, 38]], ["음성", "양성"], "혼동행렬 (임계값 0.5)")
        explain_result(
            "이 표와 혼동행렬은 임계값에 따라 모델 판단이 어떻게 달라지는지 보여줍니다. 임계값 0.5에서는 실제 양성 54명 중 38명을 찾고 16명을 놓쳤습니다. 임계값을 0.3으로 낮추면 Recall이 88.9%로 올라가지만, 양성 오경고도 늘어납니다.",
            [
                ("임계값", "모델이 양성이라고 판단하기 위한 기준 확률입니다. 낮추면 양성 예측이 늘어납니다."),
                ("Precision", "양성이라고 예측한 사람 중 실제 양성인 비율입니다."),
                ("Recall", "실제 양성인 사람 중 모델이 양성으로 찾아낸 비율입니다."),
                ("F1 Score", "Precision과 Recall을 함께 반영한 성능 지표입니다."),
                ("False Negative", "실제 양성인데 모델이 음성으로 예측한 경우입니다."),
            ],
        )
        st.markdown(
            """
            임계값을 0.3으로 낮추면 recall이 올라가 실제 당뇨 양성 환자를 놓치는 비율을 줄일 수 있습니다.
            대신 precision은 낮아져 양성으로 잘못 경고하는 경우가 늘어납니다.
            """
        )

    with tab_note:
        st.subheader("핵심 해석")
        st.markdown(
            """
            - 가설인 **혈당(`Glucose`)이 높을수록 당뇨 양성 가능성이 높을 것**이라는 예상은 대체로 맞았습니다.
            - 당뇨 양성 그룹은 `Glucose`, `BMI`, `Age` 평균이 모두 더 높았습니다.
            - 의료 선별 목적이라면 정확도뿐 아니라 recall과 false negative를 함께 봐야 합니다.
            """
        )
        st.warning("교육용 분석 결과입니다. 실제 진단이나 치료 판단에는 의료 전문가의 상담이 필요합니다.")


def main():
    apply_sidebar_style()
    intro()

    st.sidebar.title("EDA 분석")
    page = st.sidebar.radio("분석 선택", ["심장병 EDA", "당뇨 EDA"])
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            <hr />
            <div>데이터 출처: Kaggle 공개 데이터셋 기반 미니프로젝트</div>
            <div style="margin-top: 0.7rem;">배포용 앱 폴더: health-eda-streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "심장병 EDA":
        heart_page()
    else:
        diabetes_page()


if __name__ == "__main__":
    main()
