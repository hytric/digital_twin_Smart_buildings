import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import torch
import torch.nn as nn
import os
import pickle  # 만약 스케일러를 사용하는 경우

###############################
# 1) 데이터 불러오기 및 전처리 함수
###############################
@st.cache_data
def load_and_process_data():
    # 전처리된 데이터 로드
    elec_data = pd.read_csv(
        "./data/processed_electricity.csv", 
        index_col="timestamp", parse_dates=True
    )
    weather_data = pd.read_csv(
        "./data/archive/weather.csv", 
        index_col="timestamp", parse_dates=True
    )
    weather_panther = weather_data[weather_data["site_id"] == "Panther"]
    weather_panther = weather_panther.truncate("2017-07-01", "2017-12-31")
    air_temp = weather_panther["airTemperature"].copy().ffill()

    edu_cols = [
        c for c in elec_data.columns 
        if ("Panther" in c) and ("education" in c)
    ]
    return elec_data, air_temp, edu_cols

###############################
# 2) LSTM 모델 정의 & 로드
###############################
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

@st.cache_resource
def get_valid_buildings(models_dir):
    """
    모델 디렉토리 내에 존재하는 모델 파일들을 기반으로 건물 이름 리스트를 생성합니다.
    """
    if not os.path.exists(models_dir):
        st.error(f"모델 디렉토리가 존재하지 않습니다: {models_dir}")
        return []
    
    files = os.listdir(models_dir)
    model_files = [f for f in files if f.endswith("_lstm_model.pt")]
    building_names = [f.replace("_lstm_model.pt", "") for f in model_files]
    
    return building_names

@st.cache_resource
def load_model(building_name, models_dir):
    """
    building_name: 예) "Panther_education_Teofila"
    """
    model_path = os.path.join(models_dir, f"{building_name}_lstm_model.pt")
    model = LSTMModel(input_dim=2, hidden_dim=64, output_dim=6, num_layers=1)
    try:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        print(f"모델 로드 성공: {model_path}")
        return model
    except FileNotFoundError:
        print(f"[오류] 모델 파일을 찾을 수 없음: {model_path}")
        return None
    except Exception as e:
        print(f"[오류] 모델 로드 중 문제 발생: {e}")
        return None

###############################
# 3) 예측 함수
###############################
def predict_6h(building_name, target_time, elec_data, air_temp, models_dir):
    """
    building_name (str): 건물명
    target_time (pd.Timestamp): 사용자가 지정한 '현재 시각'
    elec_data (DataFrame): 전처리된 전력 데이터 (시계열)
    air_temp (Series): 전처리된 온도 데이터 (시계열)
    models_dir (str): 모델 저장 디렉토리 경로
    
    Returns: pd.DataFrame(columns=['timestamp', 'actual', 'pred', 'temp', 'data_role'])
             - 'pred'는 T+1 ~ T+6 구간만 값이 채워져 있고, 나머지는 NaN
             - 'data_role'은 내부적으로 과거/미래 구분용
             - 최종 테이블에서는 data_role은 숨기고 temp 추가
    """
    model = load_model(building_name, models_dir)
    if model is None:
        # 모델이 없으면 빈 데이터프레임 반환
        st.error(f"{building_name}에 대한 모델을 로드할 수 없습니다.")
        return pd.DataFrame(columns=["timestamp", "actual", "pred", "temp", "data_role"])
    
    # -------- 1) 데이터 준비 --------
    # (a) 과거 24시간 입력 -> 미래 6시간 예측
    input_length = 24
    pred_length = 6
    
    earliest_time = elec_data.index.min() + pd.Timedelta(hours=input_length)
    latest_time = elec_data.index.max() - pd.Timedelta(hours=pred_length)
    
    if target_time < earliest_time or target_time > latest_time:
        # 과거 시점이 너무 이르거나, 미래 시점이 데이터 범위를 초과하면 예측 불가
        st.warning("선택한 시각으로 예측할 수 있는 데이터가 부족합니다.")
        return None
    
    # (b) 전력 & 온도 시계열에서 [T-24, T) 구간 추출
    start_input = target_time - pd.Timedelta(hours=input_length)
    end_input   = target_time
    
    usage_input = elec_data[building_name].loc[start_input:end_input].resample("H").mean().ffill()
    temp_input  = air_temp.loc[start_input:end_input].resample("H").mean().ffill()
    
    # 입력 데이터가 24개보다 적으면 예측 불가
    if len(usage_input) < input_length or len(temp_input) < input_length:
        st.warning("입력 데이터가 부족하여 예측할 수 없습니다.")
        return None
    
    # (c) 모델에 넣을 shape=(1,24,2) 텐서
    X = pd.DataFrame({
        "electricity": usage_input.values[-input_length:], 
        "temp": temp_input.values[-input_length:]
    })
    
    X_tensor = torch.tensor(X.values, dtype=torch.float).unsqueeze(0)
    
    # -------- 2) 예측 수행 --------
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze(0).numpy()
    
    # -------- 3) 예측 결과를 DataFrame에 반영 --------
    # (a) 결과 구간: [T-18, T+6]
    past_18_start = target_time - pd.Timedelta(hours=18)
    future_6_end = target_time + pd.Timedelta(hours=6)
    
    usage_24 = elec_data[building_name].loc[past_18_start:future_6_end].resample("H").mean().ffill()
    temp_24  = air_temp.loc[past_18_start:future_6_end].resample("H").mean().ffill()
    
    df_24 = pd.DataFrame({
        "timestamp": usage_24.index,
        "actual": usage_24.values,
        "temp": temp_24.values,
    })
    df_24["pred"] = np.nan
    
    # (b) 예측치 배치: T+1 ~ T+6 구간
    pred_times = [target_time + pd.Timedelta(hours=(i+1)) for i in range(pred_length)]
    
    for i in range(pred_length):
        pred_time = pred_times[i]
        if pred_time in df_24["timestamp"].values:
            df_24.loc[df_24["timestamp"] == pred_time, "pred"] = y_pred[i]
        else:
            st.warning(f"예측 시각 {pred_time}이 데이터프레임에 존재하지 않습니다.")
    
    # (c) 데이터 역할 추가: 'input'과 'prediction' 구분(차트에서 구분용)
    df_24["data_role"] = np.where(
        df_24["timestamp"] < target_time + pd.Timedelta(hours=1), 
        "input", 
        "prediction"
    )
    
    return df_24

###############################
# 4) Streamlit 메인
###############################
def main():
    st.title("전력 Usage 예측 시스템 (Panther_education)")
    
    # ----- (A) 데이터 로드 및 전처리 -----
    elec_data, air_temp, edu_columns = load_and_process_data()
    
    # ----- (B) 모델 디렉토리 설정 -----
    models_dir = "/Users/jongha/vscode/digital_twin_Smart_buildings/models"
    
    # ----- (C) 유효한 모델 리스트 가져오기 -----
    valid_buildings = get_valid_buildings(models_dir)
    
    if not valid_buildings:
        st.error("사용 가능한 모델이 없습니다. 모델 파일을 확인하세요.")
        return
    
    # ----- (D) 사이드바: 건물 / 날짜 / 시각 선택 -----
    st.sidebar.header("조회 옵션")
    selected_building = st.sidebar.selectbox("건물을 선택하세요", sorted(valid_buildings))
    
    # 날짜 선택 범위를 2017-07-02 ~ 2017-12-31로 고정
    min_dt = datetime.date(2017, 7, 2)
    max_dt = datetime.date(2017, 12, 31)
    selected_date = st.sidebar.date_input(
        "날짜를 선택하세요", 
        value=min_dt,  # 기본값
        min_value=min_dt,
        max_value=max_dt
    )
    
    # 시각 선택 (0~23)
    selected_hour = st.sidebar.slider("시각(0~23)", 0, 23, 10)
    
    # 최종 Timestamp
    target_time = pd.Timestamp(
        year=selected_date.year,
        month=selected_date.month,
        day=selected_date.day,
        hour=selected_hour
    )
    
    # ----- (E) 예측 실행 버튼 -----
    if st.sidebar.button("예측 실행"):
        with st.spinner("예측을 수행 중입니다..."):
            df_24 = predict_6h(selected_building, target_time, elec_data, air_temp, models_dir)
        
        if df_24 is None or df_24.empty:
            st.warning("입력 데이터가 부족하거나 예측 범위를 벗어났습니다.")
            return
        
        # ----- (F) 시각화 -----
        st.subheader(f"[{selected_building}] {target_time} 기준 24시간 (과거 18h + 미래 6h)")
        
        # Melt 데이터 준비 (actual vs pred 비교용)
        chart_data = df_24.melt(
            id_vars=["timestamp", "data_role", "temp"], 
            value_vars=["actual", "pred"], 
            var_name="type", 
            value_name="usage"
        )
        
        # 기본 선 그래프
        line_chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="시간"),
                y=alt.Y("usage:Q", title="전력 사용량"),
                color=alt.Color("type:N", title="데이터 종류"),
                tooltip=["timestamp:T", "type:N", "usage:Q", "temp:Q"]
            )
        )
        
        # 현재 시간을 표시하는 수직선 추가
        rule = (
            alt.Chart(pd.DataFrame({'timestamp': [target_time]}))
            .mark_rule(color='red')
            .encode(x='timestamp:T')
        )
        
        # 그래프 합성
        final_chart = line_chart + rule
        
        # 차트 속성 설정
        final_chart = final_chart.properties(
            width=700,
            height=400,
            title="18시간 과거 + 6시간 미래 (실제 vs 예측)"
        ).interactive()
        
        # 차트 출력
        st.altair_chart(final_chart, use_container_width=True)
        
        # ----- (G) 테이블 표시 (data_role 제외, temp 포함) -----
        st.subheader(f"[{selected_building}] {target_time} 기준 24시간 데이터 테이블")
        st.dataframe(
            df_24.drop(columns=["data_role"]).set_index("timestamp")
        )
        
    st.write(f"선택된 시각: {target_time}")
    

if __name__ == "__main__":
    main()