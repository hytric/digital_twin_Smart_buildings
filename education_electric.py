import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------
# 1) 데이터 불러오기
# ------------------------------------------------
elec_data = pd.read_csv("./data/archive/electricity.csv",
                        index_col='timestamp', parse_dates=True)
weather_data = pd.read_csv("./data/archive/weather.csv",
                           index_col='timestamp', parse_dates=True)

# weather_data에서 Panther 사이트만 필터
weather_data_panther = weather_data[weather_data['site_id'] == 'Panther']
# 2016~2017 데이터만 사용한다고 가정
weather_data_panther_2years = weather_data_panther.truncate(before='2016-07-01', after='2017-12-31')

# 예: airTemperature만 사용
air_temp = weather_data_panther_2years["airTemperature"].copy()
air_temp.ffill()

# ------------------------------------------------
# 2) 여러 건물을 한꺼번에 학습하기 위해 건물 리스트 추출
# ------------------------------------------------
# "Panther_education"이 들어가는 컬럼만 추출
edu_columns = [
    'Panther_education_Teofila', 'Panther_education_Jerome', 'Panther_education_Misty',
    'Panther_education_Tina', 'Panther_education_Janis', 'Panther_education_Quintin',
    'Panther_education_Violet', 'Panther_education_Edna', 'Panther_education_Sophia',
    'Panther_education_Hugh', 'Panther_education_Annetta', 'Panther_education_Ivan',
    'Panther_education_Alecia', 'Panther_education_Rosalie', 'Panther_education_Jonathan'
]

# 2년치 데이터만 사용
elec_data_2years = elec_data.truncate(before='2016-07-01', after='2017-12-31')

# ------------------------------------------------
# 2-1) 컬럼 단위로 이상치 제거 및 결측치 보간 처리 함수 정의
# ------------------------------------------------
def process_column(df, column):
    """
    데이터프레임에서 특정 컬럼의 이상치를 제거하고 결측치를 보간하여 처리.
    """
    col_data = df[column].copy()
    
    # IQR 방식으로 이상치 제거
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    col_data = col_data.mask((col_data < lower_bound) | (col_data > upper_bound))
    
    col_data = col_data.ffill().bfill()
    
    return col_data


# "Panther_education" 컬럼만 필터링
filtered_columns = edu_columns

# 컬럼별로 이상치 제거 및 보간 처리
processed_columns = {}
for col in filtered_columns:
    processed_columns[col] = process_column(elec_data_2years, col)

# 처리된 컬럼들을 합쳐서 새로운 DataFrame 생성
processed_elec_data = pd.DataFrame(processed_columns, index=elec_data_2years.index)


# ------------------------------------------------
# 3) 공통으로 사용할 LSTM 모델 정의
# ------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # 초기 hidden, cell
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]   # 마지막 타임스텝
        out = self.fc(out)
        return out


# ------------------------------------------------
# 4) 데이터셋 구성용 함수
# ------------------------------------------------
def make_sequence_dataset(df, input_length=24, pred_length=6):
    x_list = []
    y_list = []
    
    for i in range(len(df) - input_length - pred_length + 1):
        x_window = df.iloc[i:i+input_length]
        y_window = df.iloc[i+input_length : i+input_length+pred_length]
        
        # X: (time_steps, features=[전력, 온도])
        x_list.append(x_window[['electricity', 'temp']].values)
        # Y: (pred_length,) = 전력만
        y_list.append(y_window['electricity'].values)
    
    return np.array(x_list), np.array(y_list)


# ------------------------------------------------
# 5) 모델 학습/평가를 묶은 함수 (한 건물 단위)
# ------------------------------------------------
def train_and_evaluate_for_one_building(building_name,
                                        df_elec,
                                        df_temp,
                                        input_length=24,
                                        pred_length=6,
                                        test_ratio=0.1,
                                        val_ratio=0.1,
                                        hidden_size=64,
                                        epochs=20,
                                        batch_size=32):
    """
    building_name: 예) "Panther_education_Teofila"
    df_elec: 해당 건물의 전력 시계열 (Series)
    df_temp: 온도 시계열 (Series)
    """
    # 1) merge
    df = pd.merge(df_elec, df_temp, left_index=True, right_index=True, how='inner')
    df.columns = ['electricity', 'temp']
    df.sort_index(inplace=True)
    
    # 2) train/val/test 분할
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    # 3) Sequence Dataset 만들기
    train_X, train_Y = make_sequence_dataset(train_df, input_length, pred_length)
    val_X,   val_Y   = make_sequence_dataset(val_df,   input_length, pred_length)
    test_X,  test_Y  = make_sequence_dataset(test_df,  input_length, pred_length)
    
    # Tensor 변환
    train_X_t = torch.from_numpy(train_X).float()
    train_Y_t = torch.from_numpy(train_Y).float()
    val_X_t   = torch.from_numpy(val_X).float()
    val_Y_t   = torch.from_numpy(val_Y).float()
    test_X_t  = torch.from_numpy(test_X).float()
    test_Y_t  = torch.from_numpy(test_Y).float()
    
    # DataLoader
    train_ds = TensorDataset(train_X_t, train_Y_t)
    val_ds   = TensorDataset(val_X_t,   val_Y_t)
    test_ds  = TensorDataset(test_X_t,  test_Y_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    # 4) 모델 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_dim=2,    # [전력, 온도]
                      hidden_dim=hidden_size,
                      output_dim=pred_length,
                      num_layers=1)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 5) 학습 루프
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, epochs+1):
        # (a) train
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        train_epoch_loss = running_loss / len(train_loader.dataset)
        
        # (b) val
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, Y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
        
        val_epoch_loss = running_val_loss / len(val_loader.dataset)
        
        print(f"[{building_name}] Epoch {epoch}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
        
        # Best 모델 업데이트
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = model.state_dict()
    
    # 학습 종료 후 Best 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 6) 테스트 셋 성능
    model.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            pred_list.append(preds.cpu().numpy())
            true_list.append(Y_batch.numpy())
    
    pred_test = np.concatenate(pred_list, axis=0)
    true_test = np.concatenate(true_list, axis=0)
    
    # Flatten
    pred_test_flat = pred_test.flatten()
    true_test_flat = true_test.flatten()
    
    mae = mean_absolute_error(true_test_flat, pred_test_flat)
    mse = mean_squared_error(true_test_flat, pred_test_flat)
    rmse = np.sqrt(mse)
    mape = 100 * np.mean(np.abs((true_test_flat - pred_test_flat) / (true_test_flat + 1e-6)))
    
    print(f"[{building_name}] Test MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
    
    # 7) 피크 경고 로직 (범용 버전)
    warnings = []

    # 건물별 평균 및 표준편차 기반 임계값 설정
    mean_usage = true_test.mean(axis=1)  # 건물별 평균 사용량
    std_usage = true_test.std(axis=1)    # 건물별 표준편차
    threshold_factor = 3  # 평균 + 3 * 표준편차 (범위를 넓히고 싶으면 증가 가능)

    # 알람 생성 함수
    def calc_diff(true_arr, pred_arr):
        return true_arr - pred_arr

    for i in range(len(pred_test)):
        # 각 시점별 임계값 계산
        dynamic_threshold = mean_usage[i] + threshold_factor * std_usage[i]
        
        # 최대값과 차이를 계산
        peak_pred = pred_test[i].max()  # 예측값 중 최대값
        max_diff = calc_diff(true_test[i], pred_test[i]).max()  # 예측값과 실제값 차이
        
        # 임계값 초과 여부 확인
        if peak_pred > dynamic_threshold or abs(max_diff) > dynamic_threshold * 0.3:
            warnings.append({
                'index': i,
                'peak_pred': peak_pred,
                'dynamic_threshold': dynamic_threshold,
                'max_diff': max_diff
            })

    
    # 결과 리턴
    result_dict = {
        "building_name": building_name,
        "model": model,
        "test_metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        },
        "test_predictions": pred_test,  # (samples, 6)
        "test_actuals": true_test,      # (samples, 6)
        "warnings": warnings
    }
    return result_dict


import os

# 모델 파라미터 저장 디렉토리 생성
model_save_dir = "./models"
os.makedirs(model_save_dir, exist_ok=True)

# ------------------------------------------------
# 6) 실제로 루프 돌리면서 각 건물별 결과 저장
# ------------------------------------------------
results = []

for bld_col in edu_columns:
    # 1) 해당 건물의 전력 시계열만 추출
    elec_series = processed_elec_data[bld_col].copy()
    try:
        # 2) 학습/평가
        r = train_and_evaluate_for_one_building(
            building_name=bld_col,
            df_elec=elec_series.to_frame(),  # merge 편의를 위해 DF 형태
            df_temp=air_temp.to_frame(),
            input_length=24,
            pred_length=6,
            test_ratio=0.2,
            val_ratio=0.1,
            hidden_size=64,
            epochs=30,
            batch_size=32
        )
        results.append(r)
        
        # 3) 학습된 모델 저장
        model_path = os.path.join(model_save_dir, f"{bld_col}_lstm_model.pt")
        torch.save(r["model"].state_dict(), model_path)
        print(f"Model for {bld_col} saved at: {model_path}")
    
    except Exception as e:
        print(f'''
-----------------------------------------
Error processing building: {bld_col}
Error: {e}
-----------------------------------------
        ''')

# ------------------------------------------------
# 7) 최종 결과 종합
# ------------------------------------------------
for res in results:
    bld_name = res["building_name"]
    mae = res["test_metrics"]["MAE"]
    rmse = res["test_metrics"]["RMSE"]
    mape = res["test_metrics"]["MAPE"]
    n_warn = len(res["warnings"])
    
    print(f"\n=== [{bld_name}] 결과 요약 ===")
    print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%, Warnings={n_warn}")

# 필요 시 모델 파라미터, 예측값 등을 후속 분석이나 시각화에 활용 가능