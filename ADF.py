'''
['Panther_education_Teofila', 'Panther_education_Jerome', 'Panther_education_Misty', 
'Panther_education_Tina', 'Panther_education_Janis', 'Panther_education_Quintin', 
'Panther_education_Violet', 'Panther_education_Edna', 'Panther_education_Sophia', 
'Panther_education_Hugh', 'Panther_education_Annetta', 'Panther_education_Ivan', 
'Panther_education_Alecia', 'Panther_education_Rosalie', 'Panther_education_Jonathan', 
'Panther_education_Neal', 'Panther_education_Mohammad', 'Panther_education_Enriqueta', 
'Panther_education_Shelton', 'Panther_education_Aurora', 'Panther_education_Vincent', 
'Panther_education_Mattie', 'Panther_education_Genevieve', 'Panther_education_Diann', 
'Panther_education_Emily', 'Panther_education_Scarlett', 'Panther_education_Zelda', 
'Panther_education_Gina', 'Panther_education_Karri', 'Panther_education_Cleopatra']
'''

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
air_temp.fillna(method='ffill')

# ------------------------------------------------
# 2) 여러 건물을 한꺼번에 학습하기 위해 건물 리스트 추출
# ------------------------------------------------
# "Panther_education"이 들어가는 컬럼만 추출
edu_columns = [col for col in elec_data.columns if ("Panther" in col) and ("education" in col)]

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
    
    # 결측치 처리 수정 (ffill과 bfill 사용)
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

# 결과 출력
print("컬럼별 이상치 제거 및 결측치 보간 처리 완료!")
print(processed_elec_data.head())

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# ------------------------------------------------
# 1) ACF 및 ADF 검정 함수 정의
# ------------------------------------------------
def validate_column(data, column, adf_threshold=0.05, plot_acf_flag=False):
    """
    ACF와 ADF 검정을 실행하여 컬럼을 검증.
    
    Parameters:
    data (pd.DataFrame): 검증할 데이터프레임
    column (str): 컬럼명
    adf_threshold (float): ADF 검정의 p-value 임계값 (기본값: 0.05)
    plot_acf_flag (bool): ACF 그래프를 출력할지 여부
    
    Returns:
    bool: 컬럼이 검증을 통과하면 True, 그렇지 않으면 False
    """
    col_data = data[column].dropna()  # 결측치 제거
    
    # 1. ACF 검정 (옵션으로 그래프 출력)
    if plot_acf_flag:
        print(f"ACF for column: {column}")
        plot_acf(col_data, lags=30)
        plt.show()
    
    # 2. ADF 검정
    adf_result = adfuller(col_data)
    p_value = adf_result[1]
    
    print(f"Column: {column}")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("-" * 40)
    
    # p-value가 임계값(adf_threshold)보다 크면 비정상 데이터로 판단
    return p_value < adf_threshold

# ------------------------------------------------
# 2) ACF 및 ADF 검증 실행
# ------------------------------------------------
validated_columns = []

for col in processed_elec_data.columns:
    # 컬럼 검증: ADF 임계값 0.05, ACF 그래프 비활성화
    if validate_column(processed_elec_data, col, adf_threshold=0.05, plot_acf_flag=False):
        validated_columns.append(col)

# ------------------------------------------------
# 3) 검증을 통과한 컬럼만 선택
# ------------------------------------------------
validated_data = processed_elec_data[validated_columns]

print("검증을 통과한 컬럼:")
print(validated_columns)

# ------------------------------------------------
# 4) 검증 결과 시각화 (옵션)
# ------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=validated_data)
plt.title("Validated Electricity Data")
plt.xlabel("Timestamp")
plt.ylabel("Electricity Usage")
plt.legend(validated_columns)
plt.show()