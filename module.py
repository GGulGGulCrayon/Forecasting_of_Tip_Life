#!/usr/bin/env python
# coding: utf-8

# # 해당쿼리로 데이터 추출
# 
# SELECT datetime, shot_no, tool_number, current_spindle, current_x, current_z, vibration, rpm, rpm_set, feed, feed_set, load_1, servo_load_x, servo_load_z, servo_current_x, servo_current_z from TPOP_MACHINE_PARAMETER
# WHERE mc_id = 22 and date between '2022-07-13' AND '2022-10-12'
# 
# - 날짜 기준 범위는 1개월 반씩 분리해서 Query 수행

# In[1]:


# !jupyter nbconvert --to python preprocessing_anormaly_detection.py


# # 준비

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')
import joblib


# In[2]:


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', 200)


# # 함수 정의

# ## 일반

# ### 컬럼 데이터 형변환 (64bit → 32bit)
#  * 데이터 용량 줄이기 위한 용도

# In[3]:


#주어진 dataframe에서 데이터 타입이 64bit인 경우 32bit로 변경하여 데이터 용량 축소
def change_data_type_64bit_to_32bit(data):
    column_names = list(data.columns.values)
    
    for col in column_names:
        data_type = str(data[col].dtype)
        
        if data_type == 'int64':
            data[col] = data[col].astype('int32')
        elif data_type == 'float64':
            data[col] = data[col].astype('float32')
        #LJY_20220929 : 컬럼의 data type이 'object'인데 실제 값은 numeric인 경우 int32로 변환
        # 추후 데이터 타입 판별로 변경 필요 (numeric인지 체크하는 api에서 datetime을 걸러내지 못해서 임시로 column name으로 체크하도록 처리)
        else:
            if col != "datetime":
                data[col] = data[col].replace('\\0', 0)
                data[col] = data[col].fillna(0).astype('int32')
#                data[col] = (data[col].fillna(0).astype('int32').astype(object).where(data[col].notnull()))
#                data[col] = data[col].notnull().astype('int32')
    
    column_names.clear()
    del column_names


# ### tool 정보 추출
#  * 'tool_number' 필드에서 공구 정보를 추출하여 별도의 컬럼으로 구성

# In[4]:


# tool 번호 및 상태 생성
def extract_tool_info_from_data(data):
    if 'tool_number' not in data.columns:
        return
    
    data['tool_state'] = data['tool_number']%100
    data['tool'] = (data['tool_number'] - data['tool_state'])/100
    if data['tool'].isnull().sum() > 0:
        data = data.dropna(axis = 0)
    data['tool'] = data['tool'].astype(np.int32)


# ## 데이터 불러오기

# ### 주어진 경로의 csv 파일 목록 찾기

# In[5]:


#주어진 경로의 .csv파일을 찾아서 list 형태로 반환
def find_csv_files(path):
    file_list = os.listdir(path)
    
    file_list_csv = [file_csv for file_csv in file_list if file_csv.endswith('.csv')]
    file_list_csv
    
    file_list.clear()
    del file_list
    
    return file_list_csv


# ### 주어진 경로의 대상 csv파일을 불러오기
#  * 입력인자 중 sensor_data_only : True이면 전류 센서 데이터만 포함, False이면 Focas 데이터를 포함한 파일을 의미
#  * 데이터 종류를 입력으로 하여 해당하는 데이터 타입의 .csv파일 Load
#  * 데이터 타입 형변환 (64bit → 32bit, object타입의 경우 datetime 컬럼을 제외하고 int32로 변환)

# In[6]:


#주어진 경로의 .csv파일을 찾아서 모두 병합한 dataframe 구성
def build_dataframe_from_cvs_files(path, sensor_data_only=True, display_report=False):
    file_list_csv = find_csv_files(path)
    
    df = pd.DataFrame()
    
    if len(file_list_csv) < 1:
        return df

    #전류 센서 & focas 데이터 포함 파일의 header부분(컬럼명 나열된 line) 정보
    all_data_columns = '"datetime","shot_no","tool_number","current_spindle","current_x","current_z","vibration","rpm","rpm_set","feed","feed_set","load_1","servo_load_x","servo_load_z","servo_current_x","servo_current_z"'
    # all_data_columns = ',datetime,tool_number,shot_no,current_x,current_z,current_spindle,rpm,feed,load_1,servo_load_x,servo_load_z,servo_current_x,servo_current_z'

    
    #전류 센서데이터만 포함한 파일의 header부분(컬럼명 나열된 line) 정보
    sensor_data_only_columns = '"datetime","tool_number","shot_no","current_spindle","current_x","current_z"'
    
    for csv_idx in file_list_csv:
        first_line = ""
        
        #encoding을 utf-8-sig로 해야 파일에서 읽은 문자열의 맨 앞에 '\ufeff'가 붙는 현상을 막을 수 있음
        # - utf-8로 해서 읽으면 맨 앞에 '\ufeff'가 붙어서 위에 정의한 all_data_columns, sensor_data_only_columns와 비교 시 무조건 False 발생
        with open(csv_idx, encoding="utf-8-sig") as f:
            first_line = f.readline()
            print(first_line)
            first_line = first_line.strip('\n')
            f.close()
        
        if display_report is True:
            print(first_line)
        
        #전류 센서 & focas 데이터 포함 파일 (무의미한 column은 DB에서 조회할 때 제외하고 조회 완료)
        if first_line == all_data_columns:
            if display_report is True:
                print("file {} : all data".format(csv_idx))
                
            if sensor_data_only is True:
                continue
        #전류 센서 데이터만 포함한 파일
        elif first_line == sensor_data_only_columns:
            if display_report is True:
                print("file {} : current sensor data only".format(csv_idx))
                
            if sensor_data_only is False:
                continue
        else:
            if display_report is True:
                print("file {} : invalid data".format(csv_idx))
                
            continue
            
        csv_df = pd.read_csv(csv_idx)
            
        #데이터 타입을 64bit → 32bit로 변경하여 용량 축소
        change_data_type_64bit_to_32bit(csv_df)
    
        #print(csv_df.info())

        df = pd.concat([df, csv_df], axis = 0)
        del csv_df
    
    file_list_csv.clear()
    del file_list_csv
    
    return df


# ### 주어진 경로의 대상 파일을 불러오기
#  * 입력인자 중 sensor_data_only : True이면 전류 센서 데이터만 포함, False이면 Focas 데이터를 포함한 파일을 의미
#  * current directory에서 .parquet파일을 찾아서 불러오기 
#  * .parquet파일이 없는 경우 .csv를 읽어서 통합한 후 .parquet로 저장

# In[7]:


#--------------------------------------------------------------
#☆☆ Parquet(파케이)란?
#   - Apache Parquet은 쿼리 속도를 높이기 위한 최적화를 제공하는 열 형식 파일 형식이며 CSV 또는 JSON보다 훨씬 효율적인 파일 형식입니다.
#   - <참고> : https://pearlluck.tistory.com/561
#--------------------------------------------------------------
#
#
#Parquet(파케이) 파일이 있는 경우 해당 파일을 읽고, 그렇지 않은 경우 .csv 파일을 순회하면서 읽어서 merge
#pyarrow 모듈 설치해야 .parquet 파일 입출력 기능 사용 가능
# !pip install pyarrow

def read_data(sensor_data_only=True, extract_tool_info=True, display_report=False):
    df = pd.DataFrame()
    
    #전류 센서로만 구성된 데이터를 사용하려고 하는 경우
    if sensor_data_only is True:
        if os.path.isfile('./VL04_data_sensor_only.parquet'): #파일이 있는 경우
            df = pd.read_parquet('./VL04_data_sensor_only.parquet')  #.parquet 데이터 ()
    #focas 데이터까지 포함된 데이터를 사용하려고 하는 경우
    else:    
        if os.path.isfile('./VL04_data.parquet'): #파일이 있는 경우
            df = pd.read_parquet('./VL04_data.parquet')  #.parquet 데이터 ()
    
    if df.empty:
        #현재 디렉토리의 .csv파일을 모두 읽어서 하나의 dataframe으로 구성
        df = build_dataframe_from_cvs_files('./', sensor_data_only, display_report)

        df = df.reset_index(drop=True)

        #dataframe을 .pqrquet 파일로 저장하여 추후 상대적으로 적은 메모리의 파일을 빠르게 읽어서 사용 할 수 있도록 처리
        df.to_parquet('./VL04_data.parquet', compression='gzip') #압축 파일 형식(gzip, snappy, ..)을 지정하여 pqrquet 파일 저장
        
    # tool 번호 및 상태 생성
    if not df.empty and extract_tool_info:
        extract_tool_info_from_data(df)
        
    return df


# ## 전처리 : 유효한 shot 정보 도출
#  * 'shot_no' 필드 값을 기준으로 shot 구간 찾기
#  * shot 구간 내부의 공구 사용 순서 및 각 공구 별 데이터 index 구간 도출
#  * 공구 사용 순서에서 유효한 공구 사용 패턴 구간 찾기
#  * 유효한 공구 사용 패턴 구간에 대해 numbering하여 별도의 'real_shot' 필드에 값 설정 (디폴트 : -1) 

# ### dataframe에 사용된 공구 순서 도출

# In[8]:


#dataframe에 사용된 공구를 순서대로 추출해서 list형태로 반환
def extract_tool_list(data, report_result=False):
    import itertools

    tool_list = []
    #print([(k, (g)) for k, g in itertools.groupby(df_shot_specific['tool'])])
    for k, g in itertools.groupby(data['tool']):
        tool_list.append(k)

    if report_result is True:
        print(tool_list)
    
    return tool_list


# In[9]:


#dataframe에 사용된 공구를 순서대로 추출해서 list형태로 반환
def extract_tool_list_with_range(data, report_result=False):
    import itertools

    irow = 0
    tool_list = []
    range_list_of_tool = []
    
    #print([(k, (g)) for k, g in itertools.groupby(df_shot_specific['tool'])])
    for k, g in itertools.groupby(data['tool']):
        tool_list.append(k)
        nrow = len(list(g))
        range_list_of_tool.append((irow,irow+nrow-1))
        irow += nrow

    if report_result is True:
        print(tool_list)
        print(range_list_of_tool)
    
    return tool_list, range_list_of_tool


# In[10]:


#dataframe에서 (1) 주어진 shot_no에 해당하는 데이터를 추출한 후, (2) 해당 데이터에 사용된 공구를 순서대로 가져와서 반환
def extract_tool_list_of_specific_shot(data, shot_no, report_result=False):
    df_shot = data[data['shot_no'] == shot_no]
    
    if report_result is True:
        print("===== shot{} ============================\n\n(1) dataframe -----------------\n".format(shot_no))
        print(df_shot)
        print("(2) used tool -----------------\n")
        print(df_shot['tool'].unique())
        print("(3) extract order using tools -----------------\n")
    
    tool_list = extract_tool_list(df_shot, report_result)
    
    del df_shot
    
    return tool_list


# In[11]:


#dataframe에서 전체 shot에 대해 아래의 과정 수행
#   - 각 shot 별로 사용된 공구 순서를 추출
#   - shot & 공구 사용 순서정보를 column으로 하는 dataframe을 구성한 후 반환
def extract_tool_list_of_all_shots(data, report_result=False):
    column_names = ['shot_no', 'order_using_tool']
    df_by_shot = pd.DataFrame(columns=column_names)
    row = 0

    for shot in df['shot_no'].unique():
        tool_list = extract_tool_list_of_specific_shot(df, shot, report_result)
        str_tool_list = ' '.join(map(str, tool_list))
        df_by_shot.loc[row] = [shot, str_tool_list]
        
        del tool_list
        
        row += 1

    print(df_by_shot)
    
    return df_by_shot


# ### 유효 shot 구간 도출
#  * 정상 공구 사용 패턴을 가지는 구간을 하나의 유효 shot으로 판별하고 별도의 column('real_shot')에 해당 numbering

# In[12]:


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


# In[13]:


#dataframe에서 (1) 주어진 shot_no에 해당하는 데이터를 추출한 후
#(2) 해당 데이터에 사용된 공구 순서 및 각 공구 별 row index 범위를 순서대로 가져옴
#(3) 정상 패턴 가공이 이루어지는 구간을 도출 (유효한 shot으로 판단할 수 있는 구간)
#(4) (3)에서 찾은 구간 데이터의 'real_shot' 컬럼에 새로 shot numbering
def find_and_mark_valid_shot_of_specific_shot(data, shot_no, real_shot_no, report_result=False):
    df_shot = data[data['shot_no'] == shot_no]
    
    if report_result is True:
        print("===== shot{} ============================\n\n(1) dataframe -----------------\n".format(shot_no))
        print(df_shot)
        print("(2) used tool -----------------\n")
        print(df_shot['tool'].unique())
        print("(3) extract order using tools -----------------\n")
    
    tool_list, range_list_of_tool = extract_tool_list_with_range(df_shot, report_result=False)

    tool_pattern = [1, 5, 9, 11, 7, 3, 11, 7]
    sub_list = find_sub_list(tool_pattern, tool_list)
    
    if len(sub_list) < 1:
        if report_result is True:
            print("shot_no {} of raw data : invalid_shot".format(shot_no))

        return 0
    
    if report_result is True:
        print(sub_list)
    
    start_index = df_shot.index[0]
    count = 0
    
    if report_result is True: 
        if sub_list[0][0] > 0:
            print("------ invalid data\n")
            print(data.iloc[start_index: start_index+range_list_of_tool[sub_list[0][0]][0]])

    for sub in sub_list:
        real_shot_no += 1
        sub_start = start_index + range_list_of_tool[sub[0]][0]
        sub_end   = start_index + range_list_of_tool[sub[1]][1]
        print("sub({}:{}) - real_shot_no {}".format(sub_start, sub_end, real_shot_no))
        
        #========================================================================
        #LJY_20220926 : iloc 함수를 사용하여 구간 access 후 값 설정 시 pandas ver.1.3.5 이후로(1.4.0부터) 동작하지 않던 오류 수정
        #------------------------------------------------------------
        #오류 : 기존에 작성된 코드
        #data.iloc[sub_start:sub_end]['real_shot'] = real_shot_no
        
        #------------------------------------------------------------
        #방법1 : column을 특정한 후 row index range 지정하여 값 설정
        data.real_shot.iloc[sub_start:sub_end] = real_shot_no
        
        #------------------------------------------------------------
        #방법2 : row index range와 column index를 지정하여 값 설정
        # print('real_shot_no : ',real_shot_no)
        # data.iloc[sub_start:sub_end+1, data.columns.get_loc('real_shot')] = real_shot_no
        #========================================================================
        
        if report_result is True:
            print("------ valid shot[{}]\n".format(count))
            print(data.iloc[sub_start:sub_end+1])
            
        count += 1
    
    tool_list.clear()
    range_list_of_tool.clear()
    sub_list.clear()
    
    del df_shot
    
    return count


# In[14]:


#dataframe에서 유효한 공구 사용 패턴을 가지는 구간을 찾아서 별도의 컬럼('real_shot')에 새로 number 부여
# - 유효하지 않은 구간의 경우 'real_shot' 값을 -1로 설정
def find_and_mark_valid_shot(data, report_result=False):
    real_shot_no = 0
    
    data['real_shot'] = -1
    
    for shot in data['shot_no'].unique():
        real_shot_no += find_and_mark_valid_shot_of_specific_shot(data, shot, real_shot_no, report_result)
        print(real_shot_no)


# ### 유효 shot 구간 내의 시작 부위의 유휴시간 데이터 삭제
#  * 정상 공구 사용 패턴을 가지는 구간을 하나의 유효 shot으로 판별하고 별도의 column('real_shot')에 해당 numbering

# In[15]:


#dataframe의 시작 부위의 idle section에 대한 'real_shot' 정보 초기화(값: -1)
# => 실제 가공이 이루어진 유효한 구간을 찾아냄
def remove_idle_section_at_the_start_of_valid_shot(data):
    #print(data)
    
    #스핀들 전류값이 0보다 크면 가공이 이루어진 구간으로 판단하여 아무 처리하지 않고 return
    if data.iloc[0]['current_spindle'] > 0:
        return
    
    last_index_of_idle_section = data[data['current_spindle'] > 0].index[0]-1
    
    print("index : {}~{}\n".format(data.index[0], last_index_of_idle_section))
    #print(data.loc[data.index[0]:last_index_of_idle_section])
    data.loc[data.index[0]:last_index_of_idle_section, 'real_shot'] = -1 #idle 구간의 valid shot 정보 초기화
    #print(data.loc[data.index[0]:last_index_of_idle_section])
    return


# In[16]:


#dataframe의 유효한 shot 구간('real_shot' != -1인 구간) 별로 시작 부위의 idle section에 대한 'real_shot' 정보 초기화(값: -1)
# => 실제 가공이 이루어진 유효한 구간을 찾아냄
def remove_idle_sections_at_the_start_of_valid_shots(data):
    for shot in data['real_shot'].unique():
        if shot < 0:
            continue
            
#        df_by_shot = data[data['real_shot'] == shot]
        
#        if df_by_shot.iloc[0]['shot_no'] == 6517:
#            print(df_by_shot)
            
#        print(df_by_shot)
        index = data[data['real_shot'] == shot].index
    
        remove_idle_section_at_the_start_of_valid_shot(data.loc[index[0]:index[0]+len(index)-1])
        
        print(data.loc[index[0]:index[0]+len(index)-1])

    return


# ## 시각화

# In[17]:


#dataframe에서 컬럼을 다중선택하여 데이터를 차트로 가시화
#① data : dataframe 입력
#② column_names : list 형태의 column명을 입력   ex) ['col1'], ['col1', 'col2'] 
#③ dataframe에서 특정 범위만을 선택해서 차트를 가시화 하려면 index_min 또는 index_max를 지정
#   .. index_min : 디폴트값(-1)인 경우에는 데이터의 시작 row부터 포함됨
#   ..  index_max : 디폴트값(-1)인 경우에는 데이터의 끝 row까지 포함됨
#④ 차트 제목 표시 관련 설정
#   .. title : 차트의 제목 문자열
#   .. title_font_size : 차트의 제목을 표시할 font size
#⑤ 차트 크기 설정
#   .. figsize_horz, figsize_vert : 차트의 가로, 세로 크기
#⑥ 범례 표시 관련 설정
#   .. legend_font_size : 범례를 표시할 font size
#   .. legend_location : 범례 표시 위치 ("upper right", "lower right", "upper left", "lower left")
#⑦ 그래프 표현 관련 설정 : 선 or marker 표시
#   .. linestyle : 선 스타일의 이름을 "solid", "dashed", "dotted", "dashdot"와 같은 형식으로 입력하거나 아래를 참고하여 입력
'''
'-'  solid line style
'--' line style
'-.' dash-dot line style
':'  dotted line style
'.'  point marker
','  pixel marker
'o'  circle marker
'v'  triangle_down marker
'^'  triangle_up marker
'<'  triangle_left marker
'>'  triangle_right marker
'1'  tri_down marker
'2'  tri_up marker
'3'  tri_left marker
'4'  tri_right marker
's'  square marker
'p'  pentagon marker
'*'  star marker
'h'  hexagon1 marker
'H'  hexagon2 marker
'+'  plus marker
'x'  x marker
'D'  diamond marker
'd'  thin_diamond marker
'|'  vline marker
'_'  hline marker
'''
#  .. marker : 아래를 참고하여 입력 ( https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
'''
'.'  point marker
','  pixel marker
'o'  circle marker
'v'  triangle_down marker
'^'  triangle_up marker
'<'  triangle_left marker
'>'  triangle_right marker
'1'  tri_down marker
'2'  tri_up marker
'3'  tri_left marker
'4'  tri_right marker
'8'  octagon marker
's'  square marker
'p'  pentagon marker
'P'  plus (filled) marker
'*'  star marker
'h'  hexagon1 marker
'H'  hexagon2 marker
'+'  plus marker
'x'  x marker
'X'  x (filled) marker
'D'  diamond marker
'd'  thin_diamond marker
'|'  vline marker
'_'  hline marker
'''
def show_plot(data, column_names, index_min = -1, index_max = -1, title = None, title_font_size=20,
              x_axis_title = None, y_axis_title = None, axes_title_font_size=18, tick_label_font_size=15,
              figsize_horz=20, figsize_vert=10, legend_font_size=15, legend_location="upper right",
              linestyle = 'none', marker='.', marker_size=10):
    
    parameters = { 'figure.titlesize':title_font_size, 'axes.titlesize': axes_title_font_size,
                  'axes.labelsize' : tick_label_font_size, 'legend.fontsize' : legend_font_size }
    plt.rcParams.update(parameters)
    
    plt.figure(figsize = (figsize_horz, figsize_vert))
    
    str_col_names = '{}'.format(','.join(column_names))#(f"'{col}'" for col in column_name))
    print(str_col_names)
    
    if index_min > -1 and index_max > -1 and index_min < index_max:
        for column in list(column_names):
            plt.plot(data.index[index_min:index_max], data.iloc[index_min:index_max][column], marker=marker, linestyle=linestyle, markersize=marker_size)
    elif index_min > -1:
        for column in list(column_names):
            plt.plot(data.index[index_min:], data.iloc[index_min:][column], marker=marker, linestyle=linestyle, markersize=marker_size)
    elif index_max > -1:
        for column in list(column_names):
            plt.plot(data.index[:index_max], data.iloc[:index_max][column], marker=marker, linestyle=linestyles, markersize=marker_size)
    else:
        #str_col_names = ','.join('{0}'.format(col) for col in column_name)
        for column in list(column_names):
            plt.plot(data.index, data[column], marker=marker, linestyle=linestyle, markersize=marker_size)
    
    if title is not None:
        plt.title(title)

    plt.legend(column_names, loc = legend_location)
    plt.xticks(fontsize = tick_label_font_size)
    plt.yticks(fontsize = tick_label_font_size)
    
    if x_axis_title is not None:
        plt.xlabel(x_axis_title)
    if y_axis_title is not None:
        plt.ylabel(y_axis_title)


# In[18]:


#dataframe에서 컬럼을 다중선택하여 데이터를 차트로 가시화
#① data : dataframe 입력
#② target_column_name : 값을 관찰할 컬럼
#③ condition_column : 조건 판별 대상 컬럼명  ex) 'state'
#④ condition_values : 일치 여부 대상 value 목록  ex) [1], [0, 1]
#⑤ dataframe에서 특정 범위만을 선택해서 차트를 가시화 하려면 index_min 또는 index_max를 지정
#   .. index_min : 디폴트값(-1)인 경우에는 데이터의 시작 row부터 포함됨
#   ..  index_max : 디폴트값(-1)인 경우에는 데이터의 끝 row까지 포함됨
#⑥ 차트 제목 표시 관련 설정
#   .. title : 차트의 제목 문자열
#   .. title_font_size : 차트의 제목을 표시할 font size
#⑦ 차트 크기 설정
#   .. figsize_horz, figsize_vert : 차트의 가로, 세로 크기
#⑧ 범례 표시 관련 설정
#   .. legend_font_size : 범례를 표시할 font size
#   .. legend_location : 범례 표시 위치 ("upper right", "lower right", "upper left", "lower left")
#⑨ 그래프 표현 관련 설정 : 선 or marker 표시
#   .. linestyle : 선 스타일의 이름을 "solid", "dashed", "dotted", "dashdot"와 같은 형식으로 입력하거나 아래를 참고하여 입력
'''
'-'  solid line style
'--' line style
'-.' dash-dot line style
':'  dotted line style
'.'  point marker
','  pixel marker
'o'  circle marker
'v'  triangle_down marker
'^'  triangle_up marker
'<'  triangle_left marker
'>'  triangle_right marker
'1'  tri_down marker
'2'  tri_up marker
'3'  tri_left marker
'4'  tri_right marker
's'  square marker
'p'  pentagon marker
'*'  star marker
'h'  hexagon1 marker
'H'  hexagon2 marker
'+'  plus marker
'x'  x marker
'D'  diamond marker
'd'  thin_diamond marker
'|'  vline marker
'_'  hline marker
'''
#  .. marker : 아래를 참고하여 입력 ( https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
'''
'.'  point marker
','  pixel marker
'o'  circle marker
'v'  triangle_down marker
'^'  triangle_up marker
'<'  triangle_left marker
'>'  triangle_right marker
'1'  tri_down marker
'2'  tri_up marker
'3'  tri_left marker
'4'  tri_right marker
'8'  octagon marker
's'  square marker
'p'  pentagon marker
'P'  plus (filled) marker
'*'  star marker
'h'  hexagon1 marker
'H'  hexagon2 marker
'+'  plus marker
'x'  x marker
'X'  x (filled) marker
'D'  diamond marker
'd'  thin_diamond marker
'|'  vline marker
'_'  hline marker
'''
def show_plot_comparing_data_by_condition(data, target_column_name, condition_column, condition_values, data_names = ('condition is true','condition is false'),
                                           index_min = -1, index_max = -1, title = None, title_font_size=20,
                                           x_axis_title = None, y_axis_title = None, axes_title_font_size=18,
                                           tick_label_font_size=15, figsize_horz=20, figsize_vert=10,
                                           legend_font_size=15, legend_location="lower right", linestyle = 'none',
                                           marker='.', marker_size=10):
    
    parameters = { 'figure.titlesize':title_font_size, 'axes.titlesize': axes_title_font_size,
                  'axes.labelsize' : tick_label_font_size, 'legend.fontsize' : legend_font_size }
    plt.rcParams.update(parameters)
    
    plt.figure(figsize = (figsize_horz, figsize_vert))
    
    condition_true_data = pd.DataFrame()
    condition_false_data = pd.DataFrame()
      
    if index_min > -1 and index_max > -1 and index_min < index_max:
        condition_true_data = data.iloc[index_min:index_max].query(data[condition_column].isin(condition_values))
        condition_false_data = data.iloc[index_min:index_max].query(~data[condition_column].isin(condition_values))
    elif index_min > -1:
        condition_true_data = data.iloc[index_min:].query(data[condition_column].isin(condition_values))
        condition_false_data = data.iloc[index_min:].query(~data[condition_column].isin(condition_values))        
    elif index_max > -1:
        condition_true_data = data.iloc[:index_max].query(data[condition_column].isin(condition_values))
        condition_false_data = data.iloc[:index_max].query(~data[condition_column].isin(condition_values))        
    else:
        condition_true_data = data[data[condition_column].isin(condition_values)]
        condition_false_data = data[~data[condition_column].isin(condition_values)]
        
    plt.plot(condition_true_data.index, condition_true_data[target_column_name], marker=marker, linestyle=linestyle, label=data_names[0], color='red', markersize=marker_size)
    plt.plot(condition_false_data.index, condition_false_data[target_column_name], marker=marker, linestyle=linestyle, label=data_names[1], markersize=marker_size)
    
    if title is not None:
        plt.title(title)

    plt.legend(data_names, loc = legend_location)
    plt.xticks(fontsize = tick_label_font_size)
    plt.yticks(fontsize = tick_label_font_size)
    
    if x_axis_title is not None:
        plt.xlabel(x_axis_title, fontsize = axes_title_font_size)
    if y_axis_title is not None:
        plt.ylabel(y_axis_title, fontsize = axes_title_font_size)


# ## 모델 수립

# ### 유효한 shot을 기준으로 주어진 공구에 대한 주어진 field의 대푯값을 계산하여 dataframe 구성

# In[19]:


# 유효한 shot을 기준으로 주어진 공구에 대한 주어진 field의 대푯값을 계산하여  dataframe으로 구성
def build_representative_data_for_tool_machining(data, tool_no, target_column_name, apply_robust_scaler=False, except_min_value=False, except_max_value=False, delete_outlier=False,
                                                 delete_zero_value=False, except_tool_cancel_state=False, except_tool_end_state=False):
    if target_column_name not in data.columns:
        print("{} is not in columns\nPlease input valid column name".format(target_column_name))
        return
    
    condition = (data['tool']==tool_no) & (data['real_shot'] != -1)
    
    #LJY_20221005 : tool취소 상태 데이터 제거 옵션 추가
    if except_tool_cancel_state is True:
        condition = condition & (data['tool_state'] != 0)
        
    if except_tool_end_state is True:
        condition = condition & (data['tool_state'] != 9)
        
    #data_tool = data[(data['tool']==tool_no) & (data['real_shot'] != -1)]
    data_tool = data[condition]
    
    if delete_zero_value is True:
        data_tool.drop(data_tool[data_tool[target_column_name] == 0].index, inplace=True)
        
    if delete_outlier is True:
        print('before deleting outliers : {}'.format(len(data_tool.index)))
        delete_outliers(data_tool, target_column_name)
        print('after deleting outliers : {}'.format(len(data_tool.index)))
    
    if except_min_value is True:
        min_value = data_tool[target_column_name].min()
        data_tool.drop(data_tool[data_tool[target_column_name] == min_value].index, inplace=True)
        
    if except_max_value is True:
        max_value = data_tool[target_column_name].max()
        data_tool.drop(data_tool[data_tool[target_column_name] == max_value].index, inplace=True)
    
    if apply_robust_scaler is True:
        from sklearn import metrics
        #from sklearn.preprocessing import MinMaxScaler
        #from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import RobustScaler
        
        rs = RobustScaler()
        #ms = MinMaxScaler()
        data_tool[[target_column_name]] = rs.fit_transform(data_tool[[target_column_name]])
    
    #real shot no.
#    data_shot_no = data_tool['real_shot'].to_frame(name='real_shot')
    group_by_shot = data_tool.groupby(['real_shot'])
    #print(group_by_shot.groups.keys())
    
    #표준편차
    data_std = (group_by_shot.std()[target_column_name].to_frame(name='std'))
    #표준오차
    data_sem = (group_by_shot.sem()[target_column_name].to_frame(name='sem'))
    #합
    data_sum = (group_by_shot.sum()[target_column_name].to_frame(name='sum'))
    #평균
    data_mean = (group_by_shot.mean()[target_column_name].to_frame(name='mean'))
    #중앙값
    data_median = (group_by_shot.median()[target_column_name].to_frame(name='median'))
    #분산
    data_var = (group_by_shot.var()[target_column_name].to_frame(name='var'))
    #최대값
    data_max = (group_by_shot.max()[target_column_name].to_frame(name='max')) 
    #0.25분위수
    data_quantile_1_per_4 = (group_by_shot.quantile(0.25)[target_column_name].to_frame(name='quantile(0.25)')) 
    #0.75분위수
    data_quantile_3_per_4 = (group_by_shot.quantile(0.75)[target_column_name].to_frame(name='quantile(0.75)')) 
    #왜도
    data_skew = (group_by_shot.skew()[target_column_name].to_frame(name='skew')) 

    data_representative = pd.concat([data_mean, data_std, data_median, data_max, data_var, data_sem, data_sum, data_quantile_1_per_4, data_quantile_3_per_4, data_skew],axis=1)
    
    data_representative.set_index(pd.Series(group_by_shot.groups.keys()))
    
    return data_representative


# ### 주어진 공구의 주어진 필드에 대한 valid shot 별 대푯값 도출 & 시각화

# In[20]:


#representative_value_types : 'std', 'sem', 'sum', 'mean', 'median', 'var', 'max', 'quantile(0.25)', 'quantile(0.75)', 'skew'를 리스트 형태로 입력
#  ex) ['std', 'mean']
def build_and_display_representative_data_for_tool_machining(data, tool_no, target_column_name, representative_value_types, apply_robust_scaler=False, except_min_value=False, except_max_value=False):
    df_representative = build_representative_data_for_tool_machining(data, tool_no, target_column_name, apply_robust_scaler, except_min_value, except_max_value)
    show_plot(df_representative, representative_value_types, title = target_column_name, x_axis_title='real shot no.', y_axis_title='representatives')


# ### 주어진 필드에 대해 Isolation Forest 기법을 적용하여 이상치 탐지

# In[21]:


# he offsetis defined in such a way we obtain the expected number of outliers (samples with decision function < 0) in training.
def check_and_mark_outlier_by_IsolationForest_org(training_data, test_data, target_column_name, training_or_test = True):
    import pickle
    from sklearn.ensemble import IsolationForest
    import joblib
    # 학습 모델    
    if training_or_test == True:
        IF = IsolationForest(random_state = 42, contamination = 0.004 , n_estimators = 500, max_samples = 90, n_jobs = -1, bootstrap = True).fit(training_data[[target_column_name]])
        score = IF.decision_function(training_data[[target_column_name]])
        training_data['IF_Outliers'] = pd.Series(IF.predict(training_data[[target_column_name]]), index = training_data.index).apply(lambda x: 1 if x == -1 else 0)
        training_data['IF_score'] = score
        joblib.dump(IF, os.path.join(os.getcwd(), 'IF_training_model.pkl'))
        # training_data['score_sample'] = IF.score_samples(training_data[[target_column_name]])
        # training_data['offset'] = training_data['IF_score'] - training_data['score_sample']
#         training_data.loc[(training_data['IF_Outliers'] == 1) & (training_data['IF_score'] < 0)]['IF_Outliers'] = 0
#         training_data.loc[(training_data['IF_Outliers'] == 0) & (training_data['IF_score'] > 0)]['IF_Outliers'] = 1 
    else:
        try:
            joblib.load(os.path.join(os.getcwd(), 'IF_training_model.pkl'))
        except FileNotFoundError:
            print('IF_training_model File does not exist.')
            print('This need a learning model.')
            print('Please training model establishment first.')
        else:
            training_model = joblib.load(os.path.join(os.getcwd(), 'IF_training_model.pkl'))
            score = training_model.score_samples(test_data[[target_column_name]])
            test_data['IF_Outliers'] = pd.Series(training_model.predict(test_data[[target_column_name]]), index = test_data.index).apply(lambda x: 1 if x == -1 else 0)
            test_data['IF_score'] = score
            # test_data['score_sample'] = IF.score_samples(test_data[[target_column_name]])
            # test_data['offset'] = test_data['IF_score'] - test_data['score_sample'] # offset = decision_function - sample_score
    #         test_data.loc[(test_data['IF_Outliers'] == 1) & (test_data['IF_score'] > -0.006), ['IF_Outliers']] = 0 
    #         test_data.loc[(test_data['IF_Outliers'] == 0) & (test_data['IF_score'] < -0), ['IF_Outliers']] = 1 


# In[22]:


# def check_and_mark_outlier_by_IsolationForest_test(test_data, target_column_name):
#     # applying test set    
#     score = check_and_mark_outlier_by_IsolationForest_org.decision_function(test_data[[target_column_name]])
#     data['IF_Outliers'] = pd.Series(check_and_mark_outlier_by_IsolationForest_org.predict(test_data[[target_column_name]])).apply(lambda x: 1 if x == -1 else 0)
#     data['IF_score'] = score
    
#     # data.loc[(data['IF_Outliers'] == 1) & (data['IF_score'] > 0), ['IF_Outliers']] = 0 #LJY_20221005 : 'IF_score' 값이 0보다 큰 경우는 '이상'으로 판별하지 않도록 처리 시도
#     # data.loc[(data['IF_Outliers'] == 0) & (data['IF_score'] < 0), ['IF_Outliers']] = 1 #LJY_20221006 : 'IF_score' 값이 0보다 작은 경우는 '정상'으로 판별하지 않도록 처리 시도


# In[23]:


#dataframe에서 컬럼을 다중선택하여 데이터를 차트로 가시화
#① data : dataframe 입력
#② target_column_name : 값을 관찰할 컬럼
#③ condition_column : 조건 판별 대상 컬럼명  ex) 'state'
#④ condition_values : 일치 여부 대상 value 목록  ex) [1], [0, 1]
#⑤ dataframe에서 특정 범위만을 선택해서 차트를 가시화 하려면 index_min 또는 index_max를 지정
#   .. index_min : 디폴트값(-1)인 경우에는 데이터의 시작 row부터 포함됨
#   ..  index_max : 디폴트값(-1)인 경우에는 데이터의 끝 row까지 포함됨
#⑥ 차트 제목 표시 관련 설정
#   .. title : 차트의 제목 문자열
#   .. title_font_size : 차트의 제목을 표시할 font size
#⑦ 차트 크기 설정
#   .. figsize_horz, figsize_vert : 차트의 가로, 세로 크기
#⑧ 범례 표시 관련 설정
#   .. legend_font_size : 범례를 표시할 font size
#   .. legend_location : 범례 표시 위치 ("upper right", "lower right", "upper left", "lower left")
#⑨ 그래프 표현 관련 설정 : 선 or marker 표시
#   .. linestyle : 선 스타일의 이름을 "solid", "dashed", "dotted", "dashdot"와 같은 형식으로 입력하거나 아래를 참고하여 입력
'''
'-'  solid line style
'--' line style
'-.' dash-dot line style
':'  dotted line style
'.'  point marker
','  pixel marker
'o'  circle marker
'v'  triangle_down marker
'^'  triangle_up marker
'<'  triangle_left marker
'>'  triangle_right marker
'1'  tri_down marker
'2'  tri_up marker
'3'  tri_left marker
'4'  tri_right marker
's'  square marker
'p'  pentagon marker
'*'  star marker
'h'  hexagon1 marker
'H'  hexagon2 marker
'+'  plus marker
'x'  x marker
'D'  diamond marker
'd'  thin_diamond marker
'|'  vline marker
'_'  hline marker
'''
#  .. marker : 아래를 참고하여 입력 ( https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
'''
'.'  point marker
','  pixel marker
'o'  circle marker
'v'  triangle_down marker
'^'  triangle_up marker
'<'  triangle_left marker
'>'  triangle_right marker
'1'  tri_down marker
'2'  tri_up marker
'3'  tri_left marker
'4'  tri_right marker
'8'  octagon marker
's'  square marker
'p'  pentagon marker
'P'  plus (filled) marker
'*'  star marker
'h'  hexagon1 marker
'H'  hexagon2 marker
'+'  plus marker
'x'  x marker
'X'  x (filled) marker
'D'  diamond marker
'd'  thin_diamond marker
'|'  vline marker
'_'  hline marker
'''
def show_plot_comparing_data_by_condition(data, target_column_name, condition_column, condition_values, data_names = ('condition is true','condition is false'),
                                           index_min = -1, index_max = -1, title = None, title_font_size=20,
                                           x_axis_title = None, y_axis_title = None, axes_title_font_size=18,
                                           tick_label_font_size=15, figsize_horz=20, figsize_vert=10,
                                           legend_font_size=15, legend_location="lower right", linestyle = 'none',
                                           marker='.', marker_size=10):
    
    parameters = { 'figure.titlesize':title_font_size, 'axes.titlesize': axes_title_font_size,
                  'axes.labelsize' : tick_label_font_size, 'legend.fontsize' : legend_font_size }
    plt.rcParams.update(parameters)
    
    plt.figure(figsize = (figsize_horz, figsize_vert))
    
    condition_true_data = pd.DataFrame()
    condition_false_data = pd.DataFrame()
      
    if index_min > -1 and index_max > -1 and index_min < index_max:
        condition_true_data = data.iloc[index_min:index_max].query(data[condition_column].isin(condition_values))
        condition_false_data = data.iloc[index_min:index_max].query(~data[condition_column].isin(condition_values))
    elif index_min > -1:
        condition_true_data = data.iloc[index_min:].query(data[condition_column].isin(condition_values))
        condition_false_data = data.iloc[index_min:].query(~data[condition_column].isin(condition_values))        
    elif index_max > -1:
        condition_true_data = data.iloc[:index_max].query(data[condition_column].isin(condition_values))
        condition_false_data = data.iloc[:index_max].query(~data[condition_column].isin(condition_values))        
    else:
        condition_true_data = data[data[condition_column].isin(condition_values)]
        condition_false_data = data[~data[condition_column].isin(condition_values)]
        
    plt.plot(condition_true_data.index, condition_true_data[target_column_name], marker=marker, linestyle=linestyle, label=data_names[0], color='red', markersize=marker_size)
    plt.plot(condition_false_data.index, condition_false_data[target_column_name], marker=marker, linestyle=linestyle, label=data_names[1], markersize=marker_size)
    plt.xticks(fontsize = tick_label_font_size)
    plt.yticks(fontsize = tick_label_font_size)
    
    if title is not None:
        plt.title(title)

    plt.legend(data_names, loc = legend_location)
    
    if x_axis_title is not None:
        plt.xlabel(x_axis_title, fontsize = 15)
    if y_axis_title is not None:
        plt.ylabel(y_axis_title, fontsize = 15)


# # Anomaly score 기반 threshould 도출

# In[83]:


def threshould_deduction(data, target_column_name, graph = True):
    # True일 경우 training_data
    # False일 경우 data
    import matplotlib.pyplot as plt
    try:
        joblib.load(os.path.join(os.getcwd(), 'IF_training_model.pkl'))
    except FileNotFoundError:
        print('IF_training_model File does not exist.')
        print('This need a learning model.')
        print('Please training model establishment first.')
    else:
        input_data = data[target_column_name].values.reshape(-1,1)
        training_model = joblib.load(os.path.join(os.getcwd(), 'IF_training_model.pkl'))
        input_data_anomaly_score = training_model.decision_function(input_data)
        input_data_outlier = training_model.predict(input_data)
        temp_input = np.concatenate((input_data.reshape(-1,1), input_data_outlier.reshape(-1,1)), axis = 1)
        result = pd.DataFrame(temp_input, columns = [target_column_name, 'outlier'])
        if graph == True:
            plt.figure(figsize = (20,12))
            plt.scatter(input_data, input_data_anomaly_score, label = 'anomaly score')
            plt.fill_between(input_data.T[0], np.min(input_data_anomaly_score), np.max(input_data_anomaly_score), where=input_data_outlier==-1, color='r', 
                             alpha=.3, label='outlier region')
            plt.axvline(min(result.query('outlier == -1')[target_column_name]),  color = 'g', linestyle = 'dashed')
            plt.text(min(result.query('outlier == -1')[target_column_name]), np.min(input_data_anomaly_score)-0.1, 'Threshold {:.4f}'.format(min(result.query('outlier == -1')[target_column_name])), color = 'r', fontsize = 12, rotation = 90)
            plt.legend()
            plt.ylabel('anomaly score', fontsize = 15)
            plt.yticks(fontsize = 13)
            plt.xlabel(target_column_name, fontsize = 15)
            plt.xticks(fontsize = 13)
        return min(result.query('outlier == -1')[target_column_name])


# # Threshould를 적용한 시각화 적용

# In[79]:


def show_plot_with_threshould(data, target_column_name, condition_column, condition_values, data_names = ('condition is true','condition is false'),
                                           index_min = -1, index_max = -1, title = None, title_font_size=20,
                                           x_axis_title = None, y_axis_title = None, axes_title_font_size=18,
                                           tick_label_font_size=15, figsize_horz=20, figsize_vert=10,
                                           legend_font_size=15, legend_location="lower right", linestyle = 'none',
                                           marker='.', marker_size=10):
    
    try:
        joblib.load(os.path.join(os.getcwd(), 'IF_training_model.pkl'))
    except FileNotFoundError:
        print('IF_training_model File does not exist.')
        print('This need a learning model.')
        print('Please training model establishment first.')
    else:
        parameters = { 'figure.titlesize':title_font_size, 'axes.titlesize': axes_title_font_size,
                      'axes.labelsize' : tick_label_font_size, 'legend.fontsize' : legend_font_size }
        plt.rcParams.update(parameters)

        plt.figure(figsize = (figsize_horz, figsize_vert))

        input_data = data[target_column_name].values.reshape(-1,1)
        training_model = joblib.load(os.path.join(os.getcwd(), 'IF_training_model.pkl'))
        input_data_anomaly_score = training_model.decision_function(input_data)
        input_data_outlier = training_model.predict(input_data)
        temp_input = np.concatenate((input_data.reshape(-1,1), input_data_outlier.reshape(-1,1)), axis = 1)

        condition_true_data = pd.DataFrame()
        condition_false_data = pd.DataFrame()

        if index_min > -1 and index_max > -1 and index_min < index_max:
            condition_true_data = data.iloc[index_min:index_max].query(data[condition_column].isin(condition_values))
            condition_false_data = data.iloc[index_min:index_max].query(~data[condition_column].isin(condition_values))
        elif index_min > -1:
            condition_true_data = data.iloc[index_min:].query(data[condition_column].isin(condition_values))
            condition_false_data = data.iloc[index_min:].query(~data[condition_column].isin(condition_values))        
        elif index_max > -1:
            condition_true_data = data.iloc[:index_max].query(data[condition_column].isin(condition_values))
            condition_false_data = data.iloc[:index_max].query(~data[condition_column].isin(condition_values))        
        else:
            condition_true_data = data[data[condition_column].isin(condition_values)]
            condition_false_data = data[~data[condition_column].isin(condition_values)]

        plt.plot(condition_true_data.index, condition_true_data[target_column_name], marker=marker, linestyle=linestyle, label=data_names[0], color='red', markersize=marker_size)
        plt.plot(condition_false_data.index, condition_false_data[target_column_name], marker=marker, linestyle=linestyle, label=data_names[1], markersize=marker_size)
        plt.axhline(threshould_deduction(data, target_column_name, graph = False), color = 'r', linestyle = 'dashed')
        plt.text(-290, threshould_deduction(data, target_column_name, graph = False)-15, 'Threshold {:.4f}'.format(min(pd.DataFrame(temp_input, columns = [target_column_name, 'outlier']).query('outlier == -1')[target_column_name])), color = 'r', fontsize = 12)

        if title is not None:
            plt.title(title)

        plt.legend(data_names, loc = legend_location)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)

        if x_axis_title is not None:
            plt.xlabel(x_axis_title, fontsize = 15)
        if y_axis_title is not None:
            plt.ylabel(y_axis_title, fontsize = 15)


# # 데이터 불러오기

# In[26]:


os.listdir()


# In[27]:


#데이터 불러오기
df = read_data(sensor_data_only = False)


# In[28]:


df.info()


# In[29]:


# df1 = pd.read_csv('Daeshin_parameter_220713_0831_VL04_raw_data.csv')
# df2 = pd.read_csv('Daeshin_parameter_220901_1013_VL04_raw_data.csv')
# df_con = pd.concat([df1, df2], axis = 0)
# df_con.info()


# In[30]:


max(df.groupby('shot_no').mean()['current_spindle'])


# In[31]:


# max(df_con.groupby('shot_no').mean()['current_spindle'])


# In[32]:


df.tail()


# In[33]:


df['shot_no'].unique()


# In[34]:


df_mod1 = df.drop(['current_x', 'current_z', 'rpm', 'rpm_set','feed', 'load_1', 'servo_load_x', 'servo_load_z', 'servo_current_x', 'servo_current_z'], axis = 1)
df_mod1


# # 전처리

# ## 유효 shot 도출

# In[35]:


#'tool_number' 필드 값이 -1인 결측 데이터 구간 drop
df_mod1 = df_mod1.drop(df_mod1[df_mod1['tool_number'] == -1].index)
df_mod1


# In[36]:


# tool_number 정보에서 공구 번호만 도출된 tool 컬럼 생성
extract_tool_info_from_data(df_mod1)
df_mod1


# In[37]:


df_mod1['shot_no'].unique()[0]


# In[38]:


find_and_mark_valid_shot(df_mod1, report_result = True)
df_mod1


# In[39]:


#검토 : 원본 shot no.
len(df_mod1['shot_no'].unique())


# In[40]:


#검토 : 새로 생성한 유효한 shot no.
len(df_mod1['real_shot'].unique())


# In[41]:


df_mod1[df_mod1['real_shot'] != -1]['real_shot'].unique()


# ## 유효한 shot 구간에서 앞부분의 idle section 제거

#  * 유효한 shot 구간의 시작 부위 idle section을 제거한 데이터 활용

# In[42]:


# remove_idle_sections_at_the_start_of_valid_shots(df)


# # 모델 수립

# ## 'current_spindle'값에 대한 shot 별 대푯값 도출 & IsolationForest 적용

# In[43]:


df_mod1


# In[44]:


os.getcwd()


# In[45]:


# train-set 데이터 불러오기
# train = pd.read_csv('D:/jupyter/01_Sangwoo_Project/03_Project/03_GyengnamTP/03_Collecting_data/Tool5_small_data_create_synchronization.csv', index_col = 0)
raw = build_representative_data_for_tool_machining(df_mod1, 5, 'current_spindle', apply_robust_scaler=False, except_min_value=False, except_max_value=False)
raw


# In[46]:


# train-set visualization
plt.figure(figsize = (20,12))
plt.plot(raw['mean'])


# In[47]:


train_stat = raw.iloc[1200:-1, :]
train_stat


# In[48]:


train_stat.index


# In[49]:


df_mod1['current_spindle'].describe()


# In[50]:


max(df_mod1.groupby(by = 'shot_no').mean()['current_spindle'])


# In[51]:


# test-set 대표값 도출
test_stat = build_representative_data_for_tool_machining(df_mod1, 5, 'current_spindle', apply_robust_scaler=False, except_min_value=False, except_max_value=False)


# In[52]:


len(train_stat)


# In[53]:


train_stat


# In[54]:


train_stat.describe()


# In[55]:


# 학습 모델 생성
check_and_mark_outlier_by_IsolationForest_org(train_stat, test_stat, target_column_name = 'mean', training_or_test = True)


# In[56]:


train_stat.describe()


# In[57]:


train_stat


# In[58]:


train_stat.query('IF_Outliers == 0')['IF_score'].mean()


# In[59]:


outlier = train_stat.query('IF_Outliers == 1')
outlier


# In[60]:


outlier['IF_score'].describe()


# In[61]:


train_stat.query('IF_Outliers == 0')['IF_score'].describe()


# In[62]:


# 학습 모델에 대한 학습 결과 시각화
show_plot_comparing_data_by_condition(train_stat, 'mean', 'IF_Outliers', [1], data_names = ('outlier', 'normal'),
                                      title = 'Spindle Current', x_axis_title='real shot no.', y_axis_title='current_spindle')


# In[63]:


check_and_mark_outlier_by_IsolationForest_org(train_stat, test_stat, target_column_name = 'mean', training_or_test = False)
test_outlier = test_stat.query('IF_Outliers == 1')
test_outlier


# In[64]:


show_plot_comparing_data_by_condition(test_stat, 'mean', 'IF_Outliers', [1], data_names = ('outlier', 'normal'),
                                      title = 'Spindle Current', x_axis_title='real shot no.', y_axis_title='current_spindle')


# In[65]:


test_stat


# In[68]:


test_stat['mean'].values.reshape(-1,1)


# In[69]:


check_and_mark_outlier_by_IsolationForest_org(train_stat, test_stat, 'mean', training_or_test = False)


# In[84]:


threshould_deduction(test_stat, 'mean', graph = True)


# In[72]:


show_plot_with_threshould(test_stat, 'mean', 'IF_Outliers', [1], data_names = ('outlier', 'normal'),
                                      title = 'Spindle Current', x_axis_title='real shot no.', y_axis_title='current_spindle')


# In[73]:


# current_spindle = test_stat['mean'].values.reshape(-1,1)
# current_spindle_anomaly_score = IF.decision_function(current_spindle)
# current_spindle_outlier = IF.predict(current_spindle)
# plt.figure(figsize = (20,12))
# plt.scatter(current_spindle, current_spindle_anomaly_score, label = 'anomaly score')
# plt.fill_between(current_spindle.T[0], np.min(current_spindle_anomaly_score), np.max(current_spindle_anomaly_score), where=current_spindle_outlier==-1, color='r', 
#                      alpha=.3, label='outlier region')
# plt.legend()
# plt.ylabel('anomaly score')
# plt.xlabel('current_spindle')
# print(np.min(current_spindle_anomaly_score))
# print(np.max(current_spindle_anomaly_score))


# In[74]:


# print(np.min(current_spindle_anomaly_score))
# print(np.max(current_spindle_anomaly_score))


# In[75]:


# # 
# a = np.concatenate((current_spindle.reshape(-1,1), current_spindle_outlier.reshape(-1,1)), axis = 1)
# min(pd.DataFrame(a, columns = ['current_spindle', 'outlier']).query('outlier == -1')['current_spindle'])


# In[76]:


# show_plot_comparing_data_by_condition(test_stat, 'mean', 'IF_Outliers', [1], data_names = ('outlier', 'normal'),
#                                       title = 'Spindle Current', x_axis_title='real shot no.', y_axis_title='current_spindle')
# plt.axhline(min(pd.DataFrame(a, columns = ['current_spindle', 'outlier']).query('outlier == -1')['current_spindle']), color = 'r', linestyle = 'dashed')
# plt.text(-290, min(pd.DataFrame(a, columns = ['current_spindle', 'outlier']).query('outlier == -1')['current_spindle'])-15, 'Threshold {:.4f}'.format(min(pd.DataFrame(a, columns = ['current_spindle', 'outlier']).query('outlier == -1')['current_spindle'])), color = 'r', fontsize = 12)


# In[77]:


test_stat[(test_stat['mean']<2615) & (test_stat['mean']>2610)].query('IF_Outliers == 1')

