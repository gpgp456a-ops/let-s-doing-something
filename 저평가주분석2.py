import urllib.request
import ssl
import zipfile
import os
import pandas as pd
import requests
import json
import io
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
from pykrx import stock
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
import numpy as np


API_KEY = os.environ["API_KEY"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
service_account_json = os.environ["SERVICE_ACCOUNT_FILE"]

# 실행 환경에 실제 파일로 생성
SERVICE_ACCOUNT_FILE = "service_account.json"

with open(SERVICE_ACCOUNT_FILE, "w", encoding="utf-8") as f:
    f.write(service_account_json)


today = stock.get_nearest_business_day_in_a_week()

# 전체 종목 리스트
tickers = stock.get_market_ticker_list(market="ALL")

# 업종 정보
KOSPI_sector_df = stock.get_market_sector_classifications(today, market = 'KOSPI')

KOSPI_sector_df.tail()

# 전체 종목 리스트
tickers = stock.get_market_ticker_list(market="ALL")

# 업종 정보
KOSDAQ_sector_df = stock.get_market_sector_classifications(today, market = 'KOSDAQ')

KOSDAQ_sector_df.head()

total_sector_df = pd.concat([KOSPI_sector_df,KOSDAQ_sector_df])

total_sector_df = total_sector_df[['종목명','업종명','시가총액']]

def get_stock_lists():
    """
    pykrx를 이용해 코스피와 코스닥의 순수 보통주와 우선주를 티커 기준으로 정확하게 분리.
    - df_stock_list : 순수 보통주
    - df_preferred_list : 우선주
    """
    print("코스피, 코스닥 전체 종목 정보를 조회합니다...")
    
    # 1. 코스피
    df_kospi = stock.get_market_fundamental(today, market="KOSPI").reset_index()
    df_kospi['시장'] = 'KOSPI'
    
    # 2. 코스닥
    df_kosdaq = stock.get_market_fundamental(today, market="KOSDAQ").reset_index()
    df_kosdaq['시장'] = 'KOSDAQ'
    
    # 3. 전체 합치기
    df_all = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
    
    # 4. 종목명 추가
    ticker_to_name = {}
    for market in ["KOSPI", "KOSDAQ"]:
        tickers = stock.get_market_ticker_list(market=market)
        for ticker in tickers:
            ticker_to_name[ticker] = stock.get_market_ticker_name(ticker)
    df_all['종목명'] = df_all['티커'].map(ticker_to_name) 
    
    
    # 5. 우선주 판단 함수
    def is_preferred(ticker):
        return ticker[-1] in ['5','6','7','8','9'] or ticker.endswith('K')
    
    # 6. 우선주/보통주 분리
    df_preferred_list = df_all[df_all['티커'].apply(is_preferred)].copy()
    remove_keywords = ["리츠", "ETF", "ETN", "스팩"]
    pattern_remove = "|".join(remove_keywords)
    df_stock_list = df_all[~df_all['티커'].apply(is_preferred)]  # 우선주 제거
    df_stock_list = df_stock_list[~df_stock_list['종목명'].str.contains(pattern_remove, na=False)] #기타 제거
    
    
    # 7. 보통주명을 기준으로 우선주 개수 확인
    stock_names = df_stock_list['종목명'].unique()
    multi_preferred_names = []
    
    for name in stock_names:
        count = df_preferred_list['종목명'].str.contains(
            name, na=False, regex=False
        ).sum()
        if count >= 2:
            multi_preferred_names.append(name)
    
    # 8. 2개 이상 우선주 종목 제거 (보통주와 우선주 모두)
    if multi_preferred_names:
        df_preferred_list = df_preferred_list[~df_preferred_list['종목명'].apply(lambda x: any(name in x for name in multi_preferred_names))]
        df_stock_list = df_stock_list[~df_stock_list['종목명'].isin(multi_preferred_names)]
    
    # 9. 컬럼 정리
    cols = ['시장', '티커', '종목명', 'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
    df_stock_list = df_stock_list[cols]
    df_preferred_list = df_preferred_list[cols]
    
    
    return df_stock_list, df_preferred_list


# 실행
if __name__ == "__main__":
    df_stock_list, df_preferred_list = get_stock_lists()




#df 병합
df_stock_list = total_sector_df.merge(
    df_stock_list[['종목명', '티커', '시장', 'BPS', 'PER', 'PBR', 'EPS']],
    on='종목명',
    how='left'
)

stock_name = df_stock_list['종목명'].squeeze()

url = "https://opendart.fss.or.kr/api/corpCode.xml"
params = {"crtfc_key": API_KEY}

res = requests.get(url, params=params)
res.raise_for_status()

with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
    xml_data = zf.read(zf.namelist()[0])

# =========================
# 2. XML → DataFrame
# =========================
root = ET.fromstring(xml_data)

corp_list = []
for child in root.findall("list"):
    corp_list.append({
        "corp_code": child.findtext("corp_code"),
        "corp_name": child.findtext("corp_name"),
        "stock_code": child.findtext("stock_code")
    })

dart_corp_df = pd.DataFrame(corp_list)

# =========================
# 3. stock_code 정리
# =========================
dart_corp_df['stock_code'] = (
    dart_corp_df['stock_code']
    .astype(str)
    .str.strip()
    .str.zfill(6)
)


# =========================
# 4. df_stock_list에 corp_code 붙이기
# =========================
df_stock_list = df_stock_list.merge(
    dart_corp_df[['corp_code','stock_code']],
    left_on='티커',
    right_on='stock_code',
    how='left'
)


df_stock_list_PER = df_stock_list[
    ['종목명', '업종명', '시장', 'PER']
]

# PER가 0이거나 NaN인 행 제거
df_stock_list_PER = df_stock_list_PER[
    df_stock_list_PER['PER'].notna() & (df_stock_list_PER['PER'] != 0)
]

# 업종명 → 업종 내 PER 오름차순 정렬
df_sorted = (
    df_stock_list_PER
    .sort_values(by=['업종명', 'PER'], ascending=[True, True])
)

per_series = df_stock_list.pop('PER')
df_stock_list['PER'] = per_series

df_stock_list = df_stock_list.merge(
    df_sorted[['종목명']],
    on='종목명',
    how='right'
)



ANNUAL_REPORT = "11011"
Q3_REPORT = "11014"
HALF_REPORT = "11012"
Q1_REPORT = "11013"


def find_latest_report(corp_code):  #가장 최근 보고서가 반기인지 분기인지 사업보고서인지 유무 확인
    # (이전과 동일, 변경 없음)
    print("--- 가장 최신 정기보고서 검색 중... ---")
    list_url = "https://opendart.fss.or.kr/api/list.json"
    end_de = date.today().strftime('%Y%m%d')
    bgn_de = (date.today() - timedelta(days=450)).strftime('%Y%m%d')
    params = {"crtfc_key": API_KEY, "corp_code": corp_code, "bgn_de": bgn_de, "end_de": end_de, "pblntf_ty": "A", "last_reprt_at": "Y"}
    res = requests.get(list_url, params=params).json()
    if res.get('status') != '000':
        print(f"공시 목록 조회 실패: {res.get('message')}")
        return None
    report_types = ["사업보고서", "반기보고서", "분기보고서"]
    reports = [item for item in res.get("list", []) if any(report_type in item["report_nm"] for report_type in report_types)]
    if not reports:
        print("최근 정기보고서를 찾을 수 없습니다.")
        return None
        
    latest_report = reports[0]
    report_nm = latest_report['report_nm']
    reception_year = int(latest_report['rcept_dt'][:4])
    if "사업보고서" in report_nm:
        bsns_year = reception_year - 1
        reprt_code = ANNUAL_REPORT
    else:
        bsns_year = reception_year
        if "반기보고서" in report_nm: reprt_code = HALF_REPORT
        elif "분기보고서" in report_nm:
            if "(3분기)" in report_nm or "09" in report_nm: reprt_code = Q3_REPORT
            else: reprt_code = Q1_REPORT
        else: return None
    print(f"최신 보고서 확인: {bsns_year}년 {report_nm} (코드: {reprt_code})")
    return {"bsns_year": bsns_year, "reprt_code": reprt_code}




def get_financial_data(corp_code, bsns_year, reprt_code):
    fs_url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(bsns_year),
        "reprt_code": reprt_code,
        "fs_div": "CFS"
    }
    res = requests.get(fs_url, params=params).json()

    if res.get('status') == '013':
        print(f"({bsns_year}년 {reprt_code}) 연결재무제표가 없어 개별재무제표 조회")
        params["fs_div"] = "OFS"
        res = requests.get(fs_url, params=params).json()

    if res.get('status') != '000':
        print(f"API 오류: {res.get('message')} ({bsns_year}년 {reprt_code})")
        return None

    try:
        df = pd.DataFrame(res.get("list", []))

        
        if reprt_code == ANNUAL_REPORT:
            th_col = 'thstrm_amount'
            fr_col = 'frmtrm_amount'
        else:
            th_col = 'thstrm_add_amount'
            fr_col = 'frmtrm_add_amount'
            

        ################################################ROIC 계산에 필요한 정보 가져오는 부분#################################################
        
        df_is = df[df['sj_nm'].isin(['손익계산서', '포괄손익계산서'])]
        df_bs = df[df['sj_nm'].isin(['재무상태표', '연결재무상태표'])]
        df_cf = df[df['sj_div'] == 'CF']

        
        ebit_accounts = df_is['account_nm'].str.contains('영업이익|영업손실|영업손익', na=False) & \
                                ~df_is['account_nm'].str.contains('계속|중단|기타', na=False)


        ebit_th = pd.to_numeric(df_is.loc[ebit_accounts, th_col]).squeeze()
        ebit_fr = pd.to_numeric(df_is.loc[ebit_accounts, fr_col]) if fr_col in df_is.columns else 0
        ebit_fr = ebit_fr.squeeze() if not isinstance(ebit_fr, int) else ebit_fr
        
        # EBT
        ebt_accounts = df_is['account_nm'].str.contains('법인세', na=False) & \
                       df_is['account_nm'].str.contains('차감전', na=False)  
        
        ebt_th = pd.to_numeric(df_is.loc[ebt_accounts, th_col]).squeeze()
        ebt_fr = pd.to_numeric(df_is.loc[ebt_accounts, fr_col]) if fr_col in df_is.columns else 0
        ebt_fr = ebt_fr.squeeze() if not isinstance(ebt_fr, int) else ebt_fr
        
        # EBT 제외
        df_is = df_is.loc[~ebt_accounts].copy()
        
        # Tax
        tax_accounts = df_is['account_nm'].str.contains('법인세비용') & \
                       ~df_is['account_nm'].str.contains('기타')
                        
        tax_th = pd.to_numeric(
            df_is.loc[tax_accounts, th_col],
            errors="coerce"
        ).sum()
        
        tax_fr = (
            pd.to_numeric(df_is.loc[tax_accounts, fr_col], errors="coerce").sum()
            if fr_col in df_is.columns else 0
        )
        
        # 음수 처리
        tax_th = abs(tax_th)
        tax_fr = abs(tax_fr)


        tax_rate_th, tax_rate_fr = tax_th / ebt_th, tax_fr / ebt_fr

        
        keywords_dict = {
            "inventory_th": "재고자산",
            "accounts_receivable_th": "매출채권",
            "accounts_payable_th": "매입채무",
            "fixed_assets_th": "유형자산",
            "intangible_assets_th": "무형자산"
        }
        
        results = {}
        
        for var_name, keyword in keywords_dict.items():
            
            mask = df_bs['account_nm'].str.contains(keyword, na=False) & \
                   ~df_bs['account_nm'].str.contains('감가', na=False)
            
            th_series = pd.to_numeric(df_bs.loc[mask, 'thstrm_amount'], errors='coerce')
            
            if th_series.empty:
                results[var_name] = 0

            else:
                results[var_name] = th_series.sum().squeeze()  # 스칼라로 변환

        # 변수로 꺼내기
        inventory = results["inventory_th"]
        accounts_receivable = results["accounts_receivable_th"]
        accounts_payable = results["accounts_payable_th"]
        fixed_assets = results["fixed_assets_th"]
        intangible_assets = results["intangible_assets_th"]

        #########################################################EV/EBIT 계산에 필요한 정보 가져오는 부분 ###########################################



                
        nci_th = pd.to_numeric(
            df_bs.loc[df_bs['account_nm'].str.contains('비지배지분', na=False), 'thstrm_amount'],
            errors='coerce'
        ).sum()  # 없으면 sum()은 0을 반환
        
        debt_th = pd.to_numeric(
            df_bs.loc[df_bs['account_nm'].str.contains('부채총계', na=False), 'thstrm_amount'],
            errors='coerce'
        ).sum()
        
        cash_th = pd.to_numeric(
            df_bs.loc[df_bs['account_nm'].str.contains('현금', na=False), 'thstrm_amount'],
            errors='coerce'
        ).sum()
        
        st_fin_th = pd.to_numeric(
            df_bs.loc[df_bs['account_nm'].str.contains('단기금융', na=False), 'thstrm_amount'],
            errors='coerce'
        ).sum()

        ######################################################이자보상배율 계산에 필요한 정보 가져오는 부분############################################

        interest_th = pd.to_numeric(
            df_cf.loc[
                df_cf["account_nm"].str.contains("이자", na=False)
                & df_cf["account_nm"].str.contains("지급", na=False),
                "thstrm_amount"
            ],
            errors="coerce"
        ).sum()

        interest_th = abs(interest_th)



        ################################################################가져 온 정보 넘기기 ####################################################
        
        data = {
            'ebit_th': ebit_th, 'ebit_fr': ebit_fr,
            'tax_rate_th': tax_rate_th, 'tax_rate_fr': tax_rate_fr,
            
            'inventory' : inventory,
            'accounts_receivable' : accounts_receivable,
            'accounts_payable' : accounts_payable,
            'fixed_assets' : fixed_assets,
            'intangible_assets' : intangible_assets,
            'nci' : nci_th,
            'total_debt': debt_th,
            'cash': cash_th + st_fin_th,
            'interest' : interest_th
        }
        print('data : '  , data)
        
        return data
        
    except Exception as e:
        print(f"데이터 처리 중 오류 발생: {e} ({bsns_year}년 {reprt_code})")
        return None




def calc_ROIC_and_EV_EBIT_and_interest_coverage(row):
    
    corp_code = row["corp_code"]
    market_cap = row["시가총액"]

    latest = find_latest_report(corp_code)
    if not latest:
        return pd.Series({"ROIC": np.nan, "EV_EBIT": np.nan, "이자보상배율" : np.nan})

    latest_fs = get_financial_data(corp_code, latest["bsns_year"], latest["reprt_code"])
    if not latest_fs:
        return pd.Series({"ROIC": np.nan, "EV_EBIT": np.nan, "이자보상배율" : np.nan})

    # 🔹 연간 / 분기 공통 처리
    if latest["reprt_code"] == ANNUAL_REPORT:
        ebit = latest_fs["ebit_th"]
        tax_rate = latest_fs['tax_rate_th']
        interest = latest_fs['interest']
        nopat = ebit * (1-tax_rate)
    
    else:
        last_annual = get_financial_data(corp_code, latest["bsns_year"] - 1, ANNUAL_REPORT)
        if not last_annual:
            return pd.Series({"ROIC": np.nan, "EV_EBIT": np.nan, "이자보상배율" : np.nan})

        
        last_frmtrm_interest = get_financial_data(corp_code, latest["bsns_year"] - 1, latest["reprt_code"])
        if not last_frmtrm_interest:
            return pd.Series({"ROIC": np.nan, "EV_EBIT": np.nan, "이자보상배율" : np.nan})


        
        ebit = latest_fs["ebit_th"] + last_annual["ebit_th"] - latest_fs["ebit_fr"]
        nopat = (
            latest_fs["ebit_th"] * (1 - latest_fs["tax_rate_th"])
            + last_annual["ebit_th"] * (1 - last_annual["tax_rate_th"])
            - latest_fs["ebit_fr"] * (1 - latest_fs["tax_rate_fr"])
        )
        interest = latest_fs["interest"] + last_annual["interest"] - last_frmtrm_interest["interest"]

 
    #  IC
    
    ic = (
        latest_fs["inventory"]
        + latest_fs["accounts_receivable"]
        - latest_fs["accounts_payable"]
        + latest_fs["fixed_assets"]
        + latest_fs["intangible_assets"]
    )

    if not np.isfinite(nopat) or not np.isfinite(ic) or ic <= 0:
        roic = np.nan
    else:
        roic = nopat / ic

    # 🔹 EV / EBIT
    ev = market_cap + latest_fs['nci'] + latest_fs["total_debt"] - latest_fs["cash"]

    if not np.isfinite(ev) or not np.isfinite(ebit) or ebit <= 0:
        ev_ebit = np.nan
    else:
        ev_ebit = ev / ebit

    # 🔹 이자보상배율

    if interest == 0:
        interest_coverage = 10000
    
    elif ebit <= 0:
        return np.nan
    
    else : 
        interest_coverage = ebit / interest


    print(df_stock_list[df_stock_list['corp_code']==corp_code], '지표들: ',  pd.Series({
        "ROIC": roic,
        "EV_EBIT": ev_ebit,
        "이자보상배율" : interest_coverage
    }))

    
    return pd.Series({
        "ROIC": roic,
        "EV_EBIT": ev_ebit,
        "이자보상배율" : interest_coverage
    })

    



# ROIC, EB/EBIT, 이자보상배율 계산

BATCH_SIZE = 150
SLEEP_PER_CALL = 0.3      # 종목 1개당 sleep
SLEEP_PER_BATCH = 20     # 배치 종료 후 sleep

roic_list = []
ev_ebit_list = []
interest_coverage_list = []

for i in range(0, len(df_stock_list), BATCH_SIZE):
    batch = df_stock_list.iloc[i:i + BATCH_SIZE]

    print(f"▶ 지표 계산 중: {i} ~ {i + len(batch) - 1}")

    for _, row in batch.iterrows():
        try:
            metrics = calc_ROIC_and_EV_EBIT_and_interest_coverage(row)

            roic_list.append(metrics.get("ROIC", np.nan))
            ev_ebit_list.append(metrics.get("EV_EBIT", np.nan))
            interest_coverage_list.append(metrics.get("이자보상배율", np.nan))

        except Exception as e:
            roic_list.append(np.nan)
            ev_ebit_list.append(np.nan)
            interest_coverage_list.append(np.nan)
            print("지표 계산 실패:", row.get("corp_code"), e)

        time.sleep(SLEEP_PER_CALL)

    # 🔴 배치 단위 휴식 (DART 차단 방지 핵심)
    if i + BATCH_SIZE < len(df_stock_list):
        print("⏸ DART 보호용 휴식...")
        time.sleep(SLEEP_PER_BATCH)



df_stock_list = (
    df_stock_list[df_stock_list["ROIC"] >= 0.1]
    .reset_index(drop=True)
)



# 1️⃣ 업종별 중앙값 계산 (PER / EV·EBIT / 이자보상배율)
industry_median = (
    df_stock_list
    .groupby('업종명')[['PER', 'EV/EBIT', '이자보상배율']]
    .median()
    .rename(columns={
        'PER': 'PER_median',
        'EV/EBIT': 'EV_EBIT_median',
        '이자보상배율': 'ICR_median'
    })
)

# 2️⃣ 원본 df에 업종 중앙값 merge
df_merged = df_stock_list.merge(
    industry_median,
    on='업종명',
    how='left'
)

# 3️⃣ 업종 중앙값 기준 필터링
df_under_price_stock_list = (
    df_merged[
        (df_merged['PER'] <= df_merged['PER_median']) &
        (df_merged['EV/EBIT'] <= df_merged['EV_EBIT_median']) &
        (df_merged['이자보상배율'] >= df_merged['ICR_median'])
    ]
    .drop(columns=['PER_median', 'EV_EBIT_median', 'ICR_median'])
    .reset_index(drop=True)
)

print("EV/EBIT, PER, ROIC, 이자보상배율 업종 중앙값 기준 스크리닝 완료")




cols = [c for c in df_under_price_stock_list.columns if c != 'EV/EBIT'] + ['EV/EBIT']
df_under_price_stock_list = df_under_price_stock_list[cols]


df_under_price_stock_list = (
    df_under_price_stock_list
    .sort_values(by=['업종명', 'EV/EBIT'], ascending=[True, True])
    .reset_index(drop=True)
)


df_under_price_stock_list['티커'] = (
    df_under_price_stock_list['티커']
    .astype(str)
    .str.strip()
)

df_under_price_stock_list = df_under_price_stock_list.drop(
    columns=['BPS', 'EPS', 'corp_code'],
    errors='ignore'
)



print(">>> 업종별 EV/EBIT 오름차순 스크리닝 완료. 선정 종목 수:", len(df_filtered))


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES
)

gc = gspread.authorize(creds)

spreadsheet = gc.open_by_key(SPREADSHEET_ID)
worksheet = spreadsheet.get_worksheet(0)  \

worksheet.clear()

set_with_dataframe(
    worksheet,
    df_under_price_stock_list,
    include_index=False,
    include_column_header=True
)

os.remove(SERVICE_ACCOUNT_FILE)
