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

        def to_numeric(series): 
            return pd.to_numeric(series.str.replace(',', ''), errors='coerce').fillna(0)
        
        if reprt_code == ANNUAL_REPORT:
            th_col = 'thstrm_amount'
            fr_col = 'frmtrm_amount'
        else:
            th_col = 'thstrm_add_amount'
            fr_col = 'frmtrm_add_amount'
            
        
        df_is = df[df['sj_nm'].isin(['손익계산서', '포괄손익계산서'])]
        df_bs = df[df['sj_nm'].isin(['재무상태표', '연결재무상태표'])]

        
        ebit_accounts = df_is['account_nm'].str.contains('영업이익|영업손실|영업손익', na=False) & \
                                ~df_is['account_nm'].str.contains('계속|중단', na=False)


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
                        
        tax_th = pd.to_numeric(df_is.loc[tax_accounts, th_col]).squeeze()
        tax_fr = pd.to_numeric(df_is.loc[tax_accounts, fr_col]) if fr_col in df_is.columns else 0
        tax_fr = tax_fr.squeeze() if not isinstance(tax_fr, int) else tax_fr
        
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

    

        data = {
            'ebit_th': ebit_th, 'ebit_fr': ebit_fr,
            'tax_rate_th': tax_rate_th, 'tax_rate_fr': tax_rate_fr,
            
            'inventory' : inventory,
            'accounts_receivable' : accounts_receivable,
            'accounts_payable' : accounts_payable,
            'fixed_assets' : fixed_assets,
            'intangible_assets' : intangible_assets
        }
        return data
        
    except Exception as e:
        print(f"데이터 처리 중 오류 발생: {e} ({bsns_year}년 {reprt_code})")
        return None




def calc_roic(row):
    results = []
    print("--- ROIC 계산 중 ---")
    
    corp_code = row["corp_code"]

    latest = find_latest_report(corp_code)
    if not latest:
        return None

    latest_fs = get_financial_data(corp_code, latest["bsns_year"], latest["reprt_code"])
    if not latest_fs:
        return None

    if latest["reprt_code"] == ANNUAL_REPORT:
        ebit = latest_fs["ebit_th"]
        tax_rate = latest_fs['tax_rate_th']
        
        inventory = latest_fs['inventory']
        accounts_receivable = latest_fs['accounts_receivable']
        accounts_payable = latest_fs['accounts_payable']
        fixed_assets = latest_fs['fixed_assets']
        intangible_assets = latest_fs['intangible_assets']

        nopat = ebit * (1-tax_rate)
        ic = inventory + accounts_receivable - accounts_payable + fixed_assets + intangible_assets
    
    else:
        last_annual = get_financial_data(corp_code, latest["bsns_year"] - 1, ANNUAL_REPORT)
        if not last_annual:
            return None
        
        nopat = (
            latest_fs["ebit_th"] * (1 - latest_fs["tax_rate_th"])
            + last_annual["ebit_th"] * (1 - last_annual["tax_rate_th"])
            - latest_fs["ebit_fr"] * (1 - latest_fs["tax_rate_fr"])
        )
        ic = latest_fs['inventory'] + latest_fs['accounts_receivable'] - latest_fs['accounts_payable'] + latest_fs['fixed_assets'] + latest_fs['intangible_assets']

    
    if nopat/ic <= 0:
        return None

    roic = nopat/ic
    
    return roic


# 1️⃣ ROIC 계산
df_filtered['ROIC'] = df_stock_list.apply(calc_roic, axis=1)

# 2️⃣ ROIC가 0.1 이상인 기업만 남기기
df_stock_list = df_stock_list[df_stock_list['ROIC'] >= 0.1].reset_index(drop=True)





def get_financial_data(corp_code, bsns_year, reprt_code):   #EV와 EBITDA를 구하는데 필요한 항목을 가져오는 과정(가장 최근 4개 분기의 데이터를 가져와서 계산)
    """보고서 종류에 따라 올바른 필드를 선택하여 재무 데이터를 조회합니다."""
    fs_url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {"crtfc_key": API_KEY, "corp_code": corp_code, "bsns_year": str(bsns_year), "reprt_code": reprt_code, "fs_div": "CFS"}
    res = requests.get(fs_url, params=params).json()

    if res.get('status') == '013':
        print(f"({bsns_year}년 {reprt_code}) 연결재무제표가 없어 개별재무제표를 조회합니다.")
        params["fs_div"] = "OFS"
        res = requests.get(fs_url, params=params).json()
    
    if res.get('status') != '000':
        print(f"API 오류: {res.get('message')} ({bsns_year}년 {reprt_code})")
        return None
    
    try:
        df = pd.DataFrame(res.get("list", []))
        def to_numeric(series): return pd.to_numeric(series.str.replace(',', ''), errors='coerce').fillna(0)

        def get_values(df, keywords, is_bs=False):
            # 재무상태표(BS)가 아니면서, 사업보고서(ANNUAL_REPORT)가 아닌 경우 누적 필드 사용
            if not is_bs and reprt_code != ANNUAL_REPORT:
                th_col, fr_col = 'thstrm_add_amount', 'frmtrm_add_amount'
            else: # 재무상태표이거나 사업보고서인 경우 기본 필드 사용
                th_col, fr_col = 'thstrm_amount', 'frmtrm_amount'

            for keyword in keywords:
                row = df[df['account_nm'].str.strip().str.startswith(keyword)]
                if not row.empty:
                    thstrm = to_numeric(row[th_col]).iloc[0]
                    frmtrm = to_numeric(row[fr_col]).iloc[0] if fr_col in row else 0
                    return thstrm, frmtrm
            return 0, 0

        df_bs = df[df['sj_nm'] == '재무상태표']
        df_is = df[df['sj_nm'].isin(['손익계산서', '포괄손익계산서'])]
        df_cf = df[df['sj_nm'] == '현금흐름표']

        
        ebit_accounts = df_is['account_nm'].str.contains('영업이익|영업손실|영업손익', na=False) & \
                                ~df_is['account_nm'].str.contains('계속|중단', na=False)


        ebit_th = pd.to_numeric(df_is.loc[ebit_accounts, th_col]).squeeze()
        ebit_fr = pd.to_numeric(df_is.loc[ebit_accounts, fr_col]) if fr_col in df_is.columns else 0
        ebit_fr = ebit_fr.squeeze() if not isinstance(ebit_fr, int) else ebit_fr


        
        nci_th, _ = get_values(df_bs, ['비지배지분'], is_bs=True)
        debt_th, _ = get_values(df_bs, ['부채총계'], is_bs=True)

        cash_accounts = df_bs[df_bs['account_nm'].str.strip().str.startswith('현금')]
        cash_th = to_numeric(cash_accounts['thstrm_amount']).sum()
        
        st_fin_accounts = df_bs[df_bs['account_nm'].str.strip().str.startswith('단기금융')]
        st_fin_th = to_numeric(st_fin_accounts['thstrm_amount']).sum()
        
        data = {
            'ebit': ebit_th, 'ebit_fr': ebit_fr,
            'nci': nci_th,
            'total_debt': debt_th,
            'cash': cash_th + st_fin_th,
        }
        return data
    except Exception as e:
        print(f"데이터 처리 중 오류 발생: {e} ({bsns_year}년 {reprt_code})")
        return None


    

def calc_ev_ebit(row):
    corp_code = row["corp_code"]
    market_cap = row["시가총액"]

    latest = find_latest_report(corp_code)
    if not latest:
        return None

    latest_fs = get_financial_data(corp_code, latest["bsns_year"], latest["reprt_code"])
    if not latest_fs:
        return None

    if latest["reprt_code"] == ANNUAL_REPORT:
        ebit = latest_fs["ebit"]
    else:
        last_annual = get_financial_data(corp_code, latest["bsns_year"] - 1, ANNUAL_REPORT)
        if not last_annual:
            return None

        ebit = (
            latest_fs["ebit"]
            + last_annual["ebit"]
            - latest_fs["ebit_fr"]
        )


    ev = market_cap + latest_fs["total_debt"] - latest_fs["cash"]
    
    if ev/ebit <= 0:
        return None
    
    
    return ev / ebit




def build_ev_ebit_table(df_stock_list):
    results = []

    for _, row in df_stock_list.iterrows():
        try:
            ratio = calc_ev_ebit(row)
            if ratio is not None:
                results.append({
                    "종목명": row["종목명"],
                    "업종명": row["업종명"],
                    "EV/EBIT": ratio
                })
        except:
            continue

    df_result = pd.DataFrame(results)
    df_ev_stock_list = df_result.sort_values("EV/EBIT")
    return df_ev_stock_list


df_ev_stock_list = build_ev_ebit_table(df_stock_list)

df_stock_list = df_stock_list.merge(
    df_ev_stock_list[['종목명','EV/EBIT']],
    on='종목명',
    how='right'
)







# 업종별 평균 계산
industry_mean = (
    df_stock_list
    .groupby('업종명')[['PER', 'EV/EBIT']]
    .mean()
    .rename(columns={
        'PER': 'PER_mean',
        'EV/EBIT': 'EV_EBIT_mean'
    })
)

# 원본 df에 업종 평균 붙이기
df_merged = df_stock_list.merge(
    industry_mean,
    on='업종명',
    how='left'
)

# 업종 평균보다 낮은 종목 제거
df_under_price_stock_list  = df_merged[
    (df_merged['PER'] <= df_merged['PER_mean']) &
    (df_merged['EV/EBIT'] <= df_merged['EV_EBIT_mean'])
].drop(columns=['PER_mean', 'EV_EBIT_mean'])


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

        def to_numeric(series): 
            return pd.to_numeric(series.str.replace(',', ''), errors='coerce').fillna(0)

        def get_values(df, keywords, is_bs=False):
            if not is_bs and reprt_code != ANNUAL_REPORT:
                th_col, fr_col = 'thstrm_add_amount', 'frmtrm_add_amount'
            else:
                th_col, fr_col = 'thstrm_amount', 'frmtrm_amount'

            for keyword in keywords:
                row = df[df['account_nm'].str.strip().str.startswith(keyword)]
                if not row.empty:
                    thstrm = to_numeric(row[th_col]).iloc[0]
                    frmtrm = to_numeric(row[fr_col]).iloc[0] if fr_col in row else 0
                    return thstrm, frmtrm
            return 0

        
        def cf_get_value(df, keywords, is_bs=False):
            
            th_col = 'thstrm_amount'
        
            for keyword in keywords:
                row = df[df['account_nm'].str.contains(keyword)]
                
                if not row.empty:
                    thstrm = to_numeric(row[th_col]).iloc[0]

                    
                    return thstrm
        
            return 0
                

        df_is = df[df['sj_nm'].isin(['손익계산서', '포괄손익계산서'])]

        
        ebit_accounts = df_is['account_nm'].str.contains('영업이익|영업손실|영업손익', na=False) & \
                                ~df_is['account_nm'].str.contains('계속|중단', na=False)


        ebit_th = pd.to_numeric(df_is.loc[ebit_accounts, th_col]).squeeze()
        ebit_fr = pd.to_numeric(df_is.loc[ebit_accounts, fr_col]) if fr_col in df_is.columns else 0
        ebit_fr = ebit_fr.squeeze() if not isinstance(ebit_fr, int) else ebit_fr



        
        df_cf = df[df['sj_div'] == 'CF']
        interest_th = cf_get_value(df_cf, ['이자의 지급', '이자지급'], is_bs=False)
        interest_th = abs(interest_th)

        data = {
            'ebit': ebit_th, 'ebit_fr': ebit_fr,
            'interest': interest_th
        }
        return data
        
    except Exception as e:
        print(f"데이터 처리 중 오류 발생: {e} ({bsns_year}년 {reprt_code})")
        return None


print("EV/BBIT 및 PER 계산 완료")



# ---------------- 이자보상배율 계산 ----------------
def calc_interest_coverage(row):
    corp_code = row["corp_code"]

    latest = find_latest_report(corp_code)
    if not latest:
        return None

    latest_fs = get_financial_data(corp_code, latest["bsns_year"], latest["reprt_code"])
    
    if not latest_fs:
        return None

    if latest["reprt_code"] == ANNUAL_REPORT:
        ebit = latest_fs["ebit"]
        interest = latest_fs["interest"]
    else:
        last_annual = get_financial_data(corp_code, latest["bsns_year"] - 1, ANNUAL_REPORT)
        last_frmtrm_interest = get_financial_data(corp_code, latest["bsns_year"] - 1, latest["reprt_code"])
        
        
        if not last_frmtrm_interest:
            return None
        
        if not last_annual:
            return None
        
        
        ebit = latest_fs["ebit"] + last_annual["ebit"] - latest_fs["ebit_fr"]
        interest = latest_fs["interest"] + last_annual["interest"] - last_frmtrm_interest["interest"]

    if interest == 0:
        return 1000

    elif interest <= 0 or ebit <= 0:
        return None

    return ebit / interest



# ---------------- 전체 스크리닝 (업종별 반복 처리로 변경) ----------------
def run_screening(df_slice):
    """주어진 업종 슬라이스에서 이자보상배율을 계산하고 업종 중위값 이상인 종목만 반환."""
    if df_slice is None or df_slice.empty:
        return pd.DataFrame(columns=df_slice.columns if df_slice is not None else [])

    df = df_slice.copy()  # 슬라이스 복사본
    print(f"--- 이자보상배율 계산 중: 업종 '{df['업종명'].iloc[0]}' ({len(df)}종목) ---")

    # 안전하게 loc 사용하여 경고 방지
    df.loc[:, "이자보상배율"] = df.apply(calc_interest_coverage, axis=1)

    # 업종 중앙값(혹은 평균) 계산 — 원래는 median 사용
    industry_median_ic = df["이자보상배율"].median()

    # 업종 평균 이상만 반환 (None/NaN 제외)
    df_filtered = df[(df["이자보상배율"].notna()) & (df["이자보상배율"] >= industry_median_ic)].copy()
    return df_filtered


# 업종 목록을 자동으로 가져와 반복 처리
industry_list = df_under_price_stock_list['업종명'].dropna().unique()
filtered_dfs = []

print(">>> 업종별 스크리닝 시작. 업종 수:", len(industry_list))
for ind in industry_list:
    slice_df = df_under_price_stock_list[df_under_price_stock_list['업종명'] == ind]
    screened = run_screening(slice_df)
    if (screened is not None) and (not screened.empty):
        filtered_dfs.append(screened)

# 결과 합치기
if filtered_dfs:
    df_filtered = pd.concat(filtered_dfs, axis=0, ignore_index=True)
else:
    df_filtered = pd.DataFrame(columns=df_under_price_stock_list.columns)

print(">>> 업종별 스크리닝 완료. 선정 종목 수:", len(df_filtered))
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
    df_filtered,
    include_index=False,
    include_column_header=True
)

os.remove(SERVICE_ACCOUNT_FILE)
