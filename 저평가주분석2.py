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
import patch_requests
from pykrx import stock
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
import time
import numpy as np


API_KEY = os.environ["API_KEY"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
service_account_json = os.environ["SERVICE_ACCOUNT_FILE"]
KRX_ID = os.environ["KRX_ID"]
KRX_PW = os.environ["KRX_PW"]


# 서비스계정 JSON을 임시 파일로 생성 (종료 시 삭제)
SERVICE_ACCOUNT_FILE = "service_account.json"
with open(SERVICE_ACCOUNT_FILE, "w", encoding="utf-8") as f:
    f.write(service_account_json)

# 보고서 코드
ANNUAL_REPORT = "11011"
HALF_REPORT   = "11012"
Q1_REPORT     = "11013"
Q3_REPORT     = "11014"

# 무차입(이자=0) 기업용 이자보상배율 표시값 (업종 중앙값 계산에서는 제외)
ICR_SENTINEL = 10000

# DART 호출 간격 (재시도 로직이 끊김을 흡수하므로 0. 10053이 자주 나면 0.1~0.3로)
SLEEP_PER_CALL = 0
SLEEP_PER_BATCH = 0
BATCH_SIZE = 150


# ===== 공통 숫자 파서 (DART 금액은 "1,234,567" 형태라 콤마 제거 필수) =====
def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce')

def first_or_zero(series):
    s = to_num(series).dropna()
    return float(s.iloc[0]) if not s.empty else 0.0

def sum_or_zero(series):
    return float(to_num(series).sum())


def dart_get(url, params, retries=4):
    """DART GET — 연결 끊김(10053/RemoteDisconnected) 시 잠깐 쉬고 재시도."""
    last_err = None
    for attempt in range(retries):
        try:
            return requests.get(url, params=params, timeout=20).json()
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  DART 요청 {retries}회 실패: {last_err}")
    return {}


# ===== 1. 기준일 / 업종 / 종목 =====
to_date = datetime.today().strftime("%Y%m%d")
from_date = (datetime.today() - timedelta(days=30)).strftime("%Y%m%d")

_df = stock.get_market_ohlcv_by_date(from_date, to_date, "005930")
today = _df.index[-1].strftime("%Y%m%d")


def get_sector_df_safe(market, base_date, retries=4, max_back=5):
    """연결 끊기면 재시도, 그래도 안 되면 과거 영업일로."""
    d = base_date
    for _ in range(max_back):
        for attempt in range(retries):
            try:
                df = stock.get_market_sector_classifications(d, market=market)
                if df is not None and not df.empty:
                    if d != base_date:
                        print(f"  ({market}) {base_date} 실패 → {d} 사용")
                    return df
            except Exception as e:
                wait = 3 * (attempt + 1)
                print(f"  ({market}) {d} 조회 실패({attempt+1}/{retries}): {e} → {wait}초 후 재시도")
                time.sleep(wait)
        prev = datetime.strptime(d, "%Y%m%d") - timedelta(days=1)
        d = stock.get_nearest_business_day_in_a_week(prev.strftime("%Y%m%d"))
    raise RuntimeError(f"{market} 업종분류 데이터를 끝내 가져오지 못했습니다.")


KOSPI_sector_df  = get_sector_df_safe('KOSPI', today)
KOSDAQ_sector_df = get_sector_df_safe('KOSDAQ', today)

total_sector_df = pd.concat([KOSPI_sector_df, KOSDAQ_sector_df])
total_sector_df = total_sector_df[['종목명', '업종명', '시가총액']]


def get_stock_lists():
    print("코스피, 코스닥 전체 종목 정보를 조회합니다...")
    df_kospi = stock.get_market_fundamental(today, market="KOSPI").reset_index()
    df_kospi['시장'] = 'KOSPI'
    df_kosdaq = stock.get_market_fundamental(today, market="KOSDAQ").reset_index()
    df_kosdaq['시장'] = 'KOSDAQ'
    df_all = pd.concat([df_kospi, df_kosdaq], ignore_index=True)

    ticker_to_name = {}
    for market in ["KOSPI", "KOSDAQ"]:
        for ticker in stock.get_market_ticker_list(today, market=market):
            ticker_to_name[ticker] = stock.get_market_ticker_name(ticker)
    df_all['종목명'] = df_all['티커'].map(ticker_to_name)

    def is_preferred(ticker):
        return ticker[-1] in ['5', '6', '7', '8', '9'] or ticker.endswith('K')

    df_preferred_list = df_all[df_all['티커'].apply(is_preferred)].copy()
    remove_keywords = ["리츠", "ETF", "ETN", "스팩"]
    pattern_remove = "|".join(remove_keywords)
    df_stock_list = df_all[~df_all['티커'].apply(is_preferred)]
    df_stock_list = df_stock_list[~df_stock_list['종목명'].str.contains(pattern_remove, na=False)]

    multi_preferred_names = []
    for name in df_stock_list['종목명'].unique():
        if df_preferred_list['종목명'].str.contains(name, na=False, regex=False).sum() >= 2:
            multi_preferred_names.append(name)
    if multi_preferred_names:
        df_preferred_list = df_preferred_list[
            ~df_preferred_list['종목명'].apply(lambda x: any(n in x for n in multi_preferred_names))]
        df_stock_list = df_stock_list[~df_stock_list['종목명'].isin(multi_preferred_names)]

    cols = ['시장', '티커', '종목명', 'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
    return df_stock_list[cols], df_preferred_list[cols]


# ===== 2. DART 고유번호 =====
def get_dart_corp_df():
    res = requests.get("https://opendart.fss.or.kr/api/corpCode.xml", params={"crtfc_key": API_KEY})
    res.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        xml_data = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml_data)
    corp_list = [{"corp_code": c.findtext("corp_code"),
                  "corp_name": c.findtext("corp_name"),
                  "stock_code": c.findtext("stock_code")} for c in root.findall("list")]
    dart_corp_df = pd.DataFrame(corp_list)
    dart_corp_df['stock_code'] = dart_corp_df['stock_code'].astype(str).str.strip().str.zfill(6)
    return dart_corp_df


# ===== 3. 최신 정기보고서 =====
def find_latest_report(corp_code):
    list_url = "https://opendart.fss.or.kr/api/list.json"
    end_de = date.today().strftime('%Y%m%d')
    bgn_de = (date.today() - timedelta(days=450)).strftime('%Y%m%d')
    params = {"crtfc_key": API_KEY, "corp_code": corp_code,
              "bgn_de": bgn_de, "end_de": end_de, "pblntf_ty": "A", "last_reprt_at": "Y"}
    res = dart_get(list_url, params)
    if res.get('status') != '000':
        return None
    report_types = ["사업보고서", "반기보고서", "분기보고서"]
    reports = [it for it in res.get("list", []) if any(rt in it["report_nm"] for rt in report_types)]
    if not reports:
        return None
    latest = reports[0]
    report_nm = latest['report_nm']
    reception_year = int(latest['rcept_dt'][:4])
    if "사업보고서" in report_nm:
        return {"bsns_year": reception_year - 1, "reprt_code": ANNUAL_REPORT}
    bsns_year = reception_year
    if "반기보고서" in report_nm:
        reprt_code = HALF_REPORT
    elif "분기보고서" in report_nm:
        reprt_code = Q3_REPORT if ("(3분기)" in report_nm or "09" in report_nm) else Q1_REPORT
    else:
        return None
    return {"bsns_year": bsns_year, "reprt_code": reprt_code}


# ===== 4. 재무 데이터 =====
def get_financial_data(corp_code, bsns_year, reprt_code):
    fs_url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {"crtfc_key": API_KEY, "corp_code": corp_code,
              "bsns_year": str(bsns_year), "reprt_code": reprt_code, "fs_div": "CFS"}
    res = dart_get(fs_url, params)
    if res.get('status') == '013':
        params["fs_div"] = "OFS"
        res = dart_get(fs_url, params)
    if res.get('status') != '000':
        return None

    try:
        df = pd.DataFrame(res.get("list", []))
        if df.empty:
            return None

        if reprt_code == ANNUAL_REPORT:
            th_col, fr_col = 'thstrm_amount', 'frmtrm_amount'
        else:
            th_col, fr_col = 'thstrm_add_amount', 'frmtrm_add_amount'
        if th_col not in df.columns: th_col = 'thstrm_amount'
        if fr_col not in df.columns: fr_col = 'frmtrm_amount'

        df_is = df[df['sj_nm'].isin(['손익계산서', '포괄손익계산서'])]
        df_bs = df[df['sj_nm'].isin(['재무상태표', '연결재무상태표'])]
        df_cf = df[df['sj_div'] == 'CF']

        ebit_mask = df_is['account_nm'].str.contains('영업이익|영업손실|영업손익', na=False) & \
                    ~df_is['account_nm'].str.contains('계속|중단|기타', na=False)
        ebit_th = first_or_zero(df_is.loc[ebit_mask, th_col])
        ebit_fr = first_or_zero(df_is.loc[ebit_mask, fr_col])

        ebt_mask = df_is['account_nm'].str.contains('법인세', na=False) & \
                   df_is['account_nm'].str.contains('차감전', na=False)
        ebt_th = first_or_zero(df_is.loc[ebt_mask, th_col])
        ebt_fr = first_or_zero(df_is.loc[ebt_mask, fr_col])

        df_is_tax = df_is.loc[~ebt_mask]
        tax_mask = df_is_tax['account_nm'].str.contains('법인세비용', na=False) & \
                   ~df_is_tax['account_nm'].str.contains('기타', na=False)
        tax_th = abs(sum_or_zero(df_is_tax.loc[tax_mask, th_col]))
        tax_fr = abs(sum_or_zero(df_is_tax.loc[tax_mask, fr_col]))

        tax_rate_th = tax_th / ebt_th if ebt_th else 0.0
        tax_rate_fr = tax_fr / ebt_fr if ebt_fr else 0.0

        def bs_sum(keyword):
            mask = df_bs['account_nm'].str.contains(keyword, na=False) & \
                   ~df_bs['account_nm'].str.contains('감가', na=False)
            return sum_or_zero(df_bs.loc[mask, 'thstrm_amount'])

        inventory           = bs_sum('재고자산')
        accounts_receivable = bs_sum('매출채권')
        accounts_payable    = bs_sum('매입채무')
        fixed_assets        = bs_sum('유형자산')
        intangible_assets   = bs_sum('무형자산')

        nci    = sum_or_zero(df_bs.loc[df_bs['account_nm'].str.contains('비지배지분', na=False), 'thstrm_amount'])
        debt   = sum_or_zero(df_bs.loc[df_bs['account_nm'].str.contains('부채총계', na=False), 'thstrm_amount'])
        cash   = sum_or_zero(df_bs.loc[df_bs['account_nm'].str.contains('현금', na=False), 'thstrm_amount'])
        st_fin = sum_or_zero(df_bs.loc[df_bs['account_nm'].str.contains('단기금융', na=False), 'thstrm_amount'])

        interest = abs(sum_or_zero(df_cf.loc[
            df_cf['account_nm'].str.contains('이자', na=False) &
            df_cf['account_nm'].str.contains('지급', na=False), 'thstrm_amount']))

        return {
            'ebit_th': ebit_th, 'ebit_fr': ebit_fr,
            'tax_rate_th': tax_rate_th, 'tax_rate_fr': tax_rate_fr,
            'inventory': inventory, 'accounts_receivable': accounts_receivable,
            'accounts_payable': accounts_payable, 'fixed_assets': fixed_assets,
            'intangible_assets': intangible_assets,
            'nci': nci, 'total_debt': debt, 'cash': cash + st_fin,
            'interest': interest,
        }
    except Exception as e:
        print(f"데이터 처리 중 오류: {e} ({bsns_year}년 {reprt_code})")
        return None


# ===== 5. 종목별 지표 =====
def calc_metrics(row):
    nan_series = pd.Series({"ROIC": np.nan, "EV_EBIT": np.nan, "이자보상배율": np.nan})
    corp_code = row["corp_code"]
    market_cap = row["시가총액"]
    if pd.isna(corp_code):
        return nan_series

    latest = find_latest_report(corp_code)
    if not latest:
        return nan_series
    latest_fs = get_financial_data(corp_code, latest["bsns_year"], latest["reprt_code"])
    if not latest_fs:
        return nan_series

    if latest["reprt_code"] == ANNUAL_REPORT:
        ebit = latest_fs["ebit_th"]
        nopat = ebit * (1 - latest_fs["tax_rate_th"])
        interest = latest_fs["interest"]
    else:
        last_annual = get_financial_data(corp_code, latest["bsns_year"] - 1, ANNUAL_REPORT)
        last_same   = get_financial_data(corp_code, latest["bsns_year"] - 1, latest["reprt_code"])
        if not last_annual or not last_same:
            return nan_series
        ebit = latest_fs["ebit_th"] + last_annual["ebit_th"] - latest_fs["ebit_fr"]
        nopat = (latest_fs["ebit_th"] * (1 - latest_fs["tax_rate_th"])
                 + last_annual["ebit_th"] * (1 - last_annual["tax_rate_th"])
                 - latest_fs["ebit_fr"] * (1 - latest_fs["tax_rate_fr"]))
        interest = latest_fs["interest"] + last_annual["interest"] - last_same["interest"]

    ic = (latest_fs["inventory"] + latest_fs["accounts_receivable"] - latest_fs["accounts_payable"]
          + latest_fs["fixed_assets"] + latest_fs["intangible_assets"])
    roic = nopat / ic if (np.isfinite(nopat) and np.isfinite(ic) and ic > 0) else np.nan

    ev = market_cap + latest_fs["nci"] + latest_fs["total_debt"] - latest_fs["cash"]
    ev_ebit = ev / ebit if (np.isfinite(ev) and np.isfinite(ebit) and ebit > 0) else np.nan

    if interest == 0:
        icr = ICR_SENTINEL
    elif ebit <= 0:
        icr = np.nan
    else:
        icr = ebit / interest

    return pd.Series({"ROIC": roic, "EV_EBIT": ev_ebit, "이자보상배율": icr})


# ===== 메인 =====
def main():
    df_stock_list, _ = get_stock_lists()

    df_stock_list = total_sector_df.merge(
        df_stock_list[['종목명', '티커', '시장', 'BPS', 'PER', 'PBR', 'EPS']],
        on='종목명', how='left')

    dart_corp_df = get_dart_corp_df()
    df_stock_list = df_stock_list.merge(
        dart_corp_df[['corp_code', 'stock_code']],
        left_on='티커', right_on='stock_code', how='left')

    valid_per = df_stock_list[df_stock_list['PER'].notna() & (df_stock_list['PER'] != 0)]
    df_stock_list = df_stock_list[df_stock_list['종목명'].isin(valid_per['종목명'])].reset_index(drop=True)

    roic_list, ev_ebit_list, icr_list = [], [], []
    for i in range(0, len(df_stock_list), BATCH_SIZE):
        batch = df_stock_list.iloc[i:i + BATCH_SIZE]
        print(f"▶ 지표 계산 중: {i} ~ {i + len(batch) - 1} / {len(df_stock_list)}")
        for _, row in batch.iterrows():
            try:
                m = calc_metrics(row)
                roic_list.append(m["ROIC"])
                ev_ebit_list.append(m["EV_EBIT"])
                icr_list.append(m["이자보상배율"])
            except Exception as e:
                roic_list.append(np.nan); ev_ebit_list.append(np.nan); icr_list.append(np.nan)
                print("지표 계산 실패:", row.get("corp_code"), e)
            if SLEEP_PER_CALL:
                time.sleep(SLEEP_PER_CALL)
        if SLEEP_PER_BATCH and i + BATCH_SIZE < len(df_stock_list):
            time.sleep(SLEEP_PER_BATCH)

    df_stock_list['ROIC'] = roic_list
    df_stock_list['EV/EBIT'] = ev_ebit_list
    df_stock_list['이자보상배율'] = icr_list
    df_stock_list = df_stock_list.dropna(subset=['ROIC', 'EV/EBIT', '이자보상배율']).reset_index(drop=True)

    df = df_stock_list[df_stock_list["ROIC"] >= 0.1].reset_index(drop=True)

    tmp = df.copy()
    tmp.loc[tmp['이자보상배율'] == ICR_SENTINEL, '이자보상배율'] = np.nan
    industry_median = (
        tmp.groupby(['시장', '업종명'])[['PER', 'EV/EBIT', '이자보상배율']]
        .median()
        .rename(columns={'PER': 'PER_median', 'EV/EBIT': 'EV_EBIT_median', '이자보상배율': 'ICR_median'})
        .reset_index()
    )

    df = df.merge(industry_median, on=['시장', '업종명'], how='left')
    df = df[
        (df['PER'] <= df['PER_median']) &
        (df['EV/EBIT'] <= df['EV_EBIT_median']) &
        (df['이자보상배율'] >= df['ICR_median'])
    ].drop(columns=['PER_median', 'EV_EBIT_median', 'ICR_median']).reset_index(drop=True)

    cols = [c for c in df.columns if c != 'EV/EBIT'] + ['EV/EBIT']
    df = df[cols].sort_values(by=['업종명', 'EV/EBIT'], ascending=[True, True]).reset_index(drop=True)
    df['티커'] = df['티커'].astype(str).str.strip()
    df = df.drop(columns=['BPS', 'EPS', 'corp_code', 'stock_code'], errors='ignore')

    print(">>> 스크리닝 완료. 선정 종목 수:", len(df))

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key(SPREADSHEET_ID)

    ws_kospi = spreadsheet.get_worksheet(0)
    ws_kosdaq = spreadsheet.get_worksheet(1)
    df_kospi = df[df['시장'] == 'KOSPI']
    df_kosdaq = df[df['시장'] == 'KOSDAQ']

    ws_kospi.clear()
    ws_kosdaq.clear()
    set_with_dataframe(ws_kospi, df_kospi, include_index=False, include_column_header=True)
    set_with_dataframe(ws_kosdaq, df_kosdaq, include_index=False, include_column_header=True)
    print("KOSPI → Sheet1 / KOSDAQ → Sheet2 저장 완료")


if __name__ == "__main__":
    try:
        main()
    finally:
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            os.remove(SERVICE_ACCOUNT_FILE)



print("KOSPI → Sheet1 / KOSDAQ → Sheet2 저장 완료")

os.remove(SERVICE_ACCOUNT_FILE)
