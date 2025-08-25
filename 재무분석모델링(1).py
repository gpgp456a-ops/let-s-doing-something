import urllib.request
import ssl
import zipfile
import os
import pandas as pd
import requests
import json
import io
import xml.etree.ElementTree as ET
from datetime import date, timedelta

stock_code = '005490'  ###이부분에서 종목 코드만 바꾸면 됨.


'''한국투자증권 api key'''
APP_KEY = 
APP_SECRET = 

'''dart api key'''
API_KEY = 

'''정확한 계산을 위해서는 depr과 amor에 유형과 무형 자산 상각비를 넣어주면 됨.'''        
depr = 0
amor = 0


'''주식종목코드 정제 파이썬 파일'''

base_dir = os.getcwd()

def kospi_master_download(base_dir, verbose=False):
    cwd = os.getcwd()
    if (verbose): print(f"current directory is {cwd}")
    ssl._create_default_https_context = ssl._create_unverified_context

    urllib.request.urlretrieve("https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip",
                               base_dir + "\\kospi_code.zip")

    os.chdir(base_dir)
    if (verbose): print(f"change directory to {base_dir}")
    kospi_zip = zipfile.ZipFile('kospi_code.zip')
    kospi_zip.extractall()

    kospi_zip.close()

    if os.path.exists("kospi_code.zip"):
        os.remove("kospi_code.zip")


def get_kospi_master_dataframe(base_dir):
    file_name = base_dir + "\\kospi_code.mst"
    tmp_fil1 = base_dir + "\\kospi_code_part1.tmp"
    tmp_fil2 = base_dir + "\\kospi_code_part2.tmp"

    wf1 = open(tmp_fil1, mode="w")
    wf2 = open(tmp_fil2, mode="w")

    with open(file_name, mode="r", encoding="cp949") as f:
        for row in f:
            rf1 = row[0:len(row) - 228]
            rf1_1 = rf1[0:9].rstrip()
            rf1_2 = rf1[9:21].rstrip()
            rf1_3 = rf1[21:].strip()
            wf1.write(rf1_1 + ',' + rf1_2 + ',' + rf1_3 + '\n')
            rf2 = row[-228:]
            wf2.write(rf2)

    wf1.close()
    wf2.close()

    part1_columns = ['단축코드', '표준코드', '한글명']
    df1 = pd.read_csv(tmp_fil1, header=None, names=part1_columns, encoding='cp949')

    field_specs = [2, 1, 4, 4, 4,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 9, 5, 5, 1,
                   1, 1, 2, 1, 1,
                   1, 2, 2, 2, 3,
                   1, 3, 12, 12, 8,
                   15, 21, 2, 7, 1,
                   1, 1, 1, 1, 9,
                   9, 9, 5, 9, 8,
                   9, 3, 1, 1, 1
                   ]

    part2_columns = ['그룹코드', '시가총액규모', '지수업종대분류', '지수업종중분류', '지수업종소분류',
                     '제조업', '저유동성', '지배구조지수종목', 'KOSPI200섹터업종', 'KOSPI100',
                     'KOSPI50', 'KRX', 'ETP', 'ELW발행', 'KRX100',
                     'KRX자동차', 'KRX반도체', 'KRX바이오', 'KRX은행', 'SPAC',
                     'KRX에너지화학', 'KRX철강', '단기과열', 'KRX미디어통신', 'KRX건설',
                     'Non1', 'KRX증권', 'KRX선박', 'KRX섹터_보험', 'KRX섹터_운송',
                     'SRI', '기준가', '매매수량단위', '시간외수량단위', '거래정지',
                     '정리매매', '관리종목', '시장경고', '경고예고', '불성실공시',
                     '우회상장', '락구분', '액면변경', '증자구분', '증거금비율',
                     '신용가능', '신용기간', '전일거래량', '액면가', '상장일자',
                     '상장주수', '자본금', '결산월', '공모가', '우선주',
                     '공매도과열', '이상급등', 'KRX300', 'KOSPI', '매출액',
                     '영업이익', '경상이익', '당기순이익', 'ROE', '기준년월',
                     '시가총액', '그룹사코드', '회사신용한도초과', '담보대출가능', '대주가능'
                     ]

    df2 = pd.read_fwf(tmp_fil2, widths=field_specs, names=part2_columns)

    df_stock_list = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)

    # clean temporary file and dataframe
    del (df1)
    del (df2)
    os.remove(tmp_fil1)
    os.remove(tmp_fil2)
    
    print("Done")

    return df_stock_list


kospi_master_download(base_dir)
df_stock_list = get_kospi_master_dataframe(base_dir) 


'''종목 이름'''
stock_name = df_stock_list[df_stock_list['단축코드'] == stock_code]['한글명'].squeeze()


'''주가 가져오기'''


BASE_URL = "https://openapi.koreainvestment.com:9443"

# 토큰 발급
url = f"{BASE_URL}/oauth2/tokenP"
data = {
    "grant_type": "client_credentials",
    "appkey": APP_KEY,
    "appsecret": APP_SECRET
}
res = requests.post(url, data=json.dumps(data))

if res.status_code == 200:
    token_info = res.json()
    ACCESS_TOKEN = token_info["access_token"]
    print(" 토큰 발급 성공")
else:
    print(" 토큰 발급 실패:", res.status_code, res.text)




"""지정한 종목의 보통주 현재가 반환"""

path = "/uapi/domestic-stock/v1/quotations/inquire-price"
url = f"{BASE_URL}{path}"

headers = {
    "Content-Type": "application/json",
    "authorization": f"Bearer {ACCESS_TOKEN}",
    "appKey": APP_KEY,
    "appSecret": APP_SECRET,
    "tr_id": "FHKST01010100"
}
params = {
    "fid_cond_mrkt_div_code": "J",  # 주식 구분 (코스피/코스닥 구분 자동)
    "fid_input_iscd": stock_code   
}

res = requests.get(url, headers=headers, params=params)
if res.status_code == 200:
    output = res.json()['output']
    common_stock_price = int(output.get("stck_prpr", 0))
    print("보통주 현재가", common_stock_price)




'''우선주 유무 확인하고, 있으면 현재가 반환'''
preferred_name = stock_name + "우"
preferred_stock_price = 0

if (df_stock_list["한글명"] == preferred_name).any():  
    row = df_stock_list[df_stock_list['한글명'] == preferred_name]
    preferred_code = row['단축코드'].values[0]

    path = "/uapi/domestic-stock/v1/quotations/inquire-price"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST01010100"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",  
        "fid_input_iscd": preferred_code    
    }
    
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        output = res.json()['output']
        preferred_stock_price = int(output.get("stck_prpr", 0))
        print("우선주 현재가", preferred_stock_price)
    else:
        pass


'''회사 고유번호 가져오기''' #dart에서 조회할 때 고유번호가 필요함.

url = "https://opendart.fss.or.kr/api/corpCode.xml"
params = {"crtfc_key": API_KEY}

res = requests.get(url, params=params)

with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
    file_name = zf.namelist()[0]
    xml_data = zf.read(file_name)

root = ET.fromstring(xml_data)

corp_code = None

for child in root.findall("list"):
    if child.find("stock_code").text == stock_code:
        corp_code = child.find("corp_code").text
        break

print(stock_name, "corp_code:", corp_code)



'''시가총액 구하기'''
fs_url = "https://opendart.fss.or.kr/api/stockTotqySttus.json"
params_fs = {
    "crtfc_key": API_KEY,
    "corp_code": corp_code,
    "bsns_year": "2024",     # 사업연도
    "reprt_code": "11011"    # 사업보고서
}



res_fs = requests.get(fs_url, params=params_fs).json()
df = pd.DataFrame(res_fs.get("list", []))
common_issued_shares = int(df['istc_totqy'][0].replace(',', ''))


raw_value = df['istc_totqy'][1]

if raw_value not in ["0", "-"]:  
    preferred_issued_shares = int(raw_value.replace(',', ''))
else:
    preferred_issued_shares = 0

market_value = common_issued_shares * common_stock_price  + preferred_issued_shares * preferred_stock_price
print("시가총액", market_value)


'''연결 재무제표 가져오기'''


# --- 보고서 코드 정의 ---
ANNUAL_REPORT = "11011"
Q3_REPORT = "11013"
HALF_REPORT = "11012"
Q1_REPORT = "11014"


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
        
        ebit_th, ebit_fr = get_values(df_is, ['영업이익', '영업손실'], is_bs=False)


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

def main():
    """DART 공시 목록을 기반으로 EV/EBITDA를 계산합니다."""
    latest_report_info = find_latest_report(corp_code)
    if not latest_report_info: return

    latest_year = latest_report_info["bsns_year"]
    latest_code = latest_report_info["reprt_code"]
    last_year = latest_year - 1

    # EV 계산을 위한 최신 재무상태표 데이터 조회
    latest_fs_data = get_financial_data(corp_code, latest_year, latest_code)
    if not latest_fs_data: return

    ev = market_value + latest_fs_data['nci'] + latest_fs_data['total_debt'] - latest_fs_data['cash']

    if latest_code == ANNUAL_REPORT:
        ttm_data = latest_fs_data
    else:
        print(f"{stock_name}의 TTM EBITDA 계산을 위해 추가 데이터를 조회합니다.")
        # 직전 연도 연간 실적만 추가로 조회
        last_annual_data = get_financial_data(corp_code, last_year, ANNUAL_REPORT)
        if not last_annual_data:
            print("{stock_name}의 TTM 계산에 필요한 데이터가 부족합니다.")
            return
        
        # --- 💡 핵심 수정: TTM 계산식 변경 ---
        ttm_data = {}
        for key in ['ebit']:
            # latest_fs_data에 포함된 전기(frmtrm) 값을 직접 사용
            ttm_data[key] = (latest_fs_data[key] + 
                             last_annual_data[key] - 
                             latest_fs_data[f'{key}_fr']) # 작년 동기 실적
            
    depr = 0
    amor = 0
    ebitda = ttm_data['ebit'] + depr + amor
    
    print(f"\n--- {stock_name}의 EV/EBITDA 계산 결과 ---")
    print(f"EV ({stock_name}의 기업가치) = {ev:,.0f} 원")
    print(f"{stock_name}의 TTM EBITDA = {ebitda:,.0f} 원")
    
    if ebitda > 0:
        ev_ebitda_ratio = ev / ebitda
        print(f"{stock_name}의 EV/EBITDA = {ev_ebitda_ratio:.2f} 배")
    else:
        print("EBITDA가 0 또는 음수이므로 {stock_name}의 EV/EBITDA를 계산할 수 없습니다.")
if __name__ == "__main__":
    main()




