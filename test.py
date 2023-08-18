import pytest
from app import app, home
import json
def test_mytest():
    assert True

def test_home():
    assert (home() == "Application flask")

test_data = [
    ("100139", 0),
    ("186806", 1),
    ("999", -1)
]
@pytest.mark.parametrize("SK_ID_CURR, expected", test_data)
def test_score(SK_ID_CURR, expected):
    response = app.test_client().get(f'/predict/?id_client={SK_ID_CURR}')
    res = json.loads(response.data.decode('utf-8')).get("score")
    assert (res == expected)

test_global = [
    ("2", ['PREV_NAME_CONTRACT_STATUS_Refused_MEAN', 'EXT_SOURCE_1'])
]
@pytest.mark.parametrize("n, expected", test_global)
def test_global(n, expected):
    response = app.test_client().get(f'/features/?n={n}')
    res = json.loads(response.data.decode('utf-8')).get("n")
    assert (res == expected)


test_id_data = [
    ("100139", {'ACTIVE_DAYS_CREDIT_MAX': {'0': -1252.0},
 'ACTIVE_DAYS_CREDIT_MEAN': {'0': -1252.0},
 'ACTIVE_MONTHS_BALANCE_SIZE_MEAN': {'0': 8.0},
 'BURO_CREDIT_ACTIVE_Active_MEAN': {'0': 0.5},
 'BURO_CREDIT_ACTIVE_Closed_MEAN': {'0': 0.5},
 'BURO_DAYS_CREDIT_MEAN': {'0': -733.5},
 'BURO_DAYS_CREDIT_MIN': {'0': -1252.0},
 'BURO_DAYS_CREDIT_UPDATE_MEAN': {'0': -20.0},
 'BURO_MONTHS_BALANCE_SIZE_MEAN': {'0': 8.0},
 'BURO_STATUS_1_MEAN_MEAN': {'0': 0.0},
 'CC_AMT_BALANCE_MAX': {'0': 298383.57},
 'CC_AMT_BALANCE_MEAN': {'0': 192715.0356},
 'CC_AMT_BALANCE_MIN': {'0': 1224.81},
 'CC_CNT_DRAWINGS_ATM_CURRENT_MAX': {'0': 7.0},
 'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN': {'0': 1.0666666667},
 'CC_CNT_DRAWINGS_ATM_CURRENT_VAR': {'0': 2.7657657658},
 'CC_CNT_DRAWINGS_CURRENT_MAX': {'0': 7.0},
 'CC_CNT_DRAWINGS_CURRENT_MEAN': {'0': 1.2666666667},
 'CC_CNT_DRAWINGS_CURRENT_VAR': {'0': 2.981981982},
 'CC_MONTHS_BALANCE_MIN': {'0': -75.0},
 'DAYS_BIRTH': {'0': -13286},
 'DAYS_EMPLOYED': {'0': -2305.0},
 'DAYS_EMPLOYED_PERC': {'0': 0.1734908927},
 'EXT_SOURCE_1': {'0': 0.5677754907},
 'EXT_SOURCE_2': {'0': 0.6014083651},
 'EXT_SOURCE_3': {'0': 0.0963186928},
 'PREV_CODE_REJECT_REASON_XAP_MEAN': {'0': 0.6363636364},
 'PREV_NAME_CONTRACT_STATUS_Approved_MEAN': {'0': 0.5454545455},
 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN': {'0': 0.2727272727},
 'REFUSED_DAYS_DECISION_MAX': {'0': -1868.0},
 'SK_ID_CURR': {'0': 100139}}),
]
@pytest.mark.parametrize("id, expected", test_id_data)
def test_id_data(id, expected):
    response = app.test_client().get(f'/id_data/?id={id}')
    res = json.loads(response.data.decode('utf-8')).get("data")
    assert (res == expected)