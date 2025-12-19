import pytest
from app.services.sms_parser import SMSParser, TransactionType

@pytest.fixture
def parser():
    return SMSParser()

def test_parse_hdfc_debit(parser):
    sms = "Rs 450.00 debited from A/c XX1234 on 20-12-24 to SWIGGY BANGALORE. Avl Bal: Rs 25,430.50"
    result = parser.parse(sms, "HDFCBK")
    
    assert result['parsed_successfully'] == True
    assert result['amount'] == 450.00
    assert result['transaction_type'] == TransactionType.DEBIT.value
    assert result['merchant'] == "SWIGGY BANGALORE"
    assert result['account_balance'] == 25430.50

def test_parse_icici_upi(parser):
    sms = "INR 1250 paid to Zomato via UPI. Ref No: 432198765432. Balance: Rs 18,234.00"
    result = parser.parse(sms, "ICICI")
    
    assert result['amount'] == 1250.0
    assert result['merchant'] == "Zomato"
    assert result['transaction_type'] == TransactionType.DEBIT.value

def test_parse_credit(parser):
    sms = "Amt Rs 2500.00 credited to A/c XX9012 on 19-12-2024 from SALARY ACCOUNT. Total Bal Rs 52000.00"
    result = parser.parse(sms, "AXIS")
    
    assert result['amount'] == 2500.00
    assert result['transaction_type'] == TransactionType.CREDIT.value
    assert result['account_balance'] == 52000.00

def test_is_transaction_sms(parser):
    transaction_sms = "Rs 450 debited from account"
    non_transaction_sms = "Your OTP is 123456"
    
    assert parser.is_transaction_sms(transaction_sms) == True
    assert parser.is_transaction_sms(non_transaction_sms) == False

def test_parse_phonepe(parser):
    sms = "Paid Rs.350 to UBER INDIA via PhonePe UPI on 20-12-24. UPI Ref: 434567890123"
    result = parser.parse(sms, "PHONEPE")
    
    assert result['amount'] == 350.0
    assert result['merchant'] == "UBER INDIA"