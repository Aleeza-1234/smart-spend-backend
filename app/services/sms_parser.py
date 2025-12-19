"""
SMS Parser for Indian Bank Transaction Messages
Supports major Indian banks: SBI, HDFC, ICICI, Axis, Kotak, etc.
"""

import re
from typing import Optional, Dict, Tuple
from datetime import datetime
import spacy
from enum import Enum

class TransactionType(Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"
    UNKNOWN = "UNKNOWN"

class BankType(Enum):
    UBI = "Union Bank of India"
    SBI = "State Bank of India"
    HDFC = "HDFC Bank"
    ICICI = "ICICI Bank"
    AXIS = "Axis Bank"
    KOTAK = "Kotak Mahindra Bank"
    PNB = "Punjab National Bank"
    BOB = "Bank of Baroda"
    PAYTM = "Paytm Payments Bank"
    PHONEPE = "PhonePe"
    GPAY = "Google Pay"
    UNKNOWN = "Unknown Bank"

class SMSParser:
    """
    Parses transaction SMS from Indian banks and UPI apps
    """
    
    def __init__(self):
        # Load spaCy model for NER (optional, for merchant extraction)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spaCy model not loaded. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Common patterns for different banks
        self.patterns = self._compile_patterns()
        
        # Currency symbols and keywords
        self.currency_keywords = ['rs', 'rs.', 'inr', '₹', 'rupees']
        
        # Transaction keywords
        self.debit_keywords = ['debited', 'debit', 'spent', 'paid', 'withdrawn', 'sent', 'transfer to']
        self.credit_keywords = ['credited', 'credit', 'received', 'deposited', 'refund']
        
        # UPI keywords
        self.upi_keywords = ['upi', 'vpa', '@']
    
    def _compile_patterns(self) -> Dict[str, list]:
        """
        Compile regex patterns for different SMS formats
        """
        patterns = {
            # Amount patterns
            'amount': [
                r'(?:rs\.?|inr|₹)\s*(\d+(?:,\d+)*(?:\.\d{2})?)',  # Rs 1,234.56
                r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:rs\.?|inr|₹)',  # 1,234.56 Rs
                r'(?:amount|amt)[\s:]*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',  # Amount: 1234
            ],
            
            # Account balance patterns
            'balance': [
                r'(?:avbl?\.?\s*bal|balance|bal)[\s:]*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'(?:bal|balance)[\s:]+(?:is\s+)?(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            ],
            
            # Merchant/Beneficiary patterns
            'merchant': [
                r'(?:to|at|from)\s+([A-Z][A-Za-z0-9\s&\-\.]{2,30}?)(?:\s+on|\s+via|\s+using|\.|$)',
                r'(?:merchant|beneficiary)[\s:]+([A-Za-z0-9\s&\-\.]{3,30})',
                r'(?:paid to|sent to|received from)\s+([A-Za-z0-9\s&\-\.]{3,30})',
            ],
            
            # UPI ID patterns
            'upi_id': [
                r'([a-zA-Z0-9\.\-_]+@[a-zA-Z]+)',  # user@bank
                r'vpa[\s:]+([a-zA-Z0-9\.\-_]+@[a-zA-Z]+)',
            ],
            
            # Card patterns
            'card': [
                r'card\s+(?:no\.?\s+)?[xX*]+(\d{4})',  # Card XX1234
                r'[xX*]+(\d{4})',  # XX1234
            ],
            
            # Account patterns
            'account': [
                r'(?:a\/c|account|acc)[\s:]*[xX*]*(\d{4,})',
            ],
            
            # Date/Time patterns
            'datetime': [
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # DD/MM/YYYY
                r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})',  # DD MMM YYYY
            ],
            
            # Reference number patterns
            'ref_number': [
                r'(?:ref|reference|txn|transaction)[\s:]*([A-Z0-9]+)',
                r'utr[\s:]*([A-Z0-9]+)',
            ],
        }
        
        # Compile all patterns
        compiled = {}
        for key, pattern_list in patterns.items():
            compiled[key] = [re.compile(p, re.IGNORECASE) for p in pattern_list]
        
        return compiled
    
    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract amount from SMS text"""
        for pattern in self.patterns['amount']:
            match = pattern.search(text)
            if match:
                amount_str = match.group(1)
                # Remove commas and convert to float
                amount_str = amount_str.replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return None
    
    def _extract_balance(self, text: str) -> Optional[float]:
        """Extract account balance from SMS"""
        for pattern in self.patterns['balance']:
            match = pattern.search(text)
            if match:
                balance_str = match.group(1)
                balance_str = balance_str.replace(',', '')
                try:
                    return float(balance_str)
                except ValueError:
                    continue
        return None
    
    def _extract_merchant(self, text: str) -> Optional[str]:
        """Extract merchant name from SMS"""
        for pattern in self.patterns['merchant']:
            match = pattern.search(text)
            if match:
                merchant = match.group(1).strip()
                # Clean up merchant name
                merchant = re.sub(r'\s+', ' ', merchant)
                merchant = merchant.rstrip('.')
                if len(merchant) > 3:
                    return merchant
        
        # Try using spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT']:
                    return ent.text
        
        return None
    
    def _extract_upi_id(self, text: str) -> Optional[str]:
        """Extract UPI ID from SMS"""
        for pattern in self.patterns['upi_id']:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None
    
    def _extract_card_number(self, text: str) -> Optional[str]:
        """Extract last 4 digits of card"""
        for pattern in self.patterns['card']:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None
    
    def _extract_ref_number(self, text: str) -> Optional[str]:
        """Extract transaction reference number"""
        for pattern in self.patterns['ref_number']:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None
    
    def _detect_transaction_type(self, text: str) -> TransactionType:
        """Detect if transaction is debit or credit"""
        text_lower = text.lower()
        
        # Check for debit keywords
        for keyword in self.debit_keywords:
            if keyword in text_lower:
                return TransactionType.DEBIT
        
        # Check for credit keywords
        for keyword in self.credit_keywords:
            if keyword in text_lower:
                return TransactionType.CREDIT
        
        return TransactionType.UNKNOWN
    
    def _detect_bank(self, text: str, sender: str = None) -> BankType:
        """Detect bank from SMS content or sender ID"""
        text_lower = text.lower()
        sender_lower = sender.lower() if sender else ""
        
        # Bank detection patterns
        bank_patterns = {
            BankType.UBI: ['ubi', 'union bank', 'unionbank', 'union bank of india'],
            BankType.SBI: ['sbi', 'statebank', 'state bank'],
            BankType.HDFC: ['hdfc', 'hdfcbank'],
            BankType.ICICI: ['icici', 'icicibank'],
            BankType.AXIS: ['axis', 'axisbank'],
            BankType.KOTAK: ['kotak', 'kotakbank'],
            BankType.PNB: ['pnb', 'punjab national'],
            BankType.BOB: ['bob', 'bankofbaroda', 'baroda'],
            BankType.PAYTM: ['paytm', 'paytmbank'],
            BankType.PHONEPE: ['phonepe', 'phone pe'],
            BankType.GPAY: ['googlepay', 'google pay', 'gpay'],
        }
        
        for bank, keywords in bank_patterns.items():
            for keyword in keywords:
                if keyword in text_lower or keyword in sender_lower:
                    return bank
        
        return BankType.UNKNOWN
    
    def parse(self, sms_text: str, sender: str = None) -> Dict:
        """
        Parse SMS and extract transaction information
        
        Args:
            sms_text: The SMS message text
            sender: SMS sender ID (optional)
        
        Returns:
            Dictionary with parsed information
        """
        result = {
            'raw_sms': sms_text,
            'sender': sender,
            'amount': None,
            'merchant': None,
            'transaction_type': None,
            'account_balance': None,
            'upi_id': None,
            'card_last4': None,
            'ref_number': None,
            'bank_name': None,
            'parsed_successfully': False,
            'confidence': 0.0
        }
        
        # Detect bank
        bank = self._detect_bank(sms_text, sender)
        result['bank_name'] = bank.value
        
        # Extract amount (most critical)
        amount = self._extract_amount(sms_text)
        if amount:
            result['amount'] = amount
            result['parsed_successfully'] = True
            result['confidence'] += 0.5
        
        # Extract transaction type
        trans_type = self._detect_transaction_type(sms_text)
        if trans_type != TransactionType.UNKNOWN:
            result['transaction_type'] = trans_type.value
            result['confidence'] += 0.2
        
        # Extract merchant
        merchant = self._extract_merchant(sms_text)
        if merchant:
            result['merchant'] = merchant
            result['confidence'] += 0.1
        
        # Extract balance
        balance = self._extract_balance(sms_text)
        if balance:
            result['account_balance'] = balance
            result['confidence'] += 0.1
        
        # Extract UPI ID
        upi_id = self._extract_upi_id(sms_text)
        if upi_id:
            result['upi_id'] = upi_id
            result['confidence'] += 0.05
        
        # Extract card number
        card = self._extract_card_number(sms_text)
        if card:
            result['card_last4'] = card
            result['confidence'] += 0.05
        
        # Extract reference number
        ref_num = self._extract_ref_number(sms_text)
        if ref_num:
            result['ref_number'] = ref_num
        
        # Cap confidence at 1.0
        result['confidence'] = min(result['confidence'], 1.0)
        
        return result
    
    def is_transaction_sms(self, sms_text: str, sender: str = None) -> bool:
        """
        Check if SMS is a transaction message
        
        Returns:
            True if SMS appears to be a transaction notification
        """
        text_lower = sms_text.lower()
        
        # Check for transaction keywords
        transaction_keywords = [
            'debited', 'credited', 'paid', 'received', 'withdrawn',
            'transaction', 'txn', 'upi', 'transfer', 'balance',
            'account', 'card', 'payment'
        ]
        
        has_keyword = any(keyword in text_lower for keyword in transaction_keywords)
        has_amount = self._extract_amount(sms_text) is not None
        
        return has_keyword and has_amount


# Example usage and testing
if __name__ == "__main__":
    parser = SMSParser()
    
    # Test SMS examples from different banks
    test_messages = [
        {
            'text': 'A/c *6617 Debited for Rs:250.00 on 10-12-2025 23:07:47 by Mob Bk ref no 042166517738 Avl Bal Rs:5989.97.If not you, Call 1800222243 -Union Bank of India',
            'sender': 'JK-UNIONB-S'
        },
        {
            'text': 'A/c *6617 Debited for Rs:250.00 on 10-12-2025 23:07:47 by Mob Bk ref no 042166517738 Avl Bal Rs:5989.97.If not you, Call 1800222243 -Union Bank of India',
            'sender': 'JD-UNIONB-S'
        },
        {
            'text': 'A/c *6617 Debited for Rs:250.00 on 10-12-2025 23:07:47 by Mob Bk ref no 042166517738 Avl Bal Rs:5989.97.If not you, Call 1800222243 -Union Bank of India',
            'sender': 'JS-UNIONB-S'
        },
        {
            'text': 'Rs 450.00 debited from A/c XX1234 on 20-12-24 to SWIGGY BANGALORE. Avl Bal: Rs 25,430.50',
            'sender': 'HDFCBK'
        },
        {
            'text': 'INR 1250 paid to Zomato via UPI. Ref No: 432198765432. Balance: Rs 18,234.00',
            'sender': 'ICICI'
        },
        {
            'text': 'Your A/c XX5678 is debited with Rs.850.00 on 20Dec24 Info: UPI/432156789012/swiggy@paytm. Avl Bal Rs.45678.90',
            'sender': 'SBIINB'
        },
        {
            'text': 'Amt Rs 2500.00 credited to A/c XX9012 on 19-12-2024 from SALARY ACCOUNT. Total Bal Rs 52000.00',
            'sender': 'AXIS'
        },
        {
            'text': 'Paid Rs.350 to UBER INDIA via PhonePe UPI on 20-12-24. UPI Ref: 434567890123',
            'sender': 'PHONEPE'
        },
        {
            'text': 'Rs 175 paid via Google Pay to user@oksbi. UPI transaction ID: 445678901234',
            'sender': 'GPAY'
        },
        {
            'text': 'Card XX3456 used for Rs 1899.00 at AMAZON.IN on 20-12-2024. Available limit: Rs 48101.00',
            'sender': 'HDFCBK'
        }
    ]
    
    print("="*80)
    print("SMS PARSER TEST")
    print("="*80)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n--- Test {i} ---")
        print(f"Sender: {msg['sender']}")
        print(f"SMS: {msg['text'][:80]}...")
        
        result = parser.parse(msg['text'], msg['sender'])
        
        print(f"\nParsed Results:")
        print(f"  ✓ Amount: ₹{result['amount']}")
        print(f"  ✓ Type: {result['transaction_type']}")
        print(f"  ✓ Merchant: {result['merchant']}")
        print(f"  ✓ Balance: ₹{result['account_balance']}")
        print(f"  ✓ UPI ID: {result['upi_id']}")
        print(f"  ✓ Bank: {result['bank_name']}")
        print(f"  ✓ Success: {result['parsed_successfully']}")
        print(f"  ✓ Confidence: {result['confidence']:.2f}")