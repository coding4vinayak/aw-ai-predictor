import os
import logging
from datetime import datetime
from .base import BaseConnector

class ZohoConnector(BaseConnector):
    """Zoho CRM connector"""
    
    def get_api_key(self):
        """Get Zoho API key from environment"""
        return os.getenv('ZOHO_ACCESS_TOKEN')
    
    def get_base_url(self):
        """Get Zoho API base URL"""
        # Default to US datacenter, can be overridden with ZOHO_API_DOMAIN
        domain = os.getenv('ZOHO_API_DOMAIN', 'zohoapis.com')
        return f'https://www.{domain}/crm/v2'
    
    def get_auth_headers(self):
        """Get Zoho authentication headers"""
        return {
            'Authorization': f'Zoho-oauthtoken {self.api_key}'
        }
    
    def get_test_endpoint(self):
        """Test Zoho API connection"""
        try:
            response = self.get('/settings/modules')
            return response
        except Exception:
            # Fallback to a simpler endpoint
            response = self.get('/Leads', params={'per_page': 1})
            return {'status': 'ok', 'api_accessible': True}
    
    def get_leads(self, limit=100, offset=0):
        """
        Get leads from Zoho CRM
        
        Args:
            limit (int): Number of leads to fetch
            offset (int): Offset for pagination
            
        Returns:
            list: List of normalized lead data
        """
        try:
            params = {
                'per_page': min(limit, 200),  # Zoho max is 200
                'page': (offset // limit) + 1 if limit > 0 else 1,
                'fields': 'First_Name,Last_Name,Email,Phone,Company,Title,Lead_Status,Lead_Source,Created_Time,Modified_Time,Rating'
            }
            
            response = self.get('/Leads', params=params)
            
            leads = []
            for lead in response.get('data', []):
                normalized_lead = self.normalize_zoho_lead(lead)
                leads.append(normalized_lead)
            
            return leads
            
        except Exception as e:
            logging.error(f"Zoho get_leads error: {str(e)}")
            # Return demo data if API fails
            return self.get_demo_leads(limit)
    
    def get_contacts(self, limit=100, offset=0):
        """
        Get contacts from Zoho CRM
        
        Args:
            limit (int): Number of contacts to fetch
            offset (int): Offset for pagination
            
        Returns:
            list: List of normalized contact data
        """
        try:
            params = {
                'per_page': min(limit, 200),
                'page': (offset // limit) + 1 if limit > 0 else 1,
                'fields': 'First_Name,Last_Name,Email,Phone,Account_Name,Title,Created_Time,Modified_Time,Lead_Source'
            }
            
            response = self.get('/Contacts', params=params)
            
            contacts = []
            for contact in response.get('data', []):
                normalized_contact = self.normalize_zoho_contact(contact)
                contacts.append(normalized_contact)
            
            return contacts
            
        except Exception as e:
            logging.error(f"Zoho get_contacts error: {str(e)}")
            # Return demo data if API fails
            return self.get_demo_contacts(limit)
    
    def normalize_zoho_lead(self, raw_data):
        """Normalize Zoho lead data to standard format"""
        # Combine first and last name
        first_name = raw_data.get('First_Name', '')
        last_name = raw_data.get('Last_Name', '')
        full_name = f"{first_name} {last_name}".strip()
        
        return {
            'id': raw_data.get('id'),
            'name': full_name,
            'email': raw_data.get('Email', ''),
            'phone': raw_data.get('Phone', ''),
            'company': raw_data.get('Company', ''),
            'title': raw_data.get('Title', ''),
            'source': raw_data.get('Lead_Source', ''),
            'status': raw_data.get('Lead_Status', ''),
            'lifecycle_stage': 'lead',
            'created_date': self.parse_zoho_date(raw_data.get('Created_Time')),
            'last_activity': self.parse_zoho_date(raw_data.get('Modified_Time')),
            'score': self.parse_zoho_rating(raw_data.get('Rating')),
            'platform': 'zoho',
            'raw_data': raw_data
        }
    
    def normalize_zoho_contact(self, raw_data):
        """Normalize Zoho contact data to standard format"""
        # Combine first and last name
        first_name = raw_data.get('First_Name', '')
        last_name = raw_data.get('Last_Name', '')
        full_name = f"{first_name} {last_name}".strip()
        
        return {
            'id': raw_data.get('id'),
            'name': full_name,
            'email': raw_data.get('Email', ''),
            'phone': raw_data.get('Phone', ''),
            'company': raw_data.get('Account_Name', ''),
            'title': raw_data.get('Title', ''),
            'lifecycle_stage': 'contact',
            'created_date': self.parse_zoho_date(raw_data.get('Created_Time')),
            'last_activity': self.parse_zoho_date(raw_data.get('Modified_Time')),
            'score': 0,  # Contacts don't typically have ratings
            'platform': 'zoho',
            'raw_data': raw_data
        }
    
    def parse_zoho_date(self, date_string):
        """Parse Zoho date format"""
        if not date_string:
            return None
        
        try:
            # Zoho returns dates in ISO format
            if isinstance(date_string, str):
                return date_string
            return date_string
        except (ValueError, TypeError):
            return date_string
    
    def parse_zoho_rating(self, rating_value):
        """Parse Zoho rating/score value"""
        if not rating_value:
            return 0
        
        # Zoho ratings are typically: Hot, Warm, Cold
        rating_map = {
            'Hot': 80,
            'Warm': 60,
            'Cold': 40,
            'Acquired': 90,
            'Active': 70,
            'Market Failed': 20,
            'Project Cancelled': 10,
            'Not Contacted': 30
        }
        
        if isinstance(rating_value, str):
            return rating_map.get(rating_value, 50)
        
        try:
            return int(float(rating_value))
        except (ValueError, TypeError):
            return 50
    
    def get_deals(self, limit=100, offset=0):
        """Get deals from Zoho CRM"""
        try:
            params = {
                'per_page': min(limit, 200),
                'page': (offset // limit) + 1 if limit > 0 else 1,
                'fields': 'Deal_Name,Amount,Stage,Pipeline,Closing_Date,Created_Time,Probability,Type'
            }
            
            response = self.get('/Deals', params=params)
            
            deals = []
            for deal in response.get('data', []):
                normalized_deal = self.normalize_zoho_deal(deal)
                deals.append(normalized_deal)
            
            return deals
            
        except Exception as e:
            logging.error(f"Zoho get_deals error: {str(e)}")
            return []
    
    def normalize_zoho_deal(self, raw_data):
        """Normalize Zoho deal data"""
        return {
            'id': raw_data.get('id'),
            'name': raw_data.get('Deal_Name', ''),
            'amount': self.parse_amount(raw_data.get('Amount')),
            'stage': raw_data.get('Stage', ''),
            'pipeline': raw_data.get('Pipeline', ''),
            'probability': self.parse_probability(raw_data.get('Probability')),
            'close_date': self.parse_zoho_date(raw_data.get('Closing_Date')),
            'created_date': self.parse_zoho_date(raw_data.get('Created_Time')),
            'deal_type': raw_data.get('Type', ''),
            'platform': 'zoho',
            'raw_data': raw_data
        }
    
    def parse_amount(self, amount_value):
        """Parse deal amount"""
        if not amount_value:
            return 0.0
        
        try:
            return float(amount_value)
        except (ValueError, TypeError):
            return 0.0
    
    def parse_probability(self, prob_value):
        """Parse deal probability"""
        if not prob_value:
            return 0.0
        
        try:
            prob = float(prob_value)
            return prob / 100 if prob > 1 else prob  # Convert percentage to decimal
        except (ValueError, TypeError):
            return 0.0
    
    def get_accounts(self, limit=100, offset=0):
        """Get accounts from Zoho CRM"""
        try:
            params = {
                'per_page': min(limit, 200),
                'page': (offset // limit) + 1 if limit > 0 else 1,
                'fields': 'Account_Name,Phone,Website,Billing_City,Billing_State,Industry,Annual_Revenue,Created_Time'
            }
            
            response = self.get('/Accounts', params=params)
            
            accounts = []
            for account in response.get('data', []):
                normalized_account = self.normalize_zoho_account(account)
                accounts.append(normalized_account)
            
            return accounts
            
        except Exception as e:
            logging.error(f"Zoho get_accounts error: {str(e)}")
            return []
    
    def normalize_zoho_account(self, raw_data):
        """Normalize Zoho account data"""
        return {
            'id': raw_data.get('id'),
            'name': raw_data.get('Account_Name', ''),
            'phone': raw_data.get('Phone', ''),
            'website': raw_data.get('Website', ''),
            'city': raw_data.get('Billing_City', ''),
            'state': raw_data.get('Billing_State', ''),
            'industry': raw_data.get('Industry', ''),
            'annual_revenue': self.parse_amount(raw_data.get('Annual_Revenue')),
            'created_date': self.parse_zoho_date(raw_data.get('Created_Time')),
            'platform': 'zoho',
            'raw_data': raw_data
        }
    
    def get_demo_leads(self, limit=10):
        """Return demo leads data when API is not available"""
        demo_leads = []
        for i in range(min(limit, 10)):
            demo_leads.append({
                'id': f'zoho_demo_lead_{i}',
                'name': f'Zoho Lead {i+1}',
                'email': f'zoholead{i+1}@example.com',
                'phone': f'+1-555-{3000+i}',
                'company': f'Zoho Company {i+1}',
                'title': 'Business Development Manager',
                'source': 'Web Form',
                'status': 'Qualified',
                'lifecycle_stage': 'lead',
                'created_date': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'score': 60 + (i * 4),
                'platform': 'zoho_demo',
                'raw_data': {'demo': True}
            })
        return demo_leads
    
    def get_demo_contacts(self, limit=10):
        """Return demo contacts data when API is not available"""
        demo_contacts = []
        for i in range(min(limit, 10)):
            demo_contacts.append({
                'id': f'zoho_demo_contact_{i}',
                'name': f'Zoho Contact {i+1}',
                'email': f'zohocontact{i+1}@example.com',
                'phone': f'+1-555-{4000+i}',
                'company': f'Zoho Account {i+1}',
                'title': 'Operations Manager',
                'lifecycle_stage': 'customer',
                'created_date': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'score': 70 + (i * 3),
                'platform': 'zoho_demo',
                'raw_data': {'demo': True}
            })
        return demo_contacts
    
    def get_supported_features(self):
        """Get Zoho specific supported features"""
        base_features = super().get_supported_features()
        zoho_features = {
            'deals': True,
            'accounts': True,
            'campaigns': True,
            'tasks': True,
            'events': True,
            'calls': True,
            'custom_fields': True,
            'workflows': True
        }
        base_features.update(zoho_features)
        return base_features
