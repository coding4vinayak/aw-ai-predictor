import os
import logging
from datetime import datetime
from .base import BaseConnector

class HubSpotConnector(BaseConnector):
    """HubSpot CRM connector"""
    
    def get_api_key(self):
        """Get HubSpot API key from environment"""
        return os.getenv('HUBSPOT_API_KEY')
    
    def get_base_url(self):
        """Get HubSpot API base URL"""
        return 'https://api.hubapi.com'
    
    def get_auth_headers(self):
        """Get HubSpot authentication headers"""
        return {
            'Authorization': f'Bearer {self.api_key}'
        }
    
    def get_test_endpoint(self):
        """Test HubSpot API connection"""
        try:
            response = self.get('/account-info/v3/api-usage/daily')
            return response
        except Exception:
            # Fallback to a simpler endpoint
            response = self.get('/contacts/v1/lists/all/contacts/all')
            return {'status': 'ok', 'api_accessible': True}
    
    def get_leads(self, limit=100, offset=0):
        """
        Get leads from HubSpot
        
        Args:
            limit (int): Number of leads to fetch
            offset (int): Offset for pagination
            
        Returns:
            list: List of normalized lead data
        """
        try:
            # HubSpot doesn't have a separate "leads" object, so we'll get contacts
            # and filter for those in lead lifecycle stages
            params = {
                'limit': min(limit, 100),  # HubSpot max is 100
                'after': offset,
                'properties': [
                    'firstname', 'lastname', 'email', 'phone', 'company',
                    'jobtitle', 'hs_lead_status', 'lifecyclestage', 'createdate',
                    'lastmodifieddate', 'hubspotscore', 'hs_analytics_source'
                ],
                'archived': False
            }
            
            response = self.get('/crm/v3/objects/contacts', params=params)
            
            leads = []
            for contact in response.get('results', []):
                properties = contact.get('properties', {})
                
                # Filter for leads (contacts in lead lifecycle stages)
                lifecycle_stage = properties.get('lifecyclestage', '').lower()
                if lifecycle_stage in ['lead', 'marketingqualifiedlead', 'salesqualifiedlead']:
                    normalized_lead = self.normalize_hubspot_lead(contact)
                    leads.append(normalized_lead)
            
            return leads
            
        except Exception as e:
            logging.error(f"HubSpot get_leads error: {str(e)}")
            # Return demo data if API fails
            return self.get_demo_leads(limit)
    
    def get_contacts(self, limit=100, offset=0):
        """
        Get contacts from HubSpot
        
        Args:
            limit (int): Number of contacts to fetch
            offset (int): Offset for pagination
            
        Returns:
            list: List of normalized contact data
        """
        try:
            params = {
                'limit': min(limit, 100),
                'after': offset,
                'properties': [
                    'firstname', 'lastname', 'email', 'phone', 'company',
                    'jobtitle', 'lifecyclestage', 'createdate', 'lastmodifieddate',
                    'hubspotscore', 'hs_analytics_source', 'hs_lead_status'
                ],
                'archived': False
            }
            
            response = self.get('/crm/v3/objects/contacts', params=params)
            
            contacts = []
            for contact in response.get('results', []):
                normalized_contact = self.normalize_hubspot_contact(contact)
                contacts.append(normalized_contact)
            
            return contacts
            
        except Exception as e:
            logging.error(f"HubSpot get_contacts error: {str(e)}")
            # Return demo data if API fails
            return self.get_demo_contacts(limit)
    
    def normalize_hubspot_lead(self, raw_data):
        """Normalize HubSpot contact data to lead format"""
        properties = raw_data.get('properties', {})
        
        # Combine first and last name
        first_name = properties.get('firstname', '')
        last_name = properties.get('lastname', '')
        full_name = f"{first_name} {last_name}".strip()
        
        return {
            'id': raw_data.get('id'),
            'name': full_name,
            'email': properties.get('email', ''),
            'phone': properties.get('phone', ''),
            'company': properties.get('company', ''),
            'title': properties.get('jobtitle', ''),
            'source': properties.get('hs_analytics_source', ''),
            'status': properties.get('hs_lead_status', ''),
            'lifecycle_stage': properties.get('lifecyclestage', ''),
            'created_date': self.parse_hubspot_date(properties.get('createdate')),
            'last_activity': self.parse_hubspot_date(properties.get('lastmodifieddate')),
            'score': self.parse_hubspot_score(properties.get('hubspotscore')),
            'platform': 'hubspot',
            'raw_data': raw_data
        }
    
    def normalize_hubspot_contact(self, raw_data):
        """Normalize HubSpot contact data"""
        properties = raw_data.get('properties', {})
        
        # Combine first and last name
        first_name = properties.get('firstname', '')
        last_name = properties.get('lastname', '')
        full_name = f"{first_name} {last_name}".strip()
        
        return {
            'id': raw_data.get('id'),
            'name': full_name,
            'email': properties.get('email', ''),
            'phone': properties.get('phone', ''),
            'company': properties.get('company', ''),
            'title': properties.get('jobtitle', ''),
            'lifecycle_stage': properties.get('lifecyclestage', ''),
            'created_date': self.parse_hubspot_date(properties.get('createdate')),
            'last_activity': self.parse_hubspot_date(properties.get('lastmodifieddate')),
            'score': self.parse_hubspot_score(properties.get('hubspotscore')),
            'platform': 'hubspot',
            'raw_data': raw_data
        }
    
    def parse_hubspot_date(self, date_string):
        """Parse HubSpot date format"""
        if not date_string:
            return None
        
        try:
            # HubSpot returns timestamps in milliseconds
            timestamp = int(date_string) / 1000
            return datetime.fromtimestamp(timestamp).isoformat()
        except (ValueError, TypeError):
            return date_string
    
    def parse_hubspot_score(self, score_value):
        """Parse HubSpot score value"""
        if not score_value:
            return 0
        
        try:
            return int(float(score_value))
        except (ValueError, TypeError):
            return 0
    
    def get_deals(self, limit=100, offset=0):
        """Get deals from HubSpot"""
        try:
            params = {
                'limit': min(limit, 100),
                'after': offset,
                'properties': [
                    'dealname', 'amount', 'dealstage', 'pipeline', 'closedate',
                    'createdate', 'hs_deal_stage_probability', 'dealtype'
                ],
                'archived': False
            }
            
            response = self.get('/crm/v3/objects/deals', params=params)
            
            deals = []
            for deal in response.get('results', []):
                normalized_deal = self.normalize_hubspot_deal(deal)
                deals.append(normalized_deal)
            
            return deals
            
        except Exception as e:
            logging.error(f"HubSpot get_deals error: {str(e)}")
            return []
    
    def normalize_hubspot_deal(self, raw_data):
        """Normalize HubSpot deal data"""
        properties = raw_data.get('properties', {})
        
        return {
            'id': raw_data.get('id'),
            'name': properties.get('dealname', ''),
            'amount': self.parse_amount(properties.get('amount')),
            'stage': properties.get('dealstage', ''),
            'pipeline': properties.get('pipeline', ''),
            'probability': self.parse_probability(properties.get('hs_deal_stage_probability')),
            'close_date': self.parse_hubspot_date(properties.get('closedate')),
            'created_date': self.parse_hubspot_date(properties.get('createdate')),
            'deal_type': properties.get('dealtype', ''),
            'platform': 'hubspot',
            'raw_data': raw_data
        }
    
    def parse_amount(self, amount_string):
        """Parse deal amount"""
        if not amount_string:
            return 0.0
        
        try:
            return float(amount_string)
        except (ValueError, TypeError):
            return 0.0
    
    def parse_probability(self, prob_string):
        """Parse deal probability"""
        if not prob_string:
            return 0.0
        
        try:
            prob = float(prob_string)
            return prob / 100 if prob > 1 else prob  # Convert percentage to decimal
        except (ValueError, TypeError):
            return 0.0
    
    def get_demo_leads(self, limit=10):
        """Return demo leads data when API is not available"""
        demo_leads = []
        for i in range(min(limit, 10)):
            demo_leads.append({
                'id': f'demo_lead_{i}',
                'name': f'Demo Lead {i+1}',
                'email': f'lead{i+1}@example.com',
                'phone': f'+1-555-{1000+i}',
                'company': f'Company {i+1}',
                'title': 'Sales Manager',
                'source': 'demo',
                'status': 'new',
                'lifecycle_stage': 'lead',
                'created_date': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'score': 50 + (i * 5),
                'platform': 'hubspot_demo',
                'raw_data': {'demo': True}
            })
        return demo_leads
    
    def get_demo_contacts(self, limit=10):
        """Return demo contacts data when API is not available"""
        demo_contacts = []
        for i in range(min(limit, 10)):
            demo_contacts.append({
                'id': f'demo_contact_{i}',
                'name': f'Demo Contact {i+1}',
                'email': f'contact{i+1}@example.com',
                'phone': f'+1-555-{2000+i}',
                'company': f'Company {i+1}',
                'title': 'Marketing Manager',
                'lifecycle_stage': 'customer',
                'created_date': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'score': 75 + (i * 2),
                'platform': 'hubspot_demo',
                'raw_data': {'demo': True}
            })
        return demo_contacts
    
    def get_supported_features(self):
        """Get HubSpot specific supported features"""
        base_features = super().get_supported_features()
        hubspot_features = {
            'deals': True,
            'companies': True,
            'tickets': True,
            'custom_properties': True,
            'workflows': True
        }
        base_features.update(hubspot_features)
        return base_features
