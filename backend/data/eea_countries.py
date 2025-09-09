"""
EEA Countries Classification
Provides lists and utilities for European Economic Area country classification
"""
from typing import List, Dict, Set


class EEACountries:
    """
    European Economic Area countries classification
    """
    
    # EEA countries (EU + Iceland, Liechtenstein, Norway)
    EEA_COUNTRIES = {
        # EU countries
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
        'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
        'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
        'Slovenia', 'Spain', 'Sweden',
        # EEA but not EU
        'Iceland', 'Liechtenstein', 'Norway'
    }
    
    # Alternative names and common variations
    COUNTRY_ALIASES = {
        'Czech Republic': ['Czechia', 'Czech Rep'],
        'United Kingdom': ['UK', 'Great Britain', 'Britain'],
        'United States': ['USA', 'US', 'America'],
        'United Arab Emirates': ['UAE'],
        'South Korea': ['Korea', 'South Korea'],
        'North Korea': ['Korea DPR', 'DPRK'],
        'Russia': ['Russian Federation'],
        'China': ['People\'s Republic of China', 'PRC'],
        'Taiwan': ['Republic of China', 'ROC'],
        'Hong Kong': ['Hong Kong SAR'],
        'Macau': ['Macao', 'Macau SAR'],
        'Bosnia and Herzegovina': ['Bosnia', 'BiH'],
        'North Macedonia': ['Macedonia'],
        'Moldova': ['Republic of Moldova'],
        'Ukraine': ['Ukrainian SSR'],
        'Belarus': ['Belorussia', 'White Russia'],
        'Switzerland': ['Swiss Confederation'],
        'Monaco': ['Principality of Monaco'],
        'San Marino': ['Republic of San Marino'],
        'Vatican City': ['Holy See', 'Vatican'],
        'Andorra': ['Principality of Andorra'],
        'Liechtenstein': ['Principality of Liechtenstein'],
        'Iceland': ['Republic of Iceland'],
        'Norway': ['Kingdom of Norway']
    }
    
    @classmethod
    def is_eea_country(cls, country_name: str) -> bool:
        """
        Check if a country is in the EEA
        
        Args:
            country_name: Name of the country to check
            
        Returns:
            True if country is in EEA, False otherwise
        """
        if not country_name:
            return False
        
        # Direct match
        if country_name in cls.EEA_COUNTRIES:
            return True
        
        # Check aliases
        country_lower = country_name.lower().strip()
        for country, aliases in cls.COUNTRY_ALIASES.items():
            if country_lower == country.lower() or country_lower in [alias.lower() for alias in aliases]:
                return country in cls.EEA_COUNTRIES
        
        return False
    
    @classmethod
    def get_eea_countries_list(cls) -> List[str]:
        """
        Get list of all EEA countries
        
        Returns:
            List of EEA country names
        """
        return sorted(list(cls.EEA_COUNTRIES))
    
    @classmethod
    def get_non_eea_countries(cls, all_countries: List[str]) -> List[str]:
        """
        Get list of non-EEA countries from a given list
        
        Args:
            all_countries: List of all countries to filter
            
        Returns:
            List of non-EEA countries
        """
        return [country for country in all_countries if not cls.is_eea_country(country)]
    
    @classmethod
    def classify_transactions(cls, transactions: List[Dict], country_column: str = 'merchant_country') -> Dict[str, List[Dict]]:
        """
        Classify transactions as EEA or non-EEA
        
        Args:
            transactions: List of transaction dictionaries
            country_column: Name of the column containing country information
            
        Returns:
            Dictionary with 'eea' and 'non_eea' transaction lists
        """
        eea_transactions = []
        non_eea_transactions = []
        
        for transaction in transactions:
            country = transaction.get(country_column, '')
            if cls.is_eea_country(country):
                eea_transactions.append(transaction)
            else:
                non_eea_transactions.append(transaction)
        
        return {
            'eea': eea_transactions,
            'non_eea': non_eea_transactions
        }
    
    @classmethod
    def get_country_statistics(cls, transactions: List[Dict], country_column: str = 'merchant_country') -> Dict[str, any]:
        """
        Get statistics about EEA vs non-EEA transactions
        
        Args:
            transactions: List of transaction dictionaries
            country_column: Name of the column containing country information
            
        Returns:
            Dictionary with statistics
        """
        classified = cls.classify_transactions(transactions, country_column)
        
        total_transactions = len(transactions)
        eea_count = len(classified['eea'])
        non_eea_count = len(classified['non_eea'])
        
        return {
            'total_transactions': total_transactions,
            'eea_transactions': eea_count,
            'non_eea_transactions': non_eea_count,
            'eea_percentage': (eea_count / total_transactions * 100) if total_transactions > 0 else 0,
            'non_eea_percentage': (non_eea_count / total_transactions * 100) if total_transactions > 0 else 0
        }
    
    @classmethod
    def get_sql_eea_list(cls) -> str:
        """
        Get EEA countries list formatted for SQL IN clause
        
        Returns:
            String with EEA countries for SQL queries
        """
        countries = "', '".join(cls.EEA_COUNTRIES)
        return f"'{countries}'"
    
    @classmethod
    def normalize_country_name(cls, country_name: str) -> str:
        """
        Normalize country name to standard form
        
        Args:
            country_name: Country name to normalize
            
        Returns:
            Normalized country name
        """
        if not country_name:
            return ''
        
        country_lower = country_name.lower().strip()
        
        # Check if it's already a standard EEA country
        for country in cls.EEA_COUNTRIES:
            if country_lower == country.lower():
                return country
        
        # Check aliases
        for country, aliases in cls.COUNTRY_ALIASES.items():
            if country_lower == country.lower() or country_lower in [alias.lower() for alias in aliases]:
                return country
        
        # Return original if no match found
        return country_name
