import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SHLCatalogScraper:
    """
    Scrapes the SHL product catalog to extract assessment metadata.
    """
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    
    def scrape_catalog(self):
        """
        Scrapes the SHL product catalog and returns a DataFrame with assessment metadata.
        """
        logging.info("Starting to scrape SHL product catalog...")
        
        try:
            response = requests.get(self.base_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all product cards/sections
            product_sections = soup.find_all('div', class_=lambda c: c and 'product-' in c)
            
            assessments = []
            
            for section in product_sections:
                try:
                    # Extract assessment name
                    name_elem = section.find('h3')
                    if not name_elem:
                        continue
                    name = name_elem.text.strip()
                    
                    # Extract URL if available
                    url = ""
                    url_elem = section.find('a', href=True)
                    if url_elem:
                        url = url_elem['href']
                        if not url.startswith('http'):
                            url = f"https://www.shl.com{url}"
                    
                    # Extract description
                    description = ""
                    desc_elem = section.find('p')
                    if desc_elem:
                        description = desc_elem.text.strip()
                    
                    # Extract metadata like remote testing, IRT support, duration, test type
                    metadata_text = section.text.strip()
                    
                    # Check for remote testing support
                    remote_testing = "Yes" if "remote" in metadata_text.lower() else "Unknown"
                    
                    # Check for IRT (Item Response Theory) support
                    irt_support = "Yes" if "irt" in metadata_text.lower() or "item response theory" in metadata_text.lower() else "Unknown"
                    
                    # Extract duration
                    duration = "Unknown"
                    duration_match = re.search(r'(\d+)[-\s]?(?:to)?[-\s]?(\d+)?\s*(?:min|minutes)', metadata_text, re.IGNORECASE)
                    if duration_match:
                        if duration_match.group(2):
                            duration = f"{duration_match.group(1)}-{duration_match.group(2)} minutes"
                        else:
                            duration = f"{duration_match.group(1)} minutes"
                    
                    # Determine test type
                    test_type = "Unknown"
                    
                    if "personality" in metadata_text.lower():
                        test_type = "Personality Assessment"
                    elif "cognitive" in metadata_text.lower() or "ability" in metadata_text.lower():
                        test_type = "Cognitive Assessment"
                    elif "skill" in metadata_text.lower():
                        test_type = "Skill Assessment"
                    elif "behavioral" in metadata_text.lower():
                        test_type = "Behavioral Assessment"
                    elif "situational" in metadata_text.lower() or "judgment" in metadata_text.lower():
                        test_type = "Situational Judgment Test"
                    
                    assessments.append({
                        'name': name,
                        'url': url,
                        'description': description,
                        'remote_testing': remote_testing,
                        'irt_support': irt_support,
                        'duration': duration,
                        'test_type': test_type
                    })
                    
                except Exception as e:
                    logging.error(f"Error processing assessment section: {e}")
                    continue
            
            logging.info(f"Successfully scraped {len(assessments)} assessments from SHL catalog")
            
            return pd.DataFrame(assessments)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching SHL catalog: {e}")
            return pd.DataFrame()

    def get_detailed_description(self, url):
        """
        Scrapes the detailed product page to get a more complete description.
        """
        if not url or url == "":
            return ""
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for description sections, typically in paragraphs
            description_elems = soup.find_all('p')
            description = ' '.join([elem.text.strip() for elem in description_elems if len(elem.text.strip()) > 50])
            
            return description
            
        except Exception as e:
            logging.error(f"Error fetching detailed description from {url}: {e}")
            return ""

def scrape_and_save(output_path='assessments.csv'):
    """
    Scrapes the SHL catalog and saves the data to a CSV file.
    """
    scraper = SHLCatalogScraper()
    df = scraper.scrape_catalog()
    
    if not df.empty:
        # For assessments with URLs, try to get more detailed descriptions
        for idx, row in df.iterrows():
            if row['url']:
                detailed_desc = scraper.get_detailed_description(row['url'])
                if detailed_desc:
                    # Combine with existing description or replace if empty
                    if row['description']:
                        df.at[idx, 'description'] = f"{row['description']} {detailed_desc}"
                    else:
                        df.at[idx, 'description'] = detailed_desc
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logging.info(f"Saved assessment data to {output_path}")
        return df
    else:
        logging.error("Failed to scrape assessment data")
        return None

if __name__ == "__main__":
    # For testing
    scrape_and_save()
