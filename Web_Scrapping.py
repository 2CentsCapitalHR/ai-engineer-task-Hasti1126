import os
import csv
import time
import urllib.parse
import requests
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import re
import mimetypes
from urllib.parse import urlparse, parse_qs

# ========================
# CONFIG
# ========================
INPUT_CSV = "data.csv"
DOWNLOAD_DIR = "adgm_rag_reference_3"
OUTPUT_CSV = "crawl_results_3.csv"
ALLOWED_EXT = (".pdf", ".docx", ".doc", ".xlsx", ".txt")
MAX_RETRIES = 3
TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================
# SPECIALIZED DRIVER SETUP
# ========================
def setup_driver():
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--window-size=1920,1080")
    chrome_opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option('useAutomationExtension', False)
    
    # Add download preferences
    prefs = {
        "download.default_directory": os.path.abspath(DOWNLOAD_DIR),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_opts.add_experimental_option("prefs", prefs)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_opts)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

# ========================
# FILE VALIDATION
# ========================
def validate_downloaded_file(file_path, expected_min_size=1024):
    """Validate that downloaded file is actually a valid file"""
    try:
        if not os.path.exists(file_path):
            return False, "File doesn't exist"
        
        file_size = os.path.getsize(file_path)
        if file_size < expected_min_size:
            return False, f"File too small ({file_size} bytes)"
        
        # Check if it's actually HTML (error page)
        with open(file_path, 'rb') as f:
            first_bytes = f.read(512)
            if b'<html' in first_bytes.lower() or b'<!doctype html' in first_bytes.lower():
                return False, "File contains HTML (likely error page)"
        
        # Try to detect MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        logger.info(f"Detected MIME type: {mime_type}")
        
        return True, f"Valid file ({file_size} bytes, {mime_type})"
    
    except Exception as e:
        return False, f"Validation error: {e}"

# ========================
# IMPROVED FILENAME EXTRACTION
# ========================
def extract_filename_from_response(response, fallback_name="document"):
    """Extract proper filename from HTTP response headers"""
    
    # Try Content-Disposition header first
    content_disposition = response.headers.get('content-disposition', '')
    if content_disposition:
        import re
        filename_match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)["\']?', content_disposition)
        if filename_match:
            filename = filename_match.group(1).strip()
            # Clean the filename
            filename = urllib.parse.unquote(filename)
            return sanitize_filename(filename)
    
    # Try to get filename from URL
    parsed_url = urlparse(response.url)
    url_filename = os.path.basename(parsed_url.path)
    if url_filename and '.' in url_filename:
        return sanitize_filename(url_filename)
    
    # Check content type for extension
    content_type = response.headers.get('content-type', '').lower()
    if 'pdf' in content_type:
        return f"{fallback_name}.pdf"
    elif 'msword' in content_type or 'officedocument' in content_type:
        return f"{fallback_name}.docx"
    elif 'excel' in content_type or 'spreadsheet' in content_type:
        return f"{fallback_name}.xlsx"
    else:
        return f"{fallback_name}.pdf"

def sanitize_filename(filename):
    """Clean filename for safe saving"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
    filename = filename.strip('._')  # Remove leading/trailing dots and underscores
    
    # Ensure it's not too long
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:90] + ext
    
    return filename

# ========================
# URL TYPE DETECTION
# ========================
def detect_url_type(url):
    """Detect if URL is direct file download or webpage to scrape"""
    if "google.com/url" in url:
        return "google_redirect"
    elif any(url.lower().endswith(ext) for ext in ALLOWED_EXT):
        return "direct_file"
    elif "assets.adgm.com/download" in url:
        return "direct_file"
    elif any(domain in url for domain in ["adgm.com", "thomsonreuters.com"]):
        return "webpage"
    else:
        return "unknown"

def extract_real_url_from_google(google_url):
    """Extract the real URL from Google redirect URL"""
    try:
        parsed = urllib.parse.urlparse(google_url)
        query_params = urllib.parse.parse_qs(parsed.query)
        
        if 'q' in query_params:
            real_url = query_params['q'][0]
            real_url = urllib.parse.unquote(real_url)
            logger.info(f"Extracted real URL: {real_url}")
            return real_url
        else:
            return google_url
    except:
        return google_url

# ========================
# IMPROVED GOOGLE REDIRECT HANDLING
# ========================
def download_through_google_redirect(driver, google_url, save_dir, category, doc_type):
    """Download file by following Google redirect properly with better filename handling"""
    try:
        logger.info(f"Following Google redirect: {google_url}")
        
        # Navigate to Google redirect URL
        driver.get(google_url)
        time.sleep(3)
        
        # Get final URL after redirect
        final_url = driver.current_url
        logger.info(f"Redirected to: {final_url}")
        
        # If we're on ADGM assets, use the session to download
        if "assets.adgm.com" in final_url:
            cookies = driver.get_cookies()
            session = requests.Session()
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'])
            
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.google.com/',
                'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.*,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
            })
            
            # Make HEAD request first to get proper filename
            try:
                head_response = session.head(final_url, timeout=TIMEOUT)
                filename = extract_filename_from_response(head_response, f"{sanitize_filename(doc_type)}")
            except:
                filename = f"{sanitize_filename(doc_type)}.pdf"
            
            # Ensure proper directory structure
            safe_category = sanitize_filename(category)
            safe_doc_type = sanitize_filename(doc_type)
            file_dir = os.path.join(save_dir, safe_category, safe_doc_type)
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(file_dir, filename)
            
            # Now download the file
            response = session.get(final_url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                return False, "", "Still receiving HTML after redirect"
            
            with open(save_path, 'wb') as f:
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            # Validate the downloaded file
            is_valid, validation_msg = validate_downloaded_file(save_path)
            if not is_valid:
                os.remove(save_path)  # Remove invalid file
                return False, "", f"Invalid file: {validation_msg}"
            
            return True, save_path, f"Success via Google redirect ({total_size} bytes)"
        else:
            return False, "", f"Redirect didn't reach ADGM assets (ended at: {final_url})"
    
    except Exception as e:
        return False, "", f"Google redirect failed: {e}"

# ========================
# IMPROVED DIRECT FILE DOWNLOAD
# ========================
def download_direct_file(driver, file_url, save_dir, category, doc_type, retries=MAX_RETRIES):
    """Download file directly from URL with improved filename handling"""
    
    # Get cookies from current session
    cookies = driver.get_cookies()
    session = requests.Session()
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.*,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Upgrade-Insecure-Requests': '1',
    })
    
    for attempt in range(retries):
        try:
            logger.info(f"Direct download attempt {attempt + 1}/{retries}: {file_url}")
            
            # Make HEAD request first to get proper filename
            try:
                head_response = session.head(file_url, timeout=TIMEOUT, allow_redirects=True)
                filename = extract_filename_from_response(head_response, f"{sanitize_filename(doc_type)}")
            except:
                filename = f"{sanitize_filename(doc_type)}.pdf"
            
            # Ensure proper directory structure
            safe_category = sanitize_filename(category)
            safe_doc_type = sanitize_filename(doc_type)
            file_dir = os.path.join(save_dir, safe_category, safe_doc_type)
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(file_dir, filename)
            
            # Now download the file
            response = session.get(file_url, stream=True, timeout=TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            content_length = response.headers.get('content-length', '0')
            
            logger.info(f"Content-Type: {content_type}, Content-Length: {content_length}")
            
            # Skip if it's HTML (error page)
            if 'text/html' in content_type:
                logger.warning(f"Received HTML instead of file: {file_url}")
                return False, "", "Received HTML error page"
            
            # Write file
            with open(save_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
            
            if downloaded_size == 0:
                return False, "", "Downloaded file is empty"
            
            # Validate the downloaded file
            is_valid, validation_msg = validate_downloaded_file(save_path)
            if not is_valid:
                os.remove(save_path)  # Remove invalid file
                return False, "", f"Invalid file: {validation_msg}"
            
            logger.info(f"✅ Downloaded: {save_path} ({downloaded_size} bytes)")
            return True, save_path, f"Success ({downloaded_size} bytes)"
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Direct download attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                return False, "", str(e)
            time.sleep(2 ** attempt)
    
    return False, "", "Max retries exceeded"

# ========================
# WEBPAGE SCRAPING (Updated to use new download function)
# ========================
def scrape_webpage_for_files(driver, page_url, save_dir, category, doc_type):
    """Scrape webpage for downloadable files"""
    
    try:
        logger.info(f"Scraping webpage: {page_url}")
        driver.get(page_url)
        
        # Wait for page load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)
        
        # Look for download links
        download_results = []
        
        # Method 1: Find direct file links
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            try:
                href = link.get_attribute("href")
                if href and any(ext in href.lower() for ext in ALLOWED_EXT):
                    success, save_path, message = download_direct_file(
                        driver, href, save_dir, category, doc_type
                    )
                    download_results.append({
                        'url': href,
                        'success': success,
                        'path': save_path,
                        'message': message
                    })
            except Exception as e:
                continue
        
        # Method 2: Look for ADGM-specific patterns
        page_source = driver.page_source
        asset_pattern = r'https://assets\.adgm\.com/download/[^"\'>\s]+'
        asset_matches = re.findall(asset_pattern, page_source)
        
        for match in asset_matches:
            if any(ext in match.lower() for ext in ALLOWED_EXT):
                success, save_path, message = download_direct_file(
                    driver, match, save_dir, category, doc_type
                )
                download_results.append({
                    'url': match,
                    'success': success,
                    'path': save_path,
                    'message': message
                })
        
        # Method 3: Look for Thomson Reuters specific patterns
        if "thomsonreuters.com" in page_url:
            tr_pattern = r'https://[^"\'>\s]*\.pdf[^"\'>\s]*'
            tr_matches = re.findall(tr_pattern, page_source)
            
            for match in tr_matches:
                success, save_path, message = download_direct_file(
                    driver, match, save_dir, category, doc_type
                )
                download_results.append({
                    'url': match,
                    'success': success,
                    'path': save_path,
                    'message': message
                })
        
        logger.info(f"Processed {len(download_results)} download attempts")
        return download_results, None
        
    except Exception as e:
        logger.error(f"Error scraping webpage {page_url}: {e}")
        return [], str(e)

# ========================
# RESULT TRACKING
# ========================
class ADGMCrawlResults:
    def __init__(self):
        self.results = []
    
    def add_result(self, category, doc_type, page_url, file_url="", saved_path="", status="", error=""):
        self.results.append({
            'CategoryFolder': category,
            'DocumentType': doc_type,
            'PageURL': page_url,
            'FileURL': file_url,
            'SavedLocalPath': saved_path,
            'Status': status,
            'Error': error
        })
    
    def save_to_csv(self, filename):
        if self.results:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)

# ========================
# MAIN CRAWLER (Updated)
# ========================
def main_crawler():
    driver = setup_driver()
    results = ADGMCrawlResults()
    
    try:
        # Read CSV
        with open(INPUT_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        logger.info(f"Starting crawl of {len(rows)} entries...")
        
        for i, row in enumerate(tqdm(rows, desc="Processing entries"), 1):
            category = row["Category"].strip()
            doc_type = row["Document/Template Type"].strip()
            url = row["Official ADGM/Government Link"].strip()
            
            logger.info(f"\n[{i}/{len(rows)}] Processing: {category} - {doc_type}")
            logger.info(f"URL: {url}")
            
            # Detect URL type
            url_type = detect_url_type(url)
            logger.info(f"URL Type: {url_type}")
            
            if url_type == "google_redirect":
                success, save_path, message = download_through_google_redirect(
                    driver, url, DOWNLOAD_DIR, category, doc_type
                )
                
                if success:
                    results.add_result(category, doc_type, url, url, save_path, "✅ Success (Google redirect)")
                else:
                    # Try extracting real URL and direct download
                    real_url = extract_real_url_from_google(url)
                    if real_url != url:
                        success2, save_path2, message2 = download_direct_file(
                            driver, real_url, DOWNLOAD_DIR, category, doc_type
                        )
                        if success2:
                            results.add_result(category, doc_type, url, real_url, save_path2, "✅ Success (extracted URL)")
                        else:
                            results.add_result(category, doc_type, url, real_url, "", f"❌ Failed", f"{message}, {message2}")
                    else:
                        results.add_result(category, doc_type, url, url, "", f"❌ Failed", message)
            
            elif url_type == "direct_file":
                success, save_path, message = download_direct_file(
                    driver, url, DOWNLOAD_DIR, category, doc_type
                )
                
                if success:
                    results.add_result(category, doc_type, url, url, save_path, "✅ Success")
                else:
                    results.add_result(category, doc_type, url, url, "", f"❌ Failed", message)
            
            elif url_type == "webpage":
                download_results, error = scrape_webpage_for_files(
                    driver, url, DOWNLOAD_DIR, category, doc_type
                )
                
                if error:
                    results.add_result(category, doc_type, url, "", "", f"⚠️ Crawl error", error)
                elif not download_results:
                    results.add_result(category, doc_type, url, "", "", "⚠️ No files found")
                else:
                    # Add results for each download attempt
                    for result in download_results:
                        if result['success']:
                            results.add_result(category, doc_type, url, result['url'], result['path'], "✅ Success")
                        else:
                            results.add_result(category, doc_type, url, result['url'], "", "❌ Failed", result['message'])
            
            else:
                results.add_result(category, doc_type, url, "", "", "⚠️ Unknown URL type")
            
            # Respectful delay
            if i < len(rows):  # Don't delay after last item
                time.sleep(DELAY_BETWEEN_REQUESTS)
    
    except Exception as e:
        logger.error(f"Critical error: {e}")
    
    finally:
        results.save_to_csv(OUTPUT_CSV)
        driver.quit()
        logger.info(f"\n✅ Crawling complete! Results saved to {OUTPUT_CSV}")
        
        # Print summary
        success_count = sum(1 for r in results.results if "Success" in r['Status'])
        total_count = len(results.results)
        logger.info(f"Summary: {success_count}/{total_count} files downloaded successfully")

# ========================
# FILE REPAIR UTILITY
# ========================
def repair_existing_downloads():
    """Utility function to check and repair existing downloaded files"""
    logger.info("Checking existing downloads...")
    
    repaired_count = 0
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            is_valid, message = validate_downloaded_file(file_path)
            
            if not is_valid:
                logger.warning(f"Invalid file found: {file_path} - {message}")
                
                # Try to determine if it's an HTML error page
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(1000)  # Read first 1000 chars
                        if '<html' in content.lower() or 'error' in content.lower():
                            logger.info(f"Removing HTML error file: {file_path}")
                            os.remove(file_path)
                            repaired_count += 1
                except:
                    pass
    
    logger.info(f"Repaired/removed {repaired_count} invalid files")

if __name__ == "__main__":
    # Uncomment the line below to check/repair existing downloads first
    # repair_existing_downloads()
    
    main_crawler()