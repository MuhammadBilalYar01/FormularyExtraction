import numpy as np
import pandas as pd
import cv2
import math
import collections
import tempfile
import pdfplumber
from thefuzz import fuzz
import configparser
import json
import requests
import os
import timeit
import concurrent.futures
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from cloud_detect import provider
import logging
from table import Table

LOG_FILE = 'Illinois_Blue_Cross_Medicaid_Apr_22.log'
BASE_DATA_FILE = 'Illinois_Blue_Cross_Medicaid_Apr_22.pdf'
JSON_FILE = 'Illinois_Blue_Cross_Medicaid_Apr_22.json'
PROCESSED_DATA_FILE = 'Illinois_Blue_Cross_Medicaid_Apr_22.csv'
RXNORM_MAPPING_FILE = 'Illinois_Blue_Cross_Medicaid-rxnormid_drugs-Apr_22.txt'
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, filename=LOG_FILE, filemode='a+')
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check Environment & setup the base directory for the output & data files
if provider(excluded=['alibaba', 'aws', 'do', 'azure', 'oci']) == 'gcp':
    BASE_OUTPUT_DIR = '.'
    BASE_DATA_DIR = '.'
    IS_ON_GCP = True
    logger.info('Running on GCP')
else:
    BASE_OUTPUT_DIR = '../../../../output/Illinois'
    BASE_DATA_DIR = '../../../../data/Illinois/Blue_Cross_Medicaid'
    IS_ON_GCP = False
    logger.info('Running on local machine')

NUM_CORES = mp.cpu_count()
NUM_THREADS = 40

# Load Table Configuration
def load_config_section():
    config = configparser.RawConfigParser()
    config.read('./table_config.cfg')
    if not hasattr(load_config_section, 'section_dict'):
        load_config_section.section_dict = collections.defaultdict()        
        for section in config.sections():
            load_config_section.section_dict[section] = dict(config.items(section))
    return load_config_section.section_dict

# Function to get the row line positions from the page.
# arguments: page_image - the image of the page
def parse_table_first_iteration(page_image):
    list_lines = []
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        page_image.save(f.name)
        img = cv2.imread(f.name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 210)
        lines = cv2.HoughLinesP(edges, 1, math.pi/2, threshold = 12, minLineLength = 300, maxLineGap = 0) #2, None, 30, 1);
        for line1 in lines:
            for line in line1:
                pt1 = (line[0], line[1])
                pt2 = (line[2], line[3])
                # assert pt1[1] == pt2[1]
                list_lines.append(pt1[1])
    return list_lines

# Function to get the header index of the detected table, given the header columns
def get_header_index(table, table_headers):
    header_row = 0
    for row in table:
        if len(row) == 3:
            # if row equals table_headers
            if (table_headers[0] in row[0]) and (table_headers[1] in row[1]) and (table_headers[2] in row[2]):
                return header_row
            else:
                header_row += 1

## [START] Function to load the file & parse the tables from each page
# Extract the text from the pdf file
def parse_pdf(path, columns, table_headers, page_range, table_config):
    df_file_content = pd.DataFrame(columns=columns)
    logger.info('Parsing the PDF file: {}'.format(path))
    try:
        with pdfplumber.open(path) as pdf:
            for i in page_range:
                try:
                    page = pdf.pages[i]
                    page_content = page.crop(table_config.bbox)
                    # Parse table
                    table_content = page_content.extract_table(table_config.table_settings)
                    # Commenting out the 2 step parsing of the table, as the default table parsing is sufficient.
                    # page_image = page_content.to_image(resolution=288)
                    # list_horizontal_lines = parse_table_first_iteration(page_image)
                    # list_horizontal_lines = [x / 4 + table_config.bbox[1] for x in list_horizontal_lines]
                    # hlines = table_config.table_settings["explicit_horizontal_lines"].copy()
                    # table_config.update_hlines(list_horizontal_lines)
                    # # Now, parse the entire page
                    # table_content = page_content.extract_table(table_config.table_settings)
                    # # reset the table_settings explicit_horizontal_lines
                    # table_config.set_hlines(hlines)

                    # Get header index - it will be either 0 or 1
                    header_index = get_header_index(table_content, table_headers)
                    df_page_content = pd.DataFrame(table_content[header_index + 1:], columns=columns)
                    df_file_content = df_file_content.append(df_page_content, ignore_index=True)
                    logger.info('Parsed page: {}'.format(i))
                except Exception as e:
                    logger.exception("Error at page: {} - {}".format(i, e))
                    pass
    except Exception as e:
        logger.exception("Error in file open: {}".format(e))
    return df_file_content
## [END] Function to load the file & parse the tables from each page

## [START] Function to clean up the data and do further processing
# Clean up the extracted text
def process_file_content(df_file_content, drug_types):
    # Basic common cleaning /  processing
    # Replace all None and '' values with NaN
    df_file_content = df_file_content.replace(['', None], np.nan)
    df_file_content = df_file_content.replace('\n', ' ', regex=True)
    # Drop rows with all NaN values
    df_file_content = df_file_content.dropna(how='all')
    # Keep the rows where the column DrugTier is in ['P', 'NP']
    df_file_content = df_file_content[df_file_content.DrugTier.isin(['P', 'NP', 'SC'])].reset_index(drop=True)
    # Make a copy of the dataframe and do further file specific processing on it
    file_content = df_file_content.copy()
    # Here, we are splitting the drug name for multiple dosage types and creating new rows for each dosage type
    # Split the column DrugName on ';' and insert rows for each item in the split array
    for index, row in file_content.iterrows():
        if ';' in row[0]:
            drug_names = row[0].split(';')
            base_drug = ''
            # Find index of any of the items in drug_types in drug_names[0] and split on that
            if any(x in drug_names[0] for x in drug_types):
                base_drug = drug_names[0].split(max(filter(lambda x: x in drug_names[0], drug_types)))[0]
                # base_drug = drug_names[0].split(any(x in drug_names[0] for x in drug_types))[0]
            else:
                base_drug = drug_names[0]
                logging.error('No match found for {}'.format(drug_names[0]))
            for drug_index, drug_name in enumerate(drug_names):
                if drug_index == 0:
                    file_content.loc[index, 'DrugName'] = drug_name.strip()
                else:
                    new_row = file_content.iloc[index, :].copy()
                    new_row[0] = base_drug.strip() + ' ' + drug_name.strip()
                    file_content = file_content.append(new_row)

    # Reset index of file_content
    file_content = file_content.reset_index(drop=True)
    # Remove the extra spaces
    file_content['DrugName'] = file_content['DrugName'].apply(lambda x: ' '.join(x.split()))
    # In the column DrugName replace '\u2013' by '-'
    file_content['DrugName'] = file_content['DrugName'].replace("\u2013", "-", regex=True)
    # If any element of the columns Drug & Notes contains '\n' then replace it by '' in the string
    file_content['DrugName'] = file_content['DrugName'].apply(lambda x: x.replace('\n', '') if str(x).find('\n') != -1 else x)
    file_content['Requirements_Limits'] = file_content['Requirements_Limits'].apply(lambda x: x.replace('\n', '') if str(x).find('\n') != -1 else x)
    return file_content
## [END] Function to clean up the data and do further processing

## [START] Functions for building the RxNorm Id Search URLs for each DrugName
# Remove all the non-alphanumeric characters from the string
def cleanup_text(text):
    text = text.replace('*', '')
    text = text.replace('+', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('\u00ae', '')
    text = text.replace('\u2013', ' ')
    text = text.replace('\u2019', '')
    text = text.replace('\u2018', '')
    text = text.replace(',', ' ')
    text = text.replace('/', ' ')
    text = text.replace(':', ' ')
    text = text.replace('-', ' ')
    text = text.strip()
    return text

# Get the base DrugName from the DrugName-Dosage combination text
def get_base_drug_name(drug_name, drug_types):
    # Find the base drug name from drug_name having multiple strengths
    # Here, we are splitting the drug name for multiple strengths and finding the base drug name
    # find the index of any of the items in drug_types and split on that index
    base_drug = ''
    # Find index of any of the items in drug_types in drug_names[0] and split on that
    if any(x in drug_name for x in drug_types):
        try:
            # Find the index of last word in caps in drug_name
            drug_name_words = drug_name.split()
            last_word_index = [i for i, x in enumerate(drug_name_words) if x in drug_types][-1]
            base_drug = ' '.join(drug_name_words[:last_word_index + 1])
            base_strength = ' '.join(drug_name_words[last_word_index + 1:])
            logger.info('base_drug: {}, base_strength: {}'.format(base_drug, base_strength))
        except IndexError:
            logger.warning('No match found for {}'.format(drug_name))
            base_drug = drug_name
            base_strength = ''
        # base_drug = drug_name.split(max(filter(lambda x: x in drug_name, drug_types)))[0]
        # # base_drug = drug_names[0].split(any(x in drug_names[0] for x in drug_types))[0]
    else:
        base_drug = drug_name
        base_strength = ''
        logger.error('No match found for {}'.format(drug_name))
    return base_drug, base_strength

# Build the RxNorm Id Search URL for each DrugName
def build_urls(drugname, drug_types):
   # Split the Drug name on ',' only if it has any of the words from drug_types and has ', ' in it
    if any(x in drugname for x in drug_types) and (',' in drugname):
        drug_strengths = [y.strip() for y in drugname.split(',')]
        logging.info('getting base drug name for {}'.format(drugname))
        base_drug_name, base_strength = get_base_drug_name(drugname.split(',')[0], drug_types)
    else:
        if ',' in drugname:
            print('did not find any of drug_types in {}'.format(drugname))
        drug_strengths = [drugname]
        base_drug_name = drugname
        base_strength = ''
    
    # Remove * and + and fix other characters in the columns drug & dosage
    drug = cleanup_text(drug_strengths[0])
    strength = [base_strength] + drug_strengths[1:] if len(drug_strengths) > 1 else [base_strength]
    urls = []
    for index, i in enumerate(strength):
        # Remove * and + and fix other characters in the strength column
        i = cleanup_text(i)
        base_drug_name = cleanup_text(base_drug_name)
        url = 'https://rxnav.nlm.nih.gov/REST/rxcui.json?name=' + base_drug_name + '+' + i + '&search=2'
        url = '+'.join(url.split())
        urls.append(url)

    approx_url = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug + '&maxEntries=1'
    approx_url = '+'.join(approx_url.split())
    urls.append(approx_url + '&option=1')

    drug_name_len = len(drug.split())
    drug_name_partial1 = ' '.join(drug.split()[:4]) if drug_name_len > 4 else ''
    approx_url_drug_partial1 = ""
    if drug_name_partial1 != '':
        approx_url_drug_partial1 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial1 + '&maxEntries=1'
        approx_url_drug_partial1 = '+'.join(approx_url_drug_partial1.split())
        urls.append(approx_url_drug_partial1 + '&option=1')

    drug_name_partial2 = ' '.join(drug.split()[:3]) if drug_name_len > 3 else ''
    approx_url_drug_partial2 = ""
    if drug_name_partial2 != '':
        approx_url_drug_partial2 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial2 + '&maxEntries=1'
        approx_url_drug_partial2 = '+'.join(approx_url_drug_partial2.split())
        urls.append(approx_url_drug_partial2 + '&option=1')

    drug_name_partial3 = ' '.join(drug.split()[:2]) if drug_name_len > 2 else ''
    approx_url_drug_partial3 = ""
    if drug_name_partial3 != '':
        approx_url_drug_partial3 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial3 + '&maxEntries=1'
        approx_url_drug_partial3 = '+'.join(approx_url_drug_partial3.split())
        urls.append(approx_url_drug_partial3 + '&option=1')

    urls.append(approx_url + '&option=0')
    if approx_url_drug_partial1 != '':
        urls.append(approx_url_drug_partial1 + '&option=0')
    if approx_url_drug_partial2 != '':
        urls.append(approx_url_drug_partial2 + '&option=0')
    if approx_url_drug_partial3 != '':
        urls.append(approx_url_drug_partial3 + '&option=0')
    return urls
## [END] Functions for building the RxNorm Id Search URLs for each DrugName

## [SRART] Functions for validating the DrugName match with the value from the RxNorm Id Search API
# Cleanup the DrugName text
def get_cleanedup_drug_name(drug_name):
    drug_words = drug_name.split()
    drug_sub_name = ' '.join(drug_words[:min(8, len(drug_words))]).lower().strip()
    drug_sub_name = drug_sub_name.replace('/', ' ')
    drug_sub_name = drug_sub_name.replace(',', ' ')
    drug_sub_name = drug_sub_name.replace('(', ' ')
    drug_sub_name = drug_sub_name.replace(')', ' ')
    drug_sub_name = drug_sub_name.replace('-', ' ')
    drug_sub_name = drug_sub_name.replace('*', ' ')
    drug_sub_name = drug_sub_name.replace('+', ' ')
    drug_sub_name = drug_sub_name.replace('%', ' ')
    drug_sub_name = drug_sub_name.lower().strip()
    return ' '.join(drug_sub_name.split())

# Perform similarity check on the DrugName and the RxNorm Id Search API response
def check_drug_match(drug_name, rxnorm_drug):
    drug_name_len = len(drug_name.split())
    # rxnorm_drug = rxnorm_drug.lower()
    if fuzz.partial_ratio(drug_name, rxnorm_drug) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(drug_name, fuzz.partial_ratio(drug_name, rxnorm_drug)))

    if fuzz.token_sort_ratio(drug_name, rxnorm_drug) > 63:
        return True
    logger.info("{} - token sort ratio - {}".format(drug_name, fuzz.token_sort_ratio(drug_name, rxnorm_drug)))

    # if fuzz.partial_ratio(_dict_alt_drugs.get(drug_name.split()[0]), rxnorm_drug) > 70:
    #     return True
    # logger.info("{} - Alt Drug -- partial ratio - {}".format(drug_name, fuzz.partial_ratio(_dict_alt_drugs.get(drug_name.split()[0]), rxnorm_drug)))

    if (fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]), rxnorm_drug) > 70):
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(4, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]), rxnorm_drug)))
    
    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]), rxnorm_drug) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(3, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]), rxnorm_drug)))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]), rxnorm_drug) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(2, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]), rxnorm_drug)))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]), rxnorm_drug) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(1, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]), rxnorm_drug)))

    if fuzz.partial_ratio(drug_name.split()[0], rxnorm_drug) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(drug_name.split()[0], 
    fuzz.partial_ratio(drug_name.split()[0], rxnorm_drug)))

    if (fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]).lower(), rxnorm_drug.lower()) > 70):
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(4, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]).lower(), rxnorm_drug.lower())))
    
    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]).lower(), rxnorm_drug.lower()) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(3, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]).lower(), rxnorm_drug.lower())))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]).lower(), rxnorm_drug.lower()) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(2, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]).lower(), rxnorm_drug.lower())))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]).lower(), rxnorm_drug.lower()) > 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(1, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]).lower(), rxnorm_drug.lower())))

    if fuzz.partial_ratio(drug_name.split()[0].lower(), rxnorm_drug.lower()) > 70:
        logger.info("{} - partial ratio - {}".format(drug_name.split()[0].lower(), 
        fuzz.partial_ratio(drug_name.split()[0].lower(), rxnorm_drug.lower())))
        return True
    else:
        logger.info("{} - partial ratio - {}".format(drug_name.split()[0], 
        fuzz.partial_ratio(drug_name.split()[0], rxnorm_drug)))
        return False
## [END] Functions for validating the DrugName match with the value from the RxNorm Id Search API

## [START] Functions for fetching the rxnormIds and validating the DrugName match through check_drug_match function
def fetch_rxnorm_id_row(file_content_row):
    # Iterate over the URLs to fetch the RxNorm Id
    for url in file_content_row[3]:
        try:
            response = requests.get(url) #requests.get(url, params=payload)
            if 'approximateTerm.json' in url:
                file_content_row[4] = response.json()['approximateGroup']['candidate'][0]['rxcui']
                # Add check for Drug Name if URL has 'approximateTerm.json'
                url_check = 'https://rxnav.nlm.nih.gov/REST/rxcui/' + file_content_row[4] + '/property.json?propName=RxNorm%20Name'
                return_value = requests.get(url_check)
                return_val= return_value.json()['propConceptGroup']['propConcept'][0]['propValue']
                if check_drug_match(get_cleanedup_drug_name(file_content_row[0]), return_val):
                    # found a match
                    logger.info("Drug name: {}, response: {}, Found Match.".format(file_content_row[0]+": " + url, response.json()))
                    break
                else:
                    # no match found
                    logger.error("Drug name: {}, response: {}, No Match Found.".format(file_content_row[0]+": " + url, response.json()))
                    file_content_row[4] = ""
            else:
                file_content_row[4] = response.json()['idGroup']['rxnormId'][0]
                # Break if rxnorm_id was found!!!
                if file_content_row[4] != "":
                    break
        except Exception as e:
            logger.error("Drug name: {}, response: {}, Exception: {}".format(file_content_row[0]+": " + url, response.json(), e))
            file_content_row[4] = ""
    return file_content_row
## [END] Functions for fetching the rxnormIds and validating the DrugName match through check_drug_match function

## [START] Parallelized invokation of the function fetch_rxnorm_ids for each row in the dataframe
# Multiprocessing implemented by splitting the dataframe into chunks and then fetching the RxNormId for each chunk
def populate_rxnorm_ids(file_content):
    # Add a column RxnormId to the dataframe with default value as ""
    file_content['RxnormId'] = ""
    # Iterate over the dataframe and fetch the RxnormIds
    logger.info("Number of cores: {}. Number of threads per core: {}.".format(NUM_CORES, NUM_THREADS))
    file_content_split = np.array_split(file_content, NUM_CORES, axis=0)
    # create the multiprocessing pool
    pool = Pool(NUM_CORES)
    # process the DataFrame by mapping function to each file_content_split across the pool
    file_content_xtended = np.vstack(pool.map(fetch_rxnorm_ids, file_content_split))
    # close down the pool and join
    pool.close()
    pool.join()
    pool.clear()
    return file_content_xtended

# Multithreaded implementation for each core to process a chunk of the dataframe
def fetch_rxnorm_ids(file_content_chunk):
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as thread_executor:
        # First, the ThreadPoolExecutor is used to call the RxNorm API for each row.
        future_rxnormid_content = {
                                    thread_executor.submit(fetch_rxnorm_id_row, file_content_row): file_content_row for (index, file_content_row) in file_content_chunk.iterrows()
                                }
        # As each API call thread completes it returns a Future object from the thread executor
        aggregated_file_content = []
        for future in concurrent.futures.as_completed(future_rxnormid_content):
            file_row = future_rxnormid_content[future]
            try:
                aggregated_file_content.append(future.result())
            except Exception as e:
                logger.exception('Data row {} generated an exception: {}'.format(file_row, e))
            else:
                logger.info('Data row successfully processed - \n{}'.format(file_row))
    return aggregated_file_content
## [END] Parallelized invokation of the function fetch_rxnorm_ids for each row in the dataframe

## [START] Build the JSON output from the dataframe
# Generate the JSON file
def extract_output(file_content):
    _list_dict = []
    for row in file_content:
        try:
            _dict = {}
            _plans = {}
            _dict["rxnorm_id"] = row[4]
            _dict["drug_name"] = row[0]
            _plans["drug_tier"] = row[1] #drug_tier_map[row[1]]
            # This check for row[2] is to check for np.NaN values. This is the property of NaN dtype.
            # row[2] is np.nan check stopped working once the collection became a ndarray after parallelization!!!
            if row[2] != row[2]: #row[2] is np.nan:
                _plans["prior_authorization"] = False
                _plans["quantity_limit"] = False
                _plans["step_therapy"] = False
            else:
                # parse PA
                if "PA" in row[2]:
                    _plans["prior_authorization"] = True
                else:
                    _plans["prior_authorization"] = False
                # parse QL
                if "QL" in row[2]:
                    _plans["quantity_limit"] = True
                else:
                    _plans["quantity_limit"] = False
                # parse ST
                if "ST" in row[2]:
                    _plans["step_therapy"] = True
                else:
                    _plans["step_therapy"] = False
            _dict["plans"] = [_plans]
            _list_dict.append(_dict)
        except Exception as e:
            logger.exception("row - {}\nException: {}".format(row, e))

    with open(os.path.join(BASE_OUTPUT_DIR, JSON_FILE), 'w') as json_file:
        json.dump(_list_dict, json_file, indent=4)
## [END] Build the JSON output from the dataframe

## [START] Build the Table objects from the table configuration
# Load the table configuration for the pdf file
def load_table_config():
    config_details = load_config_section()
    bounding_box = (int(config_details['TABLE_PROPERTIES']['bb0']), 
                    int(config_details['TABLE_PROPERTIES']['bb1']),
                    int(config_details['TABLE_PROPERTIES']['bb2']),
                    int(config_details['TABLE_PROPERTIES']['bb3']))
    vlines = [int(config_details['TABLE_PROPERTIES']['vline0']),
              int(config_details['TABLE_PROPERTIES']['vline1']),
              int(config_details['TABLE_PROPERTIES']['vline2']),
              int(config_details['TABLE_PROPERTIES']['vline3'])]
    hlines = [int(config_details['TABLE_PROPERTIES']['hline0']),
              int(config_details['TABLE_PROPERTIES']['hline1'])]
    table_config = Table(bounding_box, vlines, hlines)
    return table_config
## [END] Build the Table objects from the table configuration

## [START] Build the map of DrugName vs. RxNormId
def build_Drug_Name_RxNormId_Map():
    list_rows = []
    f = open(os.path.join(BASE_OUTPUT_DIR, JSON_FILE), 'r')
    data = json.load(f)
    for row in data:
        _dict = {}
        _dict["rxnorm_id"] = "-1" if row["rxnorm_id"] == "" else row["rxnorm_id"]
        _dict["drug_name"] = row["drug_name"]
        list_rows.append(_dict)
    f.close()
    rxnormid_drugs = pd.DataFrame(list_rows)
    rxnormid_drugs['rxnorm_id'] = rxnormid_drugs['rxnorm_id'].astype(int)
    # sort rxnormid_drugs by rxnorm_id
    rxnormid_drugs = rxnormid_drugs.sort_values(by=['rxnorm_id'])
    # write out the list to a file
    with open(os.path.join(BASE_OUTPUT_DIR, RXNORM_MAPPING_FILE), 'w') as outfile:
        for row in rxnormid_drugs.iterrows():
            print(row[1]['rxnorm_id'], row[1]['drug_name'], file=outfile)
## [END] Build the map of DrugName vs. RxNormId

if __name__ == '__main__':
    path = os.path.join(BASE_DATA_DIR, BASE_DATA_FILE)
    list_header_cols = ['Drug Name', 'Preferred Status', 'Drug Status / Restriction']
    list_cols = ['DrugName', 'DrugTier', 'Requirements_Limits']
    drug_types = ['SOLR', 'TABS', 'CAPS', 'SOLN', 'TB24', 'SUSR', 'SUSP', 'CHEW', 'PACK', 'POWD', 'CONC', 
              'CPEP', 'CP12', 'GRAN', 'CP24', 'PT24', 'CREA', 'CSDR', 'LIQD', 'SOAJ', 'PTTW', 'ELIX', 
              'SYRP', 'ENEM', 'CPCR', 'chew', 'OINT', 'LOTN', 'GEL','PRSY','SOCT','CPSP','CPDR','KIT',
              'SOPN','SOCT','SUBL','EMUL','FILM','INJ','SOLG','FOAM','OIL','PT72','LPOP','T24A','SUPP',
              'TBCR','T12A','T12','TBEC','AEPB','AERO','SUSY','SRER','SOSY','PTWK','TB12','TBDP','GUM',
              'PSKT','AERS','NEBU','AERB','TBPK']
    drug_types.extend([x.lower() for x in drug_types])
    page_range = range(8, 242)
    try:
        start = timeit.default_timer()
        logger.info("Start TimeStamp - {} secs.".format(start))
        # Load the table configuration for the pdf file
        table_config = load_table_config()
        logger.info("Loaded table configuration.")
        # Load the pdf file
        file_content = parse_pdf(path, list_cols, list_header_cols, page_range, table_config)
        logger.info('Parsed the PDF file.')
        # Process the file content
        processed_file_content = process_file_content(file_content, drug_types)
        # Build the URL column
        processed_file_content['URL'] = processed_file_content['DrugName'].apply(lambda drugname: build_urls(drugname, drug_types))
        milestone1 = timeit.default_timer()
        logger.info("Processed file. Time taken - {} secs.".format(milestone1 - start))
        logger.info("Processed file content. Populating RxNormIds now.")
        # Populate RxNormIds
        xtended_file_content = populate_rxnorm_ids(processed_file_content)
        milestone2 = timeit.default_timer()
        logger.info("RxNormId population took {} secs.".format(milestone2 - milestone1))
        logger.info('Loaded the processed data. Building JSON now!')
        # Extract the output
        extract_output(xtended_file_content)
        logger.info('JSON file saved. Building RxNormId - DrugName mapping now.')
        # Generate the DrugName vs. RxNorm Id map
        build_Drug_Name_RxNormId_Map()
        stop = timeit.default_timer()
        logger.info("Total Time taken - {} secs.".format(stop - start))
        logger.info('Finished processing!!!')
    except Exception as e:
        logger.exception("Error: {}".format(e))
        pass