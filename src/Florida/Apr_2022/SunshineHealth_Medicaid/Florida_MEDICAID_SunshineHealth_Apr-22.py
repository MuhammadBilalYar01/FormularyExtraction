import pandas as pd
import numpy as np
import cv2
import math
import itertools
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
from cloud_detect import provider
# import re
import logging
# import sys
from table import Table

LOG_FILE = 'Florida_MEDICAID_SunshineHealth_Apr22.log'
BASE_DATA_FILE = 'FL_MEDICAID_SunshineHealth-Apr2022.pdf'
JSON_FILE = 'Florida_MEDICAID_SunshineHealth_Apr22.json'
PROCESSED_DATA_FILE = 'Florida_MEDICAID_SunshineHealth_Apr22.csv'
RXNORM_MAPPING_FILE = 'Florida_MEDICAID_SunshineHealth-rxnormid_drugs-Apr_22.txt'
NUM_CORES = mp.cpu_count()
NUM_THREADS = 40
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
    BASE_OUTPUT_DIR = '../../../../output/Florida/SunshineHealth'
    BASE_DATA_DIR = '../../../../data/Florida/Apr_2022'
    IS_ON_GCP = False
    logger.info('Running on local machine')

# Load Table Configuration
def load_config_section():
    config = configparser.RawConfigParser()
    config.read('./table_config.cfg')
    if not hasattr(load_config_section, 'section_dict'):
        load_config_section.section_dict = collections.defaultdict()        
        for section in config.sections():
            load_config_section.section_dict[section] = dict(config.items(section))
    return load_config_section.section_dict

## [START] Function to load the file & parse the tables from each page
# Extract the text from the pdf file
def parse_pdf(path, page_range, table_config):
    df_file_content = pd.DataFrame() #columns=columns)
    logger.info('Parsing the PDF file: {}'.format(path))
    try:
        with pdfplumber.open(path) as pdf:
            for i in page_range:
                try:
                    page = pdf.pages[i]
                    page_content = page.crop(table_config.bbox)
                    # Now, parse the entire page
                    table_content = page_content.extract_table(table_config.table_settings)
                    df_page_content = pd.DataFrame(table_content[1:], columns=table_content[0])
                    if i== 0:
                        df_file_content.columns = table_content[0]
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
def process_file_content(df_file_content):
    try:
        # Basic common cleaning /  processing
        # Replace all None and '' values with NaN
        df_file_content = df_file_content.replace(['', None], np.nan)
        df_file_content = df_file_content.replace('\n', ' ', regex=True)
        # if columns 0, 1, 2, 3, 4 are not NaN, but column 6 is NaN, then fill column 6 with the value in next row
        for (index, row) in df_file_content.iterrows():
            if (row[0] is not np.NaN) and (row[1] is not np.NaN) and (row[2] is not np.NaN) and \
                (row[3] is not np.NaN) and (row[4] is not np.NaN) and (row[6] is np.NaN):
                df_file_content.iloc[index, 6] = df_file_content.iloc[index + 1, 6]
        # Drop the unnecessary columns
        df_columns = df_file_content.columns
        df_file_content.drop([df_columns[0], df_columns[1], df_columns[3], df_columns[4], df_columns[5]], axis=1, inplace=True)
        # Drop rows with all NaN values
        df_file_content = df_file_content.dropna(how='all')
        # # Drop the rows where the column DrugTier is NaN AND Requirements_Limits is NaN
        # df_file_content = df_file_content[~(df_file_content.DrugTier.isna() & df_file_content.Requirements_Limits.isna())].reset_index(drop=True)
        # Make a copy of the dataframe and do further file specific processing on it
        file_content = df_file_content.copy()
        # Rename columns to DrugName & Requirements_Limits
        file_content.columns = ['DrugName', 'Requirements_Limits']
        # There are some rows where the column Requirements_Limits is NaN. Drop those.
        file_content = file_content[~(file_content.Requirements_Limits.isna() | file_content.DrugName.isna())].reset_index(drop=True)
        # In the column DrugName replace '\u2013' by '-'
        file_content['DrugName'] = file_content['DrugName'].replace("\u2013", "-", regex=True)
        #  If any element of the columns Drug & Notes contains '\n' then replace it by '' in the string
        file_content['DrugName'] = file_content['DrugName'].apply(lambda x: x.replace('\n', ' ') if str(x).find('\n') != -1 else x)
        file_content['Requirements_Limits'] = file_content['Requirements_Limits'].apply(lambda x: x.replace('\n', ' ') if str(x).find('\n') != -1 else x)
    except Exception as e:
        logger.exception("Error in file processing: {}".format(e))
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

# Build the RxNorm Id Search URL for each DrugName
def build_urls(x):
    # No need to split / process the drug name - use it as is
    # drug_strengths = [x]
    # Remove * and + and fix other characters in the columns drug & dosage
    drug = cleanup_text(x)
    
    urls = []
    # for i in strength:
    # Remove * and + and fix other characters in the strength column
    # i = cleanup_text(i)
    # base_drug_name = cleanup_text(base_drug_name)
    url = 'https://rxnav.nlm.nih.gov/REST/rxcui.json?name=' + drug + '&search=2'
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
    for url in file_content_row[2]:
        try:
            response = requests.get(url)
            if 'approximateTerm.json' in url:
                file_content_row[3] = response.json()['approximateGroup']['candidate'][0]['rxcui']
                # Add check for Drug Name if URL has 'approximateTerm.json'
                url_check = 'https://rxnav.nlm.nih.gov/REST/rxcui/' + file_content_row[3] + '/property.json?propName=RxNorm%20Name'
                return_value = requests.get(url_check)
                return_val= return_value.json()['propConceptGroup']['propConcept'][0]['propValue']
                if check_drug_match(get_cleanedup_drug_name(file_content_row[0]), return_val):
                    # found a match
                    logger.info("Drug name: {}, response: {}, Found Match.".format(file_content_row[0]+": " + url, response.json()))
                    break
                else:
                    # no match found
                    logger.error("Drug name: {}, response: {}, No Match Found.".format(file_content_row[0]+": " + url, response.json()))
                    file_content_row[3] = ""
            else:
                file_content_row[3] = response.json()['idGroup']['rxnormId'][0]
                # Break if rxnorm_id was found!!!
                if file_content_row[3] != "":
                    break
        except Exception as e:
            logger.error("Drug name: {}, response: {}, Exception: {}".format(file_content_row[0]+": " + url, response.json(), e))
            file_content_row[3] = ""
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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        file_content_xtended = executor.map(fetch_rxnorm_ids, file_content_split)
    # file_content_xtended is returned as a generator object. 
    # Each item in the generator is a list of file_content.shape[0] / NUM_CORES rows.
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
# arguments: file_content - the generator object after the RxNorm Ids are populated
def extract_output(file_content):
    _list_dict = []
    for generator_row in file_content:
        # each generator_row is a list of original file_content.shape[0] / NUM_CORES rows.
        # Iterate over the list of rows and build the JSON output
        for row in generator_row:
            try:
                _dict = {}
                _plans = {}
                _dict["rxnorm_id"] = row[3]
                _dict["drug_name"] = row[0]
                _plans["drug_tier"] = 'default' #_drug_tier_map[row[1]]
                # This check for row[1] is to check for np.NaN values. This is the property of NaN dtype.
                # row[1] is np.nan check stopped working once the collection became a ndarray after parallelization!!!
                if row[1] != row[1]: # if row[1] is np.NaN:
                    _plans["prior_authorization"] = False
                else:
                    # parse PA
                    if str(row[1]) != "No":
                        _plans["prior_authorization"] = True
                    else:
                        _plans["prior_authorization"] = False
                _plans["quantity_limit"] = False
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
                  int(config_details['TABLE_PROPERTIES']['vline3']),
                  int(config_details['TABLE_PROPERTIES']['vline4']),
                  int(config_details['TABLE_PROPERTIES']['vline5']),
                  int(config_details['TABLE_PROPERTIES']['vline6']),
                  int(config_details['TABLE_PROPERTIES']['vline7'])]
    table_config = Table(bounding_box, vlines)
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
    page_range = range(1, 152)
    try:
        start = timeit.default_timer()
        logger.info("Start TimeStamp - {} secs.".format(start))
        # Load the table configuration for the pdf file
        table_config = load_table_config()
        logger.info("Loaded table configuration.")
        # Load the pdf file
        file_content = parse_pdf(path, page_range, table_config)
        # Process the file content
        processed_file_content = process_file_content(file_content)
        # Build the URL column
        processed_file_content['URL'] = processed_file_content['DrugName'].apply(lambda x: build_urls(x))
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
        logger.info('Loaded the processed data. Building JSON now!')
        # Generate the DrugName vs. RxNorm Id map
        build_Drug_Name_RxNormId_Map()
        stop = timeit.default_timer()
        logger.info("Total Time taken - {} secs.".format(stop - start))
        logger.info('Finished processing!!!')
    except Exception as e:
        logger.exception("Error: {}".format(e))
        pass