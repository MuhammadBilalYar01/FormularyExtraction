import pandas as pd
import numpy as np
import math
import itertools
import collections
import tempfile
import cv2
import pdfplumber
from thefuzz import fuzz
import configparser
import json    
import requests
import os
from cloud_detect import provider
# import re
import logging
# import sys
from table import Table

LOG_FILE = 'Nevada_health_plan_4tier_sg.log'
BASE_DATA_FILE = '4 tier SG.pdf'
JSON_FILE = 'Nevada_health_plan_4tier_sg_May_22.json'
PROCESSED_DATA_FILE = 'Nevada_health_plan_4tier_sg_May_22.csv'
RXNORM_MAPPING_FILE = 'Nevada_health_plan_4tier_sg-rxnormid_drugs-May_22.txt'
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
    BASE_OUTPUT_DIR = '../../../../../output/Nevada/Health_Plan_of_Nevada/'
    BASE_DATA_DIR = '../../../../../data/Nevada/Health_Plan_of_Nevada/May_22/'
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
            # if row equals list_header_cols
            if (table_headers[0] in row[0]) and (table_headers[1] in row[1]) and (table_headers[2] in row[2]):
                return header_row
            else:
                header_row += 1

## [START] Function to load the file & parse the tables from each page
# Extract the text from the pdf file
def parse_pdf(path, columns, table_headers, page_range, table_config):
    df_file_content = pd.DataFrame(columns=columns)
    logger.info('Parsing the PDF file: {}'.format(path))
    with pdfplumber.open(path) as pdf:
        for i in page_range:
            try:
                page = pdf.pages[i]
                page_content = page.crop(table_config.bbox)
                # Skipping CV2 search for horizontal lines as it was not working for the pages 12 & 13.
                # Normal PDFPlumber search for horizontal lines is working fine.
                table_content = page_content.extract_table(table_config.table_settings)
                # # Parse table
                # list_horizontal_lines = parse_table_first_iteration(page_content.find_tables(table_config.table_settings))
                # hlines_base = table_config.table_settings["explicit_horizontal_lines"].copy()
                # table_config.update_hlines(list_horizontal_lines)
                # table_config.set_hstrategy("explicit")
                # # Now, parse the entire left page
                # table_content = page_content.extract_table(table_config.table_settings)
                # # reset the table_settings explicit_horizontal_lines
                # table_config.set_hlines(hlines_base)
                # table_config.set_hstrategy("text")
                # # Get header index - it will be either 0 or 1
                # header_index = get_header_index(table_content, table_headers)
                # df_page_content = pd.DataFrame(table_content[header_index + 1:], columns=columns)

                # Removing the search for header index. Loading all rows into the DataFrame and doing the post processing
                df_page_content = pd.DataFrame(table_content, columns=columns)
                df_file_content = df_file_content.append(df_page_content, ignore_index=True)
                logger.info('Parsed page: {}'.format(i))
            except Exception as e:
                logger.exception("Error at page: {} - {}".format(i, e))
                pass
    return df_file_content
## [END] Function to load the file & parse the tables from each page

## [START] Function to clean up the data and do further processing
# Build a column DrugName by appending the columns Brand Name and Generic Name like this: 
# return Brand Name if present, else Generic Name
def build_drugname_list(df_file_content_row):
    if df_file_content_row['Brand Name'] in ['', None]:
        return df_file_content_row['Generic Name']
    else:
        return df_file_content_row['Brand Name']

# Clean up the extracted text
def process_file_content(df_file_content):
    # Build a column DrugName by appending the columns Brand Name and Generic Name    
    df_file_content['DrugName'] = df_file_content.apply(build_drugname_list, axis=1)
    # Basic common cleaning /  processing
    # Replace all None and '' values with NaN
    df_file_content = df_file_content.replace(['', None], np.nan)
    df_file_content = df_file_content.replace('\n', ' ', regex=True)
    # Drop rows with all NaN values
    df_file_content = df_file_content.dropna(how='all')
    # Keep only the rows where the column DrugTier is in ['1', '2', '3', '4']
    df_file_content = df_file_content[df_file_content.DrugTier.isin(['1', '2', '3', '4'])].reset_index(drop=True)
    # Make a copy of the dataframe and do further file specific processing on it
    file_content = df_file_content.copy()
    # Drop the columns 'Brand name' & 'Generic Name' from file_content
    file_content = file_content.drop(columns=['Generic Name', 'Brand Name'])
    # Reorder the columns
    reordered_cols = ['DrugName', 'DrugTier', 'Requirements_Limits']
    file_content = file_content[reordered_cols]

    # In the column DrugName replace '\u2013' by '-'
    file_content['DrugName'] = file_content['DrugName'].replace("\u2013", "-", regex=True)
    #  If any element of the columns DrugName contains '*' then replace it by '' in the string
    file_content['DrugName'] = file_content['DrugName'].replace("\*", "", regex=True)
    #  If any element of the columns Drug & Notes contains '\n' then replace it by '' in the string
    file_content['DrugName'] = file_content['DrugName'].apply(lambda x: x.replace('\n', ' ') if str(x).find('\n') != -1 else x)
    file_content['Requirements_Limits'] = file_content['Requirements_Limits'].apply(lambda x: x.replace('\n', ' ') if str(x).find('\n') != -1 else x)
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
def build_urls(drug_strengths):
    # No need to split / process the drug name - use it as is
    # drug_strengths =  x #.split(',')
    # Remove * and + and fix other characters in the columns drug & dosage
    drug = cleanup_text(drug_strengths)
    urls = []
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
    if fuzz.partial_ratio(drug_name, rxnorm_drug) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(drug_name, fuzz.partial_ratio(drug_name, rxnorm_drug)))

    if fuzz.token_sort_ratio(drug_name, rxnorm_drug) >= 63:
        return True
    logger.info("{} - token sort ratio - {}".format(drug_name, fuzz.token_sort_ratio(drug_name, rxnorm_drug)))

    # if fuzz.partial_ratio(_dict_alt_drugs.get(drug_name.split()[0]), rxnorm_drug) > 70:
    #     return True
    # logger.info("{} - Alt Drug -- partial ratio - {}".format(drug_name, fuzz.partial_ratio(_dict_alt_drugs.get(drug_name.split()[0]), rxnorm_drug)))

    if (fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]), rxnorm_drug) >= 70):
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(4, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]), rxnorm_drug)))
    
    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]), rxnorm_drug) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(3, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]), rxnorm_drug)))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]), rxnorm_drug) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(2, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]), rxnorm_drug)))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]), rxnorm_drug) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(1, drug_name_len)]), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]), rxnorm_drug)))

    if fuzz.partial_ratio(drug_name.split()[0], rxnorm_drug) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(drug_name.split()[0], 
    fuzz.partial_ratio(drug_name.split()[0], rxnorm_drug)))

    if (fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]).lower(), rxnorm_drug.lower()) >= 70):
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(4, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(4, drug_name_len)]).lower(), rxnorm_drug.lower())))
    
    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]).lower(), rxnorm_drug.lower()) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(3, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(3, drug_name_len)]).lower(), rxnorm_drug.lower())))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]).lower(), rxnorm_drug.lower()) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(2, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(2, drug_name_len)]).lower(), rxnorm_drug.lower())))

    if fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]).lower(), rxnorm_drug.lower()) >= 70:
        return True
    logger.info("{} - partial ratio - {}".format(' '.join(drug_name.split()[:min(1, drug_name_len)]).lower(), 
    fuzz.partial_ratio(' '.join(drug_name.split()[:min(1, drug_name_len)]).lower(), rxnorm_drug.lower())))

    if fuzz.partial_ratio(drug_name.split()[0].lower(), rxnorm_drug.lower()) >= 70:
        logger.info("{} - partial ratio - {}".format(drug_name.split()[0].lower(), 
        fuzz.partial_ratio(drug_name.split()[0].lower(), rxnorm_drug.lower())))
        return True
    else:
        logger.info("{} - partial ratio - {}".format(drug_name.split()[0], 
        fuzz.partial_ratio(drug_name.split()[0], rxnorm_drug)))
        return False
## [END] Functions for validating the DrugName match with the value from the RxNorm Id Search API

## [START] Build the JSON output from the dataframe
# Since initially all the rows were loaded, hence the data type of the DrugTier column is object.
# Hence, we need to keep the string version of tiers as the dict key.
_drug_tier_map = {'1': 'Tier 1', 
                  '2': 'Tier 2', 
                  '3': 'Tier 3',
                  '4': 'Tier 4',}

# Generate the JSON file
def extract_output(file_content):
    _list_dict = []
    for (index, row) in file_content.iterrows():
        _dict = {}
        _plans = {}
        for url in row[3]:
            try:
                response = requests.get(url) #requests.get(url, params=payload)
                if 'approximateTerm.json' in url:
                    _dict["rxnorm_id"] = response.json()['approximateGroup']['candidate'][0]['rxcui']
                    # Add check for Drug Name if URL has 'approximateTerm.json'
                    url_check = 'https://rxnav.nlm.nih.gov/REST/rxcui/' + _dict["rxnorm_id"] + '/property.json?propName=RxNorm%20Name'
                    return_value = requests.get(url_check)
                    return_val= return_value.json()['propConceptGroup']['propConcept'][0]['propValue']
                    if check_drug_match(get_cleanedup_drug_name(row[0]), return_val):
                        # found a match
                        logger.info("Drug name: {}, response: {}, Found Match.".format(row[0]+": " + url, response.json()))
                        break
                    else:
                        # no match found
                        logger.error("Drug name: {}, response: {}, No Match Found.".format(row[0]+": " + url, response.json()))
                        _dict["rxnorm_id"] = ""
                else:
                    _dict["rxnorm_id"] = response.json()['idGroup']['rxnormId'][0]
                    # Break if rxnorm_id was found!!!
                    if _dict["rxnorm_id"] != "":
                        break
            except Exception as e:
                logger.error("Drug name: {}, response: {}, Exception: {}".format(row[0]+": " + url, response.json(), e))
                _dict["rxnorm_id"] = ""

        _dict["drug_name"] = row[0]
        _plans["drug_tier"] = _drug_tier_map[row[1]]
        if row[2] is np.NaN:
            _plans["prior_authorization"] = False
            _plans["quantity_limit"] = False
            _plans["step_therapy"] = False
        else:
            # parse PA
            if "PA" in str(row[2]):
                # Commenting out the value extraction for now
                # pa_plan = str(get_plan(row[2], 'PA')).removeprefix('PA')
                # _plans["prior_authorization"] = pa_plan if ((pa_plan != "nan") and (len(pa_plan) > 0)) else True
                _plans["prior_authorization"] = True
            else:
                _plans["prior_authorization"] = False
            # parse QL
            if "QL" in str(row[2]):
                # print(row[2].split('; ')[0])
                # Commenting out the value extraction for now
                # ql_plan = get_plan(row[2], 'QL')
                ## _plans["quantity_limit"] = [''.join(t) for t in re.findall(r'\((.+?)\)|(\w)', row[1].split('; ')[0].split(', ')[1])][-1:]
                # _plans["quantity_limit"] = ql_plan.removeprefix('QL  (').removesuffix(')')
                _plans["quantity_limit"] = True
            else:
                _plans["quantity_limit"] = False
            # parse ST
            if "ST" in str(row[2]):
                _plans["step_therapy"] = True
            else:
                _plans["step_therapy"] = False
        _dict["plans"] = [_plans]
        _list_dict.append(_dict)

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
                  int(config_details['TABLE_PROPERTIES']['vline4'])]
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
    list_header_cols = ['Generic Name', 'Brand Name', 'Tier', 'Notes']
    list_cols = ['Generic Name', 'Brand Name', 'DrugTier', 'Requirements_Limits']
    page_range = range(6, 52)
    try:
        # Load the table configuration for the pdf file
        table_config = load_table_config()
        logger.info("Loaded table configuration.")
        # Load the pdf file
        file_content = parse_pdf(path, list_cols, list_header_cols, page_range, table_config)
        # Process the file content
        processed_file_content = process_file_content(file_content)
        # Build the URL column
        processed_file_content['URL'] = processed_file_content['DrugName'].apply(lambda x: build_urls(x))
        logger.info('Loaded the processed data. Building JSON now!')
        # Extract the output
        extract_output(processed_file_content)
        # Generate the DrugName vs. RxNorm Id map
        logger.info('Extracted JSON. Building RxNorm Id - Drug name mapping now!')
        build_Drug_Name_RxNormId_Map()
        logger.info('Finished processing!!!')
    except Exception as e:
        logger.error("Error: {}".format(e))
        pass