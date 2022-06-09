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
from cloud_detect import provider
# import re
import logging
# import sys
from table import Table

LOG_FILE = 'Nevada_Ambetter_HealthPlan_May22.log'
BASE_DATA_FILE = '2022-nv-formulary_May-22.pdf'
JSON_FILE = 'Nevada_Ambetter_HealthPlan_May_2022.json'
PROCESSED_DATA_FILE = 'Nevada_Ambetter_HealthPlan-May_22.csv'
RXNORM_MAPPING_FILE = 'Nevada_Ambetter_HealthPlan-rxnormid_drugs-May_22.txt'
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
    BASE_OUTPUT_DIR = '../../../../output/Nevada/Ambetter_HealthPlan/May_2022'
    BASE_DATA_DIR = '../../../../data/Nevada/Ambetter_HealthPlan'
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
# arguments: page_image - the detected tables in the page
def parse_table_first_iteration(list_detected_tables):
    list_horizontal_lines = []
    for detected_table in list_detected_tables:
        parsed_rows = detected_table.rows
        for parsed_row in parsed_rows:
            row_bottom_position = None
            # Assumption being that all cells will never be None for a row!
            for parsed_cell in parsed_row.cells:
                if parsed_cell is None:
                    continue # This cell was not captured, look for next cell!
                else:
                    # row position captured, but don't exit!! Need to check if any other cell has a smaller
                    # value for bottom pixel position
                    if row_bottom_position is None:
                        row_bottom_position = float(parsed_cell[3])
                    else:
                        if float(parsed_cell[3]) < row_bottom_position:
                            row_bottom_position = float(parsed_cell[3])
            list_horizontal_lines.append(row_bottom_position)
    return list_horizontal_lines

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
def parse_pdf(path, columns, page_range, table_config_left, table_config_right):
    df_file_content = pd.DataFrame(columns=columns)
    logger.info('Parsing the PDF file: {}'.format(path))
    try:
        with pdfplumber.open(path) as pdf:
            for i in page_range:
                try:
                    page = pdf.pages[i]
                    page_content_left = page.crop(table_config_left.bbox)
                    page_content_right = page.crop(table_config_right.bbox)
                    # Parse left table
                    list_horizontal_lines_left = parse_table_first_iteration(page_content_left.find_tables(table_config_left.table_settings))
                    hlines_left = table_config_left.table_settings["explicit_horizontal_lines"].copy()
                    table_config_left.update_hlines(list_horizontal_lines_left)
                    # Now, parse the entire left page
                    table_content_left = page_content_left.extract_table(table_config_left.table_settings)
                    # reset the table_settings explicit_horizontal_lines
                    table_config_left.set_hlines(hlines_left)
                    # Parse right table
                    list_horizontal_lines_right = parse_table_first_iteration(page_content_right.find_tables(table_config_right.table_settings))
                    hlines_right = table_config_right.table_settings["explicit_horizontal_lines"].copy()
                    table_config_right.update_hlines(list_horizontal_lines_right)
                    # Now, parse the entire right page
                    table_content_right = page_content_right.extract_table(table_config_right.table_settings)
                    # reset the table_settings explicit_horizontal_lines
                    table_config_right.set_hlines(hlines_right)

                    df_page_content_left = pd.DataFrame(table_content_left[1:], columns=columns)
                    df_page_content_right = pd.DataFrame(table_content_right[1:], columns=columns)
                    df_file_content = df_file_content.append(df_page_content_left, ignore_index=True)
                    df_file_content = df_file_content.append(df_page_content_right, ignore_index=True)
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
        # # Drop rows with all NaN values
        # df_file_content = df_file_content.dropna(how='all')
        # Drop the rows where the column DrugTier is NaN AND Requirements_Limits is NaN
        df_file_content = df_file_content[~(df_file_content.DrugTier.isna() & df_file_content.Requirements_Limits.isna())].reset_index(drop=True)
        # Drop the rows where the column DrugTier is not in '0', '1A', '1B', '2', '3', '4', 'NF'
        df_file_content = df_file_content[df_file_content.DrugTier.isin(['0', '1A', '1B', '2', '3', '4', 'NF'])].reset_index(drop=True)
        # Make a copy of the dataframe and do further file specific processing on it
        file_content = df_file_content.copy()
        # Find the index of '(Use' in the column DrugName and trim all the text after it
        file_content['DrugName'] = file_content['DrugName'].apply(lambda x: x.split('(Use')[0])
        #  If any element of the columns Drug & Notes contains '\n' then replace it by '' in the string
        file_content['DrugName'] = file_content['DrugName'].apply(lambda x: x.replace('\n', ' ') if str(x).find('\n') != -1 else x)
        file_content['Requirements_Limits'] = file_content['Requirements_Limits'].apply(lambda x: x.replace('\n', ' ') if str(x).find('\n') != -1 else x)
    except Exception as e:
        logger.exception("Error in file processing: Drug name - {}, Detail - {}".format(file_content.loc[:, 'DrugName'], e))
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
    try:
        # Find the base drug name from drug_name having multiple strengths
        # Here, we are splitting the drug name for multiple strengths and finding the base drug name
        # find the index of any of the items in drug_types and split on that index
        base_drug = ''
        # Find index of any of the items in drug_types in drug_names[0] and split on that
        if any(x in drug_name for x in drug_types):
            base_drug = drug_name.split(max(filter(lambda x: x in drug_name, drug_types)))[0]
            # base_drug = drug_names[0].split(any(x in drug_names[0] for x in drug_types))[0]
            logger.info('base_drug: {}'.format(base_drug))
        else:
            base_drug = drug_name
            logger.error('No match found for {}'.format(drug_name))

        return base_drug
    except Exception as e:
        logger.warning('No match found for {} - {}'.format(drug_name, e.with_traceback()))
        return drug_name

# Build the RxNorm Id Search URL for each DrugName
def build_urls(drugname):
    drug_types = ['SOLR', 'TABS', 'CAPS', 'SOLN', 'TB24', 'SUSR', 'SUSP', 'CHEW', 'PACK', 'POWD', 'CONC', 
        'CPEP', 'CP12', 'GRAN', 'CP24', 'PT24', 'CREA', 'CSDR', 'LIQD', 'SOAJ', 'PTTW', 'ELIX', 
        'SYRP', 'ENEM', 'CPCR', 'chew', 'OINT', 'LOTN', 'GEL', 'TBCR', 'PNKT', 'tbcr or', 'solg', 
        'lpop bu', 'pt72 td', 'T12A', 'TB12', 'subl sl', 'FILM', 'subl', 'ptwk td', 'nebu in', 'SOSY', 
        'TBDP', 'aepb', 'nebu', 'tb12 or', 'tbec', 'SOPN', 'tbso', 'SUBL', 'OIL', 'cpdr', 'tbpk', 'soct sc']
    # Extend drug_types to include the lower case versions of the drug types
    drug_types.extend([x.lower() for x in drug_types])
    # Here, we are splitting the DrugName on ',' to handle multiple stengths of the same drug
    # Split the Drug name on ',' only if it has any of the words from drug_types and has ', ' in it

    if any(x in drugname for x in drug_types) and (',' in drugname):
        drug_strengths = [y.strip() for y in drugname.split(',')]
        logger.info('getting base drug name for {}'.format(drugname))
        base_drug_name = get_base_drug_name(drugname.split(',')[0], drug_types)
    else:
        if ',' in drugname:
            logger.warning('did not find any of drug_types in {}'.format(drugname))
        drug_strengths = [drugname]
        base_drug_name = drugname
    
    # Remove * and + and fix other characters in the columns drug & dosage
    drug = cleanup_text(drug_strengths[0])
    strength = [''] + drug_strengths[1:] if len(drug_strengths) > 1 else ['']
    urls = []
    for index, i in enumerate(strength):
        # Remove * and + and fix other characters in the strength column
        i = cleanup_text(i)
        base_drug_name = cleanup_text(base_drug_name)
        url = 'https://rxnav.nlm.nih.gov/REST/rxcui.json?name=' + (drug if index == 0 else base_drug_name) + '+' + i + '&search=2'
        url = '+'.join(url.split())
        urls.append(url)

    approx_url = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug + '&maxEntries=1'
    approx_url = '+'.join(approx_url.split())
    urls.append(approx_url + '&option=1')

    drug_name_len = len(drug.split())
    drug_name_partial1 = ' '.join(drug.split()[:5]) if drug_name_len > 5 else ''
    approx_url_drug_partial1 = ""
    if drug_name_partial1 != '':
        approx_url_drug_partial1 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial1 + '&maxEntries=1'
        approx_url_drug_partial1 = '+'.join(approx_url_drug_partial1.split())
        urls.append(approx_url_drug_partial1 + '&option=1')

    drug_name_partial2 = ' '.join(drug.split()[:4]) if drug_name_len > 4 else ''
    approx_url_drug_partial2 = ""
    if drug_name_partial2 != '':
        approx_url_drug_partial2 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial2 + '&maxEntries=1'
        approx_url_drug_partial2 = '+'.join(approx_url_drug_partial2.split())
        urls.append(approx_url_drug_partial2 + '&option=1')

    drug_name_partial3 = ' '.join(drug.split()[:3]) if drug_name_len > 3 else ''
    approx_url_drug_partial3 = ""
    if drug_name_partial3 != '':
        approx_url_drug_partial3 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial3 + '&maxEntries=1'
        approx_url_drug_partial3 = '+'.join(approx_url_drug_partial3.split())
        urls.append(approx_url_drug_partial3 + '&option=1')

    drug_name_partial4 = ' '.join(drug.split()[:2]) if drug_name_len > 2 else ''
    approx_url_drug_partial4 = ""
    if drug_name_partial4 != '':
        approx_url_drug_partial4 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial4 + '&maxEntries=1'
        approx_url_drug_partial4 = '+'.join(approx_url_drug_partial4.split())
        urls.append(approx_url_drug_partial4 + '&option=1')

    urls.append(approx_url + '&option=0')
    if approx_url_drug_partial1 != '':
        urls.append(approx_url_drug_partial1 + '&option=0')
    if approx_url_drug_partial2 != '':
        urls.append(approx_url_drug_partial2 + '&option=0')
    if approx_url_drug_partial3 != '':
        urls.append(approx_url_drug_partial3 + '&option=0')
    if approx_url_drug_partial4 != '':
        urls.append(approx_url_drug_partial4 + '&option=0')
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

## [START] Build the JSON output from the dataframe
_drug_tier_map = {
                    '0': 'Tier 0',
                    '1A': 'Tier 1A',
                    '1B': 'Tier 1B',
                    '2': 'Tier 2',
                    '3': 'Tier 3',
                    '4': 'Tier 4',
                    'NF': 'Non-formulary'
                 }

# Generate the JSON file
def extract_output(file_content):
    _list_dict = []
    for (index, row) in file_content.iterrows():
        _dict = {}
        _plans = {}
        # _drug_details = {}
        for url in row[3]:
            try:
                response = requests.get(url)
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
            if "PA" in row[2]:
                # Commenting out the value extraction for now
                # pa_plan = str(get_plan(row[2], 'PA')).removeprefix('PA')
                # _plans["prior_authorization"] = pa_plan if ((pa_plan != "nan") and (len(pa_plan) > 0)) else True
                _plans["prior_authorization"] = True
            else:
                _plans["prior_authorization"] = False
            # parse QL
            if "QL" in row[2]:
                # print(row[2].split('; ')[0])
                # Commenting out the value extraction for now
                # ql_plan = get_plan(row[2], 'QL')
                ## _plans["quantity_limit"] = [''.join(t) for t in re.findall(r'\((.+?)\)|(\w)', row[1].split('; ')[0].split(', ')[1])][-1:]
                # _plans["quantity_limit"] = ql_plan.removeprefix('QL  (').removesuffix(')')
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
    with open(os.path.join(BASE_OUTPUT_DIR, JSON_FILE), 'w') as json_file:
        json.dump(_list_dict, json_file, indent=4)
## [END] Build the JSON output from the dataframe

## [START] Build the Table objects from the table configuration
# Load the table configuration for the pdf file
def load_table_config():
    config_details = load_config_section()
    bounding_box_left = (int(config_details['TABLE_PROPERTIES']['bbleft0']), 
                        int(config_details['TABLE_PROPERTIES']['bbleft1']),
                        int(config_details['TABLE_PROPERTIES']['bbleft2']),
                        int(config_details['TABLE_PROPERTIES']['bbleft3']))
    bounding_box_right = (int(config_details['TABLE_PROPERTIES']['bbright0']), 
                        int(config_details['TABLE_PROPERTIES']['bbright1']),
                        int(config_details['TABLE_PROPERTIES']['bbright2']),
                        int(config_details['TABLE_PROPERTIES']['bbright3']))
    vlines_left = [int(config_details['TABLE_PROPERTIES']['vlineleft0']),
                  int(config_details['TABLE_PROPERTIES']['vlineleft1']),
                  int(config_details['TABLE_PROPERTIES']['vlineleft2']),
                  int(config_details['TABLE_PROPERTIES']['vlineleft3'])]
    vlines_right = [int(config_details['TABLE_PROPERTIES']['vlineright0']),
                  int(config_details['TABLE_PROPERTIES']['vlineright1']),
                  int(config_details['TABLE_PROPERTIES']['vlineright2']),
                  int(config_details['TABLE_PROPERTIES']['vlineright3'])]
    hlines = [int(config_details['TABLE_PROPERTIES']['hline0']),
              int(config_details['TABLE_PROPERTIES']['hline1'])]
    table_config_left = Table(bounding_box_left, vlines_left, hlines)
    table_config_right = Table(bounding_box_right, vlines_right, hlines)
    return table_config_left, table_config_right
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
    # list_header_cols = ['Prescription Drug Name', 'Drug Tier', 'Coverage Requirements and \nLimits']
    list_cols = ['DrugName', 'DrugTier', 'Requirements_Limits']
    page_range = range(3, 134)
    try:
        # # If running on local machine, parse the PDF and build the data file
        # if not IS_ON_GCP:
        # Load the table configuration for the pdf file
        table_config_left, table_config_right = load_table_config()
        logger.info("Loaded table configuration.")
        # Load the pdf file
        file_content = parse_pdf(path, list_cols, page_range, table_config_left, table_config_right)
        # Process the file content
        processed_file_content = process_file_content(file_content)
        # Build the URL column
        processed_file_content['URL'] = processed_file_content['DrugName'].apply(lambda x: build_urls(x))
        # # Persist the dataframe to a file
        # processed_file_content.to_csv(os.path.join(BASE_OUTPUT_DIR, PROCESSED_DATA_FILE), index=False)
        # logger.info('Persisted the processed data!')
        # else:
        # # If running on GCP, load the data file from the bucket
        # processed_file_content = pd.read_csv(os.path.join(BASE_OUTPUT_DIR, PROCESSED_DATA_FILE))
        # processed_file_content['URL'] = processed_file_content['URL'].apply(lambda x: x[1:-1].replace('\'', '').split(', '))
        logger.info('Loaded the processed data. Building JSON now!')
        # Extract the output
        extract_output(processed_file_content)
        # Generate the DrugName vs. RxNorm Id map
        build_Drug_Name_RxNormId_Map()
        logger.info('Finished processing!!!')
    except Exception as e:
        logger.error("Error: {}".format(e))
        pass