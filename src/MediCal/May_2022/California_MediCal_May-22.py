import pandas as pd
import numpy as np
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

LOG_FILE = 'CA_Medical_Medicaid_May_22.log'
BASE_DATA_FILE = 'Medi-Cal_Rx_Contract_Drugs_List_FINAL.pdf'
JSON_FILE = 'CA_Medical_Medicaid_May_22.json'
PROCESSED_DATA_FILE = 'CA_Medical_Medicaid_May_22.csv'
RXNORM_MAPPING_FILE = 'CA_Medical_Medicaid-rxnormid_drugs-May_22.txt'
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, filename=LOG_FILE, filemode='a+')
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check Environment & setup the base directory for the output & data files
if provider(excluded=['alibaba', 'aws', 'do', 'azure', 'oci']) == 'gcp':
    BASE_OUTPUT_DIR = '.'
    BASE_DATA_DIR = '.'
else:
    BASE_OUTPUT_DIR = '../../../output/MediCal/'
    BASE_DATA_DIR = '../../../data/MediCal/05-22/'
    
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
# arguments: list_detected_tables - the tables detected in the page
def parse_table_first_iteration(list_detected_tables):
    list_lines = []
    try:
        for detected_table in list_detected_tables:
            parsed_rows = detected_table.rows
            for parsed_row in parsed_rows:
                row_bottom_position = None
                # print(parsed_row.cells[0], parsed_row.cells[1])
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
                list_lines.append(row_bottom_position)
    except Exception as e:
        logger.exception('Error in parse_table_first_iteration: {}'.format(e))
    return list_lines

# Function to get the header index of the detected table, given the header columns
def get_header_index(table, table_headers):
    header_row = 0
    for row in table:
        if len(row) == 6:
            # if row equals list_header_cols
            if row[2] == table_headers[2] and row[3] == table_headers[3] and row[4] == table_headers[4]:
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
                # Parse table
                list_horizontal_lines = parse_table_first_iteration(page_content.find_tables(table_config.table_settings))
                hlines_base = table_config.table_settings["explicit_horizontal_lines"].copy()
                table_config.update_hlines(list_horizontal_lines)
                table_config.set_hstrategy("explicit")
                # Now, parse the entire left page
                table_content = page_content.extract_table(table_config.table_settings)
                # reset the table_settings explicit_horizontal_lines
                table_config.set_hlines(hlines_base)
                table_config.set_hstrategy("text")
                # Get header index - it will be either 0 or 1
                header_index = get_header_index(table_content, table_headers)
                df_page_content = pd.DataFrame(table_content[header_index + 1:], columns=columns)
                df_file_content = df_file_content.append(df_page_content, ignore_index=True)
                logger.info('Parsed page: {}'.format(i))
            except Exception as e:
                logger.exception("Error at page: {} - {}".format(i, e))
                pass
    return df_file_content
## [END] Function to load the file & parse the tables from each page

## [START] Function to clean up the data and do further processing
# Clean up the extracted text
def process_file_content(df_file_content):
    # Basic common cleaning /  processing
    # Replace all None and '' values with NaN
    df_file_content = df_file_content.replace(['', None], np.nan)
    # df_file_content = df_file_content.replace('\n', ' ', regex=True)
    # Drop rows with all NaN values
    df_file_content = df_file_content.dropna(how='all')

    #  If the value in column DrugName equals '(continued)' then replace it by ''
    df_file_content['DrugName'].replace('(continued)', np.nan, inplace=True)

    # Drop the rows wehre the column BillingUnit is not in [np.nan, 'ea', 'ml', 'gm', 'each']
    df_file_content = df_file_content[df_file_content.BillingUnit.isin(
        [np.nan, 'ea', 'ml', 'm', 'Ml', 'gm', 'each', 'Each', '60each', 'capsule', 'gram','package', 'kit'])].reset_index(drop=True)

    # Clean up bad data!!!!!!!
    # Find rows where the column DrugName of df_file_content equals 'Meningococcal' or 'Measles, Mumps, and'
    df_file_content.loc[df_file_content.loc[(df_file_content['DrugName'].isin(['Meningococcal', 'Measles, Mumps, and'])) & \
        (df_file_content['Dosage'] == 'injection')].index, 'Dosage'] = 'Injection'

    # If DrugName equals 'Ferrous Sulfate' and the Dosage column is np.NaN then replace it by 'N/A' 
    df_file_content.loc[df_file_content.loc[
        (df_file_content['DrugName'].isin(
            ['Ferrous Sulfate', 'Vinorelbine Tartrate', 'Aluminum and', 'Aluminum Hydroxide', 
            'Aluminum Hydroxide,', 'Calcium Carbonate and', 'Bismuth Subsalicylate', 'Heparin Lock Flush'])) 
        & (df_file_content['Dosage'].isna())
    ].index, 'Dosage'] = 'N/A'

    # Handle overlap of overflowing Drug name with the 2nd or higher item in the dosage column
    index_drug_overflow_1 = list(df_file_content[(df_file_content['DrugName'] == 'saccharate) *') & 
                        (df_file_content['Dosage'] == 'Capsules, Extended')].index)
    for index in index_drug_overflow_1:
        if str(df_file_content.iloc[index - 1, 0]) != 'nan':
            df_file_content.iloc[index - 1, 0] += ' ' + df_file_content.iloc[index, 0]
            df_file_content.iloc[index, 0] = np.nan

    index_drug_overflow_2 = list(df_file_content[(df_file_content['DrugName'] == 'and Caffeine') & 
                        (df_file_content['Dosage'] == 'Suppositories')].index)
    for index in index_drug_overflow_2:
        if str(df_file_content.iloc[index - 1, 0]) != 'nan':
            df_file_content.iloc[index - 1, 0] += ' ' + df_file_content.iloc[index, 0]
            df_file_content.iloc[index, 0] = np.nan

    index_drug_overflow_3 = list(df_file_content[(df_file_content['DrugName'] == 'or citrate free) *') & 
                        (df_file_content['Dosage'] == 'syringes')].index)
    for index in index_drug_overflow_3:
        if str(df_file_content.iloc[index - 1, 0]) != 'nan':
            df_file_content.iloc[index - 1, 0] += ' ' + df_file_content.iloc[index, 0]
            df_file_content.iloc[index, 0] = np.nan

    index_drug_overflow_4 = list(df_file_content[(df_file_content['DrugName'] == 'Sulfamethoxazole') & 
                        (df_file_content['Dosage'] == 'Double strength')].index)
    for index in index_drug_overflow_4:
        if str(df_file_content.iloc[index - 1, 0]) != 'nan':
            df_file_content.iloc[index - 1, 0] += ' ' + df_file_content.iloc[index, 0]
            df_file_content.iloc[index, 0] = np.nan

    return df_file_content
## [END] Function to clean up the data and do further processing

## [START] Overflow texts in strength, and dosage columns
# Overflow text for columns Strength and Dosage
overflow_text_dosages= ['suspension','injection','Suspension Pen','solution *','injectable','solution/drops', 'Release Injectable', 
                        'release','Suspension, drops','Solution/Drops','(ultramicrosize','only)','package of four',
                        'injection, single','dose delivery','system','for injection','(macrocrystals only)','(monohydrate/',
                        'macrocrystals)','capsules','release (includes','film coated tablets)','(starter pack)','concentrate',
                        'Dose Pack','(25 mg & 100 mg','capsules)','dual chamber','syringe','(Eligard®)','tablets','aqueous susp.',
                        'Pack','(42 tablets/pack)', 'chlorofluorocarbons','as the propellant)','inhalation,', 'inhalation',
                        'premixed','inhalation with','inhalation device','Inhalation *','(without','without','chlorofluorocarbons',
                        'as the propellant','inhalation with','inhalation device','disintegrating +','Release +','long-acting +',
                        'extended-release','autoinjector','disintegrating','(kit or refill)','Disintegrating','System',
                        'Patch *','dose vial','dose vial or ampule','Pellets *','Long-Acting *','Tablets extended',
                        'release','coated +','Extended Release +','release +','Release (24-hour)','Release (12-hour)','Release',
                        'release *','Patch +','once-a-day +','(no long-acting','forms) +','release*','21/2/5','Combination','Packet',
                        '(28 Tablets/Packet)', 'Packet (28','Tablets/Packet)','Packet (21','(tri-phasic)','Packet', '(21 Tablets/Packet)',
                        '(28 Tablets/Packet)', 'Monophasic Packet', '(28 tablets/packet)', 'once-weekly patch','twice-weekly patch','sensitizing base',
                        'excluded)','Pen *','Acting +',
                        'concentrated, USP','aspart protamine','70% and Insulin','aspart 30%','Insulin aspart','protamine 70% and',
                        'insulin aspart 30%','lispro protamine','75% and insulin','lispro 25%','50% and insulin','lispro 50%','Injector *','combination packet',
                        'prefilled syringe','effervescent +','long acting +','without','as the propellant *', 'Solution with Sofzia', 'Preservative', 
                        'solution, single use', 'vials', 'Ointment *','Month Box (53','release (24-hour)', 'dose vial)', 'Dose Vial)', 'gel', 'Irrigant', 
                        'Month Box (53 tablets/box)', 'Chron.-UC-HS', 'PS-UV-ADOL HS']
overflow_text_strength1 =['units/vial','mcg deliverable)','PFU/ml','from 35-Tablet Kit','tablets from 49-',
                        'piggyback', 'tablet kit','tablets from 98-','tablet kit','7 x inert','BioNtech'] #'0.025 mg','1000 mg',
overflow_text_strength2=['PFU/ml','ampule']
## [END] Overflow texts in strength, and dosage columns

## [START] Function to handle overflow texts in code1, strength, and dosage columns
def process_overflow_text(df_file_content):
    # Make a copy of the dataframe and do further file specific processing on it
    file_content = df_file_content.copy()
    # First of all, handle the Code1 column
    drop_indices_code1 = []
    for (index, row) in file_content.iterrows():
        # Drop row from file_content & Append Code1 to previous row if the columns 
        # DrugName, Dosage, Strength, BillingUnit, UM are all NaN
        if (row[0] is np.NaN) and (row[1] is np.NaN) and (row[2] is np.NaN) and \
            (row[3] is np.NaN) and (row[4] is np.NaN) and (row[5] is not np.NaN):
            drop_indices_code1.append(index)
            while index in drop_indices_code1:
                index -= 1
            file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])

    file_content = file_content.drop(drop_indices_code1).reset_index(drop=True)

    # Handle Overflow texts in Strength column
    drop_indices_overflow = {0:[], 1:[], 2:[]}
    for (index, row) in file_content.iterrows():
        # If the Strength is in overflow_text_strength1 or overflow_text_strength2, then append 
        # DrugName, Dosage, Strength, Code1 to the previous row
        if (row[2] in overflow_text_strength1) or (row[2] in overflow_text_strength2):
            drop_indices_overflow[0].append(index)
            while index in drop_indices_overflow[0]:
                index -= 1
            if row[0] is not np.NaN:
                file_content.iloc[index, 0] = ' '.join([str(file_content.iloc[index, 0]), row[0]])
            if row[1] is not np.NaN:
                file_content.iloc[index, 1] = ' '.join([str(file_content.iloc[index, 1]), row[1]])
            join_char = '' if str(file_content.iloc[index, 2]).endswith(('/', ',')) else ' '
            file_content.iloc[index, 2] = join_char.join([str(file_content.iloc[index, 2]), row[2]])
            if row[5] is not np.NaN:
                file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
                # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
            if (row[3] is not np.NaN) and (str(file_content.iloc[index, 3]).strip() != str(row[3]).strip()):
                file_content.iloc[index, 3] = ' '.join([str(file_content.iloc[index, 3]), row[3]])
            if (row[4] is not np.NaN):
                join_char = ', ' if file_content.iloc[index, 4] is not np.NaN else ''
                file_content.iloc[index, 4] = join_char.join([str(file_content.iloc[index, 4]), row[4]])

    # unique_drop_indices = list(set(list(itertools.chain(*drop_indices_overflow[0]))))
    file_content = file_content.drop(drop_indices_overflow[0]).reset_index(drop=True)

    # Handle Overflow texts in Dosage column
    for (index, row) in file_content.iterrows():
        # If the Dosage is in overflow_text_dosage, then append 
        # DrugName, Dosage, Strength, Code1 to the previous row
        if row[1] in overflow_text_dosages:
            drop_indices_overflow[1].append(index)
            while index in drop_indices_overflow[1]:
                index -= 1
            if row[0] is not np.NaN:
                file_content.iloc[index, 0] = ' '.join([str(file_content.iloc[index, 0]), row[0]])
            file_content.iloc[index, 1] = ' '.join([str(file_content.iloc[index, 1]), row[1]])
            if row[2] is not np.NaN:
                # join_char = '' if str(file_content.iloc[index, 2]).endswith('/') else ' '
                file_content.iloc[index, 2] = row[2] if pd.isna(file_content.iloc[index, 2]) else  '; '.join([str(file_content.iloc[index, 2]), row[2]])
            if row[5] is not np.NaN:
                file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
                # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
            if (row[3] is not np.NaN) and (str(file_content.iloc[index, 3]).strip() != str(row[3]).strip()):
                file_content.iloc[index, 3] = ' '.join([str(file_content.iloc[index, 3]), row[3]])
            if (row[4] is not np.NaN):
                join_char = ', ' if file_content.iloc[index, 4] is not np.NaN else ''
                file_content.iloc[index, 4] = join_char.join([str(file_content.iloc[index, 4]), row[4]])

    # unique_drop_indices2 = list(set(list(itertools.chain(*drop_indices_overflow[1]))))
    file_content = file_content.drop(drop_indices_overflow[1]).reset_index(drop=True)

    return file_content
## [END] Function to handle overflow texts in code1, strength, and dosage columns

## [START] Function to roll up rows that are overflown from previous rows
def rollup_overflown_rows(file_content):
    drop_indices = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    for (index, row) in file_content.iterrows():
        # Drop row from file_content & Append Code1 to previous row if the columns 
        # DrugName, Dosage, Strength, BillingUnit, UM are all NaN
        if (row[0] is np.NaN) and (row[1] is np.NaN) and (row[2] is np.NaN) and \
            (row[3] is np.NaN) and (row[4] is np.NaN) and (row[5] is not np.NaN):
            drop_indices[0].append(index)
            while index in list(itertools.chain(*drop_indices.values())):
                index -= 1
            file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
        # Drop row from file_content & Append Dosage to previous row if the columns 
        # DrugName, Strength, BillingUnit, UM are all NaN, but Dosage isn't NaN
        elif (row[0] is np.NaN) and (row[2] is np.NaN) and \
            (row[3] is np.NaN) and (row[4] is np.NaN) and (row[1] is not np.NaN):
            drop_indices[1].append(index)
            tmp_index = index
            while index in list(itertools.chain(*drop_indices.values())):
                index -= 1
            if pd.isna(file_content.iloc[index, 1]):
                # Found a NaN in the previous row, so skip appending
                drop_indices[1].remove(tmp_index)
            else:
                file_content.iloc[index, 1] = ' '.join([str(file_content.iloc[index, 1]), row[1]])
                if row[5] is not np.NaN:
                    file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
                    # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
        # Drop row from file_content & Append BillingUnit to previous row if the columns 
        # DrugName, Dosage, Strength, UM are all NaN, but BillingUnit isn't NaN
        elif (row[0] is np.NaN) and (row[1] is np.NaN) and \
            (row[2] is np.NaN) and (row[4] is np.NaN) and (row[3] is not np.NaN):
            drop_indices[2].append(index)
            tmp_index = index
            while index in list(itertools.chain(*drop_indices.values())):
                index -= 1
            if pd.isna(file_content.iloc[index, 3]):
                # Found a NaN in the previous row, so skip appending
                drop_indices[2].remove(tmp_index)
            else:
                if (row[3] is not np.NaN) and (str(file_content.iloc[index, 3]).strip() != str(row[3]).strip()):
                    file_content.iloc[index, 3] = ' '.join([str(file_content.iloc[index, 3]), row[3]])
                if row[5] is not np.NaN:
                    file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
                    # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
        # Drop row from file_content & Append DrugName to previous row if the columns
        # Dosage, Strength, BillingUnit, UM are all NaN, but DrugName isn't NaN
        elif (row[1] is np.NaN) and (row[2] is np.NaN) and \
            (row[3] is np.NaN) and (row[4] is np.NaN) and (row[0] is not np.NaN):
            drop_indices[3].append(index)
            tmp_index = index
            while index in list(itertools.chain(*drop_indices.values())):
                index -= 1
            if pd.isna(file_content.iloc[index, 0]):
                # Found a NaN in the previous row, so skip appending
                drop_indices[3].remove(tmp_index)
            else:
                file_content.iloc[index, 0] = ' '.join([str(file_content.iloc[index, 0]), row[0]])
                if row[5] is not np.NaN:
                    file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
                # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
        # Drop row from file_content & Append DrugName & Code1 to previous row if the columns
        # Dosage, Strength, BillingUnit, UM are all NaN, but DrugName & Code1 aren't NaN
        elif (row[0] is np.NaN) and (row[1] is np.NaN) and \
            (row[3] is np.NaN) and (row[4] is np.NaN) and (row[2] is not np.NaN):
            drop_indices[4].append(index)
            tmp_index = index
            while index in list(itertools.chain(*drop_indices.values())):
                index -= 1
            if pd.isna(file_content.iloc[index, 2]):
                # Found a NaN in the previous row, so skip appending
                drop_indices[4].remove(tmp_index)
            else:
                if (row[2] not in overflow_text_strength1) and (row[2] not in overflow_text_strength2):
                    join_char = '' if str(file_content.iloc[index, 2]).endswith((',','/')) else '; '
                else:
                    join_char = '' if str(file_content.iloc[index, 2]).endswith((',','/')) else ' '

                file_content.iloc[index, 2] = join_char.join([str(file_content.iloc[index, 2]), row[2]])
                if row[5] is not np.NaN:
                    file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
                    # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
        # Drop row from file_content & Append UM & Code1 to previous row if the columns
        # DrugName, Dosage, Strength, BillingUnit are all NaN, but UM & Code1 aren't NaN
        elif (row[0] is np.NaN) and (row[1] is np.NaN) and \
            (row[2] is np.NaN) and (row[3] is np.NaN) and (row[4] is not np.NaN):
            drop_indices[5].append(index)
            tmp_index = index
            while index in list(itertools.chain(*drop_indices.values())):
                index -= 1
            if pd.isna(file_content.iloc[index, 4]):
                # Found a NaN in the previous row, so skip appending
                drop_indices[5].remove(tmp_index)
            else:
                join_char = '' if str(file_content.iloc[index, 4]).strip().endswith((',')) else ', '
                file_content.iloc[index, 4] = join_char.join([str(file_content.iloc[index, 4]), row[4]])
                if row[5] is not np.NaN:
                    file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])

    unique_drop_indices = list(set(list(itertools.chain(*drop_indices.values()))))
    file_content = file_content.drop(unique_drop_indices).reset_index(drop=True)
    return file_content
## [END] Function to roll up rows that are overflown from previous rows

## [START] Function to handle further specific overflow texts in the dosage column
def handle_dosage_overflow_rows(file_content):
    # Fix the issue with 'Solution' in the Dosage column
    # If the Dosage column has value in ['Solution', 'solution', 'Solution or'] and 
    # the previous row of the Dosage column has value in ['Ophthalmic', 'Powder for oral']
    # then append the value in the Dosage column to the previous row of the Dosage column
    # and drop the current row of the Dosage column
    dosage_merge_indices = file_content[file_content.Dosage.isin(['Ointment', 'ointment', 'Solution', 'solution', 
                                                                'Solution or', 'Solution with Sofzia Preservative', 
                                                                'Injection', 'Injection Kit', 'Capsules', 'Capsules *', 
                                                                'Capsules +', 'Suppositories', 'suppositories'])].index
    dosage_drop_indices = []
    for index in dosage_merge_indices:
        row = file_content.iloc[index]
        # while index in dosage_merge_indices:
        #     index -= 1
        if file_content.iloc[index - 1, 1] in ['Ophthalmic', 'Powder for oral', 'Powder for', 'Subcutaneous', 
                                            'Extended-Release', 'Tablets or', 'Tablets for Oral', 'Rectal', 
                                            'Vaginal']:
            dosage_drop_indices.append(index)
            file_content.iloc[index - 1, 1] = ' '.join([str(file_content.iloc[index - 1, 1]), row[1]])
            if row[0] is not np.NaN:
                join_char = ' ' if str(file_content.iloc[index - 1, 0]) is not np.nan else ''
                file_content.iloc[index - 1, 0] = join_char.join([str(file_content.iloc[index - 1, 0]), row[0]])
            if row[2] is not np.NaN:
                join_char = '' if str(file_content.iloc[index - 1, 2]).endswith((',','/')) else '; '
                file_content.iloc[index - 1, 2] = join_char.join([str(file_content.iloc[index - 1, 2]), row[2]])
            if row[5] is not np.NaN:
                file_content.iloc[index - 1, 5] = row[5] if pd.isna(file_content.iloc[index - 1, 5]) else  ' '.join([str(file_content.iloc[index - 1, 5]), row[5]])
                # file_content.iloc[index - 1, 5] = ' '.join([str(file_content.iloc[index - 1, 5]), row[5]])
            if (row[3] is not np.NaN) and (str(file_content.iloc[index - 1, 3]).strip() != str(row[3]).strip()):
                file_content.iloc[index - 1, 3] = ' '.join([str(file_content.iloc[index - 1, 3]), row[3]])
            if (row[4] is not np.NaN):
                join_char = ', ' if file_content.iloc[index - 1, 4] is not np.NaN else ''
                file_content.iloc[index - 1, 4] = join_char.join([str(file_content.iloc[index - 1, 4]), row[4]])

    file_content = file_content.drop(dosage_drop_indices).reset_index(drop=True)

    # If the Dosage is Extended-Release and the previous row of the Dosage column has value in ['Capsules,']
    # then append the value in the Dosage column to the previous row of the Dosage column
    # and drop the current row of the Dosage column
    dosage_merge_indices2 = file_content[file_content.Dosage.isin(['Extended-Release'])].index

    dosage_drop_indices2 = []
    for index in dosage_merge_indices2:
        row = file_content.iloc[index]
        # while index in dosage_merge_indices:
        #     index -= 1
        if file_content.iloc[index - 1, 1] in ['Capsules,']:
            dosage_drop_indices2.append(index)
            file_content.iloc[index - 1, 1] = ' '.join([str(file_content.iloc[index - 1, 1]), row[1]])
            if row[0] is not np.NaN:
                join_char = ' ' if str(file_content.iloc[index - 1, 0]) is not np.nan else ''
                file_content.iloc[index - 1, 0] = join_char.join([str(file_content.iloc[index - 1, 0]), row[0]])
            if row[2] is not np.NaN:
                join_char = '' if str(file_content.iloc[index - 1, 2]).endswith((',','/')) else '; '
                file_content.iloc[index - 1, 2] = join_char.join([str(file_content.iloc[index - 1, 2]), row[2]])
            if row[5] is not np.NaN:
                file_content.iloc[index - 1, 5] = row[5] if pd.isna(file_content.iloc[index - 1, 5]) else  ' '.join([str(file_content.iloc[index - 1, 5]), row[5]])
                # file_content.iloc[index - 1, 5] = ' '.join([str(file_content.iloc[index - 1, 5]), row[5]])
            if (row[3] is not np.NaN) and (str(file_content.iloc[index - 1, 3]).strip() != str(row[3]).strip()):
                file_content.iloc[index - 1, 3] = ' '.join([str(file_content.iloc[index - 1, 3]), row[3]])
            if (row[4] is not np.NaN):
                join_char = ', ' if file_content.iloc[index - 1, 4] is not np.NaN else ''
                file_content.iloc[index - 1, 4] = join_char.join([str(file_content.iloc[index - 1, 4]), row[4]])

    file_content = file_content.drop(dosage_drop_indices2).reset_index(drop=True)

    # If the Dosage is 'Tablets' and BillingUnit is Na and the previous row of the Dosage column has value in ['Titration Pack']
    # then append the value in the Dosage column to the previous row of the Dosage column
    # and drop the current row of the Dosage column
    dosage_merge_indices3 = file_content.loc[(file_content['Dosage'].isin(['Tablets'])) 
        & (file_content['BillingUnit'].isna())].index
    dosage_drop_indices3 = []
    for index in dosage_merge_indices3:
        row = file_content.iloc[index]
        # while index in dosage_merge_indices:
        #     index -= 1
        if file_content.iloc[index - 1, 1] in ['Titration Pack']:
            dosage_drop_indices3.append(index)
            file_content.iloc[index - 1, 1] = ' '.join([str(file_content.iloc[index - 1, 1]), row[1]])
            if row[0] is not np.NaN:
                join_char = ' ' if str(file_content.iloc[index - 1, 0]) is not np.nan else ''
                file_content.iloc[index - 1, 0] = join_char.join([str(file_content.iloc[index - 1, 0]), row[0]])
            if row[2] is not np.NaN:
                join_char = ' ' #if str(file_content.iloc[index - 1, 2]).endswith((',','/')) else '; '
                file_content.iloc[index - 1, 2] = join_char.join([str(file_content.iloc[index - 1, 2]), row[2]])
            if row[5] is not np.NaN:
                file_content.iloc[index - 1, 5] = row[5] if pd.isna(file_content.iloc[index - 1, 5]) else  ' '.join([str(file_content.iloc[index - 1, 5]), row[5]])
                # file_content.iloc[index - 1, 5] = ' '.join([str(file_content.iloc[index - 1, 5]), row[5]])
            if (row[3] is not np.NaN) and (str(file_content.iloc[index - 1, 3]).strip() != str(row[3]).strip()):
                file_content.iloc[index - 1, 3] = ' '.join([str(file_content.iloc[index - 1, 3]), row[3]])
            if (row[4] is not np.NaN):
                join_char = ', ' if file_content.iloc[index - 1, 4] is not np.NaN else ''
                file_content.iloc[index - 1, 4] = join_char.join([str(file_content.iloc[index - 1, 4]), row[4]])

    file_content = file_content.drop(dosage_drop_indices3).reset_index(drop=True)
    return file_content
## [END] Function to handle further specific overflow texts in the dosage column

## [START] Function to merge rows on the drugname column
def merge_on_drugname(file_content):
    # Get rows in file_content where DrugName is not nan but Dosage is nan
    drug_name_merge_indices = file_content[(file_content['DrugName'].notnull()) & (file_content['Dosage'].isnull())].index
    for index in drug_name_merge_indices:
        row = file_content.iloc[index]
        while index in drug_name_merge_indices:
            index -= 1
        file_content.iloc[index, 0] = ' '.join([str(file_content.iloc[index, 0]), str(row[0])])
        if row[2] is not np.NaN:
            join_char = '' if str(file_content.iloc[index, 2]).endswith((',','/')) else '; '
            file_content.iloc[index, 2] = join_char.join([str(file_content.iloc[index, 2]), row[2]])
        if row[5] is not np.NaN:
            file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
            # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
        if (row[3] is not np.NaN) and (str(file_content.iloc[index, 3]).strip() != str(row[3]).strip()):
            file_content.iloc[index, 3] = ' '.join([str(file_content.iloc[index, 3]), row[3]])
        if (row[4] is not np.NaN):
            join_char = ', ' if file_content.iloc[index, 4] is not np.NaN else ''
            file_content.iloc[index, 4] = join_char.join([str(file_content.iloc[index, 4]), row[4]])

    file_content = file_content.drop(drug_name_merge_indices).reset_index(drop=True)
    return file_content
## [END] Function to merge rows on the drugname column

## [START] Function to merge rows on the Strength column
def merge_on_strength(file_content):
    # Get rows in file_content where DrugName is nan and Dosage is nan
    strength_merge_indices = file_content[(file_content['DrugName'].isnull()) & (file_content['Dosage'].isnull())].index
    for index in strength_merge_indices:
        row = file_content.iloc[index]
        while index in strength_merge_indices:
            index -= 1
        if row[2] is not np.NaN:
            join_char = '' if str(file_content.iloc[index, 2]).endswith((',','/')) else '; '
            file_content.iloc[index, 2] = join_char.join([str(file_content.iloc[index, 2]), row[2]])
        if row[5] is not np.NaN:
            file_content.iloc[index, 5] = row[5] if pd.isna(file_content.iloc[index, 5]) else  ' '.join([str(file_content.iloc[index, 5]), row[5]])
            # file_content.iloc[index, 5] = ' '.join([str(file_content.iloc[index, 5]), row[5]])
        if (row[3] is not np.NaN) and (str(file_content.iloc[index, 3]).strip() != str(row[3]).strip()):
            file_content.iloc[index, 3] = ' '.join([str(file_content.iloc[index, 3]), row[3]])
        if (row[4] is not np.NaN):
            join_char = ', ' if file_content.iloc[index, 4] is not np.NaN else ''
            file_content.iloc[index, 4] = join_char.join([str(file_content.iloc[index, 4]), row[4]])
    file_content = file_content.drop(strength_merge_indices).reset_index(drop=True)
    return file_content
## [END] Function to merge rows on the drugname column

## [START] Function to do remaining processing on Drugname & Dosage columns
def process_drugname_dosage_cols(file_content):
    # Drop the rows where DrugName contains "See:"
    file_content = file_content[~file_content['Dosage'].str.contains('See:')].reset_index(drop=True)

    # Replace 'N/A' in Dosage column with np.NaN
    file_content.loc[file_content.loc[file_content['Dosage'] == 'N/A'].index, 'Dosage'] = np.NaN
    # Now almost all the cases are covered. Jusall t need to forward fill the DrugName where DrugName is nan
    file_content['DrugName'].fillna(method='ffill', inplace=True)
    return file_content
## [END] Function to do remaining processing on Drugname & Dosage columns

## [START] Function to build Drug Description column
def build_drug_description(file_content):
    # convert all the nan values to empty string
    file_content = file_content.fillna('')
    # Append the columns DrugName, Dosage, and Strength into one column
    file_content['DrugDescription'] = file_content[['DrugName', 'Dosage', 'Strength']].apply(lambda x: ' '.join(x), axis=1)
    # Cleanup DrugDescription column
    # If the DrugDescription column contains '*' then remove it and append it to the end of the value in DrugDescription column
    file_content['DrugDescription'] = file_content['DrugDescription'].apply(lambda x: x.replace('*', '') + ' *' if str(x).find('*') != -1 else x)
    file_content['DrugDescription'] = file_content['DrugDescription'].apply(lambda x: x.replace('+', '') + ('+' if str(x).find('*') != -1 else ' +') if str(x).find('+') != -1 else x)

    #Replace \u2018 with ' from all columns
    file_content['DrugDescription'] = file_content['DrugDescription'].apply(lambda x: x.replace('\u2018', "'"))
    file_content['DrugDescription'] = file_content['DrugDescription'].apply(lambda x: x.replace('\u2019', "'"))
    file_content['DrugDescription'] = file_content['DrugDescription'].apply(lambda x: x.replace('\u2013', "-"))
    file_content['DrugDescription'] = file_content['DrugDescription'].apply(lambda x: x.replace('\u00ae', ""))
    file_content['Strength'] = file_content['Strength'].apply(lambda x: x.replace('\u2018', "'"))
    file_content['Strength'] = file_content['Strength'].apply(lambda x: x.replace('\u2019', "'"))
    file_content['Strength'] = file_content['Strength'].apply(lambda x: x.replace('\u2013', "-"))
    file_content['Strength'] = file_content['Strength'].apply(lambda x: x.replace('\u00ae', ""))

    # Remove the extra spaces
    file_content['DrugDescription'] = file_content['DrugDescription'].apply(lambda x: ' '.join(x.split()))
    return file_content
## [END] Function to build Drug Description column

## [START] Function to build Code1_merged column by forward filling Code1 column for same DrugName
def build_code1_merged(file_content):
    # Merge the values in Code1 column for the same drug
    # Join all the values in Code1 column for the same drug in column DrugName
    file_content['Code1_Merged'] = file_content.groupby('DrugName')['Code1'].transform(lambda x: ' '.join(x))
    # Remove the extra spaces
    file_content['Code1_Merged'] = file_content['Code1_Merged'].apply(lambda x: ' '.join(x.split()))
    # Replace \u2018 with ' from Code1_Merged columns
    file_content['Code1_Merged'] = file_content['Code1_Merged'].apply(lambda x: x.replace('\u201c', "\""))
    file_content['Code1_Merged'] = file_content['Code1_Merged'].apply(lambda x: x.replace('\u201d', "\""))
    file_content['Code1_Merged'] = file_content['Code1_Merged'].apply(lambda x: x.replace('\u00ae', ""))
    file_content['Code1_Merged'] = file_content['Code1_Merged'].apply(lambda x: x.replace('\u2019', "'"))
    file_content['Code1_Merged'] = file_content['Code1_Merged'].apply(lambda x: x.replace('\u203a', ">"))
    file_content['Code1_Merged'] = file_content['Code1_Merged'].apply(lambda x: x.replace('\u00e7', "c"))
    return file_content
## [END] Function to build Code1_merged column by forward filling Code1 column for same DrugName

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
    strength = [''] + x[2].split(';')
    # Remove * and + and fix other characters in the columns drug & dosage
    drug = cleanup_text(x[0])
    dosage = cleanup_text(x[1])
    urls = []
    base_strength = strength[1] if len(strength) > 1 else ''
    for index, i in enumerate(strength):
        # Remove * and + and fix other characters in the strength column
        i = cleanup_text(i)
        url = 'https://rxnav.nlm.nih.gov/REST/rxcui.json?name=' + drug + '+' + dosage + ('' if i == '' else '+') + i + '&search=2'
        url = '+'.join(url.split())
        urls.append(url)

    base_strength_partial1 = base_strength.split('/')[0] if len(base_strength.split('/')) > 1 else ''
    base_strength_partial2 = base_strength.split(',')[0] if len(base_strength.split(',')) > 1 else ''
    base_strength = cleanup_text(base_strength)
    base_strength_partial1 = cleanup_text(base_strength_partial1)
    base_strength_partial2 = cleanup_text(base_strength_partial2)
    approx_url = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug + '+' + dosage + ('' if base_strength == '' else '+') + base_strength + '&maxEntries=1'
    approx_url = '+'.join(approx_url.split())
    urls.append(approx_url + '&option=1')
    approx_url_dose_cleaned1 = ""
    if base_strength_partial1 != '':
        approx_url_dose_cleaned1 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug + '+' + dosage + '+' + base_strength_partial1 + '&maxEntries=1'
        approx_url_dose_cleaned1 = '+'.join(approx_url_dose_cleaned1.split())
        urls.append(approx_url_dose_cleaned1 + '&option=1')
    
    approx_url_dose_cleaned2 = ""
    if base_strength_partial2 != '':
        approx_url_dose_cleaned2 = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug + '+' + dosage + '+' + base_strength_partial2 + '&maxEntries=1'
        approx_url_dose_cleaned2 = '+'.join(approx_url_dose_cleaned2.split())
        urls.append(approx_url_dose_cleaned2 + '&option=1')

    approx_url_drug_dosage = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug + '+' + dosage + '&maxEntries=1'
    approx_url_drug_dosage = '+'.join(approx_url_drug_dosage.split())
    urls.append(approx_url_drug_dosage + '&option=1')

    approx_url_drug = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug + '&maxEntries=1'
    approx_url_drug = '+'.join(approx_url_drug.split())
    urls.append(approx_url_drug + '&option=1')

    drug_name_len = len(drug.split())
    drug_name_partial = ' '.join(drug.split()[:2]) if drug_name_len > 2 else ''
    approx_url_drug_partial = ""
    if drug_name_partial != '':
        approx_url_drug_partial = 'https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=' + drug_name_partial + '&maxEntries=1'
        approx_url_drug_partial = '+'.join(approx_url_drug_partial.split())
        urls.append(approx_url_drug_partial + '&option=1')

    urls.append(approx_url + '&option=0')
    if approx_url_dose_cleaned1 != '':
        urls.append(approx_url_dose_cleaned1 + '&option=0')
    if approx_url_dose_cleaned2 != '':
        urls.append(approx_url_dose_cleaned2 + '&option=0')
    
    urls.append(approx_url_drug_dosage + '&option=0')
    urls.append(approx_url_drug + '&option=0')
    if approx_url_drug_partial != '':
        urls.append(approx_url_drug_partial + '&option=0')
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

# Alternate DrugNames for some drugs
_dict_alt_drugs = {
                    'Pneumococcal': 'Streptococcus pneumoniae serotype',
                    'Echothiophate': 'ecothiopate', 
                    'Cyanocobalamin': 'vitamin B12',
                    'Meperidine': 'meperidine hydrochloride',
                    'Phytonadione': 'vitamin K1', 
                    'Testosterone': 'testosterone Injectable Suspension',
                    'Delaviridine': 'delavirdine'
}

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

    if fuzz.partial_ratio(_dict_alt_drugs.get(drug_name.split()[0]), rxnorm_drug) > 70:
        return True
    logger.info("{} - Alt Drug -- partial ratio - {}".format(drug_name, fuzz.partial_ratio(_dict_alt_drugs.get(drug_name.split()[0]), rxnorm_drug)))

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

# Generate the JSON file
def extract_output(file_content):
    _list_dict = []
    for (index, row) in file_content.iterrows():
        _dict = {}
        _plans = {}
        # _drug_details = {}
        for url in row[7]:
            try:
                response = requests.get(url) #requests.get(url, params=payload)
                if 'approximateTerm.json' in url:
                    _dict["rxnorm_id"] = response.json()['approximateGroup']['candidate'][0]['rxcui']
                    # Add check for Drug Name if URL has 'approximateTerm.json'
                    url_check = 'https://rxnav.nlm.nih.gov/REST/rxcui/' + _dict["rxnorm_id"] + '/property.json?propName=RxNorm%20Name'
                    return_value = requests.get(url_check)
                    return_val= return_value.json()['propConceptGroup']['propConcept'][0]['propValue']
                    if check_drug_match(get_cleanedup_drug_name(row[6]), return_val):
                        # found a match
                        logger.info("Drug name: {}, response: {}, Found Match.".format(row[6]+": " + url, response.json()))
                        break
                    else:
                        # no match found
                        logger.error("Drug name: {}, response: {}, No Match Found.".format(row[6]+": " + url, response.json()))
                        _dict["rxnorm_id"] = ""
                else:
                    _dict["rxnorm_id"] = response.json()['idGroup']['rxnormId'][0]
                    # Break if rxnorm_id was found!!!
                    if _dict["rxnorm_id"] != "":
                        break
            except Exception as e:
                logger.error("Drug name: {}, response: {}, Exception: {}".format(row[6]+": " + url, response.json(), e))
                _dict["rxnorm_id"] = ""

        _dict["drug_name"] = row[6]
        _plans["drug_tier"] = 'default'
        # if '*' in str(row[6]):
        if "prior authorization" in str(row[8]).lower():
            _plans["prior_authorization"] = True
            # _plans["extra"] = "Extras"
        else:
            _plans["prior_authorization"] = False
            # _plans["extra"] = ""
        if row[4] == '':
            _plans["step_therapy"] = False
            _plans["quantity_limit"] = False
        else:
            # parse QL
            if ("QL" in str(row[4])) or ("quantity limit" in str(row[8]).lower()):
                # print(row[2].split('; ')[0])
                # Commenting out the value extraction for now
                # ql_plan = get_plan(row[2], 'QL')
                ## _plans["quantity_limit"] = [''.join(t) for t in re.findall(r'\((.+?)\)|(\w)', row[1].split('; ')[0].split(', ')[1])][-1:]
                # _plans["quantity_limit"] = ql_plan.removeprefix('QL  (').removesuffix(')')
                _plans["quantity_limit"] = True
            else:
                _plans["quantity_limit"] = False
            # parse ST
            if ("ST" in str(row[4])) or ("step therapy" in str(row[8]).lower()):
                _plans["step_therapy"] = True
            else:
                _plans["step_therapy"] = False
        _plans["extra"] = row[8]
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
                  int(config_details['TABLE_PROPERTIES']['vline4']),
                  int(config_details['TABLE_PROPERTIES']['vline5']),
                  int(config_details['TABLE_PROPERTIES']['vline6'])]
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
    list_header_cols = [None, None, 'Package Size', 'Unit',	'Type', None]
    list_cols = ['DrugName', 'Dosage', 'Strength', 'BillingUnit', 'UM', 'Code1']
    page_range = range(11, 243)
    try:
        # Load the table configuration for the pdf file
        table_config = load_table_config()
        # Load the pdf file
        file_content = parse_pdf(path, list_cols, list_header_cols, page_range, table_config)
        # Process the file content
        # Basic cleaning
        processed_file_content = process_file_content(file_content)
        # handle overflow texts in code1, strength, and dosage columns
        processed_file_content = process_overflow_text(processed_file_content)
        # Roll up rows that are overflown from previous rows
        processed_file_content = rollup_overflown_rows(processed_file_content)
        # Handle further specific overflow texts in the dosage column
        processed_file_content = handle_dosage_overflow_rows(processed_file_content)
        # merge rows on the drugname column
        processed_file_content = merge_on_drugname(processed_file_content)
        # merge rows on the Strength column
        processed_file_content = merge_on_strength(processed_file_content)
        # do remaining processing on Drugname & Dosage columns
        processed_file_content = process_drugname_dosage_cols(processed_file_content)
        # Build DrugDescription column
        processed_file_content = build_drug_description(processed_file_content)
        # Build the URL column
        processed_file_content['URL'] = processed_file_content[['DrugName', 'Dosage', 'Strength']].apply(lambda x: build_urls(x), axis=1)
        # Build Code1_Merged column
        processed_file_content = build_code1_merged(processed_file_content)
        # Extract the output
        logger.info('Loaded the processed data. Building JSON now!')
        extract_output(processed_file_content)
        # Generate the DrugName vs. RxNorm Id map
        build_Drug_Name_RxNormId_Map()
        logger.info('Finished processing!!!')
    except Exception as e:
        logger.error("Error: {}".format(e))
        pass