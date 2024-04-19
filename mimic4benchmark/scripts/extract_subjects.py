import argparse
import os

import numpy as np

import mimic4benchmark.mimic4csv as m4c
from mimic4benchmark.util import dataframe_from_csv

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-IV CSV files.')
parser.add_argument('mimic4_path', type=str, help='Directory containing MIMIC-IV CSV files.')
parser.add_argument('mimic4_ed_path', type=str, help='Directory containing MIMIC-IV ED CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['EMAR', 'LABEVENTS'])
# parser.add_argument('--phenotype_definitions', '-p', type=str,
#                     default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
#                     help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except Exception:
    pass

patients = m4c.read_patients_table(args.mimic4_path)
admits = m4c.read_admissions_table(args.mimic4_path)
stays = m4c.read_stays_table(args.mimic4_ed_path)
if args.verbose:
    print(f'START:\n\tED STAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}')

stays = m4c.remove_stays_without_admission(stays)
if args.verbose:
    print(f'REMOVE ED STAYS WITHOUT ADMISSION:\n\tSTAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}')

stays = m4c.merge_on_subject_admission(stays, admits)
stays = m4c.merge_on_subject(stays, patients)
stays = m4c.filter_admissions_on_nb_stays(stays)
if args.verbose:
    print(f'REMOVE MULTIPLE ED STAYS PER ADMIT:\n\tSTAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}')

stays = m4c.add_age_to_stays(stays)
stays = m4c.add_ined_mortality_to_stays(stays)
stays = m4c.add_inhospital_mortality_to_stays(stays)
stays = m4c.filter_stays_on_age(stays)
if args.verbose:
    print(f'REMOVE PATIENTS AGE < 18:\n\tSTAY_IDs: {stays.stay_id.unique().shape[0]}\n\HADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}')

stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
diagnoses = m4c.read_icd_diagnoses_table(args.mimic4_path) # add longtitle to data
diagnoses = m4c.filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
m4c.count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))

# Removed phenotying
# phenotypes = m4c.add_hcup_ccs_2015_groups(diagnoses, yaml.safe_load(open(args.phenotype_definitions)))

# m4c.make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
#                                                       index=False, quoting=csv.QUOTE_NONNUMERIC)

if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(patients[['subject_id']], left_on='subject_id', right_on='subject_id')
    args.event_tables = [args.event_tables[0]]
    print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

subjects = stays.subject_id.unique()
m4c.break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
m4c.break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects)
items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['itemid'].unique()]) if args.itemids_file else None
for table in args.event_tables:
    m4c.read_events_table_and_break_up_by_subject(args.mimic4_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects)
