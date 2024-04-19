import csv
import datetime
import os

import icdmappings
import numpy as np
import pandas as pd
from tqdm import tqdm

from mimic4benchmark.util import dataframe_from_csv

ICD9 = 9

def read_patients_table(mimic4_path):
    pats = dataframe_from_csv(os.path.join(mimic4_path, 'patients.csv.gz'))
    pats = pats[['subject_id', 'gender', 'anchor_age', 'dod']]
    pats.anchor_age = pd.to_datetime(pats.anchor_age)
    pats.dod = pd.to_datetime(pats.dod)
    return pats


def read_admissions_table(mimic4_path):
    admits = dataframe_from_csv(os.path.join(mimic4_path, 'admissions.csv.gz'))
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'insurance', 'language', 'marital_status', 'race', 'edregtime', 'edouttime', 'hospital_expire_flag']]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischttime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    admits.edregtime = pd.to_datetime(admits.edregtime)
    admits.edouttime = pd.to_datetime(admits.edouttime)
    # add los column (in fractional days)
    diff = admits.dischtime - admits.admittime
    admits['los'] = diff.apply(lambda x: x.total_seconds() / datetime.timedelta(days=1).total_seconds())
    return admits


def read_stays_table(mimic4_ed_path):
    stays = dataframe_from_csv(os.path.join(mimic4_ed_path, 'edstays.csv'))
    stays.inttime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)    
    diff = stays.outtime - stays.inttime
    stays['los_ed'] = diff.apply(lambda x: x.total_seconds() / datetime.timedelta(days=1).total_seconds())
    return stays


def read_icd_diagnoses_table(mimic4_path):
    codes = dataframe_from_csv(os.path.join(mimic4_path, 'd_icd_diagnoses.csv.gz'))
    codes = codes[['icd_code', 'icd_version', 'long_title']]
    diagnoses = dataframe_from_csv(os.path.join(mimic4_path, 'diagnoses_icd.csv.gz'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='icd_code', right_on='icd_code')
    diagnoses[['subject_id', 'hadm_id', 'seq_num']] = diagnoses[['subject_id', 'hadm_id', 'seq_num']].astype(int)
    return diagnoses


def read_events_table_by_row(mimic4_path, table):
    nb_rows = {'emar': 330712484, 'labevents': 27854056, 'hcpcsevents': 4349219, 'vitalsign': 4349219}
    reader = csv.DictReader(open(os.path.join(mimic4_path, table.upper() + '.csv.gz')))
    for i, row in enumerate(reader):
        if 'hadm_id' not in row:
            row['hadm_id'] = ''
        yield row, i, nb_rows[table.lower()]


def count_icd_codes(diagnoses, convert_to_icd10:bool=True, output_path=None):
    codes = diagnoses[['icd_code', 'short_title', 'long_title']].drop_duplicates().set_index('icd_code')

    if convert_to_icd10:
        print('Converting ICD9 codes to ICD10')
        mapper = icdmappings.Mapper()
        codes['icd_code'] = codes[codes.icd_version==ICD9]['icd9_code'].apply(lambda x: mapper.map(x, source='icd9', target='icd10'))

    codes['count'] = diagnoses.groupby('icd_code')['hadm_id'].count()
    codes.count = codes.count.fillna(0).astype(int)
    codes = codes[codes.count > 0]
    if output_path:
        codes.to_csv(output_path, index_label='icd_code')
    return codes.sort_values('count', ascending=False).reset_index()


def remove_stays_without_admission(stays):
    stays = stays[(stays.disposition == "ADMITTED")]
    return stays[['subject_id', 'hadm_id', 'stay_id', 'inttime', 'outtime', 'disposition']]


def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id'], right_on=['subject_id'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])


def add_age_to_stays(stays):
    stays['age'] = stays.apply(lambda e: (e['admittime'].to_pydatetime()
                                          - e['anchor_age'].to_pydatetime()).total_seconds() / 3600.0 / 24.0 / 365.0,
                               axis=1)
    stays.loc[stays.age < 0, 'age'] = 90
    return stays


def add_inhospital_mortality_to_stays(stays):
    mortality = stays.dod.notnull() & ((stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime)))
    stays['mortality'] = mortality.astype(int)
    stays['mortality_inhospital'] = stays['mortality']
    return stays


def add_ined_mortality_to_stays(stays):
    mortality = stays.dod.notnull() & ((stays.inttime <= stays.dod) & (stays.outtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.inttime <= stays.deathtime) & (stays.outtime >= stays.deathtime)))
    stays['mortality_ined'] = mortality.astype(int)
    return stays


def filter_admissions_on_nb_stays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    to_keep = to_keep[(to_keep.stay_id >= min_nb_stays) & (to_keep.stay_id <= max_nb_stays)][['hadm_id']]
    stays = stays.merge(to_keep, how='inner', left_on='hadm_id', right_on='hadm_id')
    return stays


def filter_stays_on_age(stays, min_age=18, max_age=np.inf):
    # must have already added age to stays table
    stays = stays[(stays.age >= min_age) & (stays.AGE <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates(), how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'stay_id'])


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass

        stays[stays.subject_id == subject_id].sort_values(by='inttime').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass

        diagnoses[diagnoses.subject_id == subject_id].sort_values(by=['stay_id', 'seq_num'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)


def read_events_table_and_break_up_by_subject(mimic4_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    obs_header = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valueuom']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats:
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    nb_rows_dict = {'emar': 330712484, 'labevents': 27854056, 'hcpcsevents': 4349219, 'vitalsign': 4349219}
    nb_rows = nb_rows_dict[table.lower()]

    for row, _, _ in tqdm(read_events_table_by_row(mimic4_path, table), total=nb_rows,
                                                        desc=f'Processing {table} table'):

        if (subjects_to_keep is not None) and (row['subject_id'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['itemid'] not in items_to_keep):
            continue

        row_out = {'subject_id': row['subject_id'],
                   'hadm_id': row['hadm_id'],
                   'stay_id': '' if 'stay_id' not in row else row['stay_id'],
                   'charttime': row['charttime'],
                   'itemid': row['itemid'],
                   'value': row['value'],
                   'valueuom': row['valueuom']}
        if data_stats.curr_subject_id not in ('', row['subject_id']):
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['subject_id']

    if data_stats.curr_subject_id != '':
        write_current_observations()
