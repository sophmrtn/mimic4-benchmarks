import argparse
import os
import sys

from tqdm import tqdm
from utils.preprocessing import (
    assemble_episodic_data,
    clean_events,
)
from utils.subject import (
    add_hours_elapsed_to_events,
    convert_events_to_timeseries,
    get_events_for_admission,
    get_events_for_stay,
    read_diagnoses,
    read_events,
    read_stays,
)

parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
# parser.add_argument('dict_path', type=str, help='Directory to MIMIC-IV dict - needed to load item dictionary.')

# parser.add_argument('--variable_map_file', type=str,
#                     default=os.path.join(os.path.dirname(__file__), 'resources/itemid_to_variable_map.csv'),
#                     # help='CSV containing ITEMID-to-VARIABLE map.')
# parser.add_argument('--reference_range_file', type=str,
#                     default=os.path.join(os.path.dirname(__file__), 'resources/variable_ranges.csv'),
                    # help='CSV containing reference ranges for VARIABLEs.')

args, _ = parser.parse_known_args()

# var_map = read_itemid_to_variable_map(args.variable_map_file)
# variables = var_map.VARIABLE.unique()

if args.verbose:
        print(f'EXTRACTING EPISODES FROM {len(os.listdir(args.subjects_root_path))} subjects...')

failed_to_read = 0
filter_by_nb_stays = 0
filter_by_nb_events = 0

for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    dn = os.path.join(args.subjects_root_path, subject_dir)

    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except Exception:
        continue
    
    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        # diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except Exception:
        sys.stderr.write(f'Error reading data for subject: {subject_id}. Events table likely to be missing/empty. \n')
        failed_to_read +=1
        continue

    # Filter by number of stays per subject
    if stays.shape[0] < args.min_stays:
        filter_by_nb_stays+=1
        continue
    
    episodic_data = assemble_episodic_data(stays)

    if episodic_data.shape[0] == 0:
        sys.stderr.write(f'Failed to get stay data for: {subject_id}. May cause issues\n')

    # cleaning and converting to time series
    # events = map_itemids_to_labels(events, args.dict_path)
    # events = map_itemids_to_variables(events, var_map)
    variables = events.label.unique()

    # clean events
    events = clean_events(events)

    if events.shape[0] == 0:
        # no valid events for this subject
        continue
    timeseries = convert_events_to_timeseries(events, variable_column='label', variables=variables)

    # extracting separate episodes (by hospital admission since using labevents)
    for i in range(stays.shape[0]):
        # for ed data
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime) # get events during this ed stay

        if episode.shape[0] < args.min_events or episode.shape[0] < args.max_events:
            # if no data for this episode (or less than min or more than max) then continue
            # only keep stays with nb of datapoints within specific range
            continue

        episode = add_hours_elapsed_to_events(episode, intime).set_index('hours').sort_index(axis=0)

        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              f'episode{i+1}.csv'),
                                                                 index_label='stay_id')
        
        
        # for hosp data
        # hadm_id = stays.hadm_id.iloc[i]
        # admittime = stays.admittime.iloc[i]
        # dischtime = stays.dischtime.iloc[i]

        # episode = get_events_for_admission(timeseries, hadm_id, admittime, dischtime)

        # if episode.shape[0] == 0:
        #     # no data for this episode
        #     continue

        # episode = add_hours_elapsed_to_events(episode, admittime).set_index('hours').sort_index(axis=0)

        # episodic_data.loc[episodic_data.index == hadm_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
        #                                                                       f'episode{i+1}.csv'),
        #                                                          index_label='stay')

       
        columns_str = [str(x) for x in list(episode.columns)]
        columns_str = map(lambda x: "" if x == "hours" else x, columns_str)
        sorted_indices = [i[0] for i in sorted(enumerate(columns_str), key=lambda x:x[1])]
        # columns_sorted = sorted(columns_str, key=(lambda x: "" if x == "hours" else x))

        episode = episode[[episode.columns[i] for i in sorted_indices]]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, f'episode{i+1}_timeseries.csv'),
                       index_label='hours')
