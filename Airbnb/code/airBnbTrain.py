# coding:UTF-8

"""
  @Date 2018-04-16
  @author tracy
"""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_users_path = '../data/train_users_2.csv'
test_users_path = '../data/test_users.csv'
sessions_path = '../data/sessions.csv'
# Note: age_gender_bkts.csv and countries.csv files are not used.

# -----Loading data----#
# train_users
train_users = pd.read_csv(train_users_path)
target = train_users['country_destination']
# note the target then drop it!
train_users.drop('country_destination', axis=1)

# test_users
test_users = pd.read_csv(test_users_path)
# note the test ids wait for next step split
id_test = test_users['id']

# sessions
sessions = pd.read_csv(sessions_path)
sessions['id'] = sessions['user_id']
sessions.drop('user_id', axis=1)

print ('###################Data loaded###################')
# ----prepare session data----- #
# fill null
sessions.action = sessions.action.fillna('NAN')
sessions.action_type = sessions.action_type.fillna('NAN')
sessions.action_detail = sessions.action_detail.fillna('NAN')
sessions.device_type = sessions.device_type.fillna('NAN')

# action value has too low frequency are changed to OTHER

act_freq = 100  # Threshold for frequency
# np.unique returns one array,but zip need 2 inputs. So just want a '*' to split to 2 arrays
act = dict(zip(*np.unique(sessions.action, return_counts=True)))
sessions.action = sessions.action.apply(lambda x: 'OTHER' if act[x] < act_freq else x)

# argsort is sort the value in the array and return the value mapped index
# argsort()return a dict key is the key grouped and value is the array index
f_act = sessions.action.value_counts().argsort()
f_act_detail = sessions.action_detail.value_counts().argsort()
f_act_type = sessions.action_type.value_counts().argsort()
f_dev_type = sessions.device_type.value_counts().argsort()

# group by all session data by each user
dgr_sess = sessions.groupby(['id'])

# Loop dgr_sess to created all the features
samples = []
count = 0
ln = len(dgr_sess)
for g in dgr_sess:
    if count % 1000 == 0:
        print ("%s form %s " % (count, ln))
    # the index 1 is the data of each user
    gr = g[1]
    l = []
    # the index 0 is the id of each user
    l.append(g[0])
    # first feature is the actions each user makes
    l.append(len(gr))
    sev = gr.secs_elapsed.fillna(0).values  # fill secs_elapsed

    # 2.action features
    # 2.1 how many times each action occurs.and number of unique values \ mean \std
    c_act = [0] * len(f_act)
    for i, v in enumerate(gr.action.values):
        # v is the action name.So f_act[v] returns the index of action unique counts array
        # c_act can position the each action times
        c_act[f_act[v]] += 1

    _, c_act_uqc = np.unique(gr.action.values, return_counts=True)
    c_act += [len(c_act_uqc), np.mean(c_act_uqc), np.std(c_act_uqc)]
    l += c_act

    # action_detail features
    # (how many times each value occurs, numb of unique values, mean and std)
    c_act_detail = [0] * len(f_act_detail)
    for i, v in enumerate(gr.action_detail.values):
        c_act_detail[f_act_detail[v]] += 1
    _, c_act_det_uqc = np.unique(gr.action_detail.values, return_counts=True)
    c_act_detail += [len(c_act_det_uqc), np.mean(c_act_det_uqc), np.std(c_act_det_uqc)]
    l += c_act_detail

    # action_type features
    # (how many times each value occurs, numb of unique values, mean and std
    # + log of the sum of secs_elapsed for each value)
    l_act_type = [0] * len(f_act_type)
    c_act_type = [0] * len(f_act_type)
    for i, v in enumerate(gr.action_type.values):
        l_act_type[f_act_type[v]] += sev[i]
        c_act_type[f_act_type[v]] += 1
    l_act_type = np.log(1 + np.array(l_act_type)).tolist()
    _, c_act_type_uqc = np.unique(gr.action_type.values, return_counts=True)
    c_act_type += [len(c_act_type_uqc), np.mean(c_act_type_uqc), np.std(c_act_type_uqc)]
    l = l + c_act_type + l_act_type

    # device_type features
    # (how many times each value occurs, numb of unique values, mean and std)
    c_dev_type = [0] * len(f_dev_type)
    for i, v in enumerate(gr.device_type.values):
        c_dev_type[f_dev_type[v]] += 1
    c_dev_type.append(len(np.unique(gr.device_type.values)))
    _, c_dev_type_uqc = np.unique(gr.device_type.values, return_counts=True)
    c_dev_type += [len(c_dev_type_uqc), np.mean(c_dev_type_uqc), np.std(c_dev_type_uqc)]
    l += c_dev_type

    # secs_elapsed features
    l_secs = [0] * 5
    l_log = [0] * 15
    if len(sev) > 0:
        # Simple statistics about the secs_elapsed values.
        l_secs[0] = np.log(1 + np.sum(sev))
        l_secs[1] = np.log(1 + np.mean(sev))
        l_secs[2] = np.log(1 + np.std(sev))
        l_secs[3] = np.log(1 + np.median(sev))
        l_secs[4] = l_secs[0] / float(l[1])

        # value group in 15 intervals. Compute the value of number
        log_sev = np.log(1 + sev).astype(int)
        l_log = np.bincount(log_sev, minlength=15).tolist()
    l = l + l_secs + l_log
    # above value l just one sample of session group
    samples.append(l)
    count += 1

# after build a sample features. create a data_frame with computed values
col_names = []
for i in range(len(samples[0]) - 1):
    col_names.append('c_' + str(i))

samples = np.array(samples)
samp_ar = samples[:, 1:].astype(np.float16)
samp_id = samples[:, 0]  # The first element in obs is the id of the sample.

# creating the dataframe
df_agg_sess = pd.DataFrame(samp_ar, columns=col_names)
df_agg_sess['id'] = samp_id
df_agg_sess.index = df_agg_sess.id

# ------------------------Working on train and test data------------------------#
print('Working on users data...')
# concat and fill na
print('Shape train_users = %s, Shape test_users = %s' % (train_users.shape, test_users.shape))
df_tt = pd.concat(train_users, test_users, ignore_index=True)
print('Shape df_tt = %s' % df_tt.shape)
df_tt.index = df_tt.id
df_tt = df_tt.fillna(-1)
df_tt = df_tt.replace('-unknown-', -1)

# -------------------------Creating features for train+test----------------------#
# Removing date_first_booking
df_tt = df_tt.drop(['date_first_booking'], axis=1)
# number of nulls
df_tt = np.array([sum(r == -1) for r in df_tt.values])
# date_account_created
# Computing year, month, day, week_number, weekday
dac = np.vstack(df_tt['date_account_created'].astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
# year
df_tt['dac_y'] = dac[:, 0]
# month
df_tt['dac_m'] = dac[:, 1]
# day
df_tt['dac_d'] = dac[:, 2]
dac_dates = [datetime(x[0], x[1], x[2]) for x in dac]
# isocalendar  return 3-tuple, (ISO year, ISO week number, ISO weekday)
df_tt['dac_wn'] = np.array([d.isocalendar()[1] for d in dac_dates])
# weekday returns the day's week.Monday is 1 and Sunday is 7.
df_tt['dac_w'] = np.array([d.weekday() for d in dac_dates])
# One hot code
df_tt_wd = pd.get_dummies(df_tt['dac_w'], prefix='dac_w')
df_tt = df_tt.drop(['date_account_created', 'dac_w'], axis=1)
df_tt = pd.concat((df_tt, df_tt_wd), axis=1)

# timestamp_first_active
# (Computing year, month, day, hour, week_number, weekday)
tfa = np.vstack(df_tt.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4], x[4:6], x[6:8], x[8:10], x[10:12], x[12:14]]))).values)
df_tt['tfa_y'] = tfa[:, 0]
df_tt['tfa_m'] = tfa[:, 1]
df_tt['tfa_d'] = tfa[:, 2]
df_tt['tfa_h'] = tfa[:, 3]
tfa_dates = [datetime(x[0], x[1], x[2], x[3], x[4], x[5]) for x in tfa]
df_tt['tfa_wn'] = np.array([d.isocalendar()[1] for d in tfa_dates])
df_tt['tfa_w'] = np.array([d.weekday() for d in tfa_dates])
df_tt_wd = pd.get_dummies(df_tt.tfa_w, prefix='tfa_w')
df_tt = df_tt.drop(['timestamp_first_active', 'tfa_w'], axis=1)
df_tt = pd.concat((df_tt, df_tt_wd), axis=1)

# timespans between dates
# (Computing absolute number of seconds of difference between dates, sign of the difference)
df_tt['dac_tfa_secs'] = np.array([np.log(1 + abs((dac_dates[i] - tfa_dates[i]).total_seconds())) for i in range(len(dac_dates))])
df_tt['sig_dac_tfa'] = np.array([np.sign((dac_dates[i] - tfa_dates[i]).total_seconds()) for i in range(len(dac_dates))])

# age
# if age between 1990~2000, it maybe just the year of their birthday.So just use 2014(the game year) minutes it.
av = df_tt['age'].values
av = np.where(np.logical_and(av < 2000, av > 1990), 2014 - av, av)
av = np.where(np.logical_and(av < 14, av > 0), 4, av)  # Using specific value=4 for age values below 14
av = np.where(np.logical_and(av < 2016, av > 2010), 9, av)  # This is the current year instead of age (using specific value = 9)
av = np.where(av > 99, 110, av)  # Using specific value=110 for age values above 99
df_tt['age'] = av

interv = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]


def get_interv_value(age):
    iv = 20
    for i in range(len(interv)):
        if age < interv[i]:
            iv = i
            break
    return iv


# generate age to a range list and dummies the age
df_tt['age_interv'] = df_tt.age.apply(lambda x: get_interv_value(x))
df_tt_ai = pd.get_dummies(df_tt.age_interv, prefix='age_interv')
df_tt = df_tt.drop(['age_interv'], axis=1)
df_tt = pd.concat((df_tt, df_tt_ai), axis=1)

# One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_tt_dummy = pd.get_dummies(df_tt[f], prefix=f)
    df_tt = df_tt.drop([f], axis=1)
    df_tt = pd.concat((df_tt, df_tt_dummy), axis=1)

# -----------------Merging train-test with session data-----------------#
df_all = pd.merge(df_tt, df_agg_sess, how='left')
df_all = df_all.drop(['id'], axis=1)
df_all = df_all.fillna(-2)  # Missing features for samples without session data.
# All types of null
df_all['all_null'] = np.array([sum(r < 0) for r in df_all.values])

# ---------------Computing X, y and X_test ---------------#
# target list is train list.Each row is a sample.At the first we concat the  train and test \
# than the row number 0~len(target) is the train set and len(target)~ end is the test set
piv_train = len(target)
vals = df_all.values
le = LabelEncoder()
X = vals[:piv_train]
y = le.fit_transform(target.values)
X_test = vals[piv_train:]
print('Shape X = %s, Shape X_test = %s' % (X.shape, X_test.shape))
X.to_cvs('../data/clean_X.cvs')
y.to_cvs('../data/clean_y.cvs')
X_test.to_cvs('../data/X_test.cvs')
