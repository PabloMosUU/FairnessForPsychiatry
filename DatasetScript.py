#!/usr/bin/env python
# coding: utf-8

# This notebook generates the dataset(s) to be used for classification, starting from the datasets that contain the basic information about patients

# In[ ]:


import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing


# In[ ]:


# data
DATA_DIR = '/media/bigdata/10. Stages/3. Afgerond/2020-08 Jesse Kuiper/'
TRANQ_DIAZEPAM_FILE = 'TranquilizerDiazepamFactor.csv'
SAVE_OUTPUT = False


# In[ ]:


# selecting usefull var 
admission_columns = ["OpnameID", "PseudoID","AfdelingOmschrijving", "Opnamedatum",  "Ontslagdatum",  "Opnametijd", 
                     "Ontslagtijd",    "Spoed",  "EersteOpname",  "Geslacht", "Leeftijd_opname", 
                     "OpnamestatusOmschrijving", "Duur"]

administering_columns = ["PseudoID", "VoorschriftID", "ATC_code_omschr", "Medicijnnaam_ingevuld","Dosis", 
                         "Eenheid", "ToedienDatum", "ToedienTijd", "Toegediend", "DosisVerbruikt", 
                         "DosisOrigineel", "ToedieningIsOpgeschort", "NietToegediend"  ]

dbc_columns = [ "PseudoID", "dbcnummer","Startdatum", "Einddatum" ,"hoofddiagnose_groep", "zorgvraagzwaarte", 
               "MeervoudigeProblematiekInd", "persoonlijkheidsstoornis", "Opname", "DiagnoseDatum" ]

violent_columns = ["PseudoID", "hantering_datum", "begin_incident"]

patient_columns = ["PseudoID", "Leeftijd_startdatum_dossier" ]

def drop_by_pseudo_id(df: pd.DataFrame, pseudo_ids: list) -> pd.DataFrame:
    return df[df['PseudoID'].apply(lambda x: x not in pseudo_ids)].reset_index()

# # Load the original datasets

# In[ ]:


# load opnamens
admission = pd.read_csv(DATA_DIR + "werkbestanden-opnames/latest/werkbestand_opnames.csv", sep=';', 
                        usecols=admission_columns)


# In[ ]:


# load administered
administering = pd.read_csv(DATA_DIR + "werkbestanden-medicatie/latest/werkbestand_medicatie_toediening.csv", sep=';',
                        decimal=',', usecols=administering_columns)


# In[ ]:


# load dbc
dbc = pd.read_csv(DATA_DIR + "werkbestanden-dbc/latest/werkbestand_dbc.csv", sep=';', usecols=dbc_columns)


# In[ ]:


# load map
violent = pd.read_csv(DATA_DIR + "werkbestanden-map/latest/werkbestand_map.csv", sep=';', usecols=violent_columns)


# In[ ]:


#load patient or patient uniek
patient = pd.read_csv(DATA_DIR + "werkbestanden-patient/latest/werkbestand_patient_uniek.csv", sep=';', 
                  usecols=patient_columns)


# In[ ]:


# load conversion factors from various tranquilizers to diazepam
tranq_diazepam = {k:v for k,v in pd.read_csv(TRANQ_DIAZEPAM_FILE, sep=';')[['tranquilizer', 'factor']].values}


# # Filter datasets and fix null values

# ### Admissions

# In[ ]:


# remove incomplete admissions 
admission = admission[admission.OpnamestatusOmschrijving == "Ontslagen"]


# In[ ]:


# check for na values
assert admission.isnull().sum().sum() == 0


# In[ ]:


# change Opnamedatum Ontslagdatum to date times        
admission["OpnamedatumTijd"] = pd.to_datetime(admission["Opnamedatum"] + ' ' + admission["Opnametijd"])
admission["OntslagdatumTijd"] = pd.to_datetime(admission["Ontslagdatum"] + ' ' + admission["Ontslagtijd"])


# In[ ]:


# DateTime checks for the agression and the dbc
# these datetimes make sure it only covers the 
# If the duration of admission is less than the time check, it will take the whole admission
admission["DaysF"] = np.where(admission["Duur"]>= 14, 14, admission["Duur"])
admission["DaysP"] = np.where(admission["Duur"]>= 3, 3, admission["Duur"])

# create date time checks #these should have a max value
admission["DateTimeCheckF"] = admission["OpnamedatumTijd"] + pd.to_timedelta(admission["DaysF"], unit='d')
admission["DateTimeCheckP"] = admission["OpnamedatumTijd"] + pd.to_timedelta(admission["DaysP"], unit='d')

afd = ["Klin. Affectieve & Psychotische stoorn.","Klinische Acuut & Intensieve Zorg","Klin.Acuut & Intensieve. Zorg Jeugd", "Klin Diagn & Vroege Psychose"]

# AfdelingOmschrijving
adm_afd = pd.get_dummies(admission["AfdelingOmschrijving"])
adm_afd_sel = pd.concat([admission, adm_afd[afd]], axis=1)


# In[ ]:


del adm_afd_sel["AfdelingOmschrijving"]


# In[ ]:


# create admission1 - 8663, admission2 - 3192 and admission3 - 4685
admission1 = adm_afd_sel.copy()
admission2 = adm_afd_sel[admission["Duur"]> 14].reset_index().copy()
admission3 = adm_afd_sel[admission["Duur"]> 3].reset_index().copy()


# In[ ]:


print('All discharged admissions from the four nursing wards:', len(admission1))
print('Only admissions lasting 3 or more days:', len(admission3))
print('Only admissions lasting 14 or more days:', len(admission2))


# ### Diagnoses

# In[ ]:


# change NaN in hoofddiagnose_groep to "Lege hoofddiagnose" as this is already a variable in the table with the same meaning
dbc["hoofddiagnose_groep"] = dbc["hoofddiagnose_groep"].replace(np.nan, "Lege hoofddiagnose", regex=True)
dbc["hoofddiagnose_groep"] = dbc["hoofddiagnose_groep"].str.replace("Bijkomende codes/geen diagnose","Lege hoofddiagnose")


# In[ ]:


# create a diagnose date
# this is a limitation to be mentioned in the paper
def get_diagnosis_date(row):
    if type(row.DiagnoseDatum) == str:
        return row.DiagnoseDatum
    elif type(row.Einddatum) == str:
        return row.Einddatum
    else:
        return row.Startdatum

dbc["diagnosis_date"] = dbc.apply(lambda row: get_diagnosis_date(row), 1)
# Uncomment the following line to keep only diagnoses with a DiagnoseDatum
# dbc = dbc[dbc['DiagnoseDatum'].notnull()].reset_index(drop=True)

# In[ ]:


dbc.drop(columns=['DiagnoseDatum', 'Einddatum', 'Startdatum'], inplace=True)


# In[ ]:


# Drop rows that do not have a PseudoID, as there is no way to couple them with admissions
dbc = dbc[dbc['PseudoID'].notnull()].reset_index(drop=True)


# In[ ]:


assert dbc.isnull().sum().sum() == 0


# ### Violence incidents

# In[ ]:


# Drop rows that do not have a PseudoID, as there is no way to couple them with admissions
violent = violent[violent['PseudoID'].notnull()].reset_index(drop=True)


# In[ ]:


# change hantering_datum to date time with begin_incident
violent["hantering_datumTijd"] = pd.to_datetime(violent["hantering_datum"] + ' ' + violent["begin_incident"])


# In[ ]:


assert violent.isnull().sum().sum() == 0


# ### Patient

# In[ ]:


# Select only patients for which we also have admissions
patient = admission[['PseudoID']].merge(patient, on='PseudoID', how='left').drop_duplicates()


# In[ ]:


assert len(patient) == admission['PseudoID'].nunique()


# In[ ]:


assert patient.isnull().sum().sum() == 0


# ### Administered medication

# In[ ]:


# we are only interested in administered medicine
administering = administering[administering["Toegediend"]==1]


# In[ ]:


# list of agreed upon tranquilizers
administering = administering[administering["ATC_code_omschr"].isin(tranq_diazepam)]


# In[ ]:


# 1 administering does not contain a toediendatum and toedientijd (corrupted data)
administering_drop_patients = []
# Uncomment the following line to remove that patient
# administering_drop_patients = administering[administering['ToedienDatum'].isnull()]['PseudoID'].tolist()
administering = administering.dropna(subset=["ToedienDatum"])


# In[ ]:


assert administering.isnull().sum().sum() == 0


# In[ ]:


# create datetime
administering["ToedienDatumTijd"] = pd.to_datetime(administering["ToedienDatum"] + ' ' + administering["ToedienTijd"])
administering.drop(columns=['ToedienDatum', 'ToedienTijd'], inplace=True)


# In[ ]:


# merge administering
# based on the dosis
def InDiazepam(row):
    omschr = row['ATC_code_omschr']
    dosis = row['Dosis']
    return tranq_diazepam[omschr] * dosis


# In[ ]:


administering["DoseDiazepam"] = administering.apply(InDiazepam, axis=1)
administering.drop(columns=['ATC_code_omschr', 'Dosis'], inplace=True)


# # Join tables

# ### Patient onto Admissions

# In[ ]:


# first add the patient data to the opnamens
def get_adm_pat(frame: pd.DataFrame) -> pd.DataFrame:
    adm_adm1 = frame.merge(patient, on="PseudoID", how="left")
    assert np.sum(adm_adm1.isnull().sum().values) == 0
    assert len(adm_adm1) == len(frame)
    return adm_adm1    


# In[ ]:


# merge patient data with admission data
adm_pat1 = get_adm_pat(admission1)  

adm_pat2 = get_adm_pat(admission2)

adm_pat3 = get_adm_pat(admission3)


# ### Violence incidents onto Admissions

# In[ ]:


# merge violence data, this is where the datasets start to differ (time period where you count violence incidents)
# give violence data an unique identifier
violent["IncidentID"] = np.arange(len(violent))


# In[ ]:


# merge dataset with mapdata
map_adm1 = adm_pat1.merge(violent[["PseudoID", "IncidentID", "hantering_datumTijd"]], how="left", on="PseudoID")
map_adm2 = adm_pat2.merge(violent[["PseudoID", "IncidentID", "hantering_datumTijd"]], how="left", on="PseudoID")
map_adm3 = adm_pat3.merge(violent[["PseudoID", "IncidentID", "hantering_datumTijd"]], how="left", on="PseudoID")


# In[ ]:


# whole dataset
opname_ids, incidents_during_admission, incidents_before_admission = [], [], []
for opname_id, grp in map_adm1.groupby("OpnameID"):
    # opname_id -> single OpnameID from adm_map table                                                                                                                                                      
    # grp -> a dataframe containing only the rows that have OpnameID == opname_id                                                                                                                          
    opname_ids.append(opname_id)
    if len(grp[grp["IncidentID"].notnull()]) == 0:
        # No incidents                                                                                                                                                                                     
        incidents_during_admission.append(0)
        incidents_before_admission.append(0)
    else:
        # At least one incident                                                                                                                                                                                                                                                                                                                                    
        n_during = len(grp[grp.apply(lambda row: row.OntslagdatumTijd >= row.hantering_datumTijd >= row.OpnamedatumTijd, 1)])
        n_before = len(grp[grp.apply(lambda row: row.hantering_datumTijd < row.OpnamedatumTijd, 1)])
        incidents_during_admission.append(n_during)
        incidents_before_admission.append(n_before)

# Create a new incidents-counts dataframe
admission_incidents = pd.DataFrame()
admission_incidents["OpnameID"] = opname_ids
admission_incidents["incidents_during_admission"] = incidents_during_admission
admission_incidents["incidents_before_admission"] = incidents_before_admission
assert len(admission_incidents) == len(adm_pat1)

# Merge the incidents-counts dataframe onto opnames
adm_map1 = adm_pat1.merge(admission_incidents, on="OpnameID", how="inner")
assert len(adm_map1) == len(adm_pat1)


# In[ ]:


#  dataset2 
opname_ids, incidents_during_admission, incidents_before_admission = [], [], []
for opname_id, grp in map_adm2.groupby("OpnameID"):
    # opname_id -> single OpnameID from adm_map table                                                                                                                                                      
    # grp -> a dataframe containing only the rows that have OpnameID == opname_id                                                                                                                          
    opname_ids.append(opname_id)
    if len(grp[grp["IncidentID"].notnull()]) == 0:
        # No incidents                                                                                                                                                                                     
        incidents_during_admission.append(0)
        incidents_before_admission.append(0)
    else:
        # At least one incident                                                                                                                                                                                                                                                                                                                                    
        n_during = len(grp[grp.apply(lambda row: row.DateTimeCheckF >= row.hantering_datumTijd >= row.OpnamedatumTijd, 1)])
        n_before = len(grp[grp.apply(
            lambda row: row.hantering_datumTijd < row.OpnamedatumTijd, 1
        )])
        incidents_during_admission.append(n_during)
        incidents_before_admission.append(n_before)

# Create a new incidents-counts dataframe
admission_incidents = pd.DataFrame()
admission_incidents["OpnameID"] = opname_ids
admission_incidents["incidents_during_admission"] = incidents_during_admission
admission_incidents["incidents_before_admission"] = incidents_before_admission
assert len(admission_incidents) == len(adm_pat2)

# Merge the incidents-counts dataframe onto opnames
adm_map2 = adm_pat2.merge(admission_incidents, on="OpnameID", how="inner")
assert len(adm_map2) == len(adm_pat2)


# In[ ]:


#  dataset3 
opname_ids, incidents_during_admission, incidents_before_admission = [], [], []
for opname_id, grp in map_adm3.groupby("OpnameID"):
    # opname_id -> single OpnameID from adm_map table                                                                                                                                                      
    # grp -> a dataframe containing only the rows that have OpnameID == opname_id                                                                                                                          
    opname_ids.append(opname_id)
    if len(grp[grp["IncidentID"].notnull()]) == 0:
        # No incidents                                                                                                                                                                                     
        incidents_during_admission.append(0)
        incidents_before_admission.append(0)
    else:
        # At least one incident                                                                                                                                                                                                                                                                                                                                    
        n_during = len(grp[grp.apply(lambda row: row.DateTimeCheckP >= row.hantering_datumTijd >= row.OpnamedatumTijd, 1)])
        n_before = len(grp[grp.apply(lambda row: row.hantering_datumTijd < row.OpnamedatumTijd, 1)])
        incidents_during_admission.append(n_during)
        incidents_before_admission.append(n_before)

# Create a new incidents-counts dataframe
admission_incidents = pd.DataFrame()
admission_incidents["OpnameID"] = opname_ids
admission_incidents["incidents_during_admission"] = incidents_during_admission
admission_incidents["incidents_before_admission"] = incidents_before_admission
assert len(admission_incidents) == len(adm_pat3)

# Merge the incidents-counts dataframe onto opnames
adm_map3 = adm_pat3.merge(admission_incidents, on="OpnameID", how="inner")
assert len(adm_map3) == len(adm_pat3)


# ### DBC onto Admissions

# In[ ]:


#function to merge adm_dbc with the right dbc

hoofddiagnoses = [
    "Aandachtsstoornis",
    "Andere problemen die een reden van zorg kunnen zijn",
    "Angststoornissen",
    "Autismespectrumstoornis",
    "Bipolaire stoornissen",
    "Cognitieve stoornissen",
    "Depressieve stoornissen",
    "Dissociatieve stoornissen",
    "Gedragsstoornissen",
    "Middelgerelateerde en verslavingsstoornissen",
    "Obsessieve-compulsieve en verwante stoornissen",
    "Overige psychische stoornissen",
    "Overige stoornissen op zuigelingen of kinderleeftijd",
    "Persoonlijkheidsstoornissen",
    "Psychische stoornissen door een somatische aandoening",
    "Schizofrenie en andere psychotische stoornissen",
    "Somatisch-symptoomstoornis en verwante stoornissen",
    "Trauma- en stressorgerelateerde stoornissen",
    "Voedings- en eetstoornissen",]

def get_adm_dbc(dataset: int, frame: pd.DataFrame) -> pd.DataFrame:
    dataset_end_date_column = {1: 'OntslagdatumTijd', 2: 'DateTimeCheckF', 3: 'DateTimeCheckP'}
    end_date = dataset_end_date_column[dataset]
    adm_dbc = frame.merge(dbc, how='inner', on='PseudoID')
    
    # Opnamedatum, Ontslagdatum, Startdatum, Einddatum
    adm_dbc["diagnosis_date"]= pd.to_datetime(adm_dbc["diagnosis_date"])
    
    adm_dbc['DbcWithin'] = adm_dbc.apply(lambda row: int(row.OpnamedatumTijd <= row.diagnosis_date <= row[end_date]), 1)
    
    # selecting only the rows that are DbcWithin
    dbc_within = adm_dbc[adm_dbc.DbcWithin != 0]
    
    # create columns of interesting data, this could not be dont in one go
    dbc_within2 = dbc_within[["OpnameID", "hoofddiagnose_groep"]]
    
    # only select with a diagnoses
    dbc_within2 = dbc_within2[dbc_within2["hoofddiagnose_groep"] != "Lege hoofddiagnose"]
    dbc_within2 = dbc_within2.drop_duplicates()
    
    # create columns of the values of hoofddiagnose_groep
    diagnoses = pd.get_dummies(dbc_within2["hoofddiagnose_groep"])

    dbc_within3 = pd.concat([dbc_within2, diagnoses], axis=1)
    
    if "Overige psychische stoornissen" not in dbc_within3:
        dbc_within3["Overige psychische stoornissen"] = np.nan
            
    dbc_hd = pd.DataFrame(dbc_within3.groupby('OpnameID')[hoofddiagnoses].max())
    
    # mvpi and persoonlijkheidsstoornis
    dbc_mp = pd.DataFrame(dbc_within.groupby('OpnameID')[['MeervoudigeProblematiekInd', 
                                                          'persoonlijkheidsstoornis']].max().reset_index())
    
    # zorgvraagtevraag
    dbc_zv = pd.DataFrame(dbc_within.assign(ZorgvraagzwaarteMin = dbc_within['zorgvraagzwaarte'].abs(),
                                            ZorgvraagzwaarteMax = dbc_within['zorgvraagzwaarte'].abs())
                          .groupby('OpnameID')
                          .agg({'ZorgvraagzwaarteMin':'min','ZorgvraagzwaarteMax':'max'}).reset_index())
    assert len(dbc_mp) == dbc_within.OpnameID.nunique() and len(dbc_zv) == len(dbc_mp)
    
    #merge interesting columns
    ds_dbc = dbc_mp.merge(dbc_zv, how="inner", on="OpnameID")
    
    #check that you didn't insert unnecessary rows
    assert len(dbc_mp) == len(ds_dbc)
    
    #merge hoofddiagnoses
    ds1_dbc = ds_dbc.merge(dbc_hd, how="inner", on="OpnameID")
    
    #merge interesting columns with adm_map
    adm_dbc = frame.merge(ds1_dbc, how= "left", on="OpnameID")
    
    #check that you still have the same number of admissions
    assert len(adm_dbc) == len(frame)
    
    #fill na values
    adm_dbc = adm_dbc.replace(np.nan,0)
    
    # test for missing values
    assert np.sum(adm_dbc.isnull().sum().values) == 0
    return adm_dbc


# In[ ]:


# merge dbc based on date contraints with adm_map1
adm_dbc1 = get_adm_dbc(1,adm_map1)


# In[ ]:


# merge dbc based on date contraints with adm_map2
adm_dbc2 = get_adm_dbc(2,adm_map2)


# In[ ]:


# merge dbc based on date contraints with adm_map3
adm_dbc3 = get_adm_dbc(3,adm_map3)


# ### Administered medication onto Admissions

# In[ ]:


# create past and future tranq prescriptions

var_pat = ["PseudoID",
        "Spoed",
        "EersteOpname",
        "Geslacht",
        "Leeftijd_opname",
        "Duur",
        "Leeftijd_startdatum_dossier",
        "incidents_during_admission",
        "incidents_before_admission",
        "MeervoudigeProblematiekInd",
        "persoonlijkheidsstoornis",
        "ZorgvraagzwaarteMin",
        "ZorgvraagzwaarteMax",
        "DoseDiazepam",
        "DoseDiazepamPre",
        "DoseDiazepamPost",   
]

UsefulVariables = var_pat + afd + hoofddiagnoses

# left join on PseudoID
# has no end date restriction for the post prescriptions
# TODO: rename this method
def get_adm_dbc(dataset: int, frame ):
    dataset_end_date_column = {1: 'OntslagdatumTijd', 2: 'DateTimeCheckF', 3: 'DateTimeCheckP'}
    end_date = dataset_end_date_column[dataset]
    adm_adm1 = frame.merge(administering[["PseudoID", "ToedienDatumTijd", "DoseDiazepam" ]], how="left", 
                           on="PseudoID")  

    # remove rows where the ToedienDatumTijd is outside of the OpnamedatumTijd and OntslagdatumTijd
    adm_adm1 = adm_adm1[adm_adm1["ToedienDatumTijd"] >= adm_adm1["OpnamedatumTijd"]]
    adm_adm1 = adm_adm1[adm_adm1["ToedienDatumTijd"] <= adm_adm1["OntslagdatumTijd"]]

    # create past and future prescriptions
    adm_adm1["Past"] = adm_adm1.apply(lambda row: int(row.ToedienDatumTijd <= row[end_date] and                                                       row.ToedienDatumTijd >= row.OpnamedatumTijd), 1)

    # diazepam prescribed in the past
    adm_adm1["DoseDiazepamPre"] = np.where(adm_adm1["Past"]== 1, adm_adm1["DoseDiazepam"], 0 )

    # diazepam prescribed in the future
    adm_adm1["DoseDiazepamPost"] = np.where(adm_adm1["Past"]== 0, adm_adm1["DoseDiazepam"], 0 )
    
    # groupby
    groupby_columns = ["OpnameID", "OpnamedatumTijd", "OntslagdatumTijd"]
    selection_columns = ["DoseDiazepam","DoseDiazepamPre","DoseDiazepamPost"]
    adm_adm2 = pd.DataFrame(adm_adm1.groupby(groupby_columns)[selection_columns].sum())
                                      
    Dataset = frame.merge(adm_adm2, on=["OpnameID", "OpnamedatumTijd","OntslagdatumTijd"], how="left")

    #na values should be 0 because there are no prescriptions for tranquilizers
    Dataset["DoseDiazepam"] = Dataset["DoseDiazepam"].replace(np.nan, 0, regex=True)
    Dataset["DoseDiazepamPre"] = Dataset["DoseDiazepamPre"].replace(np.nan, 0, regex=True)
    Dataset["DoseDiazepamPost"] = Dataset["DoseDiazepamPost"].replace(np.nan, 0, regex=True)
    Dataset['sum'] = Dataset.apply(lambda row: row['DoseDiazepamPre'] + row['DoseDiazepamPost'], 1)
    Dataset['diff'] = Dataset.apply(lambda row: abs(row['DoseDiazepam'] - row['sum']), 1)
    Dataset['avg'] = Dataset.apply(lambda row: 0.5 * (row['DoseDiazepam'] + row['sum']), 1)
    Dataset['fracdiff'] = Dataset.apply(lambda row: row['diff'] / row['avg'] if row['avg'] > 0 else 0, 1)
    assert len(Dataset[Dataset['fracdiff'] > 0.001]) == 0
    Dataset.drop(columns=['sum', 'diff', 'avg', 'fracdiff'], inplace=True)

    # true dataset 
    DatasetWhole = Dataset[UsefulVariables]
    
    assert np.sum(DatasetWhole.isnull().sum().values) == 0
    return DatasetWhole


# In[ ]:


DatasetWhole = get_adm_dbc(1 , adm_dbc1)

del DatasetWhole["DoseDiazepamPost"]
del DatasetWhole["DoseDiazepamPre"]


# In[ ]:


Dataset14Days = get_adm_dbc(2 , adm_dbc2)

del Dataset14Days["DoseDiazepam"]


# In[ ]:


Dataset3Days = get_adm_dbc(3 , adm_dbc3)

del Dataset3Days["DoseDiazepam"]

if len(administering_drop_patients) != 0:
    DatasetWhole = drop_by_pseudo_id(DatasetWhole,
                                      administering_drop_patients)
    Dataset14Days = drop_by_pseudo_id(Dataset14Days,
                                       administering_drop_patients)
    Dataset3Days = drop_by_pseudo_id(Dataset3Days,
                                      administering_drop_patients)
        

# ### Save the dataset
# 
# To avoid overwriting unnecessarily, I included a flag at the beginning of this file

# In[ ]:

if SAVE_OUTPUT:
    Dataset14Days.to_csv(DATA_DIR + 'Dataset14Days.csv',
                         sep=';',
                         index=False)

