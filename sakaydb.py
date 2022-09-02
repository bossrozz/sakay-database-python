import os.path as path   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt_format = '%H:%M:%S,%d-%m-%Y'
sep = ','

trips_cols = [
              'trip_id',
              'driver_id',
              'pickup_datetime',
              'dropoff_datetime',
              'passenger_count',
              'pickup_loc_id',
              'dropoff_loc_id',
              'trip_distance',
              'fare_amount'
             ]
drivers_cols = [
                'driver_id',
                'last_name',
                'given_name'
              ]
locs_cols = [
            'location_id',
            'loc_name'
           ]
trips_dtypes = {
                'trip_id': int,
                'driver_id': int,
                'pickup_datetime': 'datetime64[ns]',
                'dropoff_datetime': 'datetime64[ns]',
                'passenger_count': int,
                'pickup_loc_id': int,
                'dropoff_loc_id': int,
                'trip_distance': float,
                'fare_amount': float
               }
drivers_dtypes = {
                  'driver_id': int,
                  'last_name': str,
                  'given_name': str
                }
locs_dtypes = {
              'location_id': int,
              'loc_name': str
              }

def check_driver(driver):
        
        special_characters = '''!\"#$%&'()*+-/:;<=>?@[\]^_`{|}~'''
        if type(driver) is not str or driver.strip() == '': 
            return 'Name should not be empty or must be in string format'
        if len(driver.split(',')) != 2:
            return 'Invalid name format should be (First Name, Last Name)'
        if any(c in special_characters for c in driver):
            return 'Name contains special characters'
        return None


def check_dt(dt, dt_col): 
    
    if type(dt) is not str: 
        return (f'{dt_col} should be in string format')
    if pd.isnull(pd.to_datetime(dt.strip(), format=dt_format, errors='coerce')): 
        return (f'{dt_col} should be in valid format')
    return None


def check_pass_cnt(cnt):
 
    if type(cnt) is not int: 
        return 'Passenger count should be integer value'
    if cnt < 0:
        return 'Passenger count should be a positive value >= 0'
    return None


def check_loc(loc, loc_col):

    special_characters = '''!\"%&'()+-/:;<=>?@[\]^_`{|}~'''
    if type(loc) is not str or loc.strip() == '': 
        return(f'{loc_col} should not be empty or must be in string format')   
    elif any(c in special_characters for c in loc):
            return 'Location contains special characters'
    return None


def check_trip_d(val):
    
    if type(val) is not float and type(val) is not int:
        return 'Trip Distance should be a numeric value'
    if val < 0:
        return 'Trip Distance should be a positive number'
    return None


def check_fare(val):
    
    if type(val) is not float and type(val) is not int:
        return 'Fare Amount should be a numeric value'
    if val < 0:
        return 'Fare Amount should be a positive number'
    return None

def check_start_end(p_dt, d_dt):
    p_dt = pd.to_datetime(p_dt.strip(), format=dt_format, errors='coerce')
    d_dt = pd.to_datetime(d_dt.strip(), format=dt_format, errors='coerce')

    if (p_dt > d_dt):
        return 'Pickup Date Time should be lower than Dropoff Date Time'
    return None
 

def add_trip_checks(param_list):
    lst_validator = [check_driver, check_dt, check_dt,
                     check_pass_cnt, check_loc, check_loc,
                     check_trip_d, check_fare, check_start_end]

    for key, func in enumerate(lst_validator):
        if key == 1:
            chk = func(param_list[key], 'Pickup Date Time')
        elif key == 2:
            chk = func(param_list[key], 'Dropoff Date Time')
        elif key == 4:
            chk = func(param_list[key], 'Pickup Location')
        elif key == 5:
            chk = func(param_list[key], 'Dropoff Location')
        elif key == 8:
            chk = func(param_list[1] ,param_list[2])
        else:
            chk = func(param_list[key])
        if chk is not None:
            return chk
        
def create_file(dir, cols):
    
    df = pd.DataFrame(columns=cols)
    df.to_csv(dir, sep=sep, header=True, index=False)
    return None


def check_driver_id(name, f_dir):
    
    df = pd.read_csv(f_dir, sep=sep)
    df.given_name = df.given_name.str.lower()
    df.last_name = df.last_name.str.lower()

    df_res = df.query('last_name==@name[0].lower() & given_name==@name[1].lower()')
    
    if len(df_res) == 0:
        last_val = df.tail(1)
        if len(last_val) == 1:
            return last_val.driver_id.values[0]+1, True
        return 1, True
    else:
        return df_res.driver_id.values[0], False
    
    
def check_loc_id(loc, f_dir):
  
    df = pd.read_csv(f_dir, sep=sep)
    df_res = df.query('loc_name==@loc')
    
    if len(df_res) == 0:
        last_val = df.tail(1)
        if len(last_val) == 1:
            return last_val.location_id.values[0]+1, True
        return 1, True
    else:
        return df_res.location_id.values[0], False
    
def check_trip_id(data, f_dir):

    df = pd.read_csv(f_dir, sep=sep)
    df_res = df.query('driver_id == @data[0]'
                   '& pickup_datetime == @data[1]'
                   '& dropoff_datetime == @data[2]'
                   '& passenger_count == @data[3]'
                   '& pickup_loc_id == @data[4]'
                   '& dropoff_loc_id == @data[5]' 
                   '& trip_distance == @data[6]'
                   '& fare_amount == @data[7]')
    
    if len(df_res) == 0:
        last_val = df.tail(1)
        if len(last_val) == 1:
            return last_val.trip_id.values[0]+1, False
        return 1, False
    else:
        return 0, True


def insert_data(data, file_p, cols_lst):
    insert_df = pd.DataFrame(data, columns=cols_lst)
    insert_df.to_csv(file_p, mode='a', index=False, header=False, sep=sep)
    return None

def del_trip(trip_id, f_dir):
 
    df = pd.read_csv(f_dir, sep=sep)
    df_res = df.query('trip_id != @trip_id')

    if len(df) == len(df_res):
        return False
    else:
        df_res.to_csv(f_dir, sep=sep, header=True, index=False)
        return True
        
class SakayDBError(ValueError):
    
    def __init__(self, message):
        self.message = 'Error encountered: ' + str(message)
        super().__init__(self.message)



class SakayDB:
        
    def __init__(self, f_dir):
        if path.exists(f_dir) == False:
            raise SakayDBError(f'Directory {f_dir} does not exist')
            
        self.data_dir = f_dir
        self.__trips_dir = path.join(f_dir, 'trips.csv')
        self.__drivers_dir = path.join(f_dir, 'drivers.csv')
        self.__locs_dir = path.join(f_dir, 'locations.csv')
        
        if path.exists(self.__trips_dir) == False:
            create_file(self.__trips_dir, trips_cols)
            
        if path.exists(self.__drivers_dir) == False:
            create_file(self.__drivers_dir, drivers_cols)
            
        if path.exists(self.__locs_dir) == False:
            create_file(self.__locs_dir, locs_cols)
        
        
    def add_trip(self, driver, pickup_datetime, dropoff_datetime, passenger_count,
                 pickup_loc_name, dropoff_loc_name, trip_distance, fare_amount, is_trips=None):
        
        param_lst = [driver, pickup_datetime, dropoff_datetime, passenger_count,
                     pickup_loc_name, dropoff_loc_name, trip_distance, fare_amount]
        
        chk = add_trip_checks(param_lst)
        
        if chk is not None:
            if is_trips is None:
                raise SakayDBError(chk)
            else:
                print(f'Warning: trip index {is_trips} has invalid or incomplete information. Skipping...')
                return None
            
        split_name = [x.strip() for x in driver.split(',')]
        driver_id, is_new_driver = check_driver_id(split_name, self.__drivers_dir)
        
        pickup_loc_name = pickup_loc_name.strip()
        dropoff_loc_name = dropoff_loc_name.strip()
        
        p_loc_id, is_new_p_loc = check_loc_id(pickup_loc_name, self.__locs_dir)
        d_loc_id, is_new_d_loc = check_loc_id(dropoff_loc_name, self.__locs_dir)
        
        param_lst[0] = driver_id
        param_lst[4] = p_loc_id
        
        if is_new_p_loc and is_new_d_loc:
                d_loc_id += 1
        
        param_lst[5] = d_loc_id
        
        trip_id, is_dup = check_trip_id(param_lst, self.__trips_dir)
    
        if is_dup:
            if is_trips is None:
                raise SakayDBError('Duplicate Trip Entry')
            else:
                print(f'Warning: trip index {is_trips} is already in the database. Skipping...')
                return None
            
        if is_new_driver:
            driver_data = [[driver_id, split_name[0], split_name[1]]]
            insert_data(driver_data, self.__drivers_dir, drivers_dtypes)
            
        if is_new_p_loc:
            loc_data = [[p_loc_id, pickup_loc_name]]
            insert_data(loc_data, self.__locs_dir, locs_dtypes)
            
            
        if is_new_d_loc:
            loc_data = [[d_loc_id, dropoff_loc_name]]
            insert_data(loc_data, self.__locs_dir, locs_dtypes)

        
        trip_data = [[
                      trip_id,
                      param_lst[0],
                      param_lst[1],
                      param_lst[2],
                      param_lst[3],
                      param_lst[4], 
                      param_lst[5],
                      param_lst[6],
                      param_lst[7]
                    ]]
            
        insert_data(trip_data, self.__trips_dir, trips_dtypes)
        
        return trip_id
    
    
    def add_trips(self, trip_list):
        
        if type(trip_list) is not list:
            raise SakayDBError('Trips should be a valid list')
        
        for i, val in enumerate(trip_list):
            self.add_trip(val.get('driver'),
                          val.get('pickup_datetime'),
                          val.get('dropoff_datetime'),
                          val.get('passenger_count'),
                          val.get('pickup_loc_name'),
                          val.get('dropoff_loc_name'),
                          val.get('trip_distance'),
                          val.get('fare_amount'),
                          i)
        
        
    def delete_trip(self, trip_id):
        
        if type(trip_id) is not int:
            raise SakayDBError('Trip ID should be integer value')
        
        if del_trip(trip_id, self.__trips_dir) == False:
            raise SakayDBError('Trip ID not found')



