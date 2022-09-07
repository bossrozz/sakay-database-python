import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt_format = '%H:%M:%S,%d-%m-%Y'
sep = ','
trips_cols = ['trip_id',
              'driver_id',
              'pickup_datetime',
              'dropoff_datetime',
              'passenger_count',
              'pickup_loc_id',
              'dropoff_loc_id',
              'trip_distance',
              'fare_amount']
drivers_cols = ['driver_id',
                'last_name',
                'given_name']
locs_cols = ['location_id', 'loc_name']
trips_dtypes = {'trip_id': int,
                'driver_id': int,
                'pickup_datetime': 'datetime64[ns]',
                'dropoff_datetime': 'datetime64[ns]',
                'passenger_count': int,
                'pickup_loc_id': int,
                'dropoff_loc_id': int,
                'trip_distance': float,
                'fare_amount': float}
drivers_dtypes = {'driver_id': int,
                  'last_name': str,
                  'given_name': str}
locs_dtypes = {'location_id': int, 'loc_name': str}
dow_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
             'Friday': 4, 'Saturday': 5, 'Sunday': 6}
trips_dates = ['pickup_datetime', 'dropoff_datetime']
trips_dtypes_read = {'trip_id': int,
                     'driver_id': int,
                     'passenger_count': int,
                     'pickup_loc_id': int,
                     'dropoff_loc_id': int,
                     'trip_distance': float,
                     'fare_amount': float}
aim_colors = ['#393357', '#a98e26', '#dacd1f', '#4ab396',
              '#00a2af', '#009c69', '#306e64', '#6e3476']

plt.style.use('seaborn-white')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.fancybox'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


def check_driver(driver):
    """
    Validate string entry of driver for trip

    Parameters
    ----------
    driver : str
        String to be validated

    Returns
    -------
    validation : str or None
       str message if the entry is invalid
       None if valid
    """
    special_characters = '''!\"#$%&'()*+-/:;<=>?@[\\]^_`{|}~'''
    if type(driver) is not str or driver.strip() == '':
        return 'Name should not be empty or must be in string format'
    if len(driver.split(',')) != 2:
        return 'Invalid name format should be (First Name, Last Name)'
    if any(c in special_characters for c in driver):
        return 'Name contains special characters'
    return None


def check_dt(dt, dt_col):
    """
    Validate string value of date for trip

    Parameters
    ----------
    dt : str
        Date in string to be validated valid format '%H:%M:%S,%d-%m-%Y'
    dt_col : dt_col
        Column name for the error message

    Returns
    -------
    validation : str or None
       str message if the entry is invalid
       None if valid
    """
    if type(dt) is not str:
        return (f'{dt_col} should be in string format')
    if pd.isnull(pd.to_datetime(dt.strip(),
                                format=dt_format,
                                errors='coerce')):
        return (f'{dt_col} should be in valid format')
    return None


def check_pass_cnt(cnt):
    """
    Validate int value of passenger count for trip

    Parameters
    ----------
    cnt : int
        Should be in integer value

    Returns
    -------
    validation : str or None
       str message if the entry is invalid
       None if valid
    """
    if type(cnt) is not int:
        return 'Passenger count should be integer value'
    if cnt < 0:
        return 'Passenger count should be a positive value >= 0'
    return None


def check_loc(loc, loc_col):
    """
    Validate string value of location for trip

    Parameters
    ----------
    loc : str
        String to be validated
    loc_col : dt_col
        Column name for the error message

    Returns
    -------
    validation : str or None
       str message if the entry is invalid
       None if valid
    """
    special_characters = '''!\"%&'()+-/:;<=>?@[\\]^_`{|}~'''
    if type(loc) is not str or loc.strip() == '':
        return (f'{loc_col} should not be empty or must be in string format')
    elif any(c in special_characters for c in loc):
        return 'Location contains special characters'
    return None


def check_trip_d(val):
    """
    Validate numeric value of trip_distance for trip

    Parameters
    ----------
    val : int / float
        Should be in integer value

    Returns
    -------
    validation : str or None
       str message if the entry is invalid
       None if valid
    """
    if type(val) is not float and type(val) is not int:
        return 'Trip Distance should be a numeric value'
    if val < 0:
        return 'Trip Distance should be a positive number'
    return None


def check_fare(val):
    """
    Validate numeric value of fare for trip

    Parameters
    ----------
    val : int / float
        Should be in integer value

    Returns
    -------
    validation : str or None
       str message if the entry is invalid
       None if valid
    """
    if type(val) is not float and type(val) is not int:
        return 'Fare Amount should be a numeric value'
    if val < 0:
        return 'Fare Amount should be a positive number'
    return None


def check_start_end(p_dt, d_dt):
    """
    Validate if pick-up date is less than drop-off date

    Parameters
    ----------
    p_dt : str
        date/time value in string format '%H:%M:%S,%d-%m-%Y'
    d_dt : str
        date/time value in string format '%H:%M:%S,%d-%m-%Y'

    Returns
    -------
    validation : str or None
       str message if the entry is invalid
       None if valid
    """
    p_dt = pd.to_datetime(p_dt.strip(), format=dt_format, errors='coerce')
    d_dt = pd.to_datetime(d_dt.strip(), format=dt_format, errors='coerce')

    if (p_dt > d_dt):
        return 'Pickup Date Time should be lower than Dropoff Date Time'
    return None


def add_trip_checks(param_list):
    """
    Custom validation loop where all entries of param_list
    for the trip entry will go through the lst_validator
    list of validaiton functions. It will return a value
    if any of the validation fails

    Parameters
    ----------
    param_list : list
        List of values to be validated for the trip entry

    Returns
    -------
    chk : str or None
       Value will be dependendent on the result of the validation
       per entry
    """
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
            chk = func(param_list[1], param_list[2])
        else:
            chk = func(param_list[key])
        if chk is not None:
            return chk


def create_file(dir_path, cols):
    """
    Creates a csv file based on data frame columns and path defined

    Parameters
    ----------
    dir_path : str
        Full path and filename where the data frame would be written
    cols : list
        Column names for the data frame headers

    Returns
    -------
    None
    """
    df = pd.DataFrame(columns=cols)
    df.to_csv(dir_path, sep=sep, header=True, index=False)
    return None


def check_driver_id(name, f_dir):
    """
    Check if driver exist and returns driver id otherwise returns the
    last driver id + 1 else if its the first entry returns 1.

    Parameters
    ----------
    name : list
        First value in list is the last name, second value is the first name
    f_dir : str
        Full path and filename where the driver file would be read

    Returns
    -------
    driver_id: int
        Driver id based on the driver file entry
    flag: boolean
        False if driver exists, True if driver is new
    """
    df = pd.read_csv(f_dir, sep=sep)
    df.given_name = df.given_name.str.lower()
    df.last_name = df.last_name.str.lower()

    df_res = df.query('last_name==@name[0].lower()'
                      '& given_name==@name[1].lower()')

    if len(df_res) == 0:
        last_val = df.tail(1)
        if len(last_val) == 1:
            return last_val.driver_id.values[0] + 1, True
        return 1, True
    else:
        return df_res.driver_id.values[0], False


def check_loc_id(loc, f_dir):
    """
    Check if location exist and returns loc id otherwise returns the
    last loc id + 1 else if its the first entry returns 1.

    Parameters
    ----------
    loc : list
        First value in list is the last name, second value is the first name
    f_dir : str
        Full path and filename where the location file would be read

    Returns
    -------
    loc_id: int
        Location id based on the location file entry
    flag: boolean
        False if location exists, True if location is new
    """
    df = pd.read_csv(f_dir, sep=sep)
    df.loc_name = df.loc_name.str.lower()
    df_res = df.query('loc_name==@loc.lower()')

    if len(df_res) == 0:
        last_val = df.tail(1)
        if len(last_val) == 1:
            return last_val.location_id.values[0] + 1, True
        return 1, True
    else:
        return df_res.location_id.values[0], False


def check_trip_id(data, f_dir):
    """
    Check if trip does not exist it returns trip id + 1 else if its the first
    entry returns 1, 0 if its a duplicate.

    Parameters
    ----------
    data : list
        driver_id, pickup_datetime, dropoff_datetime, passenger_count
        pickup_loc_id, dropoff_loc_id, trip_distance, fare_amount

    f_dir: str
        Full path and filename where the trip file would be read

    Returns
    -------
    trip_id: int
        trip_id + 1 if there's an entry in the file, 1 if its the first
    flag: boolean
        True if trip exists, False if trip is new
    """
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
            return last_val.trip_id.values[0] + 1, False
        return 1, False
    else:
        return 0, True


def insert_data(data, file_p, cols_lst):
    """
    Appends all entries into the csv file based on the data parameter
    mapped to the column lists

    Parameters
    ----------
    data : list
        List of values to be written
    file_p : string
        Full path and filename where the file would be written
    cols_lst : dict / list
        list - column names for the data frame
        dict - column names : data type for the data frame
    Returns
    -------
    None
    """
    insert_df = pd.DataFrame(data, columns=cols_lst)
    insert_df.to_csv(file_p, mode='a', index=False, header=False, sep=sep)
    return None


def del_trip(trip_id, f_dir):
    """
    Deletes the trip based on the trip id in the file, query will exclude the
    trip id defined in the new data frame and overwrites it into the file

    Parameters
    ----------
    trip_id : int
        trip_id to be deleted
    f_dir : string
        Full path and filename of the trip entries

    Returns
    -------
    None
    """
    df = pd.read_csv(f_dir, sep=sep)
    df_res = df.query('trip_id != @trip_id')

    if len(df) == len(df_res):
        return False
    else:
        df_res.to_csv(f_dir, sep=sep, header=True, index=False)
        return True


def read_data(dataset, fname):
    """
    Read csv file based on input dataset and filename

    Parameters
    ----------
    dataset : str
        Dataset to be read
    fname : str
        Name of the csv file to be read

    Returns
    -------
    read_data : pandas DataFrame
        Dataframe containing the data from the csv file
    """
    if dataset == 'trips':
        return (pd.read_csv(fname, dtype=trips_dtypes_read,
                            parse_dates=trips_dates,
                            date_parser=lambda x:
                            pd.to_datetime(x, format=dt_format)))
    elif dataset == 'drivers':
        return (pd.read_csv(fname, dtype=drivers_dtypes,
                date_parser=lambda x: pd.to_datetime(x, format=dt_format)))
    elif dataset == 'locations':
        return (pd.read_csv(fname, dtype=locs_dtypes,
                date_parser=lambda x: pd.to_datetime(x, format=dt_format)))
    else:
        raise SakayDBError("Request dataset not found. Choose among 'trips', \
                           'drivers', or 'locations'")


class SakayDBError(ValueError):
    """
    Class to be raised for custom errors encountered in the

    Parameters
    ----------
    ValueError : class
        Inherits ValueError class for Error Handling

    Returns
    -------
    None
    """
    def __init__(self, message):
        self.message = 'Error encountered: ' + str(message)
        super().__init__(self.message)


class SakayDB:

    def __init__(self, f_dir):
        """
        Constructor function that initiates with the path where the files
        of SakayDB would be created, read and written

        Parameters
        ----------
        f_dir : string
            Path where all the files would read and written

        Returns
        -------
        None
        """
        if path.exists(f_dir) is False:
            raise SakayDBError(f'Directory {f_dir} does not exist')

        self.data_dir = f_dir
        self.__trips_dir = path.join(f_dir, 'trips.csv')
        self.__drivers_dir = path.join(f_dir, 'drivers.csv')
        self.__locs_dir = path.join(f_dir, 'locations.csv')

        if path.exists(self.__trips_dir) is False:
            create_file(self.__trips_dir, trips_cols)

        if path.exists(self.__drivers_dir) is False:
            create_file(self.__drivers_dir, drivers_cols)

        if path.exists(self.__locs_dir) is False:
            create_file(self.__locs_dir, locs_cols)

    def add_trip(self, driver, pickup_datetime, dropoff_datetime,
                 passenger_count, pickup_loc_name, dropoff_loc_name,
                 trip_distance, fare_amount, is_trips=None):
        """
        Adds a new trip if its not a duplicate
        Adds a new driver if it does not exist yet
        Adds a new location if it does not exist yet
        Function is also utilzied by add_trips based on is_trips value

        Parameters
        ----------
        driver : str
            Trip driver as a string in Last name, Given name format
        pickup_datetime : str
            datetime of pickup as string with format "hh:mm:ss,DD-MM-YYYY"
        dropoff_datetime : str
            datetime of dropoff as string with format "hh:mm:ss,DD-MM-YYYY"
        passenger_count : int
            number of passengers as integer
        pickup_loc_name : str
            zone as a string, (e.g., Pine View, Legazpi Village)
        dropoff_loc_name : str
            zone as a string, (e.g., Pine View, Legazpi Village)
        trip_distance : float
            distance in meters (float)
        fare_amount : float
            amount paid by passenger (float)
        is_trips : int
            flag if to determine if its for add_trips entry

        Returns
        -------
        None
        """
        param_lst = [driver, pickup_datetime, dropoff_datetime,
                     passenger_count, pickup_loc_name, dropoff_loc_name,
                     trip_distance, fare_amount]

        chk = add_trip_checks(param_lst)

        if chk is not None:
            if is_trips is None:
                raise SakayDBError(chk)
            else:
                print(f'Warning: trip index {is_trips} has invalid or '
                      'incomplete information. Skipping...')
                return None

        split_name = [x.strip() for x in driver.split(',')]
        driver_id, is_new_driver = check_driver_id(split_name,
                                                   self.__drivers_dir)

        pickup_loc_name = pickup_loc_name.strip().lower()
        dropoff_loc_name = dropoff_loc_name.strip().lower()

        p_loc_id, is_new_p_loc = check_loc_id(pickup_loc_name,
                                              self.__locs_dir)

        if pickup_loc_name == dropoff_loc_name:
            d_loc_id = p_loc_id
        else:
            d_loc_id, is_new_d_loc = check_loc_id(dropoff_loc_name,
                                                  self.__locs_dir)
            if (is_new_p_loc and is_new_d_loc):
                d_loc_id += 1

        param_lst[0] = driver_id
        param_lst[4] = p_loc_id
        param_lst[5] = d_loc_id

        trip_id, is_dup = check_trip_id(param_lst, self.__trips_dir)

        if is_dup:
            if is_trips is None:
                raise SakayDBError('Duplicate Trip Entry')
            else:
                print(f'Warning: trip index {is_trips} is already in the '
                      'database. Skipping...')
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

        trip_data = [[trip_id, param_lst[0], param_lst[1],
                      param_lst[2], param_lst[3], param_lst[4],
                      param_lst[5], param_lst[6], param_lst[7]]]

        insert_data(trip_data, self.__trips_dir, trips_dtypes)
        return trip_id

    def add_trips(self, trip_list):
        """
        Process a list of dict of trip entries and goes through same
        processing as an add_trip entry

        Parameters
        ----------
        trip_list : list
            List of dictionary that should contain parameter entry for trip

        Returns
        -------
        None
        """
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
        """
        Deletes trip entry based on trip_id if it exists if not it will
        raise an error

        Parameters
        ----------
        trip_id : int
            trip id to be deleted from the file

        Returns
        -------
        None
        """
        if type(trip_id) is not int:
            raise SakayDBError('Trip ID should be integer value')

        if del_trip(trip_id, self.__trips_dir) is False:
            raise SakayDBError('Trip ID not found')

    def search_trips(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs:
            driver_id : int, tuple of (int, int)
            pickup_datetime : str, tuple of (str, str)
            dropoff_datetime : str, tuple of (str, str)
            passenger_count : int, tuple of (int, int)
            trip_distance : int, tuple of (int, int)
            fare_amount : int, tuple of (int, int)

            For example:
            For single value search:
                driver_id=1
            For range search:
                driver_id=(1, 5) : from 1 to 5, inclusive
                driver_id=(None, 5) : All entries up to 5, inclusive
                driver_id=(1, None) : All entries from 5, inclusive
        Returns
        -------
        df : pandas.DataFrame
           Origin-Destination Matrix as a pandas.DataFrame containing
           the average daily number of trips within the specified
           date_range.
        """
        df = read_data('trips', self.__trips_dir)
        col_list = df.columns.tolist()
        case = 0

        if kwargs == {}:
            raise SakayDBError('Invalid keyword')
        if len(df) == 0:
            return []
        if kwargs != {}:
            for i, j in kwargs.items():
                if i not in col_list:
                    raise SakayDBError('Invalid keyword')
                elif i in col_list:
                    if (type(j) is not tuple) and (type(j) is not int):
                        raise SakayDBError('Invalid value')
                    elif type(j) is not tuple and type(j) is int:
                        case = 1
                    elif (type(j) is tuple) and (len(j) == 2):
                        if (all(isinstance(n, str) for n in j) is True):
                            case = 2
                        elif (all(isinstance(n, str) for n in j) is False):
                            case = 3

                if case == 1:
                    df = df.loc[df[i] == j]
                if case == 2:
                    try:
                        pudt1 = 'pickup_datetime_1'
                        dodt1 = 'dropoff_datetime_1'
                        df[pudt1] = pd.to_datetime(df[i],
                                                   format='%H:%M:%S,%d-%m-%Y')
                        df[dodt1] = pd.to_datetime(df[i],
                                                   format='%H:%M:%S,%d-%m-%Y')
                        start_date = pd.to_datetime(j[0],
                                                    format='%H:%M:%S,%d-%m-%Y')
                        end_date = pd.to_datetime(j[1],
                                                  format='%H:%M:%S,%d-%m-%Y')
                    except ValueError:
                        raise SakayDBError('Invalid values for range')
                    if (j[0] is None) and (j[1] is None):
                        raise SakayDBError('Invalid values for range')
                    elif (j[0] is not None) and (j[1] is not None):
                        if start_date > end_date:
                            raise SakayDBError('Invalid values for range')
                        elif start_date <= end_date:
                            df = df.loc[(df[i + '_1'] >= start_date)
                                        & (df[i + '_1'] <= end_date)]
                    elif (j[0] is None) and (j[1] is not None):
                        df = df.loc[df[i + '_1'] <= end_date]
                    elif (j[0] is not None) and (j[1] is None):
                        df = df.loc[df[i + '_1'] >= start_date]
                    df = df.iloc[:, :-2]
                elif case == 3:
                    try:
                        if (j[0] is None) and (j[1] is None):
                            raise SakayDBError('Invalid values for range')
                        elif (j[0] is not None) and (j[1] is not None):
                            if j[0] > j[1]:
                                raise SakayDBError('Invalid values for range')
                            elif j[0] <= j[1]:
                                df = df.loc[(df[i] >= j[0]) & (df[i] <= j[1])]
                        elif (j[0] is None) and (j[1] is not None):
                            df = df.loc[df[i] <= j[1]]
                        elif (j[0] is not None) and (j[1] is None):
                            df = df.loc[df[i] >= j[0]]
                    except TypeError:
                        raise SakayDBError('Invalid values for range')
        df = df.sort_values(by=i)
        df['pickup_datetime'] = df['pickup_datetime'].dt.strftime(dt_format)
        df['dropoff_datetime'] = df['dropoff_datetime'].dt.strftime(dt_format)
        return df

    def generate_statistics(self, stat, df=None):
        """
        Return a dictionary depending on the input `stat` parameter

        Parameters
        ----------
        stat : str
            Statistics to be generated. Can be details of the trip, passenger,
            driver, or all of the above
        df : pandas dataframe
            Dataframe to be used for creating the statistics.
            Uses the `trips` and `drivers` database by default.

        Returns
        -------
        generate_statistics : dict
            Dictionary containing the statistics requested based on `stat`
            parameter
        """
        # if df is None, use default dfs, else use input param
        if stat == 'trip':
            if df is None:
                df_trips = read_data('trips', self.__trips_dir)
                if len(df_trips) == 0:
                    return {}
                else:
                    dow = df_trips.pickup_datetime.dt.strftime('%A')
                    return ((df_trips.groupby(dow).trip_id.nunique()
                            / df_trips.groupby(dow).pickup_datetime
                                     .apply(lambda x: x.dt.date.nunique()))
                            .sort_index(key=lambda x: x.map(dow_order))
                            .to_dict())
            else:
                dow = df.pickup_datetime.dt.strftime('%A')
                return ((df.groupby(dow).trip_id.nunique()
                        / df.groupby(dow).pickup_datetime
                           .apply(lambda x: x.dt.date.nunique()))
                        .sort_index(key=lambda x: x.map(dow_order))
                        .to_dict())
        elif stat == 'passenger':
            df_trips = read_data('trips', self.__trips_dir)
            if len(df_trips) == 0:
                return {}
            else:
                return {k: self.generate_statistics('trip', v)
                        for k, v in df_trips.groupby('passenger_count')}
        elif stat == 'driver':
            df_drivers = read_data('drivers', self.__drivers_dir)
            df_trips = read_data('trips', self.__trips_dir)
            if (len(df_drivers) == 0):
                return {}
            else:
                df_temp = (df_drivers.merge(df_trips[['driver_id', 'trip_id',
                           'pickup_datetime']], how='left', on='driver_id'))
                df_temp['driver_name'] = (df_temp[['last_name', 'given_name']]
                                          .apply(lambda x: ', '.join(x),
                                          axis=1))
                return {k: self.generate_statistics('trip', v)
                        for k, v in df_temp.groupby('driver_name')}
        elif stat == 'all':
            return {'trip': self.generate_statistics('trip'),
                    'passenger': self.generate_statistics('passenger'),
                    'driver': self.generate_statistics('driver')}
        else:
            raise SakayDBError('Input parameter is unknown.')

    def plot_statistics(self, stat):
        """
        Return a plot depending on the input `stat` parameter

        Parameters
        ----------
        stat : str
            Plot to be generated. Can be details of the trip, passenger,
            or driver

        Returns
        -------
        ax : matplotlib Axes
        fig : matplotlib Figure
        """
        if stat == 'trip':
            df = (pd.DataFrame.from_dict(self.generate_statistics(stat),
                                         orient='index',
                                         columns=['avg_trips'])
                              .sort_index(key=lambda x: x.map(dow_order)))
            if len(df) == 0:
                print('{}')
            else:
                ax = df.plot.bar(y='avg_trips', color=aim_colors[0],
                                 width=0.7, legend=False, rot=False,
                                 xlabel='Day of week', ylabel='Ave Trips',
                                 figsize=(12, 8), align='center',
                                 title='Average trips per day')
                plt.show()
                return ax
        elif stat == 'passenger':
            df = (pd.DataFrame(self.generate_statistics(stat))
                    .sort_index(key=lambda x: x.map(dow_order)))
            if len(df) == 0:
                print('{}')
            else:
                ax = df.plot(marker='o', markersize=7, figsize=(12, 8),
                             lw=3, xlabel='Day of week',
                             ylabel='Ave Trips',
                             color=aim_colors)
                plt.show()
                return ax
        elif stat == 'driver':
            df = pd.DataFrame(self.generate_statistics(stat))
            if len(df) == 0:
                print('{}')
            else:
                df = (df.apply(pd.Series.nlargest, axis=1, n=5).stack()
                        .rename_axis(['dayofweek', 'driver_name'])
                        .reset_index(name='avg_trips')
                        .sort_values('dayofweek', key=lambda x:
                                     x.map(dow_order))
                        .reset_index(drop=True))

                fig, ax = plt.subplots(7, 1, sharex=True, figsize=(8, 25))
                for i, (r, d) in enumerate(df.groupby('dayofweek',
                                                      sort=False)):
                    d = d.sort_values(['avg_trips', 'driver_name'],
                                      ascending=[True, False])
                    (d.plot.barh(x='driver_name', y='avg_trips', ax=ax[i],
                                 label=r, color=aim_colors[i], width=0.6,
                                 align='center').set(ylabel=''))
                    ax[i].legend(loc='lower right')

                plt.xlabel('Ave Trips')
                plt.show()
                return fig
        else:
            raise SakayDBError('Input parameter is unknown.')

    def export_data(self):
        """
        Retuns all trips inner joined to drivers and locations. It will
        only show the trip if both location and driver exists in the
        lookup tables.

        Parameters
        ----------
        None

        Returns
        -------
        df_export : pandas.DataFrame
            Displays trips with joined Driver Name and Location Names
        """
        df_trips = read_data('trips', self.__trips_dir)
        df_drivers = read_data('drivers', self.__drivers_dir)
        df_locations = read_data('locations', self.__locs_dir)

        df_export = df_trips.merge(df_drivers, how='inner',
                                   left_on='driver_id',
                                   right_on='driver_id')
        df_export = df_export.merge(df_locations, how='inner',
                                    left_on='pickup_loc_id',
                                    right_on='location_id',
                                    suffixes=('', '_p'))
        df_export = df_export.merge(df_locations, how='inner',
                                    left_on='dropoff_loc_id',
                                    right_on='location_id',
                                    suffixes=('', '_d'))
        df_export.sort_values(by='trip_id', inplace=True)
        df_export.given_name = df_export.given_name.str.title()
        df_export.last_name = df_export.last_name.str.title()
        df_export.pickup_datetime = (df_export.pickup_datetime
                                     .astype('datetime64[ns]'))
        df_export.dropoff_datetime = (df_export.dropoff_datetime
                                      .astype('datetime64[ns]'))
        df_export['pickup_datetime'] = (df_export['pickup_datetime']
                                        .dt.strftime(dt_format))
        df_export['dropoff_datetime'] = (df_export['dropoff_datetime']
                                         .dt.strftime(dt_format))
        df_export = df_export[['given_name', 'last_name', 'pickup_datetime',
                               'dropoff_datetime', 'passenger_count',
                               'loc_name', 'loc_name_d', 'trip_distance',
                               'fare_amount']]
        df_export.columns = ['driver_givenname', 'driver_lastname',
                             'pickup_datetime', 'dropoff_datetime',
                             'passenger_count', 'pickup_loc_name',
                             'dropoff_loc_name', 'trip_distance',
                             'fare_amount']

        return df_export

    def generate_odmatrix(self, date_range=None):
        """
        Return an Origin-Destination matrix as a pandas.DataFrame containing
        the average daily number of trips that occured within the date_range
        specified (default to None, in which case all dates are included).

        Parameters
        ----------
        date_range : tuple of (str, str), Default `None`
            Takes a tuple of datetime strings, and filters trips based on
            pickup_datetime. Defaults to None, in which case all dates
            are included.

        For Example:
        For range search:
                    date_range=(1, 5) : from 1 to 5, inclusive
                    date_range=(None, 5) : All entries up to 5, inclusive
                    date_range=(1, None) : All entries from 5, inclusive
        Returns
        -------
        df : pandas.DataFrame
           Origin-Destination Matrix as a pandas.DataFrame containing
           the average daily number of trips within the specified
           date_range.
        """
        df_trips = read_data('trips', self.__trips_dir)
        df_loc = pd.read_csv('locations.csv')

        if len(df_trips) == 0:
            return df_trips
        else:
            pass
        df = df_trips.merge(df_loc,
                            left_on='pickup_loc_id',
                            right_on='location_id',
                            how='left')
        df = df.merge(df_loc,
                      left_on='dropoff_loc_id',
                      right_on='location_id',
                      how='left')
        df['pickup_datetime_1'] = pd.to_datetime(df['pickup_datetime'],
                                                 format='%H:%M:%S,%d-%m-%Y')
        df['count'] = 1
        col_dict = {'loc_name_x': 'pickup_loc_name',
                    'loc_name_y': 'dropoff_loc_name'}
        df.rename(columns=col_dict, inplace=True)
        case = 0
        filt = 'pickup_datetime_1'

        if date_range is None:
            case = 0
        elif (date_range is not None):
            if (type(date_range) is not tuple):
                raise SakayDBError('Invalid values for range')
            elif (type(date_range) is tuple) and (len(date_range) != 2):
                raise SakayDBError('Invalid values for range')
            elif (type(date_range) is tuple) and (len(date_range) == 2):
                if (date_range[0] is None) and (date_range[1] is None):
                    raise SakayDBError('Invalid values for range')
                elif ((date_range[0] is not None)
                      and (date_range[1] is not None)):
                    try:
                        date_1 = pd.to_datetime(date_range[0],
                                                format='%H:%M:%S,%d-%m-%Y')
                        date_2 = pd.to_datetime(date_range[1],
                                                format='%H:%M:%S,%d-%m-%Y')
                    except ValueError:
                        raise SakayDBError('Invalid values for range')
                    if date_1 > date_2:
                        raise SakayDBError('Invalid values for range')
                    elif date_1 <= date_2:
                        case = 1
                elif ((date_range[0] is None)
                      and (date_range[1] is not None)):
                    case = 2
                elif ((date_range[0] is not None)
                      and (date_range[1] is None)):
                    case = 3

        if case == 0:
            df = df
        elif case != 0:
            try:
                start_date = pd.to_datetime(date_range[0],
                                            format='%H:%M:%S,%d-%m-%Y')
                end_date = pd.to_datetime(date_range[1],
                                          format='%H:%M:%S,%d-%m-%Y')
            except ValueError:
                raise SakayDBError('Invalid values for range')
            if case == 1:
                df = df.loc[(df[filt] >= start_date) & (df[filt] <= end_date)]
            elif case == 2:
                df = df.loc[df[filt] <= end_date]
            elif case == 3:
                df = df.loc[df[filt] >= start_date]
        df = df.sort_values(by=filt)
        df = df.groupby(['pickup_loc_name',
                         'dropoff_loc_name',
                         pd.Grouper(key='pickup_datetime_1',
                                    freq='D')]).sum().reset_index()
        df = df.pivot_table(values='count',
                            index='dropoff_loc_name',
                            columns='pickup_loc_name',
                            aggfunc='mean',
                            fill_value=0)
        return df
