import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

dow_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
             'Friday': 4, 'Saturday': 5, 'Sunday': 6}
trips_dates = ['pickup_datetime', 'dropoff_datetime']
trips_dtypes_read = {
                'trip_id': int,
                'driver_id': int,
                'passenger_count': int,
                'pickup_loc_id': int,
                'dropoff_loc_id': int,
                'trip_distance': float,
                'fare_amount': float
               }


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


class SakayDB:
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
                    return ((df_trips.groupby(dow).trip_id.nunique() /
                             df_trips.groupby(dow).pickup_datetime
                                     .apply(lambda x: x.dt.date.nunique()))
                            .sort_index(key=lambda x: x.map(dow_order))
                            .to_dict())
            else:
                dow = df.pickup_datetime.dt.strftime('%A')
                return ((df.groupby(dow).trip_id.nunique() /
                         df.groupby(dow).pickup_datetime
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
                                         orient='index', columns=['avg_trips'])
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
                ax = df.plot(marker='o', markersize=7, figsize=(12, 8), lw=3,
                             xlabel='Day of week', ylabel='Ave Trips',
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
