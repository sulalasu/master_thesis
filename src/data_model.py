#%% Make DATA class for dataframe
# so i can add methods for plotting etc.
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

class Data():
    # Class for the processed data, which contains methods for plotting
    # and transforming, which will be passed into class 'Model'
    # TODO: change parameter of model to be of 'Data' type 

    def __init__(self, data: pd.DataFrame):
        #index is datetime 
        self.data = data

    #Methods:
    def print_head(self): 
        print(type(self.data))
        print(self.data.head())


    def plot_line(self):
        #simple line plot
        fig, ax = plt.subplots()
        ax.plot(self.data[0])
        plt.show()

    def plot_seasonal(self, plot_type: list[str], col_name):
        #seasonal plot (days of week, week of year, years)
        # 'column': str name of column to plot. Column values must be float or integer

        accepted_types = ["daily", "weekly", "yearly"]
        if plot_type not in accepted_types:
            raise ValueError("'plot_type' must be 'daily', 'weekly' or 'yearly")
        
        #drop all except 'date' and column
        series = self.data[col_name].to_frame()
        # df = self.data[["date", column]]

        #Get data depending on 'plot_type'
        # Days in a week
        if plot_type == 'daily':
            #Set plotting & naming values:
            x = 'day_of_week'
            ref_frame = 'week' #comparison period; name of column
            ref_frame_str = 'week_str' #comparison period string;  name of column
            xlabel = 'Day of week'
            xticks_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            title = 'Daily'
            print(type(series))
            print(series.head())
            #Resample daily:
            df = series[col_name].resample("D").sum()
            df = df.reset_index()
            #Add new columns:
            df[x] = df['date'].dt.day_of_week
            df[ref_frame] = df['date'].dt.isocalendar().week #need to make it string later
            df[ref_frame_str] = df[ref_frame].astype(str) #need string for 'hue'

        #Weeks in year
        elif plot_type == 'weekly':
            #Set plotting & naming values:
            x = 'week_of_year'
            ref_frame = 'year' #comparison period; name of column
            ref_frame_str = 'year_str' #comparison period string;  name of column
            xlabel = 'Week number'
            xticks_labels = [str(week) for week in range(1,53)]
            title = 'Weekly'

            #Resample weekly:
            df = series.resample('W').sum()
            df = df.reset_index()
            #Add new columns:
            df[x] = df['date'].dt.isocalendar().week
            df[ref_frame] = df['date'].dt.year
            df[ref_frame_str] = df[ref_frame].astype('str')


        elif plot_type == 'yearly':
            pass


        #Plotting:
        ax = sns.lineplot(x=x, y=col_name, data=df, hue=ref_frame_str, errorbar=('ci', False))
        ax.set_title(f'{title} seasonality plot')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('value')
        ax.set_xticks(ticks=range(len(xticks_labels)), labels=xticks_labels)
        #if more than 12 xticks, hide every second label
        # if len(ax.get_xticklabels()) > 12:
        #     ax = sns.setp(ax.axes.get_xticklabels(), visible=False)
        #     ax = sns.setp(ax.axes.get_xticklabels()[::4], visible=True)
        if len(ax.get_xticklabels()) > 12:
            for i, label in enumerate(ax.xaxis.get_major_ticks()):
                if i % 3 != 0:
                    label.set_visible(False)
                    

                
        ax.legend([],[], frameon=False)
        ax.grid(True)
        #fig.tight_layout()
        #fig.show()


    def plot_seasonal_subseries(self):
        #plot seasonal subseries
        pass
 
    def plot_acf(self):
        pass

    def plot_pacf(self):
        pass

    def plot_overview(self):
        # aggregate method to plot multiple functions
        pass
