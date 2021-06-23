
#Temporal variable of trip purposes from (predicted) SCD
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

 
dfH=pd.read_csv(r'C:\Users\eng d\Google Drive\TD\Data\Activities.csv')
# extract dfH='HOME activities',dfW= 'WORK activities', dfEAT= 'EAT activities', dfENT = 'ENT activities', dfSHO= 'SHO activities'
# dfPT_W = 'PTW activities', dfD_P= 'D/P activities'

from mpl_toolkits.axes_grid1 import make_axes_locatable
df_StartEnd_Time=pd.read_csv(r'C:\Users\eng d\Google Drive\TD\Data\TD_ActivityStartEndTime.csv')
dfH.head()
dfW.head()
dfEAT.head()
dfSHO.head()
df_StartEnd_Time.head()
#######################################
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(categories=cats, ordered=True)
dfH['ActivityDay'] = dfH['ActivityDay'].astype(cat_type)
dfW['ActivityDay'] = dfW['ActivityDay'].astype(cat_type)
dfENT['ActivityDay'] = dfENT['ActivityDay'].astype(cat_type)
dfEAT['ActivityDay'] = dfEAT['ActivityDay'].astype(cat_type)
dfSHO['ActivityDay'] = dfSHO['ActivityDay'].astype(cat_type)
dfD_P['ActivityDay'] = dfD_P['ActivityDay'].astype(cat_type)
dfPT_W['ActivityDay'] = dfPT_W['ActivityDay'].astype(cat_type)

H_matrix=pd.pivot_table(dfH, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
W_matrix=pd.pivot_table(dfW, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
ENT_matrix=pd.pivot_table(dfENT, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
EAT_matrix=pd.pivot_table(dfEAT, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
SHO_matrix=pd.pivot_table(dfSHO, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
D_P_matrix=pd.pivot_table(dfD_P, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
PT_W_matrix=pd.pivot_table(dfPT_W, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)

sns.set_context('paper')
sns.set_style("white")
###start, end = ax.get_xlim()
###ax.xaxis.set_ticks(np.arange(0, 24, 4))

f,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12),(ax13,ax14,ax15,ax16)) = plt.subplots(4,4,sharex=False, sharey=False,figsize=(21, 14))

ax1 = sns.heatmap(H_matrix,cmap="YlGnBu",linewidths=0.5, cbar=False, linecolor='white',xticklabels=4,annot_kws={"size": 24},ax=ax1)
ax1.set_ylabel('')
ax1.set_xlabel('Duration (Hr)')
ax1.set_title('H',fontsize=12)
plt.xlim([0,24])
#start, end = ax.get_xlim(0,24)
ax1.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax1.set_xticklabels(['0', '4', '8', '12','16','20', '24'])
#ax10.set_xticklabels(['0', '4', '8', 'Duration (Hr)','20', '24'])

ax2 = sns.heatmap(W_matrix,cmap="YlGnBu",cbar=False,linewidths=0.5, linecolor='white',xticklabels=4,ax=ax2)
ax2.set_ylabel('')
ax2.set_xlabel('Duration (Hr)')
ax2.set_title('W',fontsize=12)
#plt.xlim([0,24])
ax2.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax2.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax2.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax3 = sns.heatmap(D_P_matrix,cmap="YlGnBu",linewidths=0.5, linecolor='white', cbar_kws={'label':'Counts'}, xticklabels=4, cbar=False, ax=ax3)
ax3.set_ylabel('')
ax3.set_xlabel('Duration (Hr)')
ax3.set_title('D/P',fontsize=12)
#plt.xlim([0,24])
ax3.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax3.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax3.set_xticklabels(['0', '4', '8','12','16','20', '24'])

ax4 = sns.heatmap(PT_W_matrix,cmap="YlGnBu",linewidths=0.5, linecolor='white', cbar_kws={'label':'Counts'}, xticklabels=4, cbar=True, ax=ax4)
ax4.set_ylabel('')
ax4.set_xlabel('Duration (Hr)')
ax4.set_title('PTW',fontsize=12)
#plt.xlim([0,24])
ax4.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax3.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax4.set_xticklabels(['0', '4', '8','12','16','20', '24'])

#f,((ax5,ax6,ax7,ax8)) = plt.subplots(1,4,sharex=True, sharey=False,figsize=(14, 5))
ax5=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['H_StartTime_Count']),data=df_StartEnd_Time,label = 'Start Time', color = '#225ea8',ax=ax5)
ax5=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['H_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax5)
ax5.set_title('')
ax5.set_ylabel('Counts', fontsize=12)
ax5.set_xlabel('')
#ax2.legend(loc='upper left', loc=0)
#f.legend(loc="upper left")
#ax5.set_title('', size =10)
ax5.legend(loc='upper left', fontsize=8)
plt.xlim([0,24])
#ax5.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax5.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax5.set_xticklabels(['4', '8', '12', '16', '20', '24'])
ax5.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax6=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['W_StartTime_Count']),data=df_StartEnd_Time,  label = 'Start Time',color = '#225ea8',ax=ax6)
ax6=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['W_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax6)
ax6.set_ylabel('')
ax6.set_xlabel('')
ax6.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
#ax6.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax6.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax6.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax7=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['D_P_StartTime_Count']),data=df_StartEnd_Time, label = 'Start Time',color = '#225ea8',ax=ax7)
ax7=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['D_P_EndTime_Count']),data=df_StartEnd_Time,label = 'End Time', color = '#DD4968', ax=ax7)
ax7.set_ylabel('')
ax7.set_xlabel('')
ax7.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
ax7.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax7.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax7.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax8=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['PT_W_StartTime_Count']),data=df_StartEnd_Time, label = 'Start Time',color = '#225ea8',ax=ax8)
ax8=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['PT_W_EndTime_Count']),data=df_StartEnd_Time,label = 'End Time', color = '#DD4968', ax=ax8)
ax8.set_ylabel('')
ax8.set_xlabel('Time (Hr)')
ax8.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
ax8.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax7.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax8.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

#f,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex=True, sharey=False,figsize=(14, 5))

ax9 = sns.heatmap(ENT_matrix,cmap="YlGnBu",linewidths=0.5, cbar=False, linecolor='white',xticklabels=4,annot_kws={"size": 24},ax=ax9)
ax9.set_ylabel('')
ax9.set_xlabel('Duration (Hr)')
ax9.set_title('ENT',fontsize=12)
plt.xlim([0,24])
#start, end = ax.get_xlim(0,24)
ax9.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax9.set_xticklabels(['0', '4', '8', '12','20', '24'])
#ax10.set_xticklabels(['0', '4', '8', 'Duration (Hr)','20', '24'])

ax10 = sns.heatmap(EAT_matrix,cmap="YlGnBu",cbar=False,linewidths=0.5, linecolor='white',xticklabels=4,ax=ax10)
ax10.set_ylabel('')
ax10.set_xlabel('Duration (Hr)')
ax10.set_title('EAT',fontsize=12)
#plt.xlim([0,24])
ax10.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax2.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax10.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax11 = sns.heatmap(SHO_matrix,cmap="YlGnBu",linewidths=0.5, linecolor='white', cbar_kws={'label':'Counts'}, xticklabels=4, cbar=True, ax=ax11)
ax11.set_ylabel('')
ax11.set_xlabel('Duration (Hr)')
ax11.set_title('SHO',fontsize=12)
#plt.xlim([0,24])
ax11.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax3.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax11.set_xticklabels(['0', '4', '8','12','16','20', '24'])

#f,((ax5,ax6,ax7,ax8)) = plt.subplots(1,4,sharex=True, sharey=False,figsize=(14, 5))
ax13=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['ENT_StartTime_Count']),data=df_StartEnd_Time,label = 'Start Time', color = '#225ea8',ax=ax13)
ax13=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['ENT_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax13)
ax13.set_title('')
ax13.set_ylabel('Counts', fontsize=12)
ax13.set_xlabel('Time (Hr)')
#ax2.legend(loc='upper left', loc=0)
#f.legend(loc="upper left")
#ax5.set_title('', size =10)
ax13.legend(loc='upper left', fontsize=8)
plt.xlim([0,24])
#ax5.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax13.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax13.set_xticklabels(['4', '8', '12', '16', '20', '24'])
ax13.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax14=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['EAT_StartTime_Count']),data=df_StartEnd_Time,  label = 'Start Time',color = '#225ea8',ax=ax14)
ax14=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['EAT_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax14)
ax14.set_ylabel('')
ax14.set_xlabel('Time (Hr)')
ax14.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
#ax6.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax14.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax14.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax15=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['SHO_StartTime_Count']),data=df_StartEnd_Time, label = 'Start Time',color = '#225ea8',ax=ax15)
ax15=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['SHO_EndTime_Count']),data=df_StartEnd_Time,label = 'End Time', color = '#DD4968', ax=ax15)
ax15.set_ylabel('')
ax15.set_xlabel('Time (Hr)')
ax15.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
ax15.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax7.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax15.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax12.set_visible(False)
ax16.set_visible(False)

plt.savefig('C:\\Users\\eng d\\Google Drive\\TD\\Figures\\TD_all.tiff',dpi = 300)