
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import interpolate

df = pd.read_csv('/home/jyx/Master_Project/Code/Results/2021-05-20 18:17:08.254340.csv')
x = df.index
yolo_config = 608
# a_BSpline = interpolate.make_interp_spline(df.index, df['velocity VX'])
# y_new = a_BSpline(x)
plot_center = True
plot_length = False
plot_velocity = False

if plot_center:
    plt.plot(df.index,df['bbcx'],label='bounding box center x')
    plt.plot(df.index,df['bbcy'], label='bounding box center y')
    plt.plot([0,len(df['bbcx'])], [yolo_config*0.5,yolo_config*0.5],'k-',lw=1)
    plt.title('Change of Bounding Box Center with Respect to Time')

if plot_length:
    plt.plot(df.index,df['bbw'],label='bounding box width')
    plt.plot(df.index,df['bbh'], label='bounding box height')
    plt.plot([0,len(df['bbw'])], [yolo_config*0.4,yolo_config*0.4],'k-',lw=1)
    # plt.plot([0,len(df['bbw'])], [416*0.2,416*0.2],'k-',lw=1)
    plt.title('Change of Bounding Box Length with Respect to Time')

if plot_velocity:
    plt.plot(df.index, df['vx'],label='linear velocity in x axis (m/s)')
    plt.plot(df.index, df['vy'],label='linear velocity in y axis (m/s)')
    plt.plot(df.index, df['vz'],label='linear velocity in y axis (m/s)')
    plt.plot(df.index, df['wx'],label='angular velocity in x axis (m/s)')
    plt.plot(df.index, df['wy'],label='angular velocity in y axis (m/s)')
    plt.plot(df.index, df['wz'],label='angular velocity in z axis (m/s)')
    plt.plot([0,len(df['bbcx'])], [0,0],'k-',lw=1)
    plt.title('Change of Robot Velocity with Respect to Time')
    

# plt.plot(x,y_new)
# plt.xlabel('robot velocity in x axis (m/s)')
# plt.ylabel('bounding box center position (pixel)')
plt.xlabel('time')

plt.legend()
plt.show()