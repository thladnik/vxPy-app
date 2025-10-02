import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt
import functions as fc
import pandas as pd

file_path = ('dpf2_6_Fischdip')
cleaned = f'C:\\Users\\hiwi\\PyCharmProjects\\vxPy-app\\recordings\\{file_path}\\Cleaned\\Clean_Fischdip_Analyse_2_6.hdf5'

Cleaned_dict = fc.load_dict_from_hdf5(cleaned)
data = pd.read_hdf(cleaned)

unique_ap = list(data.angular_period.unique())
unique_ap = unique_ap[~np.isnan(unique_ap)]
unique_ap = np.sort(unique_ap)

unique_av = list(data.angular_velocity.unique())
unique_av.remove(0)
unique_av = np.array(unique_av)
unique_av = unique_av[~np.isnan(unique_av)]
unique_av = np.sort(unique_av)

for ap in unique_ap:
    ap_df = data.loc[data.angular_period == ap]
    for av in unique_av:
        av_df = ap_df.loc[ap_df.angular_velocity == av]
        unique_phase = av_df.group.unique()
        for phase in unique_phase:
            phase_df = av_df.loc[av_df.group == phase]
            plt.plot(phase_df.time, phase_df.left_eye_pos, label = "left eye", color="plum")
            plt.plot(phase_df.time, phase_df.right_eye_pos, label= "right eye", color="cyan")
            plt.legend(title="angular velocity: " + str(av))
            plt.xlabel("Zeit [s]")
            plt.ylabel("Winkel [°]")
            plt.title("angular period: " + str(ap))
            plt.show()