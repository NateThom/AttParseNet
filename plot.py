import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


### READING DATA ###
path_to_data_1 = "../output_predictions/metrics/all_transforms_unaligned_test_metrics.txt"
# path_to_data_2 = "../output_predictions/metrics/final_baseline_unaligned_test_metrics.txt"
# path_to_data_3 = "../output_predictions/metrics/final_baseline_aligned_test_metrics.txt"
path_to_data_4 = "../output_predictions/metrics/final_baseline_aligned_test_metrics.txt"

attributes_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                                'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                                'Eyeglasses', 'Goatee', 'Grey_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Side_Burns',
                                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

# metrics_1 = pd.read_csv(path_to_data_1, usecols=range(1, 41))
# metrics_1_list = metrics_1.values.tolist()

# metrics_2 = pd.read_csv(path_to_data_2, usecols=range(1, 41))
# metrics_2_list = metrics_2.values.tolist()

metrics_1 = pd.read_csv(path_to_data_1, usecols=range(1, 41))
metrics_1_list = metrics_1.values.tolist()

# metrics_2 = pd.read_csv(path_to_data_2, usecols=range(1, 41))
# metrics_2_list = metrics_2.values.tolist()

# metrics_3 = pd.read_csv(path_to_data_3, usecols=range(1, 41))
# metrics_3_list = metrics_3.values.tolist()

metrics_4 = pd.read_csv(path_to_data_4, usecols=range(1, 41))
metrics_4_list = metrics_4.values.tolist()

# acc_1 = metrics_1_list[0]
# pos_acc_1 = metrics_1_list[1]
# neg_acc_1 = metrics_1_list[2]
# precision_1 = metrics_1_list[3]
# recall_1 = metrics_1_list[4]

# acc_2 = metrics_2_list[0]
# pos_acc_2 = metrics_2_list[1]
# neg_acc_2 = metrics_2_list[2]
# precision_2 = metrics_2_list[3]
# recall_2 = metrics_2_list[4]

acc_1 = metrics_1_list[0]
pos_acc_1 = metrics_1_list[1]
neg_acc_1 = metrics_1_list[2]
precision_1 = metrics_1_list[3]
recall_1 = metrics_1_list[4]

# acc_2 = metrics_2_list[0]
# pos_acc_2 = metrics_2_list[1]
# neg_acc_2 = metrics_2_list[2]
# precision_2 = metrics_2_list[3]
# recall_2 = metrics_2_list[4]

# acc_3 = metrics_3_list[0]
# pos_acc_3 = metrics_3_list[1]
# neg_acc_3 = metrics_3_list[2]
# precision_3 = metrics_3_list[3]
# recall_3 = metrics_3_list[4]

acc_4 = metrics_4_list[0]
pos_acc_4 = metrics_4_list[1]
neg_acc_4 = metrics_4_list[2]
precision_4 = metrics_4_list[3]
recall_4 = metrics_4_list[4]

# PLOTTING

fig = plt.figure(figsize=(10,5))

# # Line 1
# plt.subplot()
# plt.plot(attributes_list, acc_1, marker='o', label="Old Baseline")
# # plt.ylabel("Precision")
# plt.xticks(rotation=90)
#
# # Line 2
# plt.subplot()
# plt.plot(attributes_list, acc_2, marker='o', label="New Baseline", color="orange")
# # plt.ylabel("Precision")
# plt.xticks(rotation=90)

# Line 1
plt.subplot()
plt.plot(attributes_list, acc_1, marker='o', label="Segmentation")
# plt.ylabel("Precision")
plt.xticks(rotation=90)

# # Line 2
# plt.subplot()
# plt.plot(attributes_list, recall_2, marker='o', label="Without Segmentation - Unaligned", color="orange")
# # plt.ylabel("Precision")
# plt.xticks(rotation=90)

# # Line 3
# plt.subplot()
# plt.plot(attributes_list, recall_3, marker='o', label="Without Segmentation - Aligned", color="red")
# # plt.ylabel("Precision")
# plt.xticks(rotation=90)

# Line 4
plt.subplot()
plt.plot(attributes_list, acc_4, marker='o', label="Without Segmentation", color="orange")
# plt.ylabel("Recall")
plt.xticks(rotation=90)

# P & R
# recall_1, precision_1 = zip(*sorted(zip(recall_1, precision_1)))

# plt.subplot()
# plt.plot(recall_1, precision_1, marker='o', label="AttParseNet Unaligned Test P&R", color="green")
# plt.ylabel("Precision")
# plt.xlabel("Recall")
# plt.xticks(rotation=90)

# Create grid background and legend/key centered on top of plot
plt.grid()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
plt.legend(loc='best', ncol=3, fancybox=True, shadow=True)
plt.title("AttParseNet on Unaligned Data vs. Aligned Baseline on Aligned Data")

# Save or show the plot
# plt.show()
# plt.savefig('../output_predictions/metrics/attparsenet_vs_baseline_cropped_umd_acc_fixed_2.jpg', bbox_inches="tight", dpi=1000)
# plt.savefig('../output_predictions/metrics/attparsenet_vs_baseline_cropped_lfwa_metrics_final.jpg', bbox_inches="tight", dpi=1000)
plt.savefig('../output_predictions/metrics/attparsenet_unaligned_vs_baseline_aligned_acc_test_final.jpg', bbox_inches="tight", dpi=1000)
