import joblib
import os
import pickle
# 指定文件夹路径
#folder_path = 'E:\\LUAD_WSI\\luad_wsi'  #datas\luad\luad_256_features\np
folder_path = './upload_svs'
# folder_path = '../xiangya2.3'
# 存储文件名的列表
file_names = []
# 遍历文件夹中的所有文件
for file in os.listdir(folder_path):
    # 取文件名的前12个字符
    # file_name = file[:12]
    file_name = (file.split('.'))[0]#[:23]
    # 将文件名添加到列表中
    file_names.append(file_name)

# 保存列表为.pkl文件
output_file = './stas_patients.pkl'
# output_file = './stas_patients_xiangya2.3.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(file_names, f)


patients = joblib.load('./stas_patients.pkl')#('P:\\湘雅二\\STAS\\stas_patients_1.pkl') #("./stas_patients.pkl")#
print(len(patients))
print(patients)
import os
import csv

# 指定文件夹路径
folder_path = "./upload_svs"

# 获取文件夹中的文件名
file_names = os.listdir(folder_path)

# 构建 CSV 文件的数据
data = [("slide_id", "label")]

# 对文件名进行处理并添加到数据中
for file_name in file_names:
    if file_name.endswith(".svs"):
        slide_id = file_name.split(".")[0]
        label = "LUAD"  # 这里的标签可以根据需要更改
        data.append((slide_id, label))

# 指定要保存的 CSV 文件路径
csv_file_path = "./heatmaps/process_lists/heatmap_demo_dataset.csv"

# 将数据写入 CSV 文件
with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

print("CSV 文件已创建：", csv_file_path)
