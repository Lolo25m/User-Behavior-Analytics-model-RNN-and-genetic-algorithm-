import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# تحميل البيانات
data = pd.read_csv('D:\\ processed_logon.csv')

#  datetime تحويل التواريخ إلى كائنات 
data['logontime'] = pd.to_datetime(data['logontime'], format='%d/%m/%Y %H:%M:%S')
data['logofftime'] = pd.to_datetime(data['logofftime'], format='%d/%m/%Y %H:%M:%S')

# حساب الفارق الزمني (عدد الثواني منذ بداية اليوم)
data['logontime'] = data['logontime'].dt.hour * 3600 + data['logontime'].dt.minute * 60 + data['logontime'].dt.second
data['logofftime'] = data['logofftime'].dt.hour * 3600 + data['logofftime'].dt.minute * 60 + data['logofftime'].dt.second

# التطبيع الكامل للقيم العددية
scaler = MinMaxScaler()
numerical_columns = ['Duration', 'logontime', 'logofftime']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# ترميز القيم الفئوية وتحويلها إلى نطاق [0, 1]
categorical_columns = ['User', 'Pc', 'Weekday']
for column in categorical_columns:
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column])
    # تطبيع القيم الفئوية إلى نطاق [0, 1]
    data[column] = MinMaxScaler().fit_transform(data[column].values.reshape(-1, 1))

# indexالاحتفاظ بالترتيب الأصلي عن طريق الحفاظ على الـ 
data.reset_index(drop=True, inplace=True)

# حفظ البيانات المعالجة في ملف
data.to_csv('D:\\normalized_logon.csv', index=False)
print(" save the normalizid data in:'normalized_logon.csv'")

# تقسيم البيانات إلى تدريب واختبار (70% تدريب، 30% اختبار) مع الاحتفاظ بالترتيب
train_data = data.iloc[:int(0.7 * len(data))]
test_data = data.iloc[int(0.7 * len(data)):]

# حفظ بيانات التدريب والاختبار في ملفات CSV
train_data.to_csv('D:\\train_data.csv', index=False)
test_data.to_csv('D:\\test_data.csv', index=False)
print("The training and testing data have been saved in CSV files.")

# تحويل البيانات إلى الشكل ثلاثي الأبعاد (سجل-سجل دون خلط)
train_sequences = train_data.to_numpy().reshape(-1, 1, train_data.shape[1])  # كل سجل بشكل مستقل
test_sequences = test_data.to_numpy().reshape(-1, 1, test_data.shape[1])

# حفظ بيانات التدريب والاختبار في ملفات NPY
np.save('D:\\train_data.npy', train_sequences)
np.save('D:\\test_data.npy', test_sequences)
print(f"The training ({len(train_sequences)}) and testing ({len(test_sequences)}) data have been saved in NPY files.")