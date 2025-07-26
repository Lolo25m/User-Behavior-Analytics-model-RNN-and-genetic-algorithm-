import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.integrate import trapz
from scipy.stats import gaussian_kde

# تحميل النموذج المحفوظ وبيانات الاختبار
model = load_model('D:\\gru_model_genetic.keras')
test_data = np.load('D:\\test_data.npy').astype(np.float32)  # تقليل استهلاك الذاكرة باستخدام float32

# التنبؤ وإعادة بناء المدخلات
batch_size = 1024
reconstructed_data = []

# معالجة البيانات على دفعات لتقليل استهلاك الذاكرة
for i in range(0, test_data.shape[0], batch_size):
    batch_data = test_data[i:i+batch_size]
    reconstructed_batch = model.predict(batch_data, verbose=0)
    reconstructed_data.extend(reconstructed_batch)

reconstructed_data = np.array(reconstructed_data, dtype=np.float32)  # تقليل استهلاك الذاكرة باستخدام float32

# حساب الخطأ لكل سجل على دفعات
mse_per_record = []

for i in range(0, test_data.shape[0], batch_size):
    batch_data = test_data[i:i+batch_size]
    reconstructed_batch = reconstructed_data[i:i+batch_size]
    mse_batch = np.mean(np.square(batch_data - reconstructed_batch), axis=(1, 2))
    mse_per_record.extend(mse_batch)

mse_per_record = np.array(mse_per_record)

# حساب الوسيط والانحراف
Median_error = np.median(mse_per_record)
Std_error = np.std(mse_per_record)
Mean_error = np.mean(mse_per_record)
Q1 = np.percentile(mse_per_record, 25)  # الربيع الأول
Q3 = np.percentile(mse_per_record, 75)  # الربيع الثالث
print(f"Median Error: {Median_error}")
print(f"std Error: {Std_error}")
print(f"Mean Error: {Mean_error}")
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
#print(2.5*Mean_error-(0.5*Median_error+1.5*Q1))
#print(Mean_error+(0.5*Median_error-0.5*Q1))
#print(0.5*Mean_error+ Std_error+(0.5*Median_error+Q1))
#print(2.5*Q3)
print(Mean_error+ 2*Std_error)

# تصنيف السجلات بناءً على الخطأ فقط (بدون العتبة)
# إذا كان الخطأ كبيرًا، فإن السجل سيكون شاذًا (1)، وإذا كان صغيرًا، فإن السجل سيكون غير شاذ (0)
initial_labels = np.where(mse_per_record > Mean_error + 2*Std_error , 1, 0)  # تصنيف مبدئي بناءً على الخطأ

# حفظ النتائج المبدئية (التصنيف بناءً على الخطأ فقط)
initial_results = pd.DataFrame({
    "Error": mse_per_record,
    "Initial Label": initial_labels
})
initial_results.to_csv("D:\\initial_test_results.csv", index=False)
print("Initial test results saved to D:\\initial_test_results.csv")

# تحديد العتبة باستخدام النسبة المئوية (مثلاً: 80%)
threshold = np.percentile(mse_per_record,95)
#threshold = 2*Mean_error + 3*Std_error 
print(f"Selected Threshold: {threshold}")

# إعادة التصنيف بناءً على مقارنة الخطأ مع العتبة
final_labels = np.where(mse_per_record > threshold, 1, 0)  # إذا كان الخطأ أكبر من العتبة، السجل شاذ

# حفظ النتائج النهائية بعد مقارنة الخطأ بالعتبة
final_results = pd.DataFrame({
    "Error": mse_per_record,
    "Final Label": final_labels
})
final_results.to_csv("D:\\final_test_results.csv", index=False)
print("Final test results saved to D:\\final_test_results.csv")

# # رسم توزيع السلوك الشاذ والطبيعي
# normal_errors = mse_per_record[mse_per_record <= threshold]
# anomalous_errors = mse_per_record[mse_per_record > threshold]

# print(f"Normal count: {len(normal_errors)}")
# print(f"Anomalous count: {len(anomalous_errors)}")
# print(f"Normal min/max: {normal_errors.min()} / {normal_errors.max()}")
# print(f"Anomalous min/max: {anomalous_errors.min()} / {anomalous_errors.max()}")



# #رسم منحنيات الطبيعي والشاذ
# plt.figure(figsize=(10, 6))
# sns.kdeplot(normal_errors, label="Normal Errors", color="blue", fill=True)
# sns.kdeplot(anomalous_errors, label="Anomalous Errors", color="red", fill=True)
# plt.title("Comparison of Normal vs Anomalous Errors")
# plt.xlabel("Reconstruction Error")
# plt.ylabel("Density")
# plt.legend()
# plt.savefig("D:\\Normal_vs_Anomalous.png")  # حفظ الرسم
# plt.show()


 # بيانات الخطأ حسب الفئات
normal_errors = mse_per_record[final_labels == 0]
anomalous_errors = mse_per_record[final_labels == 1]
 # توليد KDE يدويًا
x_vals = np.linspace(min(mse_per_record), max(mse_per_record), 1000)
kde_normal = gaussian_kde(normal_errors)
kde_anomalous = gaussian_kde(anomalous_errors)
 # الكثافات
density_normal = kde_normal(x_vals)
density_anomalous = kde_anomalous(x_vals)
 # منطقة التداخل: حيث الكثافتان > 0.01 (قيمة صغيرة لتحديد التقاطع الحقيقي)
intersection_mask = (density_normal > 0.01) & (density_anomalous > 0.01)
intersection_range = x_vals[intersection_mask]
min_intersection = intersection_range.min()
max_intersection = intersection_range.max()
 # ترتيب البيانات
sorted_indices = np.argsort(mse_per_record)
sorted_errors = mse_per_record[sorted_indices]
sorted_labels = final_labels[sorted_indices]
# تلوين حسب منطقة التداخل
colors = []
for error, label in zip(sorted_errors, sorted_labels):
    if min_intersection <= error <= max_intersection:
        colors.append('yellow')  # تداخل كثافي فعلي
    else:
        colors.append('blue' if label == 0 else 'red')
# الرسم
plt.figure(figsize=(14, 6))
plt.scatter(range(len(sorted_errors)), sorted_errors, c=colors, alpha=0.6, s=10)
plt.title(" Scatter Plot of All Errors with Overlap Region")
plt.xlabel("Sorted Record Index")
plt.ylabel("Reconstruction Error")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Normal (0)', markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Anomalous (1)', markerfacecolor='red', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Overlap Region ', markerfacecolor='yellow', markersize=8)
])
plt.tight_layout()
plt.savefig("D:\\Scatter_Anomalies_vs_Normal.png")
plt.show()



# رسم توزيع الخطأ
plt.figure(figsize=(10, 6))
plt.hist(mse_per_record, bins=50, color='blue', alpha=0.7, label='Reconstruction Error')
#plt.hist(normal_errors, bins=50, color='blue', alpha=0.7, label='Normal Errors')
#plt.hist(anomalous_errors, bins=50, color='red', alpha=0.7, label='Anomalous Errors')
plt.axvline(x=np.percentile(mse_per_record,95), color='red', linestyle='--', label='Threshold (95th Percentile)')
plt.title("Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("D:\\error_distribution.png")  # حفظ الرسم
plt.show()