import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# تحميل النتائج
initial_results = pd.read_csv("D:\\initial_test_results.csv")  # تصنيف الخطأ فقط
final_results = pd.read_csv("D:\\final_test_results.csv")  # تصنيف بناءً على العتبة

# استخراج الخطأ والتصنيفات
initial_labels = initial_results["Initial Label"].values
final_labels = final_results["Final Label"].values
mse_per_record = final_results["Error"].values

# (Confusion Matrix) حساب مصفوفة الارتباك 
cm = confusion_matrix(initial_labels, final_labels)
print(f"Confusion Matrix:\n{cm}")

#(Precision, Recall, F1-Score) حساب تقرير التصنيف 
report = classification_report(initial_labels, final_labels)
print(f"Classification Report:\n{report}")
# استخراج القيم من مصفوفة الارتباك
TN, FP, FN, TP = cm.ravel()
# حساب نسبة الفشل الكلية
failure = (FP + FN) / (TN + FP + FN + TP)
print(f"Failure:\n{failure}")

#   بناءً على العتبة المحددة  ROC منحنى
fpr, tpr, thresholds = roc_curve(final_labels, initial_labels)
roc_auc = auc(fpr, tpr)

#  ROC رسم منحنى 
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("D:\\roc_curve.png")  # حفظ منحنى AUC
plt.show()