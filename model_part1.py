import pandas as pd
from datetime import datetime

# حساب مدة الجلسة بين وقت تسجيل الدخول والخروج
def count_time (logon_time, logoff_time):
    return( logoff_time-logon_time).total_seconds()/3600

# حساب يوم الأسبوع من تاريخ معين
def find_weekday(time):
    return time.strftime('%A')

#  csv من ملف pandas قراءة البيانات باستخدام 
def log_feature(file_input):
    df=pd.read_csv(file_input, names=['id','date','user','pc','activity'],skiprows=1)
    # datetime الى  date تحويل من
    df['date']=pd.to_datetime(df['date'],format='%m/%d/%Y %H:%M:%S')
    return df

# logoffمع logon مطابقة سجلات
def mach_logon_logoff(df):
    session = []
    df = df.sort_values(by='date')  # فرز السجلات حسب التاريخ
    logon_dict = {}  # قاموس لتخزين أوقات تسجيل الدخول
    for index, row in df.iterrows():
        key = (row['user'], row['pc'])  # المفتاح هو المستخدم والجهاز

        if row['activity'].lower() == 'logon':  # التحقق من النشاط إذا كان Logon
            logon_dict[key] = row['date']  # إضافة وقت تسجيل الدخول إلى القاموس
        elif row['activity'].lower() == 'logoff':  # التحقق من النشاط إذا كان Logoff
            if key in logon_dict:  # التحقق إذا كان هناك وقت Logon مسبقًا للمستخدم والجهاز
                logon_time = logon_dict[key]
                logoff_time = row['date']
                duration = count_time(logon_time, logoff_time)  # حساب المدة
                weekday = find_weekday(logon_time)  # العثور على اليوم

                # إضافة الجلسة إلى القائمة
                session.append({
                    'User': row['user'],
                    'Pc': row['pc'],
                    'logontime': logon_time.strftime('%d/%m/%Y %H:%M:%S'),
                   'logofftime':logoff_time.strftime('%d/%m/%Y %H:%M:%S'),
                    'Duration': duration,
                    'Weekday': weekday
                })
                print(f"Session added: {session[-1]}")  # طباعة الجلسة المضافة

                del logon_dict[key]  # حذف دخول المستخدم بعد معالجته
            else:
                print(f"Logoff without Logon: {key}")  # طباعة رسالة في حالة وجود Logoff بدون Logon سابق
    return pd.DataFrame(session)


# حفظ البيانات المعالجة في ملف
def save_processed_logs(processed_df, file_output):
    processed_df.to_csv(file_output, index=False)
    print(f"Data saved to {file_output}")

# الخطوة الأساسية لمعالجة البيانات
def data_preprocess(file_input,file_output):
    df=log_feature(file_input)
    processed_df=mach_logon_logoff(df)
    save_processed_logs(processed_df,file_output)

file_input='D:\\logon.csv' # مسار الملف الأساسي
file_output='D:\\ processed_logon.csv' # مسار الملف الأول للمعالجة
data_preprocess(file_input,file_output)