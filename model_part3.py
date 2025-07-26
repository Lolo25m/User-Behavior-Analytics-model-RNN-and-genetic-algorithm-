import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
import pygad
import pygad.kerasga
import matplotlib.pyplot as plt

# تحميل بيانات التدريب 
train_data = np.load('D:\\train_data.npy')

# Keras بناء النموذج باستخدام 
model = Sequential()
model.add(GRU(24, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True, activation='tanh'))
model.add(GRU(12, return_sequences=False, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(6, activation='tanh'))
model.add(Dense(train_data.shape[2], activation='sigmoid'))

# تلخيص النموذج
model.summary()
print("Model built successfully")

# KerasGA  باستخدام  PyGAD-compatible تحويل النموذج إلى 
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=8)

# قائمة لتخزين قيم الخطأ لكل جيل
error_list = []

# دالة حساب الخطأ (fitness function)
def fitness_function(ga_instance , solution, solution_idx ):
    print(f"Start Fitness Calculation for Solution {solution_idx}")
    global train_data, keras_ga

    # تعيين الأوزان الحالية للنموذج
    model_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(model_weights)

    # التنبؤ ومعالجة البيانات في دفعات لتقليل الذاكرة
    batch_size = 1024
    mse_list = []
    for i in range(0, train_data.shape[0], batch_size):
        batch_data = train_data[i:i+batch_size]
        predictions = model.predict(batch_data, verbose=0)
        mse_batch = np.mean(np.square(batch_data - predictions), axis=(1, 2))
        mse_list.extend(mse_batch)
    
    # حساب الخطأ (MSE)
    loss = np.mean(mse_list)
    print(f"End Fitness Calculation for Solution {solution_idx}:",loss)
    return -loss  #   fitness يحاول زيادة قيمة  PyGAD لأن 

# وظيفة تستدعى بعد كل جيل
def on_generation(ga_instance): 
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    error_list.append(-best_solution_fitness)  # تخزين قيمة الخطأ
    print(f"Completed Generation {ga_instance.generations_completed}")
    print(f"Best fitness in generation {ga_instance.generations_completed}: {-best_solution_fitness}")

# إعداد الخوارزمية الجينية
ga_instance = pygad.GA(
    num_generations=40,
    num_parents_mating=4,
    fitness_func=fitness_function,
    initial_population=keras_ga.population_weights,
    sol_per_pop=8,
    mutation_percent_genes=5 ,
    allow_duplicate_genes=False,
    on_generation=on_generation  # استدعاء الدالة بعد كل جيل
)

# تشغيل الخوارزمية الجينية
ga_instance.run()
print("Genetic algorithm completed")

# استخراج أفضل حل (أوزان النموذج)
best_solution, best_solution_fitness, _ = ga_instance.best_solution()
model_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=best_solution)
model.set_weights(model_weights)

# حفظ النموذج مع الأوزان المثلى
model.save('D:\\gru_model_genetic.keras')
print("Model saved as gru_model_genetic")

# رسم دالة الخطأ بعد إتمام الخوارزمية
plt.plot(error_list)
plt.title("Error Reduction per Generation")
plt.xlabel("Generation")
plt.ylabel("Error")
plt.savefig("D:\\error_plot_final.png")  # حفظ الرسم في ملف
plt.show()  # لعرض الرسم