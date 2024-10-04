import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import os
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Try adjusting this value

# 读取txt文件，返回标签和序列
def load_data_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    labels = []
    sequences = []
    for line in lines:
        parts = line.strip().split('\t')  # 假设文件以tab分隔
        if len(parts) == 3:
            labels.append(parts[1])  # 第二列是分类标签
            sequences.append(parts[2])  # 第三列是蛋白质序列
    return labels, sequences

char_to_int = {
    'A': 1,  # Alanine
    'C': 2,  # Cysteine
    'D': 3,  # Aspartic Acid
    'E': 4,  # Glutamic Acid
    'F': 5,  # Phenylalanine
    'G': 6,  # Glycine
    'H': 7,  # Histidine
    'I': 8,  # Isoleucine
    'K': 9,  # Lysine
    'L': 10, # Leucine
    'M': 11, # Methionine
    'N': 12, # Asparagine
    'P': 13, # Proline
    'Q': 14, # Glutamine
    'R': 15, # Arginine
    'S': 16, # Serine
    'T': 17, # Threonine
    'V': 18, # Valine
    'W': 19, # Tryptophan
    'Y': 20  # Tyrosine
}

# 序列编码，将序列转换为数字表示
def encode_sequences(sequences, char_to_int, max_length):
    encoded_sequences = [[char_to_int.get(char, 0) for char in seq] for seq in sequences]  # 对未知字符映射为0
    padded_sequences = pad_sequences(encoded_sequences, maxlen=max_length, padding='post')
    return padded_sequences

# 创建LSTM模型（用于二分类）
def create_lstm_model(input_length, vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=input_length),
        LSTM(64, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 二分类用sigmoid
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 训练单类二分类模型
from tensorflow.keras.callbacks import EarlyStopping

def train_single_class_model(train_sequences, train_labels, valid_sequences, valid_labels, max_length, char_to_int, class_name, model_dir, strategy):
    # 将标签转换为numpy数组，确保可以使用astype方法
    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)

    # 将标签编码为0/1（针对该类训练）
    binary_labels = (train_labels == class_name).astype(int)
    valid_binary_labels = (valid_labels == class_name).astype(int)

    # 在分布式策略的作用域内创建模型
    with strategy.scope():
        model = create_lstm_model(input_length=max_length, vocab_size=len(char_to_int))

    # 定义 Early Stopping 回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # 训练模型，并保存训练历史
    history = model.fit(
        train_sequences,
        binary_labels,
        epochs=200,  # 设置足够大的初始值
        batch_size=32,
        validation_data=(valid_sequences, valid_binary_labels),
        verbose=1,
        callbacks=[early_stopping]  # 加入 EarlyStopping 回调
    )

    # 输出每个 epoch 的损失值
    print(f"Training loss for class {class_name}:")
    for epoch, loss in enumerate(history.history['loss']):
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    # 保存模型
    model.save(os.path.join(model_dir, f'{class_name}_model.h5'))

    # 保存训练历史为独立的PDF文件
    pdf_filename = os.path.join(model_dir, f'{class_name}_training_history.pdf')
    plot_history_to_pdf(history, class_name, pdf_filename)


# 可视化训练历史并保存到PDF
def plot_history_to_pdf(history, class_name, pdf_filename):
    # 创建PDF文件
    with PdfPages(pdf_filename) as pdf:
        # 创建图形
        plt.figure(figsize=(10, 4))

        # 损失图
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss for {class_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 准确率图
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy for {class_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # 保存图形到PDF
        pdf.savefig()  # 将当前页保存到PDF
        plt.close()  # 关闭当前图，防止它在下次循环时重复显示

    print(f"Training history for {class_name} saved to {pdf_filename}")

# 主函数：针对每个类别训练独立的二分类模型
def train_models_for_all_classes(train_labels, train_sequences, valid_labels, valid_sequences, max_length, char_to_int, model_dir, strategy):
    unique_classes = set(train_labels)

    for class_name in unique_classes:
        print(f"\nTraining model for class: {class_name}")
        train_single_class_model(train_sequences, train_labels, valid_sequences, valid_labels, max_length, char_to_int, class_name, model_dir, strategy)

# 设置分布式策略
strategy = tf.distribute.MirroredStrategy()

# 加载训练数据
train_file = 'train.txt'
valid_file = 'Validation.txt'
train_labels, train_sequences = load_data_from_txt(train_file)
valid_labels, valid_sequences = load_data_from_txt(valid_file)

# 假设序列长度上限为100
max_length = 3000
train_sequences = encode_sequences(train_sequences, char_to_int, max_length)
valid_sequences = encode_sequences(valid_sequences, char_to_int, max_length)

# 模型保存目录
model_dir = 'models_all_gpu/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 训练所有类别的二分类模型
train_models_for_all_classes(train_labels, train_sequences, valid_labels, valid_sequences, max_length, char_to_int, model_dir, strategy)

