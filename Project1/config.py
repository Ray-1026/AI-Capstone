# hyperparameters and configurations
num_epochs = 50  # 50 for resnet, 100 for vae
batch_size = 16
learning_rate = 1e-4

# for KNN
K = 3

model_type = "cnn"  # choose model type from ["cnn", "vae", "knn", "fcn"]

# output filename
output_filename = "prediction.csv"
