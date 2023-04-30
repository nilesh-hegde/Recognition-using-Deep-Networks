import tensorflow as tf
import matplotlib.pyplot as plt
import Task1A_E
import numpy as np
import cv2

'''
Arguments - Tensorflow model
Return - None
Description - This function is used to predict and plot the first 10 examples in test data using the trained model
'''
def predict_and_plot_test(model):
    # Read and process data
    mnist_dataset = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
    test_images = test_images.astype("float32") / 255
    test_images = np.expand_dims(test_images, -1)
    num_labels = 10
    test_labels = tf.keras.utils.to_categorical(test_labels, num_labels)
    
    # Predict the data
    predictions = model.predict(test_images[:10])
    
    # Plot the data
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('MNIST Test Set Predictions')
    print("\nMNSIT testing data set")
    for i in range(10):
        if i!=9:
            ax = axs[i//3, i%3]
            ax.imshow(test_images[i], cmap='gray')
            ax.set_title(f'Prediction:{tf.argmax(predictions[i])}')
            ax.axis('off')
        print("Prediction : {} , Truth : {}".format(tf.argmax(predictions[i]),list(test_labels[i]).index(1)))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.8,wspace=1,hspace=0.4)
    plt.show()
    return

'''
Arguments - Tensorflow model
Return - None
Description - This function reads the custom handwritten data and predicts them using the trained model
'''
def predict_new(model):
    # Read and process custom data
    test_files = []
    for i in range(10):
        img = cv2.imread("mydigits/" + str(i) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        img = np.array(img).astype("float32") / 255
        test_files.append(img)
    test_files = np.array(test_files)
    test_files = np.expand_dims(test_files, -1)
    test_files_labels = list(range(10))
    test_files_labels = tf.keras.utils.to_categorical(test_files_labels, 10)
    
    # Predict the data
    predictions = model.predict(test_files)
    
    # Plot the data
    fig, axs = plt.subplots(2, 5)
    fig.suptitle('New Input Predictions')
    print("\nCustom testing data set")
    for i in range(10):
        ax = axs[i//5, i%5]
        ax.imshow(test_files[i], cmap='gray')
        ax.set_title(f'Prediction:{tf.argmax(predictions[i])}')
        ax.axis('off')
        print("Prediction : {} , Truth : {}".format(tf.argmax(predictions[i]),list(test_files_labels[i]).index(1)))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=1,hspace=0.4)
    plt.show()
    return

'''
Arguments - None
Return - None
Description - This function starts execution and calls function to perform all sub-questions F and G of Task 1
'''
def main():
    # Load the model
    model = Task1A_E.MyModel()
    model.load_weights('DNNweights').expect_partial()
    
    # Predict test data
    predict_and_plot_test(model)
    
    # Predict custom test data
    predict_new(model)

if __name__ == "__main__":
   main()
