from matplotlib import pyplot as plt
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.legend(['value'],loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_curve(data,data_test):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color='blue', label="Train Loss")
    plt.plot(range(len(data_test)), data_test, color='red', label="Val Loss")
    plt.legend(['train loss','test'],loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()