import matplotlib.pyplot as plt
from myTool.myTools import DrawClassyPic
import numpy as np
from myNet.myNet import MyNeuralNet
from myMakeDate.myDateSets import loadRBF01, load_planar_dataset, load_extra_DataSets



if __name__ == '__main__':
#加载数据
    # X, Y = load_planar_dataset()
    X, Y = load_extra_DataSets("noisy_moons")
    # X,Y = loadRBF01()
#模型参数配置 以及 训练
    learning_rate = 0.1
    nn_model = MyNeuralNet(layers = np.array([10, 1]), hid_active = 'tanh')
    nn_model.nn_fit(X, Y, n_iterations = 20000, learningRate = learning_rate, PrintCost=True, GradCheck=False)
    predictions = nn_model.predict(X)
    score = predictions.ravel() == Y.ravel()
    print ('Net分类正确率:'+str(np.mean(score)*100)+'%')
#可视化
    nn_model.printParam()
    plt.plot(np.squeeze(nn_model.getLoss()))
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    DrawClassyPic(X, Y.ravel(), classy_fun=nn_model.predict, title='Net测试')

    plt.show()