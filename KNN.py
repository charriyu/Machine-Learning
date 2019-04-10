from numpy import *
import operator


# 建立数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lable = ['A','A','B','B']
    return group, lable;


# 控制输入inX 表示要测试的顶点数据
# dateSet表示二维数据集的数据进行训练
# K值表示进行KNN算法时进行的前k项选择
def classify(inX,dataSet,lable,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances**0.5
    print(distance)
    sortedDistance = distance.argsort()
    print(sortedDistance)
    classCount = {}
    for i in range(k):
        voteLable = lable[sortedDistance[i]]
        classCount[voteLable] = classCount.get(voteLable,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 将文本转换为NumPy的解析程序
def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOlines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOlines)
    # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    retuenMat = zeros((numberOfLines, 3))
    # 返回的分类标签向量
    classLableVector = []
    #行的索引值
    index = 0
    for line in arrayOlines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        listFromLine = line.split("\t")
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        retuenMat[index, :] = listFromLine[0:3]
        classLableVector.append(int(listFromLine[-1]))
        index += 1
    return retuenMat, classLableVector


# 归一化特征值，目的是为了将任意的取值范围值转换为0-1之间的值，使所有的影响相同
# 归一化公式：newValue = (oldValue - min)/(max - min)
def autoNorm(dataSet):
    # min中的参数0使得函数可以从列中选取最小值而不是选取当前行中最小值
    # 对numpy中参数为axis，其中axis没有设置时取所有值中的最小值，axis=0时取所有列的最小值，axsi=1时取所有行的最小值
    # 对于numpy中的max函数参数设置和前面min函数一样
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 取值为最大值和最小值之间的差值，结果是一个集合
    rangs = maxVals - minVals
    # zeros函数
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(rangs, (m, 1))
    return normDataSet, rangs, minVals
