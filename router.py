"""
https://blog.csdn.net/XpxiaoKr/article/details/51153259
https://blog.csdn.net/XpxiaoKr/article/details/51153259
    solve the logistic problem in geneti algorithms
"""
# -*- coding : uft-8 -*-
import math
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import copy

"""
    with reading the file named 'logistic.csv'
    use "with open()"
"""
itemList = \
    [[0, 0.0, 'Vijayawada', 19.4, 15.1, 0, 0, 0.0],
 [1, 1.9323814116266074, 'Tanguturu', 14.5, 5.3, 6600, 2, 1.0],
 [2, 20.857190790708088, 'Podili', 10.7, 7.0, 24000, 2, 0.5],
 [3, 33.077041744388232, 'Ongole', 14.5, 6.2, 305000, 4, 2.5],
 [4, 33.077041744388232, 'Markapur', 7.7, 8.2, 60000, 2, 0.5],
 [5, 34.486376839557948, 'KaniGiri', 9.6, 5.1, 24000, 2, 2.5],
 [6, 34.486376839557948, 'Kondukur', 13.2, 3.5, 90000, 2, 1.0],
 [7, 35.762110362784796, 'Giddalur', 3.8, 5.0, 25000, 2, 1.0],
 [8, 35.762110362784796, 'Chirala', 17.2, 9.0, 98000, 4, 2.0],
 [9, 41.826126709510191, 'Bestavapetta', 6.3, 6.3, 25000, 2, 0.5],
 [10, 46.397278485704312, 'Addanki', 13.9, 8.8, 60000, 2, 0.5],
 [11, 47.372912618077464, 'Chilakalurupet', 15.4, 11.4, 92000, 2, 1.0],
 [12, 59.32442405620133, 'Narasaraopet', 14.5, 12.5, 100000, 4, 1.0],
 [13, 62.987927732225053, 'Vinukonda', 11.8, 11.0, 65000, 2, 1.0],
 [14, 62.987927732225053, 'Tadikonda', 18.1, 14.3, 60000, 2, 1.0],
 [15, 66.744129468890392, 'Sattenapalle', 15.2, 14.0, 45000, 2, 1.0],
 [16, 68.538275316497419, 'Repalie', 21.3, 10.6, 50000, 2, 1.0],
 [17, 73.544821959944969, 'Guntur', 18.0, 13.0, 450000, 4, 3.0],
 [18, 73.544821959944969, 'Vuyyuru', 21.3, 13.6, 39000, 4, 1.0],
 [19, 74.453128626270612, 'Tenali', 19.7, 12.5, 140000, 4, 1.0],
 [20, 75.585513278670021, 'Pamarru', 22.3, 13.2, 62000, 2, 1.0],
 [21, 75.795182234229088, 'Nuzvid', 21.3, 17.5, 37000, 2, 0.5],
 [22, 75.795182234229088, 'Machilipatnam', 23.8, 12.0, 108000, 4, 1.0],
 [23, 77.065444537483828, 'Kaikalur', 24.4, 15.5, 48000, 2, 1.0],
 [24, 88.605535249215663, 'Jaggayyapeta', 14.9, 18.5, 37000, 2, 0.5],
 [25, 88.605535249215663, 'HanumenJunction', 19.5, 15.2, 50000, 2, 1.0],
 [26, 89.19358507516111, 'Gudivada', 22.7, 14.3, 180000, 2, 1.0],
 [27, 89.19358507516111, 'Bapatia', 18.2, 9.7, 82000, 2, 1.0],
 [28, 107.98018686407246, 'Rajahmundry', 29.5, 19.6, 470000, 4, 3.5],
 [29, 107.98018686407246, 'Mandapeta', 30.8, 18.3, 170000, 2, 2.0],
 [30, 114.27222071107218, 'Narasapur', 28.7, 14.5, 160000, 2, 1.0],
 [31, 117.99400024882618, 'Amaiapuram', 31.5, 15.6, 90000, 2, 1.0],
 [32, 119.69071254729833, 'Kakinada', 33.5, 19.1, 228000, 4, 2.0],
 [33, 119.69071254729833, 'Kovvur', 29.0, 19.7, 45000, 2, 1.0],
 [34, 127.33938989016715, 'Tanuku', 28.8, 17.4, 134000, 2, 1.0],
 [35, 132.23053168765526, 'Nidavole', 28.5, 18.7, 50000, 2, 1.0],
 [36, 133.71883894919219, 'Tadepallegudem', 27.7, 17.9, 130000, 4, 1.5],
 [37, 138.82247427963529, 'Eluru', 23.6, 17.0, 198000, 4, 2.0],
 [38, 138.82247427963529, 'Palakolu', 25.9, 15.7, 180000, 4, 1.0],
 [39, 145.45583114719054, 'Bhimavaram', 27.3, 15.3, 148000, 4, 1.5]]

for i in range(len(itemList)):
    if itemList[i][-2] == 4:
        temp = copy.deepcopy(itemList[i])
        itemList.append(temp)
for i in range(len(itemList)):
    itemList[i].append(i)

# sort the itemList with the distance
# save as item2
item2 = copy.deepcopy(itemList)
item2.sort(key = lambda x:x[1])

def getDistance(a, b, item=itemList):
    """
    get the distance between two points
    :param a: the first point
    :param b: the second point
    :param item: the itemList
    :return: the distance(float) between point a and point b
    """
    x1 = item[a][3]
    y1 = item[a][4]
    x2 = item[b][3]
    y2 = item[b][4]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

dis = []
for i in range(len(itemList)):
    for j in range(len(itemList)):
        dis.append(getDistance(i,j))

# the matrix of distance
distance = np.array(dis).reshape((53, 53)) * 12.2 * 1.12

def drawMap(item=itemList):
    for i in range(len(item)):
        if item[i][-3] == 4:
            plt.scatter(item[i][3],item[i][4],color='green',s=item[i][5]/3000.0)
        elif item[i][-3] == 2:
            plt.scatter(item[i][3],item[i][4],color='blue',s=item[i][5]/3000.0)
        else:
            plt.scatter(item[i][3],item[i][4],color='red',marker='*',s=500)
    plt.show()

def cost(routine, dis=distance):
    """
    get the cost of the routine
    the labour cost and the delivery cost
    :param routine : the routine
    :return: the cost(float) of the routine
    """
    # the labour cost
    labour_cost = 0.0
    for i in range(len(routine)):
        if i < 4:
            if len(routine[i]) != 0:
                labour_cost += 13500
        else:
            if len(routine[i]) != 0:
                labour_cost += 7000

    # the delivery cost
    delivery_cost = 0.0
    for i in range(len(routine)):
        if len(routine[i]) == 0:
            continue
        if i < 4:
            for j in range(1,len(routine[i])):
                delivery_cost += dis[routine[i][j - 1],routine[i][j]] * 5
        else:
            for j in range(1,len(routine[i])):
                delivery_cost += dis[routine[i][j - 1],routine[i][j]] * 3
    return labour_cost+delivery_cost

def generateRoutine(item=itemList,dis=distance):
    # 1-4 for T407
    # 5-8 FOR T310
    x1list = []
    x2list = []
    x3list = []
    x4list = []
    x5list = []
    x6list = []
    x7list = []
    x8list = []
    routine = [x1list,x2list,x3list,x4list,x5list,x6list,x7list,x8list]

    for i in range(1,len(item)): # except 0

        while True:
            # put 52 destinations into random 8 routines
            k = rnd.randint(0,7)
            # k = rnd.randint(0,5) # 6 trucks

            if item[i][5] > 350000:
                k = rnd.randint(0,3)

            # do not use T470 or T307
            # you can choose which car will be used
            if k == 3 or k == 2 or k == 1:
                continue
            if k == 6:
                continue


            # init the origin
            if len(routine[k]) == 0:
                routine[k].append(0)
                break
            # time constrain
            time = 0.0
            for j in range(1,len(routine[k])):
                time += getDistance(routine[k][j-1],routine[k][j]) / 40.0\
                        + item[routine[k][j]][-2]

            if time < 75 :
                if item[i][5] > 350000:
                    if k < 4:
                        break
                else:
                    break
                break

        # T407
        if k < 4:
            # judge whether the truck can go next destination
            if routine[k][-1] != 0:
                # loading constrain
                last = item[routine[k][-2]][5]
                now = 500000 - last
                if now < item[routine[k][-1]][5]:
                    routine[k].append(0)
                # time constrain
                past = dis[routine[k][-2]][routine[k][-1]] / 40.0 + item[routine[k][-2]][-2]
                future = item[routine[k][-1]][-2] + dis[routine[k][-1]][0] / 40.0
                if past + future > 8.5:
                    routine[k].append(0)

        # T310
        else:
            # judge whether the truck can go next destination
            if routine[k][-1] != 0:
                # loading constrain
                last = item[routine[k][-2]][4]
                now = 350000 - last
                if now < item[routine[k][-1]][4]:
                    routine[k].append(0)
                # time constrain
                past = dis[routine[k][-2]][routine[k][-1]] / 40.0 + item[routine[k][-2]][-2]
                future = item[routine[k][-1]][-2] + dis[routine[k][-1]][0] / 40.0
                if past + future > 8.5:
                    routine[k].append(0)


        routine[k].append(item[i][0])

        routine[k].append(0)

    return routine

def randomoptimize():
    """
    random searching
    :return: the best routine and the best cost
    """
    best = 999999999
    bestr = None
    for i in range(0,10000):
        # Create a random solution
        r = generateRoutine()
        # Get the cost
        simplify(r)
        c = cost(r)
        # Compare it to the best one so far
        if c < best:
            best = c
            bestr = r
            print best
            print bestr
            for t in range(len(bestr)):
                print t,'  >>>   ',bestr[t]
    return bestr,best

def getXY(num, item=itemList):
    """
    get the X and Y of a point
    :param num: the num of the point
    :param item: itemList
    :return: the X and Y of the point
    """
    return item[num][3],item[num][4]

def draw(routine,item=itemList):
    """
    draw the routine
    :param routine: the input routine
    :param item: itemList
    :return: the picture of the routine
    """
    # for i in range(len(itemList)):
    #     plt.scatter(getXY(itemList[i][0],item),'yellow')
    style = ['c-','r-','g-','b-','c--','r--','g--','b--']
    for i in range(len(routine)):
        pointX = []
        pointY = []
        for j in range(len(routine[i])):
            pointX.append(getXY(routine[i][j])[0])
            pointY.append(getXY(routine[i][j])[1])
        # plt.plot(point,color[i])
        # plt.scatter(pointX,pointY,style=style[i])
        plt.plot(pointX,pointY,style[i])

def p(routine):
    """
    print the routine in some way
    :param routine: the input routine
    :return: None. Print the routine in the console
    """
    for t in range(len(routine)):
        print t,'  >>>   ',routine[t]

def isValid(routine,dis=distance,item=itemList):
    """
    judge the routine whether is valid
    :param routine: the input routine
    :param dis: the distance matrix
    :return: if the routine is valid, return True. Otherwise return false
    """
    for k in range(len(routine)):
        for num in range(1,len(routine[k])):
            if routine[k][num] == 0:
                continue
            last = 0.0
            past = 0.0
            for i in range(len(routine[k])):
                if routine[k][num-1-i] == 0:
                    break
                last += item[routine[k][num-1-i]][5]
                past += dis[routine[k][num-2-i],routine[k][num-1-i]] / 40.0 + item[routine[k][num-1-i]][-2]
            if k < 4:
                now = 500000 - last
            else:
                now = 350000 - last
            future = item[routine[k][num]][-2] + float(dis[routine[k][num],0] / 40.0 + dis[routine[k][num-1],routine[k][num]] / 40.0)
            if now < item[routine[k][num]][5]:
                # print 'loading error at ',k,' : ',routine[k]
                # print 'at ',num
                # print 'last: ',last
                # print 'now: ',now
                # print 'need: ',itemList[routine[k][num]][5]
                return False
            if past + future > 14:
                # print 'time error at ',k,' : ',routine[k]
                # print 'at ',num
                # print 'past: ',past
                # print 'future: ',future
                return False
        sum = 0.0
        if num in range(1,len(routine[k])):
            sum += item[routine[k][num]][-2] + dis[routine[k][num-1],routine[k][num]] / 40.0
        if sum > 140:
            # print 'time out of limit at ',k,'--',num
            # print 'sum time is ',sum
            return False

    return True



def simplify(routine,dis=distance,item=itemList):
    """
    simplify the routine
    remove the 0 between the point which can be reached at one time
    add the 0 between the point which can not be reached at one time
    :param routine: the input routine
    :param dis: the distance matrix
    :return: the simplified routine
    """
    # remove 0
    for k in range(len(routine)):
        if len(routine[k]) > 0:
            record = []
            for num in range(2,len(routine[k])):
                if routine[k][num-1] == 0:
                    if routine[k][num-2] != 0:
                        last = 0.0
                        past = 0.0
                        for i in range(len(routine[k])):
                            if routine[k][num-2-i] == 0:
                                break
                            last += item[routine[k][num-2-i]][5]
                            past += float(dis[routine[k][num-3-i],routine[k][num-2-i]] / 40.0) + item[routine[k][num-2-i]][-2]
                        if k < 4:
                            now = 500000 - last
                        else:
                            now = 350000 - last
                        future = item[routine[k][num]][-2] + float(dis[routine[k][num],[routine[k][num-2]]] / 40.0 + dis[routine[k][num],0] / 40.0)
                        if now >= item[routine[k][num]][5]:
                            if past + future < 8.5:
                                if routine[k][num-1] == 0:
                                    record.append(num-1)
                    else:
                        record.append(num-1)
            if len(record) != 0:
                record.reverse()
                for num in record:
                    del(routine[k][num])
    # add 0
    for k in range(len(routine)):
        if len(routine[k]) > 0:
            num = 2
            while True:
                if num >= len(routine[k]):
                    break
                if routine[k][num-1] != 0:
                    last = 0.0
                    past = 0.0
                    for i in range(len(routine[k])):
                        last += item[routine[k][num-1-i]][5]
                        past += float(dis[routine[k][num-2-i],routine[k][num-1-i]] / 40.0) + \
                                item[routine[k][num-1-i]][-2]
                        if routine[k][num-2-i] == 0:
                            break
                    if k < 4:
                        now = 500000 - last
                    else:
                        now = 350000 - last
                    future = item[routine[k][num]][-2] + float(dis[routine[k][num], routine[k][num-1]] / 40.0 +
                                                               dis[routine[k][num], 0] / 40.0)
                    if now < item[routine[k][num]][5]:
                        routine[k].insert(num, 0)
                    elif past + future > 8.5:
                        routine[k].insert(num, 0)
                num += 1
    return routine




def crossover(r):
    """
    the operation of crossovering a routine
    choose 2 way to crossover part of the them and check the validity of the routine
    :param r : the input routine
    :return : the crossovered routine
    """
    num = 0
    while True:
        while True:
            k1 = rnd.randint(0,len(r)-1)
            k2 = rnd.randint(0,len(r)-1)
            if len(r[k1]) != 0 and len(r[k2]) != 0 and k1 != k2:
                break
        length = len(r[k1]) if len([k1])<len(r[k2]) else len(r[k2])
        # for j in range(2):
        i = rnd.randint(1,length)
        temp1 = r[k1][0:i]+r[k2][i:]
        temp2 = r[k2][0:i]+r[k1][i:]
        r[k1] = temp1
        r[k2] = temp2
        r = simplify(r)
        # print '-----go on crossover-----'
        if num > 100:
            print
            r = generateRoutine()
        if isValid(r):
            # r = simplify(r)
            return r
        num += 1

def geneticOptimize(item = itemList, popsize = 100,
                    elite = 0.7, maxiter = 100):
    """
    """

    # Build the initial population
    pop = []
    for i in range(popsize):
        routine = generateRoutine()
        pop.append(routine)
    saves = 99999999
    saver = []
    for i in range(maxiter):

        print '===============',i,'==============='

        scores = [(cost(v),v) for v in pop]
        scores.sort(key=lambda x:x[1])
        ranked = [v for (s,v) in scores]

        if saves > cost(scores[0][1]):
            saves = cost(scores[0][1])
            saver = scores[0][1]
            print saver
            print saves

        topelite = int(popsize*elite)
        pop = ranked[0:topelite]


        while True:
            c = rnd.randint(0,topelite-1)
            new = crossover(pop[c])
            print 'new--->',new
            pop.append(new)
            if len(pop) >= popsize:
                break

    return saver,saves,scores



a, b, c = geneticOptimize()
print a
print b
print '=============scores============='
print c[0]
print c[1]
print c[2]
--------------------- 
作者：XpxiaoKr 
来源：CSDN 
原文：https://blog.csdn.net/XpxiaoKr/article/details/51153259 
版权声明：本文为博主原创文章，转载请附上博文链接！