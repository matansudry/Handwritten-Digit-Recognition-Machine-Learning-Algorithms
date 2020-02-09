import pandas as pd
import networkx as nx

def take3(elem):
    return elem[2]

def create_factor(left, right):
    factor=[[left,right,'label','cnt'],[0,0,0,1],
    [0,0,1,1],[0,0,2,1],[0,0,3,1],[0,0,4,1],
    [0,0,5,1],[0,0,6,1],[0,0,7,1],[0,0,8,1],
    [0,0,9,1],[1,0,0,1],[1,0,1,1],[1,0,2,1],
    [1,0,3,1],[1,0,4,1],[1,0,5,1],[1,0,6,1],
    [1,0,7,1],[1,0,8,1],[1,0,9,1],[0,1,0,1],
    [0,1,1,1],[0,1,2,1],[0,1,3,1],[0,1,4,1],
    [0,1,5,1],[0,1,6,1],[0,1,7,1],[0,1,8,1],
    [0,1,9,1],[1,1,0,1],[1,1,1,1],[1,1,2,1],
    [1,1,3,1],[1,1,4,1],[1,1,5,1],[1,1,6,1],
    [1,1,7,1],[1,1,8,1],[1,1,9,1]]
    return factor

def create_factor_one_varible(left):
    factor=[[left,'label','cnt'],[0,0,0],
    [0,1,0],[0,2,0],[0,3,0],[0,4,0],
    [0,5,0],[0,6,0],[0,7,0],[0,8,0],
    [0,9,0],[1,0,0],[1,1,0],[1,2,0],
    [1,3,0],[1,4,0],[1,5,0],[1,6,0],
    [1,7,0],[1,8,0],[1,9,0]]
    return factor

def create_graphs(X_test, y_test,cnt,limit):
    train_size = len(X_test)
    list_of_graphs=[]
    labels_cnt={}
    for i in range(10):
        labels_cnt[i]=0
    for i in range(train_size):
        if labels_cnt[y_test[i]]>=cnt:
            continue
        labels_cnt[y_test[i]] +=1
        list_of_graphs.append((create_graph(X_test[i],limit),y_test[i]))
    return (list_of_graphs)

def create_graph(vec_of_pic,limit):
    G_t = nx.Graph()
    for i in range(784):
        if vec_of_pic[i]<=limit:
            x=0
        else:
            x=1
        G_t.add_node(i+1,value=x)
    for i in range(1,784):
        if(i%28!=0):
            factor=create_factor(i,i+1)
            G_t.add_edge(i, i+1,value=factor)
        if(i<=756):
            factor=create_factor(i,i+28)
            G_t.add_edge(i, i+28,value=factor)
    return G_t

def Normalized_Cuts_all_graphs(list_of_graphs):
    list_of_graphs2=[]
    for image_graph in list_of_graphs:
        list_of_graphs2.append((Normalized_Cuts(image_graph[0]),image_graph[1]))
    return (list_of_graphs2)

def Normalized_Cuts(image_graph):
    del_list=[]
    for edge in image_graph.edges.data():
        node_1=image_graph.nodes._nodes[edge[0]]['value']
        node_2=image_graph.nodes._nodes[edge[1]]['value']
        if (node_1>0 and node_2 == 0) or (node_1==0 and node_2 >0): 
            del_list.append((edge[0],edge[1]))
    image_graph.remove_edges_from(del_list)      
    return image_graph

def create_pixals_counter2(list_of_graphs):
    pixals_counter={}
    edges_to_add=[]
    for i in range(1,784):
        if(i%28!=0):
            edges_to_add.append((i,i+1))
        if(i<=756):
            edges_to_add.append((i,i+28))
    for i in range (1512):
        temp=create_factor(edges_to_add[i][0], edges_to_add[i][1])
        pixals_counter[edges_to_add[i]]=temp
    for graph in list_of_graphs:
        for edge in graph[0].edges():
            left=graph[0].nodes._nodes[edge[0]]['value']
            right=graph[0].nodes._nodes[edge[1]]['value']
            cnt=0
            for row in pixals_counter[edge]:
                if cnt==0:
                    cnt+=1
                    continue
                if(row[0]==left and row[1]== right and row[2]==graph[1]):
                    row[3]+=1
    pixals_counter=normalized_factor(pixals_counter)
    return pixals_counter

def normalized_factor(pixals_counter):
    for edge in pixals_counter:
        sum_00=0
        sum_01=0
        sum_10=0
        sum_11=0
        cnt=0
        for row in pixals_counter[edge]:
            if cnt == 0:
                cnt+=1
                continue
            if(row[0]==0 and row[1]==0):
                sum_00+=row[3]
            elif (row[0]==0 and row[1]==1):
                sum_01+=row[3]
            elif (row[0]==1 and row[1]==0):
                sum_10+=row[3]
            elif (row[0]==1 and row[1]==1):
                sum_11+=row[3]
        for row in pixals_counter[edge]:
            if cnt == 1:
                cnt+=1
                continue
            if(row[0]==0 and row[1]==0):
                row[3]=row[3]/sum_00
            elif (row[0]==0 and row[1]==1):
                row[3]=row[3]/sum_01
            elif (row[0]==1 and row[1]==0):
                row[3]=row[3]/sum_10
            elif (row[0]==1 and row[1]==1):
                row[3]=row[3]/sum_11
    return pixals_counter

def getting_msg(temp_belief,temp_belief_one_varible,i):
    if temp_belief[0][0]==i:
        index=0
    else:
        index= 1
    cnt=0
    for j in temp_belief:
        if cnt==0:
            cnt+=1
            continue
        temp_value=j[index]
        temp_label=j[2]
        temp_preb=j[3]
        for m in temp_belief_one_varible:
            if (m[0]==temp_value and m[1]==temp_label):
                m[2]+=temp_preb
                break
    return temp_belief_one_varible

def sum_all_probs(temp_belief_one_varible):
    if len(temp_belief_one_varible)==0:
        return temp_belief_one_varible
    if len(temp_belief_one_varible)==1:
        for i in temp_belief_one_varible:
            return temp_belief_one_varible[i]
    else:
        cnt=0
        for i in temp_belief_one_varible:
            if cnt == 0:
                cnt+=1
                temp_answer=temp_belief_one_varible[i]
                continue
            counter=0
            cnt2=0
            for row in temp_belief_one_varible[i]:
                if cnt2==0:
                    cnt2+=1
                    counter+=1
                    continue
                temp_answer[counter][2]=temp_answer[counter][2]*row[2]
    sum=0
    cnt2=0
    for row in temp_answer:
        if cnt2==0:
            cnt2+=1
            continue
        sum+= row[2]
    cnt2=0
    for row in temp_answer:
        if cnt2==0:
            cnt2+=1
            continue
        row[2]=row[2]/sum   
    return temp_answer

def sending_msg(temp_belief,temp_belief_one_varible,i):
    if temp_belief[0][0]==i:
        index=0
    else:
        index= 1
    cnt=0
    counter=0
    for j in temp_belief:
        if cnt==0:
            cnt+=1
            counter+=1
            continue
        temp_value=j[index]
        temp_label=j[2]
        temp_preb=j[3]
        for row in temp_belief_one_varible:
                if (row[0]==temp_value and row[1]==temp_label):
                    temp_belief[counter][3]= temp_belief[counter][3]*temp_preb
                    counter+=1
                    break
    return temp_belief

def bp (pixals_counter,g_t):
    temp2=pixals_counter
    for i in range(1,785):
        temp_belief_one_varible={}
        ok=0
        neighbers=g_t.edges._adjdict[i]     
        for j in neighbers:
            if j < i:
                temp_belief=temp2[j, i]
                ok=1
                temp_belief_one_varible[j]=create_factor_one_varible(i)
                temp_belief_one_varible[j]=getting_msg(temp_belief,temp_belief_one_varible[j],i)
        total_msg=sum_all_probs(temp_belief_one_varible)
        for j in neighbers:
            if ok==0:
                break
            if j>i:
                temp_belief=temp2[i, j]
                temp_belief=sending_msg(temp_belief,total_msg,i)
                temp2[i,j]=temp_belief
    temp2=normalized_factor(temp2)   
    return temp2

def bp2 (pixals_counter,g_t):
    temp2=pixals_counter
    for i in range(1,785):
        m=785-i
        temp_belief_one_varible={}
        ok=0
        neighbers=g_t.edges._adjdict[m]     
        for j in neighbers:
            if j > m:
                temp_belief=temp2[m, j]
                ok=1
                temp_belief_one_varible[j]=create_factor_one_varible(m)
                temp_belief_one_varible[j]=getting_msg(temp_belief,temp_belief_one_varible[j],m)
        total_msg=sum_all_probs(temp_belief_one_varible)
        for j in neighbers:
            if ok==0:
                break
            if j<m:
                temp_belief=temp2[j, m]
                temp_belief=sending_msg(temp_belief,total_msg,m)
                temp2[j,m]=temp_belief
    temp2=normalized_factor(temp2)   
    return temp2
        
def normalized_predict(predict):
    sum=0
    for i in range(10):
        sum+=predict[i]
    if sum == 0:
        return predict
    for i in range(10):
        predict[i]=predict[i]/sum
    return predict

def run_test(label,img_test,test_labels,number_of_images):
    cnt_pic=0
    edges_to_add=[]
    matrix=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]]
    for i in range(1,784):
        if(i%28!=0):
            edges_to_add.append((i,i+1))
        if(i<=756):
            edges_to_add.append((i,i+28))
    for temp_img in img_test:
        if cnt_pic==number_of_images:
            break
        predict={}
        for label_loop in range (10):
            predict[label_loop]=0.1
        for j in edges_to_add:
            predict=normalized_predict(predict)
            factor=label[j]
            left=factor[0][0]
            right=factor[0][1]
            left_predict=temp_img[left-1]
            right_predict=temp_img[right-1]
            if(left_predict>0):
                left_predict=1
            if(right_predict>0):
                right_predict=1
            cnt=0
            for row in factor:
                if cnt == 0 :
                    cnt+=1
                    continue
                for label_pre in range(10):    
                    if(row[0]==left_predict and row[1]==right_predict and row[2]==label_pre):
                        predict[label_pre]=predict[label_pre]*row[3]
                        break
        label_prediction=max(predict, key=predict.get)
        matrix[label_prediction][test_labels[cnt_pic]]+=1
        cnt_pic+=1
    return(matrix)

def run_test2(upper_bound,low_bound,img_test,test_labels,number_of_images,pixals_counter,limit,number_of_edges):
    probability_matrix=[[],[],[],[],[],[],[],[],[],[]]
    matrix=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]]
    for i in pixals_counter:
        for j in range(1,41):
            if pixals_counter[i][j][3]>upper_bound or pixals_counter[i][j][3]<low_bound:
                probability_matrix[pixals_counter[i][j][2]].append((i,pixals_counter[i][j],pixals_counter[i][j][3]))
    for i in range(10):
        probability_matrix[i].sort(key=take3,reverse=True)
    for i in range(10):
        while (len(probability_matrix[i])>number_of_edges*2):
            probability_matrix[i].remove(probability_matrix[i][number_of_edges+1])
    cnt=0
    for image_test in img_test:
        predict={}
        if cnt==number_of_images:
            break
        for label_loop in range (10):    
            predict[label_loop]=0
        for label_loop in range (10):
            position_number=-1
            for i in probability_matrix[label_loop]:
                position_number+=1
                if(image_test[i[0][0]]==i[1][0] and image_test[i[0][1]]==i[1][1] and i[1][3]>0.1):
                        predict[label_loop]+=(number_of_edges/2)-position_number
                elif (image_test[i[0][0]]==i[1][0] and image_test[i[0][1]]==i[1][1] and i[1][3]<0.1):
                        predict[label_loop]-=position_number-(number_of_edges/2)
                elif((image_test[i[0][0]]!=i[1][0] and i[1][3]>0.1) or (image_test[i[0][1]]!=i[1][1] and i[1][3]>0.1)):
                        predict[label_loop]-=(number_of_edges/2)-position_number
                elif ((image_test[i[0][0]]!=i[1][0] and i[1][3]<0.1) or  (image_test[i[0][1]]==i[1][1] and i[1][3]<0.1)):
                        predict[label_loop]+=position_number-(number_of_edges/2)
        label_prediction=max(predict, key=predict.get)
        matrix[label_prediction][test_labels[cnt]]+=1
        cnt+=1
    return matrix

def image_binary(img_test,limit):
    for image_test in img_test:
        for i in range(784):
            if(image_test[i]>limit):
                image_test[i]=1
            else:
                image_test[i]=0
    return
def print_matrix(matrix,number_of_images,results_file):
    for i in range(10):
        print(matrix[i])
    sum=0
    for i in range(10):
        sum+=matrix[i][i]
    print ("accuracy = " + str(sum/number_of_images*100) +"%\n")