# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:00:40 2019

@author: wang
"""

import cvxpy as cvx
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (1.5, 1.5)
plt.rcParams['savefig.dpi'] = 600 
plt.rcParams['figure.dpi'] = 600
import seaborn as sns
import networkx as nx
import math
import scipy
import csv
from scipy.cluster import hierarchy  
from scipy import cluster
from collections import Counter
from sklearn import metrics

#Normalize matrix's value to [0,1]
def normalization(X_matrix):
    X_matrix=X_matrix-np.median(X_matrix)
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            X_matrix[i,j]=1.0/(1.0+math.exp(-X_matrix[i,j]))    
    return X_matrix


#get sample's subgroup information
def subtype_information(subtype_series):
    count=1
    subtype_list=[]
    for i in range(len(subtype_series)-1):
        if subtype_series[i]==subtype_series[i+1]:
            count=count+1
        else:
            subtype_list.append(count)
            count=1
    subtype_list.append(count)
    
    cancer_subtype=[] 
    for i in range(len(subtype_list)):
        count=0
        for j in range(0,i+1):
            count+=subtype_list[j]
        cancer_subtype.append(count)
    return cancer_subtype



'''
calculate clustering evaluation metric
purity,NMI,ARI
'''

#(1)purity

def purity(cluster_result,gold_standard,n_clusters):    

    if len(set(cluster_result))==n_clusters:
        cluster_dict={}
        for i in range(n_clusters):
            cluster_dict[i]=[]
            ele_index = np.argwhere(cluster_result == [i])
            sub_subtype=gold_standard[ele_index[:,0]]
            for j in range(len(sub_subtype)):
                cluster_dict[i].append(sub_subtype[j])
        
        cluster=np.zeros(n_clusters) 
        for i in range(n_clusters):
            cluster[i]=max(Counter(cluster_dict[i]).values())
        
        total_num=float(cluster_result.shape[0])
        purity=cluster.sum()/total_num
        return purity
    if len(set(cluster_result))!=n_clusters:
        print 'Error: incorrect cluster number!'

#(2)ARI
def ARI(labels_true,labels_pred):
    labels_pred_list=list(labels_pred)       
    labels_true_list=list(labels_true)

    return metrics.adjusted_rand_score(labels_true_list,labels_pred_list)

#(3)NMI
def NMI(labels_true,labels_pred):
    
    labels_pred_list=list(labels_pred)        
    labels_true_list=list(labels_true)

    return metrics.normalized_mutual_info_score(labels_true_list,labels_pred_list)



# functions to be used in CBP_SMF_sovler about display results

#scatter cost
def cost_scatter(cost):
    num=[i for i in range(len(cost))]
    plt.figure()
    plt.title('cost',fontsize=25)
    plt.xlabel('iteration')
    plt.ylabel('F-norm')
    plt.scatter(num,cost,10,color='r',marker='^')


#clustermap for Vl
def clustermap(V_labeled,labelsample_list,col_color):
    #plt.figure()
    #plt.title('clustermap for V(labeled)',fontsize=25)
    rank=V_labeled.shape[0]
    V_result=pd.DataFrame(columns=labelsample_list,index=[i for i in range(0,rank)]).fillna(0)
    for i in range(rank):
        for j in range(len(labelsample_list)):
            V_result.iloc[i,j]=V_labeled[i,j]   
    #return sns.clustermap(V_result,col_colors=col_color,col_cluster=None,row_cluster=None,xticklabels=False)
    sns.clustermap(V_result,col_colors=col_color,col_cluster=None,row_cluster=None,xticklabels=False).fig.suptitle('clustermap for Vl matrix',fontsize=25) 
    #plt.show()
    
#plot for every module on V    
def module_plot(V_labeled,subtype):
    plt.figure()
    plt.title('module plot',fontsize=25)
    color=['bo','go','r^','y*','k+']
    style=['solid','dashed','dashdot','dotted','solid']
    for row in range(V_labeled.shape[0]):
        plt.plot(V_labeled[row,:],color[row],markersize=2.5,linestyle=style[row],linewidth=2.2,label='module '+str(row))
    for sample in subtype:
        plt.plot([sample,sample],[0,1.0],'k--',linewidth=1.5)
    plt.legend(loc='upper left')
    #plt.show()

#module's boxplot
def module_boxplot(V_labeled,labelsample_list,gold_subtype):
    plt.figure()
    plt.title('module boxplot',fontsize=25)
    rank=V_labeled.shape[0]
    V_result=pd.DataFrame(columns=labelsample_list,index=[i for i in range(0,rank)]).fillna(0)
    for i in range(rank):
        for j in range(len(labelsample_list)):
            V_result.iloc[i,j]=V_labeled[i,j]
    n1=V_result.shape[0]
    V_boxplot=V_result.T
    V_boxplot['subtype']=gold_subtype
    #plt.figure(figsize=(7,9),facecolor='white',frameon=True)
    for n2 in range(n1):
        figname=int(str(n1)+str(1)+str(n2+1))
        plt.subplot(figname)   
        sns.boxplot(x='subtype',y=n2,width=0.5,fliersize=0.1,data=V_boxplot)
    plt.show()
        
    



'''
############################################################
#####             /* CBP_SMF_Solver */                 #####
############################################################
'''

class CBP_SMF_Solver():
    def __init__(self,X_input,rank,n_iter,beta,omega,corr_matrix,labeled_count,subgroup,multi_runtimes,consensus_runtimes,labeled_sample_list,unlabeled_sample_list,z_threshold,background_network,col_color):
        self.X_input=X_input
        self.rank=rank
        self.n_iter=n_iter
        self.beta=beta       
        self.omega=omega
        self.corr_matrix=corr_matrix
        self.labeled_count=labeled_count
        self.subgroup=subgroup
        self.multi_runtimes=multi_runtimes
        self.consensus_runtimes=consensus_runtimes
        self.labeled_sample_list=labeled_sample_list
        self.unlabeled_sample_list=unlabeled_sample_list
        self.z_threshold=z_threshold
        self.background_network=background_network
        self.col_color=col_color
        
        self.gamma=0
        self.m=[x.shape[0] for x in X_input]
        self.n=X_input[0].shape[1]
        self.Wa,self.Da,self.Wp,self.Dp,self.La,self.Lp= self.Laplacian(corr_matrix,labeled_count,labeled_count[-1], self.n-labeled_count[-1])
   
    def Laplacian(self,corr_matrix,subtype,num_label,num_nolabel):
        #a
        Wa=np.zeros((corr_matrix.shape[0],corr_matrix.shape[1]))
        Wa[0:subtype[0],0:subtype[0]]=corr_matrix[0:subtype[0],0:subtype[0]]
        for i in range(len(subtype)-1):
            Wa[subtype[i]:subtype[i+1],subtype[i]:subtype[i+1]]=corr_matrix[subtype[i]:subtype[i+1],subtype[i]:subtype[i+1]]
        Da=np.zeros((corr_matrix.shape[0],corr_matrix.shape[1]))
        for i in range(Da.shape[0]):
            Da[i,i]=np.sum(Wa[i])    
        La=Da-Wa
        
        #p
        if num_nolabel!=0:
            ratio=float(num_label)/num_nolabel
            Wp=0.1*ratio*(corr_matrix-Wa)
        else:
            Wp=0.1*(corr_matrix-Wa)
        Dp=np.zeros((corr_matrix.shape[0],corr_matrix.shape[1]))
        for i in range(Dp.shape[0]):
            Dp[i,i]=np.sum(Wp[i])
        Lp=Dp-Wp
        
        return Wa,Da,Wp,Dp,La,Lp 
    
    
    #parameter recommendation of beta , omega
    def parameter_recommendation(self):
        initial_U,initial_W,initial_V=self.initialize(self.rank)
        f_norm=0
        
        for x in range(len(self.X_input)):
            f_norm+=(1.0/len(self.X_input))*np.linalg.norm(self.X_input[x]-np.dot(initial_U[x],np.dot(initial_W[x],initial_V)),"fro")**2
       
        Vl=initial_V[:,0:self.labeled_count[-1]]
        
        affinity=np.dot(Vl,np.dot(self.La,Vl.T))
        penalty=np.dot(Vl,np.dot(self.Lp,Vl.T))    
        semi_supervised_norm=np.trace(affinity)-np.trace(penalty) 
       
        for t in range(10):
            if f_norm/(10**t) < 10:
                break
            
        recom_beta=f_norm/(100.0*semi_supervised_norm)    
        recom_gamma=10**t
        
        print 'CBP_SMF recommendation of beta : %4.4f (±50%%)'%recom_beta
        print 'CBP_SMF recommendation of gamma: %d (±50%%)'%recom_gamma
        

        
    def initialize(self,rank):
        U=[]
        for count in range(len(self.m)):
            U.append(np.random.random((self.m[count],rank)))
        W=[]
        for count in range(len(self.m)):
            W_n=np.zeros((rank,rank))+0.1/(rank-1)
            for i in range(W_n.shape[0]):
                W_n[i,i]=0.9
            W.append(W_n)
        V=np.random.random((rank,self.n))
        return U,W,V        

    def NMF(self,init_U,init_W,init_V):   
        eps = np.spacing(1)    
        X=self.X_input
        
        Xl=[]
        Xul=[]
        for n in range(len(X)):
            Xl.append(X[n][:,0:self.labeled_count[-1]])
            Xul.append(X[n][:,self.labeled_count[-1]:X[n].shape[1]])
      
        U=copy.deepcopy(init_U)
        W=copy.deepcopy(init_W)
        V=init_V
        Vl=V[:,0:self.labeled_count[-1]]
        Vul=V[:,self.labeled_count[-1]:V.shape[1]]
    
        PI=[1.0/len(X) for i in range(len(X))]
        
        cost=np.zeros(self.n_iter)
        for n in range(len(X)):
            locals()['cost_c'+str(n+1)]=np.zeros(self.n_iter)
        cost_laplacian=np.zeros(self.n_iter)
        
        constraints=[]
        constraints_sum=0
        for n in range(len(X)):
            locals()['pi'+str(n+1)]=cvx.Variable()
            constraints.append(locals()['pi'+str(n+1)]>0)
            constraints_sum+=locals()['pi'+str(n+1)]
        constraints.append(constraints_sum==1)
    
        for it in range(self.n_iter):   

            #update U,W
            for i in range(len(X)):
                substitu_1=np.dot(U[i],W[i])
                substitu_2=np.dot(V,V.T) 
                substitu=np.dot(substitu_1,substitu_2)
                U[i]=np.multiply(U[i],((np.dot(X[i],np.dot(V.T,W[i].T)))/(np.dot(substitu,W[i].T)+eps)))
                W[i]=np.multiply(W[i],((np.dot(U[i].T,np.dot(X[i],V.T)))/(np.dot(U[i].T,substitu)+eps)))        
            
            ##update Vl,Vul
            substitu_l=0;substitu_ul=0;substitu_l_down=0;substitu_ul_down=0
            for i in range(len(X)):
                substitu_l+=PI[i]*np.dot(W[i].T,np.dot(U[i].T,Xl[i]))
                substitu_ul+=PI[i]*np.dot(W[i].T,np.dot(U[i].T,Xul[i]))
                substitu_l_down+=PI[i]*np.dot(W[i].T,np.dot(U[i].T,np.dot(U[i],np.dot(W[i],Vl))))
                substitu_ul_down+=PI[i]*np.dot(W[i].T,np.dot(U[i].T,np.dot(U[i],np.dot(W[i],Vul))))                    
            judge_matrix=(substitu_l+self.beta*np.dot(Vl,(self.Wa+self.Dp)))
    
            Vl=np.multiply(Vl,(judge_matrix/(substitu_l_down+self.gamma*Vl+self.beta*np.dot(Vl,(self.Da+self.Wp))+eps)))
            Vul=np.multiply(Vul,((substitu_ul)/(substitu_ul_down+self.gamma*Vul+eps)))
            V=np.hstack((Vl,Vul))            
            
            ##update PI
            for i in range(len(X)):
                locals()['c'+str(i+1)]=np.linalg.norm(X[i]-np.dot(U[i],np.dot(W[i],V)),"fro")**2
            claplacian=np.trace(np.dot(Vl,np.dot((self.La-self.Lp),Vl.T))) 
            
            for i in range(len(X)):
                locals()['cost_c'+str(i+1)][it]=locals()['c'+str(i+1)]  
            cost_laplacian[it]=claplacian
            
            #cvx to optimize
            target_sum=0
            target_square=0
            for i in range(len(X)):
                target_sum+=locals()['c'+str(i+1)]*locals()['pi'+str(i+1)]
                target_square+=self.omega*(locals()['pi'+str(i+1)])**2
            target=target_sum+target_square
            obj=cvx.Minimize(target)
            prob = cvx.Problem(obj,constraints)
            prob.solve()
            
            for i in range(len(X)):
                PI[i]=locals()['pi'+str(i+1)].value
            
            affinity=np.dot(Vl,np.dot(self.La,Vl.T))
            penalty=np.dot(Vl,np.dot(self.Lp,Vl.T))
            cost_sum=0;cost_weight=0
            for i in range(len(X)):
                cost_sum+=PI[i]*locals()['c'+str(i+1)] 
                cost_weight+=PI[i]**2
            cost[it]=cost_sum+self.beta*(np.trace(affinity)-np.trace(penalty))+self.gamma*np.linalg.norm(V,"fro")**2+self.omega*cost_weight
            #decide self.beta if set self.beta=0        
            if cost_sum+self.beta*(np.trace(affinity)-np.trace(penalty))<0:
                self.beta=0
        cost_fnorm=[locals()['cost_c'+str(n+1)] for n in range(len(X))]
        return cost,cost_fnorm,cost_laplacian,U,W,V,PI     
    
    def multirun(self):
        cost_thres=1e10
        purity_thres=0.0
        purity_list=[]
        ARI_list=[]
        NMI_list=[]
        
        for run in range(self.multi_runtimes):
            print 'runtimes:[',run+1,'/%d]'%self.multi_runtimes
            init_U,init_W,init_V=self.initialize(self.rank)###
            cost,cost_fnorm,cost_laplacian,U,W,V,PI=self.NMF(init_U,init_W,init_V)
            
            sample_class=self.cluster_V(V)
            sample_class_labeled=sample_class[0:self.labeled_count[-1]]       
            #print len(set(sample_class_labeled))
            purity_list.append(purity(sample_class_labeled,self.subgroup,self.rank))
            ARI_list.append(ARI(self.subgroup,sample_class_labeled))
            NMI_list.append(NMI(self.subgroup,sample_class_labeled))
            
            if cost[-1]<cost_thres and purity_list[-1]>purity_thres:
                print 'optimize done'
                cost_thres=cost[-1]
                purity_thres=purity_list[-1]
                cost_min=cost
                U_min=U
                W_min=W
                V_min=V
                PI_min=PI
        return cost_min,U_min,W_min,V_min,PI_min,purity_list,ARI_list,NMI_list  


    def cluster_V(self,V_result):
        sample_class=np.zeros(V_result.shape[1])
        for sam in range(len(sample_class)):
            sample_class[sam]=np.argmax(V_result[:,sam])
        return sample_class     

   
    #consensus clustering
    def consensus(self,custom_rank):
        consensus_matrix_init=np.zeros((self.labeled_count[-1],self.labeled_count[-1]))
        for run in range(self.consensus_runtimes):
            print 'running:',run
            consensus_matrix=np.zeros((self.labeled_count[-1],self.labeled_count[-1]))
            sampleclass=np.zeros(consensus_matrix.shape[0])
            init_U,init_W,init_V=self.initialize(custom_rank)
            cost,cost_fnorm,cost_laplacian,U,W,V,PI=self.NMF(init_U,init_W,init_V)
            V_consensus=V[:,0:self.labeled_count[-1]]
            for num in range(V_consensus.shape[1]):
                sampleclass[num]=np.argmax(V_consensus[:,num])#use argmax(V's column) as cluster label
            
            for i in range(consensus_matrix.shape[0]):
                for j in range(consensus_matrix.shape[1]):
                    if sampleclass[i]==sampleclass[j]:
                        consensus_matrix[i,j]=1.0
                        consensus_matrix[j,i]=1.0
                    else:
                        consensus_matrix[i,j]=0.0
                        consensus_matrix[j,i]=0.0
            consensus_matrix_init+=consensus_matrix
        consensus_matrix_final=consensus_matrix_init/self.consensus_runtimes
        return consensus_matrix_final

    def consensus_ordered(self,custom_rank):
        consensus_matrix_init=pd.DataFrame(columns=self.labeled_sample_list,index=self.labeled_sample_list).fillna(0)
        for run in range(self.consensus_runtimes):
            #print 'running:',run
            consensus_matrix=pd.DataFrame(columns=self.labeled_sample_list,index=self.labeled_sample_list).fillna(0)
            sampleclass=np.zeros(consensus_matrix.shape[0])
            init_U,init_W,init_V=self.initialize(custom_rank)
            cost,cost_fnorm,cost_laplacian,U,W,V,PI=self.NMF(init_U,init_W,init_V)
            V_consensus=pd.DataFrame(columns=self.labeled_sample_list,index=[i for i in range(0,custom_rank)]).fillna(0)
            for i in range(custom_rank):
                for j in range(self.labeled_count[-1]):
                    V_consensus.iloc[i,j]=V[:,0:self.labeled_count[-1]][i,j]
            
            for num in range(V_consensus.shape[1]):
                sampleclass[num]=np.argmax(V_consensus.iloc[:,num])
            
            for i in range(consensus_matrix.shape[0]):
                for j in range(consensus_matrix.shape[1]):
                    if sampleclass[i]==sampleclass[j]:
                        consensus_matrix.iloc[i,j]=1.0
                        consensus_matrix.iloc[j,i]=1.0
                    else:
                        consensus_matrix.iloc[i,j]=0.0
                        consensus_matrix.iloc[j,i]=0.0
            consensus_matrix_init+=consensus_matrix
            #print consensus_matrix_init
        consensus_matrix_final=consensus_matrix_init/self.consensus_runtimes
        
        return consensus_matrix_final
    
    def cophenetic(self,rank_start,rank_end):
        
        cophenet=[]
        for i in range(rank_start,rank_end+1):
            print 'running rank:',i
            consensus_result_rank=self.consensus_ordered(i)
            Z = hierarchy.linkage(consensus_result_rank, method ='average',metric='euclidean')
            Y=scipy.spatial.distance.pdist((np.eye(self.labeled_count[-1])-np.asmatrix(consensus_result_rank)))
            c,d=cluster.hierarchy.cophenet(Z,Y)
            cophenet.append(c)
        
        plt.figure()
        plt.title('cophenetic')
        plt.xlabel('rank')
        plt.grid(True)
        plt.plot([k for k in range(rank_start,rank_end+1)],cophenet)
        plt.scatter([k for k in range(rank_start,rank_end+1)],cophenet,marker='o')
        plt.show()
        return cophenet  
     
    def subtype_module_correspond(self,V_result):
        cluster_result=self.cluster_V(V_result)
        if len(set(cluster_result))==len(set(self.subgroup)):
            cluster_result_labeled=cluster_result[0:self.labeled_count[-1]]
            cluster_dict={}
            for i in range(len(set(cluster_result))):
                cluster_dict[i]=[]
                ele_index = np.argwhere(cluster_result_labeled == [i])
                sub_subtype=self.subgroup[ele_index[:,0]]
                for j in range(len(sub_subtype)):
                    cluster_dict[i].append(sub_subtype[j])   
            cluster_subtype={}
            for i in range(len(set(cluster_result))):
                cluster_subtype[i]=[k for k, v in Counter(cluster_dict[i]).items() if v==max(Counter(cluster_dict[i]).values())][0]
            print 'Corresponding relationship between module and subtype are:'
            for i in range(len(set(cluster_result))):
                print 'CBPs_module:'+str(i+1)+' corresponds to '+'subtype:'+cluster_subtype[i]
        else:
            print 'number of module does not equal number of subtype!Run again please'
            
            
    def getmodulegene(self,genelist,U_matrix):
        list_modulelist=[]
        num_module=U_matrix.shape[1]
        for i in range(num_module):
            module_list=[]
            transf=(U_matrix[:,i]-np.average(U_matrix[:,i]))/np.std(U_matrix[:,i],ddof=1)
            indices=list(np.argwhere(transf>self.z_threshold)[:,0])
            for j in indices:
                module_list.append(genelist[j])
            list_modulelist.append(module_list)
        return list_modulelist
    
    def getCBPedges(self,module_list):
        list_CBPedges=[]
        for i in range(len(module_list[0])):
            multi_module=[]
            for j in range(len(module_list)):
                multi_module.extend(module_list[j][i])           
            #map module onto backgroud network
            list_CBPedges.append(self.background_network.subgraph(multi_module).edges())
        
        return list_CBPedges

    def predict_subtype(self,V_result):
        cluster_result=self.cluster_V(V_result)
        cluster_result_labeled=cluster_result[0:self.labeled_count[-1]]
        cluster_dict={}
        for i in range(len(set(cluster_result))):
            cluster_dict[i]=[]
            ele_index = np.argwhere(cluster_result_labeled == [i])
            sub_subtype=self.subgroup[ele_index[:,0]]
            for j in range(len(sub_subtype)):
                cluster_dict[i].append(sub_subtype[j])   
        cluster_subtype={}
        for i in range(len(set(cluster_result))):
            cluster_subtype[i]=[k for k, v in Counter(cluster_dict[i]).items() if v==max(Counter(cluster_dict[i]).values())][0]
        subtype_list=[cluster_subtype[i] for i in cluster_result]
        
        prediction_dict={}
        for i in range(len(self.unlabeled_sample_list)):
            prediction_dict[self.unlabeled_sample_list[i]]=subtype_list[i+self.labeled_count[-1]]
        return prediction_dict  
    
    def writeCBP(self,edges):
        for i in range(len(edges)):
            with open('result\\CBPs_module'+str(i+1)+'.csv', 'wb',) as csvfile:
                writer = csv.writer(csvfile)
                for edge in edges[i]:
                    writer.writerow(edge)
                    
    def writeSampleGroup(self,prediction_result):
        with open('result\\unlabeled_samples_group.csv', 'wb',) as csvfile:
            writer = csv.writer(csvfile)
            for key, value in prediction_result.items():
                writer.writerow([key, value])
            
    def display_analysis(self,V_result,cost):
        
        sample_class=self.cluster_V(V_result)
        sample_class_labeled=sample_class[0:self.labeled_count[-1]]
        print 'Purity:'
        print purity(sample_class_labeled,self.subgroup,self.rank)
        print 'ARI:'
        print ARI(self.subgroup,sample_class_labeled)
        print 'NMI:'
        print NMI(self.subgroup,sample_class_labeled)
        
        #plt.subplot(4,1,1)

        cost_scatter(cost)
        
        #plt.subplot(4,1,2)
        clustermap(V_result[:,0:self.labeled_count[-1]],self.labeled_sample_list,self.col_color)
       
        #plt.subplot(4,1,3)
        module_plot(V_result[:,0:self.labeled_count[-1]],self.labeled_count)
        
        module_boxplot(V_result[:,0:self.labeled_count[-1]],self.labeled_sample_list,self.subgroup)

        
                

if __name__=='__main__':
    
    X1_test_df=pd.read_csv('E:\\python code\\Article\\data\\final2_genematrix.csv',index_col='Keys')
    X2_test_df=pd.read_csv('E:\\python code\\Article\\data\\final_miRNAmatrix.csv',index_col='sample')
    corr_test_df=pd.read_csv('E:\\python code\\Article\\data\\final_corrmatrix.csv',index_col='Unnamed: 0')
    
    X1_feature=list(X1_test_df.index)
    X2_feature=list(X2_test_df.index)
    
    
    X1_test=X1_test_df.as_matrix()
    X2_test=X2_test_df.as_matrix()
    
    corr_test=corr_test_df.as_matrix()

    X1_test=normalization(X1_test)
    X2_test=normalization(X2_test)
    X_input=[X1_test,X2_test]

    annotation=pd.read_csv('E:\\python code\\Article\\data\\final_annotation.csv').set_index('Unnamed: 0') 
    subtype = annotation.pop("subtype")
    subtype_color={'Luminal A':'r','Luminal B':'b','Basal like':'g','HER2 enriched':'y','Normal like':'k'}
    col_color=subtype.map(subtype_color)
    
    G=nx.DiGraph()
    backgroud=open("E:\\python code\\Article\\CBP-SMF-package-withnet\\data\\backgroud_network.txt", 'rb')
    for line in backgroud.readlines():
        G.add_edge(line.split()[0],line.split()[1])
            
    
    
    labeled_count=subtype_information(subtype)
    labeled_sample_list=X1_test_df.columns[0:labeled_count[-1]]
    unlabeled_sample_list=X1_test_df.columns[labeled_count[-1]:]
    
    CBP_SMF=CBP_SMF_Solver(X_input=X_input,rank=4,n_iter=50,beta=10,gamma=0,omega=100000,
                           corr_matrix=corr_test,labeled_count=labeled_count,subgroup=subtype,
                           multi_runtimes=10,consensus_runtimes=10,
                           labeled_sample_list=labeled_sample_list,unlabeled_sample_list=unlabeled_sample_list,
                           z_threshold=2.0,col_color=col_color)
    

    
    #single run NMF framework
    #initial_U,initial_W,initial_V=CBP_SMF.initialize(rank=4)
#    cost,cost_fnorm,cost_laplacian,U,W,V,PI=CBP_SMF.NMF(initial_U,initial_W,initial_V)
    
    '''multi-run NMF framework'''
    cost_min,U_min,W_min,V_min,PI_min,purity_list,ARI_list,NMI_list=CBP_SMF.multirun()
    
    '''result display'''
    CBP_SMF.display_analysis(V_min,cost_min)
    
    '''module extract and subtype predict'''
    CBP_SMF.subtype_module_correspond(V_min)
    prediction_dict=CBP_SMF.predict_subtype(V_min)
    module_from_X1=CBP_SMF.getmodulegene(X1_feature,U_min[0])
    module_from_X2=CBP_SMF.getmodulegene(X2_feature,U_min[1])
   
    
    '''if don't know choice of k, use CBP_SMF.cophenetic '''   
    CBP_SMF.parameter_recommendation()
    cophenet = CBP_SMF.cophenetic(rank_start=3,rank_end=5)
    



