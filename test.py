#coding:utf-8
import os
import random
import time
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from speech_embedder_net import SpeechEmbedder

model_path = './speech_id_checkpoint/final_epoch_950_batch_id_141.model'
data = './test_tisv/speaker1.npy'
###one speaker utterances, some can be used for entrollment and others for evaluation.

def get_centroid(embeddings, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings):
        if utterance_id <= (utterance_num-1):
            #print(utterance.shape)
            centroid = centroid + utterance 
        else: break
    centroid = centroid/utterance_num
    return centroid

if (__name__=='__main__'):
    print('Load model successfully.')
    embedder_net = SpeechEmbedder().cuda()
    embedder_net.load_state_dict(torch.load(model_path))

    inputs = np.load(data)
    print(inputs.shape)
    entrollment_nums = 1
    ###we use two utterances for entrollment, and caculate their centroids
    inputs = torch.FloatTensor(np.transpose(inputs, axes=(0,2,1))).cuda()
    outputs = embedder_net(inputs)
    centroid_embeddings = get_centroid(outputs, entrollment_nums)
    centroid_embeddings = torch.unsqueeze(centroid_embeddings, 0).cpu()
    print(centroid_embeddings.size())
    test_nums = len(inputs) - entrollment_nums

    correct_nums = 0
    thres = 0.70
    for every_utter_idx in range(entrollment_nums, len(inputs)):
        output = embedder_net(torch.unsqueeze(inputs[every_utter_idx], dim=0)).cpu()
        print(output.size())
        score = cosine_similarity(centroid_embeddings.detach().numpy(), output.detach().numpy())
        if score >= thres:
            print('Accept!')
            correct_nums += 1
        else:
            print('Reject!')
    acc = correct_nums / (len(inputs) - entrollment_nums)
    print('the verification accuracy is: ', acc)


    # inputs1 = torch.randn(1,10,40)
    # inputs1 = Variable(inputs1).cuda()
    # outputs1 = embedder_net(inputs1).cpu()
    # print(outputs1.size())
    # inputs2 = torch.randn(1,20,40)
    # inputs2 = Variable(inputs2).cuda()
    # outputs2 = embedder_net(inputs2).cpu()
    # print(outputs2.size())
    # score = cosine_similarity(outputs1.detach().numpy(), outputs2.detach().numpy())
    # print('The score(similarity) is: ', score)
    # thres = 0.80
    # if score >= thres:
    #     print('Accept!')
    # else:
    #     print('Reject!')
    
    
