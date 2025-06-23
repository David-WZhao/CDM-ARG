import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import torch.optim as optim
import torch.nn as nn
from argparse import ArgumentParser
from data_loader import ARGDataLoader
from modules import CML
torch.backends.cudnn.enabled = False
from utils import evaluate
import warnings
import random
from random import randint
from protein_bert_pytorch import ProteinBERT
import esm
warnings.filterwarnings("ignore")


parser = ArgumentParser("CML")
# runtime args
parser.add_argument("--device", type=str, help='cpu or gpu', default="cpu")
parser.add_argument("--train_rate", type=float, help='train rate', default=0.8)
parser.add_argument("--batch_size", type=int, help='batch size', default=8)
parser.add_argument("--lr", type=float, help='learning rate', default=1e-4)
parser.add_argument("--epoch", type=int, help='epoch', default=10)
parser.add_argument("--K", type=int, help='K fold', default=1)
parser.add_argument("--X_dim", type=int, help='dimension of X', default=128)
parser.add_argument("--G_dim", type=int, help='dimension of G', default=128)
parser.add_argument("--z_dim", type=int, help='dimension of z', default=256)
parser.add_argument("--EMS_input_dim", type=int, help='dimension of EMS_input_dim', default=1280)




args = parser.parse_args()

#the configuration of args
device = "cuda:0"
if args.device != 'cpu':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

train_rate = args.train_rate
batch_size = args.batch_size
lr = args.lr
epoch = args.epoch
K = args.K
X_dim, G_dim = args.X_dim, args.G_dim
z_dim = args.z_dim
EMS_input_dim = args.EMS_input_dim
print(str(args))
dataloader = ARGDataLoader()
antibiotic_count = 15
mechanism_count  = 6
transfer_count = 2

transfer_count, mechanism_count, antibiotic_count = dataloader.get_data_shape()

alpha, beta, yita, tao= 1, 0.2, 0.2, 0.2

# load ESM-2 model
# ESMmodel, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# ESMmodel.eval()

# the initialization of ProteinBERT
modelProtein = ProteinBERT(
    num_tokens = 23,
    num_annotation=1,
    dim = 512,
    dim_global = 256,
    depth = 6,
    narrow_conv_kernel = 9,
    wide_conv_kernel = 9,
    wide_conv_dilation = 5,
    attn_heads = 8,
    attn_dim_head = 64
).to(device)

#random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_num = randint(1,1000)
setup_seed(seed_num)
# print("707")

# Train  initialization of evaluation metrics
t_transfer_acc, t_transfer_precision, t_transfer_recall, t_transfer_f1 = 0, 0, 0, 0
t_antibiotic_acc, t_antibiotic_precision, t_antibiotic_recall, t_antibiotic_f1 = 0, 0, 0, 0
t_mechanism_acc, t_mechanism_precision, t_mechanism_recall, t_mechanism_f1 = 0, 0, 0, 0

test_dataloader = dataloader.load_test_dataSet(batch_size)
omiga = 0.2
i = 0
for i in range(1):
    yita = yita + 0.2
#Repeat k cross
    for k in range(K):
        print('Cross ', k + 1, ' of ', K)
        #create model
        model = CML(X_dim, G_dim, z_dim, EMS_input_dim, antibiotic_count, mechanism_count, transfer_count)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        transfer_loss_function = nn.NLLLoss()
        antibiotic_loss_function = nn.NLLLoss()
        mechanism_loss_function = nn.NLLLoss()


        # train_dataloader = train_val_dataloader[k]['train']
        # val_dataloader = train_val_dataloader[k]['val']
        #load dataset
        train_dataloader, val_dataloader = dataloader.load_n_cross_data(k + 1, batch_size)

        running_loss = 0.0
        for e in range(epoch):

            df = pd.DataFrame()
            model.train()
            print('train batch: ', len(train_dataloader))
            for index, (seq,seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(train_dataloader):

                seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), \
                    transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)

                # process sequence by ESM
                # new_list = [(str(i + 1), element) for i, element in enumerate(seq)]
                #
                # batch_labels, batch_strs, batch_tokens = batch_converter(new_list)
                # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

                # Extract the representation of each residue
                # with torch.no_grad():
                #     results = ESMmodel(batch_tokens, repr_layers=[33], return_contacts=True)
                # token_representations = results["representations"][33]

                # By averaging the residues of each sequence, a sequence-level representation is generated

                # sequence_representations = []
                # for i, tokens_len in enumerate(batch_lens):
                #     sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

                # print(sequence_representations)
                # exit()
                # process sequence by ProteinBert
                adjusted_input = seq_map.squeeze(1) # 移除维度1 → [8, 1573]
                index_seq_map = torch.argmax(adjusted_input, dim=-1).to(device)  # 形状变为 [8, 1, 1573]

                mask = torch.ones_like(index_seq_map, dtype=torch.bool).to(device)

                annotation = torch.randint(0, 1, (index_seq_map.size(0), 1)).float().to(device)


                seq_logits,_ = modelProtein(index_seq_map, annotation, mask=mask)
                # Add one dimension to dimension 1 (the position with index 1)
                expanded_tensor = seq_logits.unsqueeze(1)
                # stacked_sequence_representations = torch.stack(sequence_representations, dim=0).to(device)

                optimizer.zero_grad()
                antibiotic_output, mechanism_output,transfer_output, diffusion_loss  = model.forward(expanded_tensor, antibiotic_label.view(-1, 1), mechanism_label.view(-1, 1), transfer_label.view(-1, 1),antibiotic_count,
                                                                                                     mechanism_count,transfer_count)
                #loss function
                loss_transfer = transfer_loss_function(torch.log(transfer_output + 0.000001), transfer_label)
                loss_mechanism = mechanism_loss_function(torch.log(mechanism_output + 0.000001), mechanism_label)

                loss_antibiotic = antibiotic_loss_function(torch.log(antibiotic_output + 0.000001), antibiotic_label)

                loss = alpha * loss_antibiotic + beta * loss_mechanism + yita * loss_transfer + tao * diffusion_loss

                # loss = alpha * loss_antibiotic + beta * loss_mechanism + yita * loss_transfer

                # print(add_mean_constraint(batch_mean_list, 0.2))
                # exit()
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()



                df = df._append({'loss_transfer': loss_transfer.item(), 'loss_antibiotic': loss_antibiotic.item(), 'loss_mechanism': loss_mechanism.item(), 'loss': loss.item(), 'running_loss': running_loss}, ignore_index=True)
                if index % 50 == 49:
                    # exit()
                    print('[%d, %2d, %5d] loss: %.3f' % (k + 1, e + 1, index + 1, running_loss / 50))
                    running_loss = 0.0



            df.to_csv('./res/loss_cross' + str(k + 1) + '_epoch' + str(e) + '.csv')
            model.eval()
            val_transfer_pred, val_transfer_label = np.empty(shape=[0, transfer_count]), np.array([])
            val_mechanism_pred, val_mechanism_label = np.empty(shape=[0, mechanism_count]), np.array([])
            val_antibiotic_pred, val_antibiotic_label = np.empty(shape=[0, antibiotic_count]), np.array([])

            for index, (seq, seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(val_dataloader):
                seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)
                antibiotic_output,  mechanism_output, transfer_output,_ = model.forward(seq_map,
                    antibiotic_label.view(-1, 1), mechanism_label.view(-1, 1), transfer_label.view(-1, 1),
                    antibiotic_count, mechanism_count, transfer_count     )

                transfer_output, transfer_label = transfer_output.cpu().detach().numpy(), transfer_label.cpu().numpy()
                val_transfer_pred = np.append(val_transfer_pred, transfer_output, axis=0)
                val_transfer_label = np.concatenate((val_transfer_label, transfer_label))

                antibiotic_output, antibiotic_label = antibiotic_output.cpu().detach().numpy(), antibiotic_label.cpu().numpy()
                val_antibiotic_pred = np.append(val_antibiotic_pred, antibiotic_output, axis=0)
                val_antibiotic_label = np.concatenate((val_antibiotic_label, antibiotic_label))

                mechanism_output, mechanism_label = mechanism_output.cpu().detach().numpy(), mechanism_label.cpu().numpy()
                val_mechanism_pred = np.append(val_mechanism_pred, mechanism_output, axis=0)
                val_mechanism_label = np.concatenate((val_mechanism_label, mechanism_label))

            print('-------------Val: epoch ' + str(e + 1) + '-----------------')
            acc, macro_p, macro_r, macro_f1 = evaluate(val_transfer_pred, val_transfer_label, transfer_count)
            print('transfer -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
            acc, macro_p, macro_r, macro_f1 = evaluate(val_mechanism_pred, val_mechanism_label, mechanism_count)
            print('mechanism -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
            acc, macro_p, macro_r, macro_f1 = evaluate(val_antibiotic_pred, val_antibiotic_label, antibiotic_count)
            print('antibiotic -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))



        # 测试
        model.eval()
        test_transfer_pred, test_transfer_label = np.empty(shape=[0, transfer_count]), np.array([])
        test_antibiotic_pred, test_antibiotic_label = np.empty(shape=[0, antibiotic_count]), np.array([])
        test_mechanism_pred, test_mechanism_label = np.empty(shape=[0, mechanism_count]), np.array([])
        for index, (seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(test_dataloader):
            seq_map, transfer_label, mechanism_label, antibiotic_label = seq_map.view(-1, 1, 1576, 23).to(device), transfer_label.to(device), mechanism_label.to(device), antibiotic_label.to(device)
            # transfer_output, mechanism_output, antibiotic_output, mean_logvar1, mean_logvar2, mean_logvar3,\
            #     mean_logvar4, prob = model.forward(seq_map,
            #     torch.zeros(transfer_label.view(-1,1).shape).to(device),
            #     torch.zeros(transfer_label.view(-1,1).shape).to(device),
            #     torch.zeros(transfer_label.view(-1,1).shape).to(device))
            antibiotic_output, mechanism_output, transfer_output, _ = model.forward(seq_map,
                                                                                    antibiotic_label.view(-1, 1),
                                                                                    mechanism_label.view(-1, 1),
                                                                                    transfer_label.view(-1, 1),
                                                                                    antibiotic_count, mechanism_count,
                                                                                    transfer_count)

            # transfer_output, mechanism_output, antibiotic_output,  mu, logvar, recon_x, x = model.forward(seq_map)
            # transfer_output, mechanism_output, antibiotic_output = model.forward(seq_map)

            transfer_output, transfer_label = transfer_output.cpu().detach().numpy(), transfer_label.cpu().numpy()
            test_transfer_pred = np.append(test_transfer_pred, transfer_output, axis=0)
            test_transfer_label = np.concatenate((test_transfer_label, transfer_label))

            antibiotic_output, antibiotic_label = antibiotic_output.cpu().detach().numpy(), antibiotic_label.cpu().numpy()
            test_antibiotic_pred = np.append(test_antibiotic_pred, antibiotic_output, axis=0)
            test_antibiotic_label = np.concatenate((test_antibiotic_label, antibiotic_label))

            mechanism_output, mechanism_label = mechanism_output.cpu().detach().numpy(), mechanism_label.cpu().numpy()
            test_mechanism_pred = np.append(test_mechanism_pred, mechanism_output, axis=0)
            test_mechanism_label = np.concatenate((test_mechanism_label, mechanism_label))

    #calculate evaluation metrics
        print('========Test: Cross ' + str(k + 1) + '===============')
        acc, macro_p, macro_r, macro_f1 = evaluate(test_transfer_pred, test_transfer_label, transfer_count)
        print('transfer -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        t_transfer_acc += acc
        t_transfer_precision += macro_p
        t_transfer_recall += macro_r
        t_transfer_f1 += macro_f1

        acc, macro_p, macro_r, macro_f1 = evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)
        print('mechanism -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        t_mechanism_acc += acc
        t_mechanism_precision += macro_p
        t_mechanism_recall += macro_r
        t_mechanism_f1 += macro_f1

        acc, macro_p, macro_r, macro_f1 = evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)
        print('antibiotic -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        t_antibiotic_acc += acc
        t_antibiotic_precision += macro_p
        t_antibiotic_recall += macro_r
        t_antibiotic_f1 += macro_f1



        torch.save(model.state_dict(), './res/model{}.pth'.format(k))
        # torch.save(model,'./res/modeltotal.pth')
    print('transfer => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_transfer_acc / K, t_transfer_precision / K,
                                                                                                     t_transfer_recall / K,
                                                                                                     t_transfer_f1 / K))
    print('mechanism => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_mechanism_acc / K, t_mechanism_precision / K,
                                                                                                     t_mechanism_recall / K,
                                                                                                     t_mechanism_f1 / K))
    print('antibiotic => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_antibiotic_acc / K, t_antibiotic_precision / K,
                                                                                                     t_antibiotic_recall / K,
                                                                                                     t_antibiotic_f1 / K))
# write the results to file
with open('./res/result.txt', 'a', encoding='utf8') as f:
    f.write(str(args))
    f.write('\n seed =>{}\n'.format(seed_num))
    f.write('\n transfer => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_transfer_acc / K, t_transfer_precision / K,
                                                                                                         t_transfer_recall / K,
                                                                                                         t_transfer_f1 / K))
    f.write('\n mechanism => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_mechanism_acc / K,
                                                                                                            t_mechanism_precision / K,
                                                                                                            t_mechanism_recall / K,
                                                                                                            t_mechanism_f1 / K))

    f.write('\n antibiotic => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_antibiotic_acc / K,
                                                                                                             t_antibiotic_precision / K,
                                                                                                             t_antibiotic_recall / K,
                                                                                                             t_antibiotic_f1 / K))

    f.write('----------------------------------------------------------------------------------------\n')

# def save_snapshot(model, filename):
#     torch.save(model.state_dict(), filename)
#     f.close()

