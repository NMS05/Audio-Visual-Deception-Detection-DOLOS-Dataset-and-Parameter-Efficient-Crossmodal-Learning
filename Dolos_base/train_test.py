import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.audio_visual_dataset import AudioVisualDataset, af_collate_fn

from models.audio_model import W2V2_Model
from models.fusion_model import Fusion
# pip install torch metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc

from models.visual_model import ViT_model


def parse_options():
    parser = argparse.ArgumentParser(description="WILTY baseline repo")
    ##### TRAINING DYNAMICS

    parser.add_argument('--device', type=str, default="cuda:0", help='the gpu id used for predict')
    parser.add_argument('--device_ID', type=str, default="cuda:0", help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')  # default=0.001
    parser.add_argument('--when', type=int, default=15,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16, help='initial batchsize')
    parser.add_argument('--num_epochs', type=int, default=20, help='total training epochs')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--multi', action='store_true', help="multitask learning with multiple losses")
    ##### UNIMODAL ARCHITECTURE

    parser.add_argument('--num_encoders', type=int, default=4, help="number of transformer encoders for each modality")
    parser.add_argument('--adapter', action='store_true', help="indicator of using adapter")
    parser.add_argument('--adapter_type', type=str, default='efficient_conv',
                        help='indicate which type of adapter to use - full_conv or efficient_conv or nlp')
    ##### DATA

    parser.add_argument('--data_root', type=str, default='/home/nithish/WILTY/data/')
    parser.add_argument('--audio_path', type=str, default='/home/nithish/WILTY/data/audio_files/',
                        help='audio dataset root dir')
    parser.add_argument('--visual_path', type=str, default='/home/nithish/WILTY/data/face_frames/',
                        help='visual (FACE) dataset root dir')

    #### TRAINING LOGS

    parser.add_argument('--log', type=str, default="logs", help='log and save model name')
    parser.add_argument('--protocols', type=list,
                        default=[['train_fold3.csv', 'test_fold3.csv']],
                        help='protocols for train/test')

    """
    parser.add_argument('--protocols', type=list,
                        default=[['train_file_1.csv', 'test_file_1.csv'], ['train_file_2.csv', 'test_file_2.csv'],
                                 ['train_file_3.csv', 'test_file_3.csv'], ['train_file_4.csv', 'test_file_4.csv'],
                                 ['train_file_5.csv', 'test_file_5.csv']],
                        help='protocols for train/test')                                 
    ['train_fold1.csv', 'test_fold1.csv'], ['train_fold2.csv', 'test_fold2.csv'],
                                 ['train_fold3.csv', 'test_fold3.csv'], ['long.csv', 'short.csv'],
                                 ['short.csv', 'long.csv'],
                                 ['male.csv', 'female.csv'], ['female.csv', 'male.csv']
                                 
    ['WILTY.csv', 'bag_of_lies.csv'], ['WILTY.csv', 'RLT_annotations.csv'], 
    ['bol_audio_visual.csv', 'bag_of_lies.csv'], ['bol_audio_visual.csv', 'RLT_annotations.csv']
    """
    parser.add_argument('--full_annotation', type=str,
                        default='/DOLOS/data/protocols/WILTY_binary.csv')

    parser.add_argument('--model_name', type=str, default='DOLOS_')
    parser.add_argument('--model_to_train', type=str, default='fusion',
                        help='specify vision or audio or fusion- indiciates the model to be trained')
    parser.add_argument('--fusion_type', type=str, default='cross2', help='modality fusion type')

    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    opts.device = torch.device(opts.device_ID)
    # if "fusion" in opts.model_to_train:
    #     opts.model_to_train = opts.model_to_train + "_" + opts.fusion_type

    if opts.adapter:
        opts.model_name = opts.model_name + opts.model_to_train + "_Encoders_" + str(
            opts.num_encoders) + "_Adapter_" + str(
            opts.adapter) + "_type_" + str(opts.adapter_type)
    else:
        opts.model_name = opts.model_name + opts.model_to_train + "_Encoders_" + str(
            opts.num_encoders) + "_Adapter_" + str(
            opts.adapter)
    if not os.path.exists(opts.log):
        os.makedirs(opts.log)

    return opts


def train_one_epoch(args, train_data_loader, model, optimizer, loss_fn, loss_audio, loss_vision, loss_multi):
    ### Local Parameters
    epoch_loss = []
    epoch_predictions = []
    epoch_predictions_multi = []
    epoch_labels = []
    epoch_labels_multi = []
    start_time = time.time()

    model.train()

    if args.model_to_train == "audio":

        ###Iterating over data loader
        for i, (waves, _, labels) in enumerate(train_data_loader):
            # required only for audio with shape [bs, 1 , sample length] ; does not affect face frames tensor
            waves = waves.squeeze(1)

            # Loading waves and labels to device
            waves = waves.to(args.device)
            labels = labels.to(args.device)

            # Reseting Gradients
            optimizer.zero_grad()

            # Forward
            preds = model(waves)

            # Calculating Loss
            _loss = loss_fn(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Backward
            _loss.backward()
            optimizer.step()

            epoch_predictions.append(torch.argmax(preds, dim=1))
            epoch_labels.append(labels)

            if i % 10 == 0:
                print("iter {}, loss {:.5f}".format(str(i), loss))
    elif args.model_to_train == "vision":

        ###Iterating over data loader
        for i, (_, faces, labels) in enumerate(train_data_loader):
            faces = faces.to(args.device)
            labels = labels.to(args.device)

            # Reseting Gradients
            optimizer.zero_grad()

            # Forward
            preds = model(faces)

            # Calculating Loss
            _loss = loss_fn(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Backward
            _loss.backward()
            optimizer.step()

            epoch_predictions.append(torch.argmax(preds, dim=1))
            epoch_labels.append(labels)

            if i % 10 == 0:
                print("iter {}, loss {:.5f}".format(str(i), loss))

    else:  # fusion models
        for i, (waves, faces, labels) in enumerate(train_data_loader):
            waves = waves.squeeze(1)

            # Loading waves and labels to device
            waves = waves.to(args.device)
            faces = faces.to(args.device)
            labels = labels.to(args.device)

            # Reseting Gradients
            optimizer.zero_grad()

            # Forward
            preds, a_preds, v_preds = model(waves, faces)

            # Calculating Loss
            _loss = loss_fn(preds, labels)

            if args.multi:
                _loss_a = loss_audio(a_preds, labels)
                _loss_v = loss_vision(v_preds, labels)
                _loss = _loss + _loss_a
                _loss = _loss + _loss_v
            loss = _loss.item()
            epoch_loss.append(loss)

            # Backward
            _loss.backward()
            optimizer.step()

            epoch_predictions.append(torch.argmax(preds, dim=1))
            epoch_labels.append(labels)

            if i % 10 == 0:
                print("iter {}, loss {:.5f}".format(str(i), loss))

    # CALCULATE ACCURACY AND CONFUSION MATRIX
    epoch_predictions = torch.cat(epoch_predictions)
    epoch_labels = torch.cat(epoch_labels)

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    # Acc and Loss
    epoch_loss = np.mean(epoch_loss)

    # return epoch_loss, acc, cfm, f1
    return epoch_loss, epoch_predictions, epoch_labels, None, None


def val_one_epoch(args, val_data_loader, model, loss_fn, loss_audio, loss_vision, loss_multi):
    #  Local Parameters
    epoch_loss = []
    epoch_predictions = []
    epoch_predictions_multi = []
    epoch_labels = []
    epoch_labels_multi = []
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        # Iterating over data loader
        if args.model_to_train == "audio":
            for waves, _, labels in val_data_loader:
                waves = waves.squeeze(1)

                # Loading waves and labels to device
                waves = waves.to(args.device)
                labels = labels.to(args.device)

                # Forward
                preds = model(waves)

                # Calculating Loss
                _loss = loss_fn(preds, labels)
                loss = _loss.item()
                epoch_loss.append(loss)

                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)
        elif args.model_to_train == "vision":
            for _, faces, labels in val_data_loader:
                # Loading waves and labels to device
                faces = faces.to(args.device)
                labels = labels.to(args.device)

                # Forward
                preds = model(faces)

                # Calculating Loss
                _loss = loss_fn(preds, labels)
                loss = _loss.item()
                epoch_loss.append(loss)

                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)

        else:
            for waves, faces, labels in val_data_loader:
                waves = waves.squeeze(1)

                # Loading waves and labels to device
                waves = waves.to(args.device)
                faces = faces.to(args.device)
                labels = labels.to(args.device)

                # Forward
                preds, a_preds, v_preds = model(waves, faces)

                # Calculating Loss
                _loss = loss_fn(preds, labels)
                if args.multi:
                    _loss_a = loss_audio(a_preds, labels)
                    _loss_v = loss_vision(v_preds, labels)
                    _loss = _loss + _loss_a
                    _loss = _loss + _loss_v
                loss = _loss.item()
                epoch_loss.append(loss)

                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)

    # CALCULATE ACCURACY AND CONFUSION MATRIX
    epoch_predictions = torch.cat(epoch_predictions)
    epoch_labels = torch.cat(epoch_labels)

    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    # return epoch_loss, acc, cfm, f1
    return epoch_loss, epoch_predictions, epoch_labels, None, None


def evaluation(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score


def evaluation_multi(labels, preds, labels2, preds2):
    acc, f1, auc_score = evaluation(labels, preds)

    acc2, f1_2, auc_2 = {}, {}, {}
    for i in range(labels2.shape[1]):
        acc2[i] = accuracy_score(labels2[:, i], preds2[:, i])
        f1_2[i] = f1_score(labels2[:, i], preds2[:, i])
        fpr, tpr, thresholds = roc_curve(labels2[:, i], preds2[:, i], pos_label=1)
        auc_2[i] = auc(fpr, tpr)
    return acc, f1, auc_score, acc2, f1_2, auc_2


def train_test(log_name, args):
    f = open(log_name, 'a')

    for P in args.protocols:
        # create log file for each run
        # device = torch.device(args.device)

        print("\n\nCurrent protocol.....................", P)
        train, test = P

        f.write("\n\nTrain file = " + train.split('.')[0])
        f.write("\nTest file = " + test.split('.')[0])

        train_anno = args.data_root + 'protocols/' + train
        test_anno = args.data_root + 'protocols/' + test

        if args.model_to_train in ['audio', 'vision', 'fusion', ]:
            train_dataset = AudioVisualDataset(train_anno, args.audio_path, args.visual_path)
            test_dataset = AudioVisualDataset(test_anno, args.audio_path, args.visual_path)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=af_collate_fn,
                                      num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=af_collate_fn,
                                     num_workers=4)

        else:
            raise Exception(" please indicate which model to train: audio, vision or fusion ...")

        print("\t Dataset Loaded")

        if args.model_to_train == 'audio':
            model = W2V2_Model(args.num_encoders, args.adapter, args.adapter_type)
            model.to(args.device)
        elif args.model_to_train == 'vision':
            model = ViT_model(args.num_encoders, args.adapter, args.adapter_type)
            model.to(args.device)
        else:
            model = Fusion(args.fusion_type, args.num_encoders, args.adapter,
                           args.adapter_type, args.multi)
            model.to(args.device)
        print("\t Model Loaded")

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Loss Function
        loss_fn = nn.CrossEntropyLoss()
        if args.multi:
            loss_audio = nn.CrossEntropyLoss()
            loss_vision = nn.CrossEntropyLoss()
            loss_multi = None
        else:
            loss_audio = loss_vision = loss_multi = None

        best_acc = 0.0
        best_mae = 100.0
        results = ""
        val_results = ""
        multi_task_result = ""

        print("\t Started Training")
        if args.multi:
            print("\t Multitask learning")
        for epoch in range(args.num_epochs):

            if (epoch + 1) % 5 == 0:
                print('\t\t Epoch....', epoch + 1)

            # Training
            loss, preds, lables, preds_multi, lables_multi = train_one_epoch(args, train_loader, model, optimizer,
                                                                             loss_fn, loss_audio, loss_vision,
                                                                             loss_multi)

            train_acc, train_f1, train_auc = evaluation(lables.detach().cpu().numpy(), preds.detach().cpu().numpy())
            # Validation
            val_loss, val_preds, val_lables, val_preds_multi, val_lables_multi = val_one_epoch(args, test_loader, model,
                                                                                               loss_fn, loss_audio,
                                                                                               loss_vision,
                                                                                               loss_multi)

            val_acc, val_f1, val_auc = evaluation(val_lables.detach().cpu().numpy(),
                                                  val_preds.detach().cpu().numpy())

            print("epoch {}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} "
                  "test_acc {:.5f}, test_f1: {:.5f}, test_auc:{:.5f}".format(epoch, train_acc, train_f1, train_auc,
                                                                             val_acc, val_f1, val_auc))

            # log the results for each epoch
            f.write(
                "epoch {}, test_acc {:.5f}, test_f1: {:.5f}, test_auc:{:.5f}".format(epoch, val_acc, val_f1, val_auc))
            f.write("\n")

            if val_acc > best_acc:
                best_acc = val_acc
                results = "best results are acc {:.5f}, f1: {:.5f}, auc:{:.5f} ".format(val_acc, val_f1, val_auc)
                val_results = classification_report(val_lables.cpu().numpy(), val_preds.cpu().numpy(),
                                                    target_names=["truth", "deception"])
        print("results:\n\n")
        print(results)
        f.write("****************\n")
        f.write(results)
        f.write("\n\n")
        f.write(val_results)
        f.write("\n\n")
        f.write(multi_task_result)

    f.close()


if __name__ == "__main__":
    opts = parse_options()

    # For every run create an new entry and register the hyperparameters
    # num_existing_files = len(os.listdir(opts.log))
    log_name = os.path.join(opts.log, str(time.time()) + '_' + opts.model_name + '.txt')
    with open(log_name, 'w') as f:
        f.write("\nOptimizer and LR = Adam, " + str(opts.lr))
        f.write("\nBatch Size = " + str(opts.batch_size))
        f.write("\nEpochs = " + str(opts.num_epochs))
        f.write("\nNum Encoders = " + str(opts.num_encoders))
        f.write("\nAdapter = " + str(opts.adapter))
        if opts.adapter == True:
            f.write("\nAdapter Type = " + opts.adapter_type)
        f.write("\n------------------------------------------")
        f.write("\n------------------------------------------")
    f.close()

    train_test(log_name, args=opts)
