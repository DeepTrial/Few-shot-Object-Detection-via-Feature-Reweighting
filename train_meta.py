from __future__ import print_function
import sys,os,time,math
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision import transforms

from data import dataset
from scripts.utils import *
from cfg.cfg import parse_cfg, cfg
from models.darknet_meta import Darknet


if len(sys.argv) != 5:
    print('Usage:')
    print('python train.py [datacfg] [darknetcfg] [learnetcfg] [weightfile]')
    exit()


##############################################################################################
#  config for the experiment (hyper parameter / path)
##############################################################################################
# cfg file
datacfg       = sys.argv[1]
darknetcfg    = parse_cfg(sys.argv[2])
learnetcfg    = parse_cfg(sys.argv[3])
weightfile    = sys.argv[4]

# read cfg file
data_options  = read_data_cfg(datacfg)
net_options   = darknetcfg[0]
meta_options  = learnetcfg[0]

# Configure options
cfg.config_data(data_options)
cfg.config_meta(meta_options)
cfg.config_net(net_options)



# Parameters from args-1
metadict      = data_options['meta']
trainlist     = data_options['train']

testlist      = data_options['valid']
backupdir     = cfg.backup#+"+_novel"+str(cfg.novelid)+"_neg"+str(cfg.neg_ratio)#data_options['backup']
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

if not os.path.exists(backupdir):
    os.mkdir(backupdir)
# Parameters from args-2
factor        = 3
batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])/factor
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
eval_interval  = 2000000 # batches
save_interval = 10  # every [save_interval] epoch to save the model

torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

# load Meta-model
model       = Darknet(darknetcfg, learnetcfg)
region_loss = model.loss
model.load_weights(weightfile)
#model.print_network()

# Meta-model parameters
region_loss.seen  = model.seen
processed_batches = 0 if cfg.tuning else model.seen/batch_size
trainlist         = dataset.build_dataset(data_options)
nsamples          = len(trainlist)
init_width        = model.width
init_height       = model.height
init_epoch        = 0 if cfg.tuning else model.seen/nsamples
max_epochs        = int(max_batches*batch_size/nsamples+1)
max_epochs        = int(math.ceil(cfg.max_epoch*1./cfg.repeat)) if cfg.tuning else max_epochs 

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

#print(cfg.repeat, nsamples, max_batches, batch_size)



##############################################################################################
#  adjust learning rate
##############################################################################################

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr


##############################################################################################
#  training method
##############################################################################################

def train(epoch):
    global processed_batches

    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset.listDataset(
                    trainlist,
                    shape=(init_width, init_height),
                    shuffle=False,
                    transform=transforms.Compose([transforms.ToTensor(), ]),
                    train=True,
                    seen=cur_model.seen,
                    batch_size=batch_size,
                    num_workers=num_workers),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


    metaset = dataset.MetaDataset(metafiles=metadict, train=True)
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=metaset.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    metaloader = iter(metaloader)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('Epoch [%d/%d], processed %d samples, lr %f' % (epoch, max_epochs, epoch * len(train_loader.dataset), lr))

    model.train()
    with tqdm(total=train_loader.__len__()) as t:
        loss_log=ParamLog()
        loss_bbox_log=ParamLog()
        loss_cls_log=ParamLog()
        recall_log=ParamLog()
        proposal_log=ParamLog()

        for batch_idx, (data, target) in enumerate(train_loader):
            metax, mask = metaloader.next()

            adjust_learning_rate(optimizer, processed_batches)
            processed_batches = processed_batches + 1

            if use_cuda:
                data = data.cuda()
                metax = metax.cuda()
                mask = mask.cuda()
            #target= target.cuda()

            data, target = Variable(data), Variable(target)
            metax, mask = Variable(metax), Variable(mask)

            optimizer.zero_grad()
            output = model(data, metax, mask)
            region_loss.seen = region_loss.seen + data.data.size(0)
            loss,recall_rate,conf_proposal,loss_bbox,loss_conf,loss_cls = region_loss(output, target)

            loss_log.log(loss.item())
            loss_bbox_log.log(loss_bbox)
            loss_cls_log.log(loss_cls)
            recall_log.log(recall_rate)
            proposal_log.log(conf_proposal)

            t.set_description('Epoch %d' % epoch)
            t.set_postfix(loss=loss_log.show(), loss_bbox=loss_bbox_log.show(),loss_cls=loss_cls_log.show(),recall=recall_log.show(),proposal=proposal_log.show())
            t.update()
            
            loss.backward()
            optimizer.step()

            del loss,output

    if (epoch+1) % cfg.save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

##############################################################################################
#  test method(Very SLOW! DO NOT USE!)
##############################################################################################

# def test(epoch,test_loader,test_metaloader):
#
#     def truths_length(truths):
#         for i in range(50):
#             if truths[i][1] == 0:
#                 return i
#
#     model.eval()
#     if ngpus > 1:
#         cur_model = model.module
#     else:
#         cur_model = model
#
#     num_classes = cur_model.num_classes
#     anchors     = cur_model.anchors
#     num_anchors = cur_model.num_anchors
#     total       = 0.0
#     proposals   = 0.0
#     correct     = 0.0
#
#     _test_metaloader = iter(test_metaloader)
#     with torch.no_grad():
#         for batch_idx, (data, target) in tqdm(enumerate(test_loader),desc="Epoch %d EVAL"%epoch):
#             s1=time.time()
#             metax, mask = _test_metaloader.next()
#             s2=time.time()
#             if use_cuda:
#                 data = data.cuda()
#                 metax = metax.cuda()
#                 mask = mask.cuda()
#
#             data = Variable(data)
#             metax = Variable(metax)
#             mask = Variable(mask)
#             s3=time.time()
#             output = model(data, metax, mask).data
#             s4=time.time()
#             all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
#             target=target.view(batch_size*len(cfg.base_classes),-1,5)
#             s5=time.time()
#             print("data loader: %d  predict: %d convert:%d"%(s2-s1,s4-s3,s5-s4))
#
#             for i in tqdm(range(output.size(0))):
#                 boxes = all_boxes[i]
#                 s1=time.time()
#                 boxes = nms(boxes, nms_thresh)
#                 s2=time.time()
#                 print("nms: %d"%(s2-s1))
#                 #print(output.size(),target.size(),target[i].size())
#                 truths = target[i].view(-1, 5)
#                 num_gts = truths_length(truths)
#
#                 total = total + num_gts
#
#                 for i in range(len(boxes)):
#                     if boxes[i][4] > conf_thresh:
#                         proposals = proposals+1
#                 s3=time.time()
#                 print("bbox: %d"%(s3-s2))
#                 for i in range(num_gts):
#                     box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
#                     best_iou = 0
#                     best_j = -1
#                     for j in range(len(boxes)):
#                         iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
#                         if iou > best_iou:
#                             best_j = j
#                             best_iou = iou
#                     if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
#                         correct = correct+1
#                 s4=time.time()
#             print("nms: %d  bbox: %d nms_gt:%d"%(s2-s1,s3-s2,s4-s3))
#
#     precision = 1.0*correct/(proposals+eps)
#     recall = 1.0*correct/(total+eps)
#     fscore = 2.0*precision*recall/(precision+recall+eps)
#     logging("Epoch %d Precision: %f, Recall: %f, F-Score: %f" % (epoch,precision, recall, fscore))



##############################################################################################
#  data loader
##############################################################################################

# test_loader = torch.utils.data.DataLoader(
#     dataset.listDataset(
#         testlist,
#         shape=(init_width, init_height),
#         shuffle=False,
#         transform=transforms.Compose([transforms.ToTensor(),]),
#         train=False
#     ),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=8,
#     pin_memory=True
# )
#
# test_metaset     =  dataset.MetaDataset(metafiles=metadict, train=True)
# test_metaloader  =  torch.utils.data.DataLoader(
#     test_metaset,
#     batch_size=test_metaset.batch_size,
#     shuffle=False,
#     num_workers=8,
#     pin_memory=True
# )

##############################################################################################
#  MAIN Function part
##############################################################################################

optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate/batch_size,
    momentum=momentum,
    dampening=0,
    weight_decay=decay*batch_size*factor
)

evaluate_or_train = False

if evaluate_or_train:
    logging('evaluating without training ...')
    #test(0,test_loader,test_metaloader)
else:
    init_epoch=int(init_epoch.item()) if type(init_epoch)!=int else init_epoch

    logging('start training from epoch %d'%init_epoch)
    for epoch in range(init_epoch, int(max_epochs)):
        train(epoch)
        # if (epoch+1)%eval_interval==0:
        #     test(epoch,test_loader,test_metaloader)
