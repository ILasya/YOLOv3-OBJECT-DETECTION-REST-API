import torch
import torch.nn as nn
import numpy as np


class Yolov3Layers(nn.Module):
    def __init__(self, mask=[], num_of_classes=0, anchors=[], num_of_anchors=1):
        super(Yolov3Layers, self).__init__()
        self.mask = mask
        self.num_of_classes = num_of_classes
        self.anchors = anchors
        self.num_of_anchors = num_of_anchors
        self.anchor_steps = len(anchors)/num_of_anchors
        self.thresh = 0.6
        self.stride = 32
        self.seen = 0

    def forward(self, output, nms_thresh):
        self.thresh = nms_thresh
        masked_anchors = []

        for m in self.mask:
            masked_anchors += self.anchors[m*self.anchor_steps:(m+1)*self.anchor_steps]

        masked_anchors = [anchor/self.stride for anchor in masked_anchors]
        boxes = get_region_boxes(output.data, self.thresh, self.num_of_classes, masked_anchors, len(self.mask))

        return boxes


class UpSample(nn.Module):
    def __init__(self, stride=2):
        super(UpSample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        b = x.data.size(0)
        c = x.data.size(1)
        h = x.data.size(2)
        w = x.data.size(3)
        x = x.view(b, c, h, 1, w, 1).expand(b, c, h, stride, w, stride).contiguous().view(b, c, h*stride, w*stride)
        return x

# For route and shortcut


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# Support route shortcut


class DarkNet(nn.Module):
    def __init__(self, cfg_file):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.models = self.create_network(self.blocks)
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, nms_thresh):
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []

        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'upsample']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                outputs[ind] = x
            elif block['type'] == 'yolo':
                boxes = self.models[ind](x, nms_thresh)
                out_boxes.append(boxes)
            else:
                print('unknown type %s' % (block['type']))

        return out_boxes

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = 3
        output_filters =[]
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size//2) if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                prev_filters = filters
                output_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                output_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                models.append(UpSample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = output_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = output_filters[layers[0]] + output_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                output_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = output_filters[ind-1]
                output_filters.append(prev_filters)
                prev_stride = out_strides[ind-1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'yolo':
                yolo_layer = Yolov3Layers()
                anchors = block['anchors'].split(',')
                mask = block['mask'].split(',')
                yolo_layer.mask = [int(i) for i in mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_of_classes = int(block['classes'])
                yolo_layer.num_of_anchors = int(block['num'])
                yolo_layer.anchor_steps = len(yolo_layer.anchors)//yolo_layer.num_of_anchors
                yolo_layer.stride = prev_stride
                output_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weight_file):
        print()
        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        counter = 3
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'yolo':
                pass
            else:
                print('unknown type %s' % (block['type']))

            percent_comp = (counter / len(self.blocks)) * 100

            print('Loading weights. Please Wait...{:.2f}% Complete'.format(percent_comp), end='\r', flush=True)

            counter += 1


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(output, conf_thresh, num_of_classes, anchors, num_of_anchors, only_object=1, validation=False):
    anchor_steps = len(anchors)//num_of_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_of_classes)*num_of_anchors)
    h = output.size(2)
    w = output.size(3)

    all_boxes = []
    output = output.view(batch*num_of_anchors, 5+num_of_classes, h*w).transpose(0, 1).contiguous().view(5+num_of_classes, batch*num_of_anchors*h*w)

    grid_x = torch.linspace(0, w-4, w).repeat(h, 1).repeat(batch*num_of_anchors, 1, 1).view(batch*num_of_anchors*h*w).type_as(output)
    grid_y = torch.linspace(0, h-2, h).repeat(w, 1).t().repeat(batch*num_of_anchors, 1, 1).view(batch*num_of_anchors*h*w).type_as(output)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_of_anchors, anchor_steps).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_of_anchors, anchor_steps).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_of_anchors*h*w).type_as(output)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_of_anchors*h*w).type_as(output)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5+num_of_classes].transpose(0,1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)


    sz_hw = h*w
    sz_hwa = sz_hw*num_of_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_of_classes))

    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_of_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_object:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_object) and validation:
                            for c in range(num_of_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes


def parse_cfg(cfg_file):
    blocks = []
    fp = open(cfg_file, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)); start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)); start = start + num_w
    return start