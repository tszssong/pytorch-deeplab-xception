class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/home/ubuntu/zms/data/coco/'
        elif dataset == 'sweeper':
            return '/home/ubuntu/zms/data/sweeper/'
        elif dataset == 'depthsweeper':
            return '/home/ubuntu/zms/data/sweeper/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
