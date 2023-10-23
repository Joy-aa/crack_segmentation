from pprint import pprint
import os
import setproctitle

class Config:
    name = 'crackformer'

    gpu_id = '0,1'

    setproctitle.setproctitle("%s" % name)

    # path
    # data_dir = '/nfs/wj/192_255_segmentation'
    data_dir = '/nfs/wj/CrackLS315/'
    checkpoint_path = '/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/result/crackls/checkpoints'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20

    # visdom
    vis_env = 'Crack'
    port = 8097
    vis_train_loss_every = 200
    vis_train_acc_every = 200
    vis_train_img_every = 200
    val_every = 400

    # training
    epoch = 20
    pretrained_model = ''
    weight_decay = 0.0000
    lr_decay = 0.1
    lr = 1e-3
    momentum = 0.9
    use_adam = False  # Use Adam optimizer
    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 1

    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1

    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
