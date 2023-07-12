from pprint import pprint
import os
import setproctitle

class Config:
    name = 'crackformer'

    gpu_id = '0,1'

    setproctitle.setproctitle("%s" % name)

    # path
    # data_dir = '/mnt/ningbo_nfs_36/wj/DamCrack'
    data_dir = '/nfs/wj/DamCrack'
    checkpoint_path = 'model'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20

    # visdom
    vis_env = 'Crack'
    port = 8097
    vis_train_loss_every = 40
    vis_train_acc_every = 40
    vis_train_img_every = 40
    val_every = 200

    # training
    epoch = 500
<<<<<<< HEAD
    pretrained_model = '/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/model/epoch(5)_acc(0.30-0.98).pth'
=======
    pretrained_model = '/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/model/crack260.pth'
>>>>>>> e9f39ef9011b2c7ec67e08d5ba7393a433da6809
    weight_decay = 0.0000
    lr_decay = 0.1
    lr = 1e-3
    momentum = 0.9
    use_adam = False  # Use Adam optimizer
    train_batch_size = 8
    val_batch_size = 4
    test_batch_size = 4

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
