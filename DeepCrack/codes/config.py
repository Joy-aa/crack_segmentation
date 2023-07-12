from pprint import pprint
import os
import setproctitle

class Config:
    name = 'DeepCrack_CT260_FT1'

    gpu_id = '0,1'

    setproctitle.setproctitle("%s" % name)

    # path
<<<<<<< HEAD
    train_data_path = '/mnt/ningbo_nfs_36/wj/DamCrack'
    val_data_path = '/mnt/ningbo_nfs_36/wj/DamCrack'
=======
    train_data_path = '/nfs/wj/DamCrack'
    val_data_path = '/nfs/wj/DamCrack'
>>>>>>> e9f39ef9011b2c7ec67e08d5ba7393a433da6809
    checkpoint_path = 'checkpoints'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20

    # visdom
    vis_env = 'DeepCrack'
    port = 8097
    vis_train_loss_every = 40
    vis_train_acc_every = 40
    vis_train_img_every = 40
    val_every = 200

    # training
    epoch = 500
<<<<<<< HEAD
    pretrained_model = 'checkpoints/DeepCrack_CT260_FT1/epoch(1)_acc(0.08527-0.99589).pth'
=======
    pretrained_model = 'checkpoints/DeepCrack_CT260_FT1/epoch(14)_acc(0.12132-0.99598).pth'
>>>>>>> e9f39ef9011b2c7ec67e08d5ba7393a433da6809
    weight_decay = 0.0000
    lr_decay = 0.5
    lr = 1e-4
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
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
